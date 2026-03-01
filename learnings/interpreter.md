# VLMEvalKit API Scheduling & Dispatching — Q&A

---

## Q1: How does VLMEvalKit handle API scheduling, retry mechanisms, and payload serialization for MLLMs?

### 1. Concurrency & Scheduling

The framework uses a **two-tier concurrency model**: distributed data sharding across GPU nodes, with a thread pool within each node for concurrent API calls.

#### Core Thread Pool — `vlmeval/utils/mp_util.py`

```python
def track_progress_rich(func, tasks, nproc=1, save=None, keys=None, **kwargs):
    with ThreadPoolExecutor(max_workers=nproc) as executor:
        futures = []
        for inputs in tasks:
            if isinstance(inputs, dict):
                future = executor.submit(func, **inputs)
            else:
                future = executor.submit(func, *inputs)
            futures.append(future)

        unfinished = set(range(len(tasks)))
        while len(unfinished):
            new_finished = set()
            for idx in unfinished:
                if futures[idx].done():
                    results[idx] = futures[idx].result()
                    new_finished.add(idx)
            if len(new_finished):
                if save is not None:
                    dump(res, save)          # incremental save with portalocker
            time.sleep(0.1)                  # 100ms polling interval
```

**Key characteristics:**

- `nproc` is controlled by `--api_nproc` CLI arg (default **4**) in `run.py:136`
- All futures are submitted immediately; a polling loop with `time.sleep(0.1)` harvests completions
- Results are **incrementally persisted** to a pickle file using `portalocker` to avoid race conditions

#### Wiring into Inference — `vlmeval/inference.py:64`, `inference_video.py:45`, `inference_mt.py:68`

All three inference entry points call identically:

```python
def infer_data_api(model, work_dir, model_name, dataset, api_nproc=4, ...):
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]
    track_progress_rich(
        model.generate, structs,
        nproc=api_nproc, save=out_file, keys=indices)
```

#### Async Alternative — `vlmeval/api/gpt_asyscio.py`

A fully async engine using `aiohttp` + `asyncio` exists for OpenAI-compatible APIs:

```python
async def generate_batch(self, batch_inputs, **kwargs):
    tasks = [asyncio.create_task(self.generate(inp, **kwargs))
             for inp in batch_inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

#### Multi-Node Sharding — `run.py:184–190`, `vlmeval/smp/misc.py:110–113`

```python
rank, world_size = get_rank_and_world_size()  # reads RANK, WORLD_SIZE env vars
sheet_indices = list(range(rank, len(dataset), world_size))
dist.init_process_group(backend='nccl', timeout=timedelta(seconds=3600))
dist.barrier()  # sync before result merging
```

---

### 2. Retry Mechanism & Fault Tolerance

#### Base Retry Loop — `vlmeval/api/base.py:216–266`

All providers inherit from `BaseAPI`. The retry strategy is **randomized constant-window backoff** (not exponential):

```python
def generate(self, message, **kwargs1):
    T = rd.random() * 0.5          # [0, 0.5s] initial jitter to desynchronize threads
    time.sleep(T)

    for i in range(self.retry):    # default: 10 attempts
        try:
            ret_code, answer, log = self.generate_inner(message, **kwargs)
            if ret_code == 0 and self.fail_msg not in answer and answer != '':
                return answer
        except Exception as err:
            self.logger.error(f'{type(err)}: {err}')

        T = rd.random() * self.wait * 2   # [0, 2*wait] seconds random delay
        time.sleep(T)

    return self.fail_msg if answer in ['', None] else answer
```

**Retry triggers:** `ret_code != 0`, `fail_msg` in answer, empty answer, or any uncaught exception.

#### Per-Provider Defaults

| Provider | File | `retry` | `wait` | `timeout` |
|---|---|---|---|---|
| BaseAPI | `api/base.py` | 10 | 3s | none |
| OpenAI/GPT | `api/gpt.py` | 5 | 5s | 60s (x1.1) |
| Gemini | `api/gemini.py` | 5 | 5s | SDK default |
| Claude | `api/claude.py` | 10 | 3s | **none (can hang!)** |

#### Context-Window Overflow Fallback — `base.py:141–153`

```python
def chat_inner(self, inputs, **kwargs):
    while len(inputs):
        try:
            return self.generate_inner(inputs, **kwargs)
        except Exception:
            inputs = inputs[1:]                      # strip oldest turn
            while len(inputs) and inputs[0]['role'] != 'user':
                inputs = inputs[1:]                  # ensure starts with 'user'
    return -1, self.fail_msg + ': Failed with all possible conversation turns.', None
```

#### HTTP 429 Handling

There is **no explicit 429 detection** — rate limits are treated identically to any other non-2xx status. No `Retry-After` header inspection. The general retry loop with random `[0, 2*wait]` sleep handles it.

#### Async Retry — `gpt_asyscio.py:360–386`

```python
async def generate(self, inputs, **kwargs):
    for attempt in range(self.retry + 1):
        status, answer, data = await self._send_single_request(input_msgs, **kwargs)
        if 200 <= status < 300:
            return status, answer, data
        await asyncio.sleep(self.wait)   # non-blocking sleep
    return 500, f"Failed after {self.retry + 1} attempts.", None
```

Catches `asyncio.TimeoutError` -> 408, `aiohttp.ClientConnectorError` -> 503.

---

### 3. Payload Serialization

#### Canonical Internal Format — `base.py:104–138`

All providers share one internal representation, normalized by `preproc_content`:

```python
[
    {'type': 'text',  'value': 'What is in this image?'},
    {'type': 'image', 'value': '/path/to/image.jpg'},
    {'type': 'text',  'value': 'And this one?'},
    {'type': 'image', 'value': '/path/to/image2.png'},
]
```

#### OpenAI / GPT-4V — `api/gpt.py:158–176`

Images -> **base64 data URI** (resized to `img_size=512`):

```python
b64 = encode_image_to_base64(img, target_size=512)
content_list.append({
    "type": "image_url",
    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}
})
```

#### Claude — `api/claude.py:62–87`

Images -> **raw base64** with explicit `media_type`, resized to `target_size=4096`:

```python
content_list.append({
    "type": "image",
    "source": {
        "type": "base64",
        "media_type": mimetypes.types_map[suffix],
        "data": encode_image_file_to_base64(pth, target_size=4096)
    }
})
```

#### Gemini — `api/gemini.py:49–66`

Images -> **native Python objects** (no manual base64):

```python
# genai backend
messages = ["What is in this image?", Image.open('/path/to/img.jpg')]

# vertex backend
messages = ["What is in this image?", Part.from_image(Image.load_from_file(path))]
```

#### Qwen2 VL — `api/qwen_vl_api.py:62–85`

Images -> **`file://` URIs** with optional pixel constraints:

```python
item = {
    'type': 'image',
    'image': 'file:///path/to/image.jpg',
    'min_pixels': 78400,
    'max_pixels': 1003520
}
```

#### Provider Comparison

| | Image Encoding | Type Key | Message Structure | System Prompt |
|---|---|---|---|---|
| **OpenAI** | `data:image/jpeg;base64,...` | `image_url` | `{role, content: [...]}` | `{role: "system"}` msg |
| **Claude** | Raw base64 + `media_type` | `image` | `{role, content: [...]}` | Top-level `"system"` key |
| **Gemini** | PIL/Part objects (SDK) | none | Flat list `[str, Image, ...]` | Prepended string |
| **Qwen2** | `file://` URI | `image` | `{role, content: [...]}` | `{role: "system"}` msg |

#### Key Portable Patterns

1. **Thread pool with futures polling** (`mp_util.py`) — drop-in concurrent dispatcher with incremental save
2. **Jittered retry with sentinel detection** (`base.py`) — initial random delay + per-retry random sleep prevents thundering herd
3. **Unified internal representation** -> per-provider `prepare_itlist` — clean adapter pattern for multi-provider support
4. **Context-window fallback** (`chat_inner`) — progressive turn-stripping for graceful degradation

---

## Q2: For OpenAI, Claude, Gemini, Qwen2, which models are used?

### Default Models Per Provider

| Provider | Class | Default Model | File : Line |
|---|---|---|---|
| **OpenAI** | `OpenAIWrapper` | `gpt-3.5-turbo-0613` | `gpt.py:41` |
| **Claude** | `Claude_Wrapper` | `claude-3-opus-20240229` | `claude.py:27` |
| **Gemini** | `GeminiWrapper` | `gemini-1.0-pro` | `gemini.py:12` |
| **Qwen2 VL** | `Qwen2VLAPI` | `qwen-vl-max-0809` | `qwen_vl_api.py:25` |
| **Qwen VL (old)** | `QwenVLWrapper` | `qwen-vl-plus` | `qwen_vl_api.py:125` |

### OpenAI Context Window Map (`gpt.py:16–29`)

| Model | Context Length |
|---|---|
| `gpt-4` | 8192 |
| `gpt-4-0613` | 8192 |
| `gpt-4-turbo-preview` | 128000 |
| `gpt-4-1106-preview` | 128000 |
| `gpt-4-0125-preview` | 128000 |
| `gpt-4-vision-preview` | 128000 |
| `gpt-4-turbo` | 128000 |
| `gpt-4-turbo-2024-04-09` | 128000 |
| `gpt-3.5-turbo` | 16385 |
| `gpt-3.5-turbo-0125` | 16385 |
| `gpt-3.5-turbo-1106` | 16385 |
| `gpt-3.5-turbo-instruct` | 4096 |

### Third-Party Models Routed Through OpenAI Wrapper (`gpt.py:63–84`)

| Model Prefix | API Key Env Var | Provider |
|---|---|---|
| `step` | `STEPAI_API_KEY` | Step AI |
| `yi-vision` | `YI_API_KEY` | Yi-Vision |
| `internvl2-pro` | `InternVL2_PRO_KEY` | InternVL2-Pro |
| `abab` | `MiniMax_API_KEY` | MiniMax ABAB |
| `doubao` | `DOUBAO_API_KEY` | Doubao (`doubao-1-5-vision-pro-32k-250115` hardcoded) |
| `qwen` | `DASHSCOPE_API_KEY` | Qwen via OpenAI-compatible endpoint |
| `llava` | `LOCAL_API_KEY` | Local LLaVA server |

### Qwen VL Allowed Models (`qwen_vl_api.py:136`)

The older `QwenVLWrapper` strictly enforces only two models:

```python
assert model in ['qwen-vl-plus', 'qwen-vl-max']
```

### Gemini Vertex Remapping (`gemini.py:98`)

On the Vertex backend, `gemini-1.0-pro` is automatically remapped to `gemini-1.0-pro-vision` for multimodal inputs:

```python
model_name = 'gemini-1.0-pro-vision' if self.model == 'gemini-1.0-pro' else self.model
```

### Notes

- **OpenAI** has the richest model list, with context window sizes explicitly mapped (4K–128K). The default is the older `gpt-3.5-turbo-0613`, but any model string can be passed.
- **Claude** defaults to `claude-3-opus-20240229` and uses two backends: Alles proxy (`openxlab.org.cn`) or official Anthropic API (`api.anthropic.com`).
- **Gemini** defaults to `gemini-1.0-pro`. On the Vertex backend, `gemini-1.0-pro` is automatically remapped to `gemini-1.0-pro-vision` for multimodal inputs.
- **Qwen2 VL** (new API) defaults to `qwen-vl-max-0809`. The older `QwenVLWrapper` strictly enforces only `qwen-vl-plus` or `qwen-vl-max`.
