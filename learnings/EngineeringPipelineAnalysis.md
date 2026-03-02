# U2-BENCH (VLMEvalKit) — Complete Engineering Pipeline Analysis

---

## 1. API Construction & Payload Serialization

### 1.1 Endpoint Construction

#### OpenAI / GPT-4V — `vlmeval/api/gpt.py`

**Static URL registry** (lines 6–12):

```python
APIBASES = {
    # 'OFFICIAL': 'https://api.openai.com/v1/chat/completions',   # commented out!
    'XIAOHU': "https://xiaohumini.site/v1/chat/completions",
    'HUOSHAN': "https://ark.cn-beijing.volces.com/api/v3",
    'ALI':    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
    'LOCAL':  'http://0.0.0.0:2333/v1/chat/completions'
}
```

**URL resolution priority** (lines 129–151):

1. Explicit `api_base` constructor arg (if starts with `http`, used verbatim)
2. `'doubao'` in model → `DOUBAO_API_BASE` env var
3. `'qwen'` in model → `DASHSCOPE_API_BASE` env var
4. `'llava'` in model → `LOCAL_API_BASE` env var
5. Generic proxy: `OPENAI_API_BASE` env var (if non-empty)
6. Falls back to `'OFFICIAL'` string → **currently broken** (commented out)

**Azure URL assembly** (lines 113–128):

```python
api_base_template = (
    '{endpoint}openai/deployments/{deployment_name}/chat/completions?api-version={api_version}'
)
self.api_base = api_base_template.format(
    endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
    deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
    api_version=os.getenv('OPENAI_API_VERSION')
)
```

**HTTP headers by variant** (lines 211–221):

```python
if self.use_azure:
    headers = {'Content-Type': 'application/json', 'api-key': self.key}
elif 'internvl2-pro' in self.model:
    headers = {'Content-Type': 'application/json', 'Authorization': self.key}
elif 'doubao' in self.model:
    headers = {'Content-Type': 'application/json', 'Authorization': f'Bearer {self.key}'}
else:
    headers = {'Accept': 'application/json',
               'Content-Type': 'application/json',
               'Authorization': f'Bearer {self.key}'}
```

#### Claude — `vlmeval/api/claude.py`

Two fixed URLs (lines 8–18):

```python
alles_url = 'https://openxlab.org.cn/gw/alles-apin-hub/v1/claude/v1/text/chat'
alles_headers = {'alles-apin-token': '', 'Content-Type': 'application/json'}

official_url = 'https://api.anthropic.com/v1/messages'
official_headers = {'x-api-key': '', 'anthropic-version': '2023-06-01', 'content-type': 'application/json'}
```

Backend selection (lines 37–39): env var `ANTHROPIC_BACKEND=official` overrides default `'alles'`.

#### Gemini — `vlmeval/api/gemini.py`

**No explicit HTTP URL** — uses SDK-driven backends (lines 11–47):

- `genai`: `google.generativeai` SDK; key from `GOOGLE_API_KEY`
- `vertex`: `vertexai` SDK; GCP ADC (no key); location hard-coded to `us-central1`

Backend override: `GOOGLE_API_BACKEND` env var.

#### Qwen2 VL — `vlmeval/api/qwen_vl_api.py`

**No explicit HTTP URL** — uses `dashscope.MultiModalConversation.call()` SDK (lines 23–60). Key from `DASHSCOPE_API_KEY`.

---

### 1.2 Payload Serialization — Interleaved Multi-Modal Data

#### Canonical Internal Format — `vlmeval/api/base.py:104–138`

All providers first normalize inputs via `preproc_content`:

```python
[
    {'type': 'text',  'value': 'What is in this image?'},
    {'type': 'image', 'value': '/path/to/ultrasound.jpg'},
    {'type': 'text',  'value': 'Describe the findings.'},
    {'type': 'image', 'value': '/path/to/ultrasound2.png'},
]
```

#### OpenAI — `gpt.py:158–191`

Images → **base64 data URI** (resized to `img_size=512`):

```python
def prepare_itlist(self, inputs):
    content_list = []
    for msg in inputs:
        if msg['type'] == 'text':
            content_list.append(dict(type='text', text=msg['value']))
        elif msg['type'] == 'image':
            img = Image.open(msg['value'])
            b64 = encode_image_to_base64(img, target_size=self.img_size)
            img_struct = dict(url=f'data:image/jpeg;base64,{b64}', detail=self.img_detail)
            content_list.append(dict(type='image_url', image_url=img_struct))
    return content_list
```

Final wire payload:

```python
payload = dict(model=self.model, messages=input_msgs, max_tokens=max_tokens, n=1, temperature=temperature)
response = requests.post(self.api_base, headers=headers, data=json.dumps(payload), timeout=self.timeout * 1.1)
```

#### Claude — `claude.py:62–124`

Images → **raw base64** with explicit `media_type`, resized to `target_size=4096`:

```python
content_list.append(dict(
    type='image',
    source={
        'type': 'base64',
        'media_type': mimetypes.types_map.get(suffix, None),  # e.g. 'image/jpeg'
        'data': encode_image_file_to_base64(pth, target_size=4096)
    }))
```

System prompt goes as a **top-level key** (not a message):

```python
payload = {'model': self.model, 'max_tokens': self.max_tokens, 'messages': self.prepare_inputs(inputs)}
if self.system_prompt is not None:
    payload['system'] = self.system_prompt
```

#### Gemini — `gemini.py:49–66`

Images → **native Python objects** (no manual base64, SDK handles encoding):

```python
# genai backend
messages = ["What is in this image?", Image.open('/path/to/img.jpg')]

# vertex backend
messages = ["What is in this image?", Part.from_image(Image.load_from_file(path))]
```

#### Qwen2 VL — `qwen_vl_api.py:62–85`

Images → **`file://` URIs** with optional pixel constraints:

```python
def _prepare_content(self, inputs, dataset=None):
    for s in inputs:
        if s['type'] == 'image':
            item = {'type': 'image', 'image': ensure_image_url(s['value'])}
            if dataset == 'OCRBench':
                item['min_pixels'] = 10 * 10 * 28 * 28   # 78400
            else:
                if self.min_pixels is not None:
                    item['min_pixels'] = self.min_pixels
                if self.max_pixels is not None:
                    item['max_pixels'] = self.max_pixels
```

`ensure_image_url` (lines 11–17): local paths get `file://` prepended; remote URLs pass through.

### Provider Comparison Table

| | Image Encoding | Resize Target | System Prompt | Message Structure |
|---|---|---|---|---|
| **OpenAI** | `data:image/jpeg;base64,...` | 512px | `{role: "system"}` msg | `{role, content: [...]}` |
| **Claude** | Raw base64 + `media_type` | 4096px | Top-level `"system"` key | `{role, content: [...]}` |
| **Gemini** | PIL/Part objects (SDK) | None | Prepended string at idx 0 | Flat list `[str, Image]` |
| **Qwen2** | `file://` URI | None (pixel budget) | `{role: "system"}` msg | `{role, content: [...]}` |

---

## 2. Concurrency & Scheduling

### Thread Pool Dispatcher — `vlmeval/utils/mp_util.py`

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

        # Polling loop harvests completions
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

**Wiring** — `vlmeval/inference.py:64`, `inference_video.py:45`, `inference_mt.py:68`:

```python
def infer_data_api(model, work_dir, model_name, dataset, api_nproc=4, ...):
    structs = [dict(message=struct, dataset=dataset_name) for struct in structs]
    track_progress_rich(model.generate, structs, nproc=api_nproc, save=out_file, keys=indices)
```

`nproc` controlled by `--api_nproc` CLI arg (default **4**, set in `run.py:136`).

### Async Alternative — `vlmeval/api/gpt_asyscio.py`

```python
async def generate_batch(self, batch_inputs, **kwargs):
    tasks = [asyncio.create_task(self.generate(inp, **kwargs)) for inp in batch_inputs]
    results = await asyncio.gather(*tasks, return_exceptions=True)
```

### Multi-Node Sharding — `run.py:184–190`

```python
rank, world_size = get_rank_and_world_size()  # reads RANK, WORLD_SIZE env vars
sheet_indices = list(range(rank, len(dataset), world_size))
dist.init_process_group(backend='nccl', timeout=timedelta(seconds=3600))
dist.barrier()  # sync before result merging
```

---

## 3. Retry Mechanism & Fault Tolerance

### Base Retry Loop — `vlmeval/api/base.py:216–266`

**Randomized constant-window backoff** (not exponential):

```python
def generate(self, message, **kwargs1):
    T = rd.random() * 0.5          # [0, 0.5s] initial jitter
    time.sleep(T)

    for i in range(self.retry):    # default: 10 attempts
        try:
            ret_code, answer, log = self.generate_inner(message, **kwargs)
            if ret_code == 0 and self.fail_msg not in answer and answer != '':
                return answer
            elif self.verbose:
                self.logger.info(f'RetCode: {ret_code}\nAnswer: {answer}\nLog: {log}')
        except Exception as err:
            if self.verbose:
                self.logger.error(f'An error occured during try {i}: ')
                self.logger.error(f'{type(err)}: {err}')

        T = rd.random() * self.wait * 2   # [0, 2*wait] random delay
        time.sleep(T)

    return self.fail_msg if answer in ['', None] else answer
```

### Per-Provider Defaults

| Provider | `retry` | `wait` | Actual delay window | `timeout` |
|---|---|---|---|---|
| BaseAPI | 10 | 3s | [0, 6s] | none |
| OpenAI/GPT | 5 | 5s | [0, 10s] | 60s (x1.1) |
| Gemini | 5 | 5s | [0, 10s] | SDK default |
| Claude | 10 | 3s | [0, 6s] | **none (can hang!)** |

### HTTP 429 Handling

**No explicit 429 detection** anywhere. The flow:

1. `generate_inner` receives `status_code == 429`
2. `ret_code` stays `429` (not normalized to 0)
3. Returns `(429, fail_msg, response)` → retry loop sleeps `[0, 2*wait]` and retries
4. No `Retry-After` header inspection

### Context-Window Overflow Fallback — `base.py:141–153`

```python
def chat_inner(self, inputs, **kwargs):
    while len(inputs):
        try:
            return self.generate_inner(inputs, **kwargs)
        except Exception:
            inputs = inputs[1:]                      # strip oldest turn
            while len(inputs) and inputs[0]['role'] != 'user':
                inputs = inputs[1:]
    return -1, self.fail_msg + ': Failed with all possible conversation turns.', None
```

### Async Retry — `gpt_asyscio.py:360–386`

```python
async def generate(self, inputs, **kwargs):
    for attempt in range(self.retry + 1):
        status, answer, data = await self._send_single_request(input_msgs, **kwargs)
        if 200 <= status < 300:
            return status, answer, data
        await asyncio.sleep(self.wait)
    return 500, f"Failed after {self.retry + 1} attempts.", None
```

Catches `asyncio.TimeoutError` → 408, `aiohttp.ClientConnectorError` → 503.

---

## 4. Result Parsing

### 4.1 Raw API Response Extraction

**OpenAI** — `gpt.py:243–254`:

```python
ret_code = response.status_code
ret_code = 0 if (200 <= int(ret_code) < 300) else ret_code
answer = self.fail_msg
try:
    resp_struct = json.loads(response.text)
    answer = resp_struct['choices'][0]['message']['content'].strip()
except Exception as err:
    self.logger.error(f'{type(err)}: {err}')
return ret_code, answer, response
```

**Claude** — `claude.py:115–124`: `resp_struct['data']['content'][0]['text'].strip()`

**Gemini** — `gemini.py:87–92`: `model.generate_content(messages).text`

**Qwen2** — `qwen_vl_api.py:109`: `response.output.choices[0]['message']['content'][0]['text']`

### 4.2 MCQ Answer Extraction — Three-Stage Cascade

**File:** `vlmeval/dataset/utils/multiple_choice.py`

**Stage 1 — Fast regex** (`extract_characters_regex`, lines 469–494):

```python
def extract_characters_regex(s, choices=['(A)', '(B)', '(C)', '(D)', '(E)']):
    s = s.strip()
    answer_prefixes = [
        'The best answer is', 'The correct answer is', 'The answer is',
        'The answer', 'The best option is', ...
    ]
    for answer_prefix in answer_prefixes:
        s = s.replace(answer_prefix, '')

    if len(s.split()) > 10 and not re.search('[ABCDE]', s):
        return ''
    matches = re.search(r'[ABCDE]', s)
    if matches is None:
        for choice in choices:
            if s.lower() in choice.lower():
                return choice[1]
        return ''
    return matches[0]
```

**Stage 2 — LLM Judge** (`extract_answer_from_item`, lines 262–298):

```python
def extract_answer_from_item(model, item, dataset_name=None):
    ret = can_infer(item['prediction'], choices)
    if ret:
        return dict(opt=ret, log=item['prediction'])      # fast path: no API call
    if model is None:
        return dict(opt='Z', log='Failed in Prefetch...')

    retry = 3
    while retry:
        ans = model.generate(prompt)                       # GPT judge call
        ret = can_infer(ans, choices)
        if ret:
            return dict(opt=ret, log=ans)
        retry -= 1

    # Stage 3 — Random fallback
    return dict(opt=rd.choice(options), log='Failed to predict, thus randomly generate one.')
```

`'Z'` is the sentinel for "no valid answer found."

### 4.3 Result Persistence — `vlmeval/smp/file.py:136–191`

```python
def dump(data, f, **kwargs):
    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl,
                    xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)
```

Format dispatch is purely by **file extension**. All six formats are symmetric for `load`/`dump`.

---

## 5. Performance Evaluation

All evaluation lives in `vlmeval/eval_func/`, with four standalone scripts:

### 5.1 Classification — `vlmeval/eval_func/cla_eval.py`

```python
from sklearn.metrics import precision_score, recall_score, f1_score

return {
    'parser_rate': total_samples / len(data),
    'acc': sum(np.array(y_true) == np.array(y_pred)) / total_samples,
    'precision': precision_score(y_true, y_pred, average='macro'),
    'recall': recall_score(y_true, y_pred, average='macro'),
    'f1': f1_score(y_true, y_pred, average='macro')
}
```

Covers 20+ ultrasound datasets (Fetal Planes, Breast US, PCOS, Knee Grading, COVID-BLUES, BI-RADS, etc.).

### 5.2 Medical Report Generation — `vlmeval/eval_func/report_eval.py`

```python
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
import bert_score

self.scorers = [(Bleu(4), 'Bleu'), (Rouge(), 'Rouge')]

def evaluate(self):
    metrics = {}
    for scorer, name in self.scorers:
        score, _ = scorer.compute_score(self.refs, self.hyps)
        if name == 'Bleu':
            metrics.update({f'Bleu-{i+1}': v*100 for i, v in enumerate(score)})
        else:
            metrics[name] = score * 100

    # BERTScore
    bert_scores = batch_bertscore(self.refs, self.hyps)
    metrics['BERTScore'] = np.mean(list(bert_scores.values())) * 100
    return metrics
```

**Metrics:** BLEU-1/2/3/4, ROUGE-L, BERTScore-F1 (all x100). Covers fetal, thyroid, and radiology report datasets.

### 5.3 Segmentation / Localization — `vlmeval/eval_func/seg_eval_v1.1.py`

Uses a **custom 9-grid spatial accuracy** metric instead of IoU/Dice:

```python
LOW_THRESHOLD = 0.45
HIGH_THRESHOLD = 0.55

def get_bounding_box_location_v1(bbox, img_w, img_h):
    center_x = (bbox[0] + bbox[2]) / 2 / img_w
    center_y = (bbox[1] + bbox[3]) / 2 / img_h

    if center_y < LOW_THRESHOLD:    y_pos = "upper"
    elif center_y < HIGH_THRESHOLD: y_pos = "middle"
    else:                           y_pos = "lower"

    if center_x < LOW_THRESHOLD:    x_pos = "left"
    elif center_x < HIGH_THRESHOLD: x_pos = "center"
    else:                           x_pos = "right"

    if y_pos == "middle" and x_pos == "center":
        return "center"
    return f"{y_pos} {x_pos}"
```

GT bounding boxes are mapped to one of 9 spatial labels; model text predictions are compared against these labels for accuracy.

### 5.4 Measurement / Regression — `vlmeval/eval_func/measure_eval_v1.2.py`

```python
# Min-max normalization per dataset
MIN_MAX_DICT = {
    '18': {'min': 10.0, 'max': 75.0},   # Cardiac EF (%)
    '27': {'min': 0.3,  'max': 1.5},    # IMT (mm)
    '50': {'min': 100.0,'max': 400.0},  # Abdominal circumference (mm)
    '57': {'min': 0,    'max': 85},     # Liver fat value
}

metrics = {
    'MAE': np.mean(np.abs(errors)),
    'RMSE': np.sqrt(np.mean(errors**2)),
    'Std': np.std(errors),
    '%_within_tolerance': np.mean(np.abs(errors) <= tolerance) * 100,  # default +/-0.1
}
```

ICC(3,1) via `pingouin` is implemented but **commented out**.

### Metrics Summary

| Task | File | Metrics | Libraries |
|---|---|---|---|
| Classification | `cla_eval.py` | Accuracy, Macro Precision/Recall/F1, Parser Rate | `sklearn` |
| Report Generation | `report_eval.py` | BLEU-1/2/3/4, ROUGE-L, BERTScore-F1 | `pycocoevalcap`, `bert_score` |
| Segmentation | `seg_eval_v1.1.py` | 9-grid Spatial Accuracy | custom |
| Measurement | `measure_eval_v1.2.py` | MAE, RMSE, Std, % Within Tolerance | `numpy`, (`pingouin` unused) |

### Batch Processing Flow (shared across all four scripts)

```
batch_evaluate_all_tasks_xlsx()
  └─ for each task_id in TSV_PATH:
       ├─ find_model_xlsxs(base_dir, dataset_id)          # scan for model output files
       └─ evaluate_task_with_xlsx(files)
            ├─ data = pd.read_excel(xlsx_path)             # load predictions
            ├─ gt = pd.read_csv(tsv_path, sep='\t')        # load ground truth
            ├─ metrics = model_eval(data) / report_eval(data) / ...
            └─ append {task_id, model, **metrics}
  └─ save_results → timestamped .txt with tab-separated columns
```

There is **no cross-dataset aggregation or overall leaderboard score** — every result is reported at the individual task-ID level.
