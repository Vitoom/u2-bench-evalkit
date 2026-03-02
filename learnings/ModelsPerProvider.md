## For OpenAI, Claude, Gemini, Qwen2, which models are used?

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

---