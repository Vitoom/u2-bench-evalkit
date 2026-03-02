# U2-BENCH Report Generation — Deep Dive Analysis

---

## 1. Dual-Image Payload Construction (Report Generation Prompting)

### Core File: `vlmeval/dataset/image_report.py`

### How Images Are Loaded from TSV

The parent class `ImageBaseDataset` (`vlmeval/dataset/image_base_0302.py:82–84`) parses the `img_path` column at init time:

```python
if 'img_path' in data:
    paths = [toliststr(x) for x in data['img_path']]
    data['img_path'] = [x[0] if len(x) == 1 else x for x in paths]
```

`toliststr` parses a stringified Python list like `"['/path/a.jpg', '/path/b.jpg']"` into an actual Python list. If only one path, it unwraps to a scalar string; if two or more, it stays as a list. This is the mechanism that enables dual-image samples.

### The `build_prompt` Method — `image_report.py:60–106`

```python
def build_prompt(self, line, task_type):
    if isinstance(line, int):
        line = self.data.iloc[line]

    if self.meta_only:
        tgt_path = toliststr(line['img_path'])
    else:
        tgt_path = self.dump_image(line)

    hint = line['hint'] if ('hint' in line and not pd.isna(line['hint'])) else None
    anatomy = line['anatomy_location']

    prompt = ''
    if hint is not None:
        prompt += f'Hint: {hint}\n'

    prompt += f"You are a radiologist analyzing an ultrasound image focused on the {anatomy}."

    if self.dataset_name == '39':
        prompt += "Your task is generate a concise and informative radiological report ..."
        prompt += "Output format: Strings, that is your report."
        prompt += "Example: The liver morphology is full with a smooth capsule. ..."
    else:
        prompt += "Your task is to generate a concise and informative caption ..."
        prompt += "Output format: A single string constituting the image caption. ..."
        prompt += "Examples:"
        if self.dataset_name == '10':
            prompt += "Example1: Fetal phantom ultrasound image showing ...\n"
            # ... more examples
        elif self.dataset_name == '11':
            prompt += "Example1: Thyroid nodule in the right lobe. TI-RADS level 3, Benign.\n"
            # ... more examples
        elif self.dataset_name == '44':
            prompt += "Example1: no single B-lines, B-lines are fused together ...\n"
            # ... more examples

    # === THE CRITICAL PAYLOAD ASSEMBLY ===
    msgs = []
    if isinstance(tgt_path, list):
        msgs.extend([dict(type='image', value=p) for p in tgt_path])
    else:
        msgs = [dict(type='image', value=tgt_path)]
    msgs.append(dict(type='text', value=prompt))

    return msgs
```

### Resulting Payload Structures

**Single-image sample** (`tgt_path` is a `str`):

```python
[
    {'type': 'image', 'value': '/data/ultrasound/liver_001.jpg'},
    {'type': 'text',  'value': 'You are a radiologist analyzing an ultrasound image focused on the liver...'}
]
```

**Dual-image sample** (`tgt_path` is a `list`):

```python
[
    {'type': 'image', 'value': '/data/ultrasound/liver_001_view1.jpg'},
    {'type': 'image', 'value': '/data/ultrasound/liver_001_view2.jpg'},
    {'type': 'text',  'value': 'You are a radiologist analyzing an ultrasound image focused on the liver...'}
]
```

**Key findings:**

- All image dicts come first, in order; the single text dict is always last.
- There is no interleaved text between images — no `<image1>` / `<image2>` tokens.
- The prompt text is identical regardless of image count — it says "an ultrasound image" (singular) even for dual-image cases.
- The `extend` loop scales to any number of images if the TSV `img_path` cell has more entries.

### Full Prompt Templates

**Task 39** (full radiology report):

```
You are a radiologist analyzing an ultrasound image focused on the {anatomy}.
Your task is generate a concise and informative radiological report based strictly
on the visual findings within the provided image. Your report should describe the
primary organ's appearance (size, shape, borders/capsule), its parenchymal echotexture
(e.g., homogeneous, heterogeneous, echogenicity relative to reference structures), and
identify any visible abnormalities (e.g., masses, cysts, fluid collections, calcifications,
ductal dilation). Comment on relevant adjacent structures if visualized. Use standard
radiological terminology.
Output format: Strings, that is your report.
Example: The liver morphology is full with a smooth capsule. The parenchymal echotexture
is fine and diffusely increased. Visualization of the portal venous system is suboptimal.
Intrahepatic and extrahepatic bile ducts are not dilated. The main portal vein diameter is
within normal limits. The gallbladder is normal in size and shape. The wall is smooth and
not thickened. No obvious abnormal echoes are seen within the lumen. The pancreas is normal
in size and shape with homogeneous parenchymal echotexture. The pancreatic duct is not
dilated. No definite space-occupying lesion is seen within the pancreas. The spleen is
normal in size and shape with homogeneous parenchymal echotexture. No obvious
space-occupying lesion is seen within the spleen.
```

**Task 10** (fetal plane captioning):

```
You are a radiologist analyzing an ultrasound image focused on the {anatomy}.
Your task is to generate a concise and informative caption that accurately describes
the key anatomical structures and any significant findings visible in the provided
ultrasound image.
Output format: A single string constituting the image caption. Output only the generated
caption text itself. Do not include any introductory phrases (like "Caption:"), labels,
explanations, or additional formatting.
Examples:
Example1: Fetal phantom ultrasound image showing standard diagnostic plane for abdominal
circumference (AC) measurement
Example2: Fetal phantom ultrasound image showing standard diagnostic plane for biparietal
diameter (BPD) measurement
Example3: Fetal phantom ultrasound image showing standard diagnostic plane for femur
length (FL) measurement
```

**Task 11** (thyroid nodule captioning):

```
You are a radiologist analyzing an ultrasound image focused on the {anatomy}.
Your task is to generate a concise and informative caption ...
Examples:
Example1: Thyroid nodule in the right lobe. TI-RADS level 3, Benign.
Example2: Thyroid nodule in the left lobe. TI-RADS level 3, Benign.
Example3: Thyroid nodule in the right lobe. TI-RADS level 4, Benign.
```

**Task 44** (lung ultrasound captioning):

```
You are a radiologist analyzing an ultrasound image focused on the {anatomy}.
Your task is to generate a concise and informative caption ...
Examples:
Example1: no single B-lines, B-lines are fused together into the picture of a white lung
Example2: liver on the right side
Example3: white lung?
```

**All other task IDs** get the generic caption prompt with "Examples:" appended but no actual examples listed.

---

## 2. Result Parsing & Evaluation for Long-Form Generation

### No LLM-as-a-Judge Exists

Despite importing `build_judge` in `image_report.py`, it is never called. There is no GPT-4 judge, no prompt-based comparison, and no clinical factuality scorer anywhere in this pipeline.

### No Text Cleaning / Post-Processing

There is no regex cleaning, no lowercasing, no punctuation normalization. Raw model output goes directly from the `prediction` column into the scorers.

### Two Divergent `MedicalReportScorer` Implementations

#### Version A — Runner-integrated (`vlmeval/dataset/image_report.py:16–49`)

```python
class MedicalReportScorer:
    def __init__(self, refs, hyps):
        """
        refs: { 'case001': ["expert report text 1", "expert report text 2"], ... }
        hyps: { 'case001': "generated report text", ... }
        """
        self.refs = refs
        self.hyps = hyps
        self.scorers = [
            (Bleu(4), 'Bleu'),
            (Rouge(), 'Rouge'),
            # (Cider(), 'Cider')        # commented out
        ]

    def evaluate(self):
        metrics = {}
        for scorer, name in self.scorers:
            score, _ = scorer.compute_score(self.refs, self.hyps)
            if name == 'Bleu':
                metrics.update({f'Bleu-{i+1}': v*100 for i, v in enumerate(score)})
            else:
                metrics[name] = score * 100
        # BERTScore — ENTIRELY COMMENTED OUT in this version
        # SentenceTransformer similarity — ENTIRELY COMMENTED OUT
        return metrics
```

Active metrics: BLEU-1/2/3/4, ROUGE-L only. BERTScore and CIDEr are commented out.

#### Version B — Standalone batch script (`vlmeval/eval_func/report_eval.py:105–135`)

```python
class MedicalReportScorer:
    def __init__(self, refs, hyps):
        self.refs = refs
        self.hyps = hyps
        self.scorers = [
            (Bleu(4), 'Bleu'),
            (Rouge(), 'Rouge')
        ]

    def evaluate(self):
        metrics = {}
        for scorer, name in self.scorers:
            try:
                score, _ = scorer.compute_score(self.refs, self.hyps)
                if name == 'Bleu':
                    for i, v in enumerate(score):
                        metrics[f'Bleu-{i+1}'] = v * 100
                else:
                    metrics[name] = score * 100
            except Exception as e:
                print(f"{name}计算失败: {str(e)}")

        # BERTScore — ACTIVE in this version
        try:
            bert_scores = batch_bertscore(self.refs, self.hyps)
            metrics['BERTScore'] = np.mean(list(bert_scores.values())) * 100
        except Exception as e:
            print(f"BERTScore计算失败: {str(e)}")
            metrics['BERTScore'] = 0.0

        return metrics
```

Active metrics: BLEU-1/2/3/4, ROUGE-L, plus BERTScore-F1.

### The `batch_bertscore` Function — `report_eval.py:62–103`

```python
def batch_bertscore(refs_dict, hyps_dict):
    cache = BertModelCache.instance()
    scores = {}

    import bert_score
    calc_params = {
        "model_type": "bert-base-multilingual-cased",
        "lang": "en",
        "device": cache.device,
        "batch_size": 8
    }

    all_hyp, all_ref = [], []
    for case_id, hyp in hyps_dict.items():
        ref = refs_dict.get(case_id, [])
        if ref:
            all_hyp.extend(hyp)
            all_ref.extend(ref)

    if len(all_hyp) > 0:
        _, _, F1 = bert_score.score(all_hyp, all_ref, **calc_params)

        idx = 0
        for case_id, hyp in hyps_dict.items():
            refs = refs_dict.get(case_id, [])
            if refs:
                scores[case_id] = F1[idx:idx+len(refs)].mean().item()
                idx += len(refs)

    BertModelCache.clear_cache()
    return scores
```

### The `BertModelCache` Singleton — `report_eval.py:23–61`

```python
class BertModelCache:
    _instance = None

    def __init__(self):
        if self._instance is not None:
            raise Exception("单例模式，请使用 instance() 方法")
        self.tokenizer, self.model, self.device = self.load_bert_model()
        BertModelCache._instance = self

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = BertModelCache()
        return cls._instance

    def load_bert_model(self):
        try:
            if HAVE_LOCAL_BERTSCORE_MODEL:
                model_path = f"{LOCAL_MODEL_DIR}/{BERTSCORE_MODEL_NAME}"
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModel.from_pretrained(model_path)
            else:
                tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
                model = AutoModel.from_pretrained("bert-base-multilingual-cased")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            return tokenizer, model, device
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            return None, None, "cpu"

    @classmethod
    def clear_cache(cls):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
```

Note: The singleton loads `bert-base-multilingual-cased` into `self.model`, but `batch_bertscore` passes `model_type="bert-base-multilingual-cased"` to the `bert_score` library which loads its own internal copy. The preloaded model is only used for `device` detection and GPU cache clearing.

### Ground Truth Loading & Matching — `image_report.py:108–126`

```python
def evaluate(self, eval_file, task_type='cla', **judge_kwargs):
    data = load(eval_file)
    data = data.sort_values(by='index')

    refs = {}
    for i, row in data.iterrows():
        if 'gt_report' in data.columns and row['gt_report']:
            refs[row['index']] = [row['gt_report']]       # primary GT source
        else:
            refs[row['index']] = [row['caption']]          # fallback GT source
    hyps = {row['index']: [row['prediction']] for _, row in data.iterrows()}

    medical_scorer = MedicalReportScorer(refs, hyps)
    report_dict = medical_scorer.evaluate()

    txt_file = eval_file.replace('.xlsx', '.txt')
    with open(txt_file, 'w') as tf:
        tf.write(str(report_dict))
    return report_dict
```

GT priority: `gt_report` column if present and non-empty, else `caption`. Only one reference per sample (the COCO API supports multiple, but only one is used).

### The Standalone `report_eval()` Function — `report_eval.py:163–177`

```python
def report_eval(data):
    gt_key = 'caption' if 'caption' in data.keys() else 'cap_ans'
    id_key = 'id' if 'id' in data.keys() else 'index'
    pred_key = 'response' if 'response' in data.keys() else 'prediction'
    refs = {row[id_key]: [row[gt_key]] for _, row in data.iterrows()}
    hyps = {row[id_key]: [row[pred_key]] for _, row in data.iterrows()}

    scorer = MedicalReportScorer(refs, hyps)
    metrics = scorer.evaluate()
    return metrics
```

More flexible column detection: tries `caption` then `cap_ans` for GT; `id` then `index` for the key; `response` then `prediction` for hypothesis.

### GT Loading from TSV (Standalone Script) — `report_eval.py:151–161`

```python
def read_jsonl_with_tsv(jsonl_path, tsv_path):
    with open(jsonl_path, 'r') as f:
        jsonl_data = [json.loads(line) for line in f]
    df_jsonl = pd.DataFrame(jsonl_data)

    df_tsv = pd.read_csv(tsv_path, sep='\t')

    df_merged = df_jsonl.assign(cap_ans=df_tsv['caption'])
    return df_merged
```

Merging is done positionally (`.assign`) — no index join, relies on row order alignment.

### TSV Paths (Standalone Script) — `report_eval.py:15–20`

```python
TSV_PATH = {
    '10': '/media/ps/data-ssd/json_processing/ale_tsv_output/10.tsv',
    '11': '/media/ps/data-ssd/json_processing/ale_tsv_output/11.Thyroid_US_Images.tsv',
    '39': '/media/ps/data-ssd/json_processing/ale_tsv_output/39_translated.tsv',
    # '44': '/media/ps/data-ssd/json_processing/ale_tsv_output/44.COVID-BLUES-frames.tsv',
}
```

Three active task IDs: `10` (fetal US), `11` (thyroid US), `39` (abdominal US). Task `44` (COVID lung US) is commented out.

### Batch Evaluation Pipeline — `report_eval.py:246–301`

```python
def find_model_xlsxs(base_dir, dataset_id):
    dataset_dir = os.path.join(base_dir, dataset_id)
    model_files = []
    for model_dir in os.listdir(dataset_dir):
        model_path = os.path.join(dataset_dir, model_dir)
        for task_dir in os.listdir(model_path):
            if task_dir.endswith('report'):
                for file in os.listdir(os.path.join(model_path, task_dir)):
                    full_path = os.path.join(model_path, task_dir, file)
                    model_files.append(full_path)
    return model_files

def evaluate_task_with_xlsx(files):
    task_results = []
    for xlsx_path in files:
        try:
            model_name = xlsx_path.split('/')[-3]
            task_id = xlsx_path.split('/')[-4]

            if model_name == 'MiniGPT-med':          # HARDCODED GUARD
                data = pd.read_excel(xlsx_path)
                metrics = report_eval(data)
                metrics.update({
                    'task_id': task_id,
                    'model': model_name,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                task_results.append(metrics)
        except Exception as e:
            print(f"处理文件失败：{xlsx_path}")
            print(f"错误信息：{str(e)}")
    return task_results

def batch_evaluate_all_tasks_xlsx():
    base_dir = '/media/ps/data-ssd/benchmark/VLMEvalKit/outputs/'
    output_file = f'report_results_{datetime.now().strftime("%Y%m%d_%H%M")}.txt'
    all_results = []
    for task_id in TSV_PATH.keys():
        task_pairs = find_model_xlsxs(base_dir, task_id)
        if not task_pairs:
            print(f"警告：任务 {task_id} 未找到任何模型结果文件")
            continue
        task_results = evaluate_task_with_xlsx(task_pairs)
        all_results.extend(task_results)
    save_results(all_results, output_file)
```

Note: `evaluate_task_with_xlsx` has a hardcoded `if model_name == 'MiniGPT-med':` guard — only files where the third-to-last path component is `MiniGPT-med` are evaluated. All other models are silently skipped.

### Output Format — `report_eval.py:222–243`

```python
def save_results(results, output_file):
    columns = ['task_id', 'model', 'Bleu-4', 'Rouge', 'BERTScore']
    with open(output_file, 'w') as f:
        f.write('\t'.join(columns) + '\n')
        for res in results:
            line = '\t'.join([
                res['task_id'],
                res['model'],
                f"{res.get('Bleu-4', 0):.2f}",
                f"{res.get('Rouge', 0):.2f}",
                f"{res.get('BERTScore', 0):.2f}"
            ]) + '\n'
            f.write(line)
```

BLEU-1/2/3 are computed internally but discarded in the output; only BLEU-4 is saved.

---

## Metrics Summary

| Metric | Library | Runner (`image_report.py`) | Batch Script (`report_eval.py`) |
|---|---|---|---|
| BLEU-1/2/3/4 | `pycocoevalcap` | Active | Active (only BLEU-4 saved) |
| ROUGE-L | `pycocoevalcap` | Active | Active |
| BERTScore (F1) | `bert_score` | Commented out | Active |
| CIDEr | `pycocoevalcap` | Commented out | Not present |
| LLM-as-Judge | — | Not present | Not present |
| Clinical factuality | — | Not present | Not present |
| Text cleaning/regex | — | Not present | Not present |

---

## 3. Report Accept/Reject & Quality Gating Analysis

### Does This Repo Have a Mechanism to Reject or Accept Generated Reports?

**No.** The report generation pipeline has no mechanism to reject, accept, filter, or gate generated reports based on quality. The pipeline is entirely fire-and-forget.

### What Exists: API-Level Retry Only

The only "rejection" logic is the generic API retry loop in `vlmeval/api/base.py:243–266`, which retries up to 10 times based on HTTP-level success — it knows nothing about report quality:

```python
# base.py, lines 243-266
for i in range(self.retry):    # default: 10 attempts
    try:
        ret_code, answer, log = self.generate_inner(message, **kwargs)
        # A response is ACCEPTED only when ALL three conditions are true:
        if ret_code == 0 and self.fail_msg not in answer and answer != '':
            return answer  # accepted
    except Exception as err:
        ...
    T = rd.random() * self.wait * 2   # [0, 2*wait] random delay
    time.sleep(T)

# After all retries exhausted:
return self.fail_msg if answer in ['', None] else answer
```

The three acceptance conditions are purely technical:
- `ret_code == 0` — HTTP-level success (2xx status)
- `self.fail_msg not in answer` — the answer does not contain the failure sentinel
- `answer != ''` — non-empty response

### What Happens After Retries Are Exhausted

```python
# base.py, line 266
return self.fail_msg if answer in ['', None] else answer
```

The literal string `"Failed to obtain answer via API."` is returned as the prediction and written into the results file verbatim. It is NOT filtered out.

### How Failed Predictions Flow Into Report Evaluation

In `vlmeval/dataset/image_report.py:108–126`, the `evaluate()` method performs zero filtering:

```python
# Every row is included unconditionally — no check for fail_msg,
# empty strings, minimum length, or any content validation
hyps = {row['index']: [row['prediction']] for _, row in data.iterrows()}

medical_scorer = MedicalReportScorer(refs, hyps)
report_dict = medical_scorer.evaluate()
```

Failed API responses like `"Failed to obtain answer via API."` are scored against the ground truth reference by BLEU/ROUGE exactly like any real report. They drag down the aggregate metrics silently.

### The Only Exception (Not for Reports)

The only place in the entire codebase where `fail_msg` responses are actively filtered out is in `vlmeval/dataset/mmbench_video.py:224`:

```python
res = {k: v for k, v in res.items() if model.fail_msg not in v}
```

This is specific to the video benchmark dataset and is NOT applied to `ReportDataset`.

### Per-API `fail_msg` Variants

| File | `fail_msg` string |
|---|---|
| `api/base.py:20` | `'Failed to obtain answer via API.'` |
| `api/gpt.py:58` | `'Failed to obtain answer via API. '` |
| `api/gemini.py:26` | `'Failed to obtain answer via API. '` |
| `api/claude.py` (inherited) | same as base |
| `api/bailingmm.py:25` | `'Failed to obtain answer via bailingMM API.'` |
| `api/stepai.py:26` | `'Fail to obtain answer via API.'` |

### What Does NOT Exist Anywhere in the Report Pipeline

| Mechanism | Present? |
|---|---|
| Quality threshold / confidence gate | No |
| Minimum length or format validation | No |
| Reject-and-re-prompt / self-refinement loop | No |
| LLM-as-Judge acceptance check | No |
| Human-in-the-loop review | No |
| Post-hoc filtering (drop low-BLEU predictions) | No |
| `parser_rate` or validity tracking metric | No |
| Dropping `fail_msg` responses before scoring | No |

### Implications

The report generation pipeline is entirely fire-and-forget: one prompt, one generation attempt (with API-level retries only), no quality gate, no re-generation based on content, no filtering. Failed responses pollute the metrics silently with no way to distinguish them in the final scores.

---

## Notable Issues for Porting to a DARD Pipeline

1. **Two implementations have drifted** — the runner version lacks BERTScore; the standalone script has it. Pick one to port.
2. **No text normalization** — consider adding lowercasing, punctuation stripping, or at minimum whitespace normalization before scoring.
3. **Single reference per sample** — COCO scorers support multi-reference; wiring in multiple expert annotations would improve metric reliability.
4. **No LLM-as-Judge** — this is a gap worth filling for clinical factuality assessment (e.g., prompting GPT-4 to grade anatomical accuracy, finding completeness, hallucination detection).
5. **Dual-image prompt doesn't acknowledge two images** — the prompt says "an ultrasound image" (singular). Adding explicit multi-view awareness to the prompt could improve generation quality.
6. **Positional TSV-JSONL merge** — `read_jsonl_with_tsv` merges ground truth by row position, not by a shared ID key, which is fragile.
7. **MiniGPT-med only guard** — the batch pipeline silently skips all models except `MiniGPT-med`.
8. **BERTScore model duplication** — the `BertModelCache` singleton loads the model but `bert_score.score()` re-loads its own copy internally; the singleton only provides `device` detection.
9. **No report quality gating** — failed API responses (`fail_msg`) are scored as-is against ground truth, silently dragging down aggregate metrics. Add at minimum a `fail_msg` filter, and ideally a confidence-based accept/reject loop or LLM-as-Judge quality gate.
