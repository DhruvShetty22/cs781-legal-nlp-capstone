# Phase 6: Abstractive Legal Summarization

This folder holds the Phase-6 code. One percent-format template (`phase6_summ_template.py`) is rendered into four Colab
notebooks, one per Phase-1 architecture (llama, bart, prefixlm, mistral_swa). `run_logs/` holds the executed copies
with their outputs. Every experiment starts from the student's own Phase-1 checkpoint. No other pretrained language
model or external API is used for summarization. The only other pretrained model is `roberta-large`, which
`bert_score` loads to compute the BERTScore metric, as the official evaluator does.

---

## 1. Task

Capstone spec §12 (`capstone.txt` lines 1873-2108).

* **Input**: a long Indian legal judgment (`document`) with its ID.
* **Output**: one concise abstractive summary per test document ID.
* **Data**: `Phase_6_Summ/Train/train.json` records are `{"id", "document", "summary"}`. `Phase_6_Summ/Test/test.json`
  records are `{"id", "document"}`. There is no public dev split (§12.3.2).
* **Submission**: `submission.zip` containing `prediction.json` at the root of the archive (§6.3). The code writes the
  list format (§12.3.5 Option A):

  ```json
  [
    {"id": "doc_001", "summary": "Generated legal summary ..."},
    {"id": "doc_002", "summary": "..."}
  ]
  ```

  There must be exactly one non-empty summary for every test ID. The evaluator also accepts the object format
  `{"doc_001": "...", ...}`.
* **Metrics** (§12.4): the primary metric is **BERTScore F1**. The secondary metrics are **ROUGE-1 F1** and **ROUGE-L F1**.
  The official local evaluator `evaluate_phase6.py` works as follows:
  * BERTScore: `bert_score.score(preds, refs, lang='en', batch_size=8)`, which uses roberta-large.
  * ROUGE-1: `rouge_score` with stemming.
  * ROUGE-L: its own LCS over `rouge_score.tokenizers.DefaultTokenizer(use_stemmer=True)` tokens.
  * All three are means over documents, computed after `strip()`.
* **Phase rules the template header cites** (capstone §1 item 4, §12.2):
  * At least 6 submissions.
  * All four Phase-1 models must be used.
  * At least 4 substantially different experiments. At least 3 of them must use different techniques from the §12.2.1
    list (Houlsby, Pfeiffer, prefix tuning, prompt tuning, LoRA, QLoRA, AdaLoRA, LoRMA, KD). LoRMA is compulsory.
  * At least 1 experiment must be knowledge distillation into a smaller student.

## 2. Files in this folder

| File / dir | What it is |
|---|---|
| `phase6_summ_template.py` | The single source of all Phase-6 code. It is percent-format (`# %%` code cells, `# %% [markdown]` text cells) with placeholders `{{ARCH}}`, `{{TAG}}`, `{{FAMILY}}`, `{{MODEL}}` and `{{INPUT}}`. |
| `build_p6_notebooks.py` | Fills the placeholders for each architecture and writes `P6_arch{1..4}_<arch>.ipynb`. `python build_p6_notebooks.py` builds all four. `python build_p6_notebooks.py bart` builds only one. |
| `build_nb.py` | Generic `.py` (percent format) to `.ipynb` converter that `build_p6_notebooks.py` uses. |
| `P6_arch1_llama.ipynb` | Rendered notebook, `ARCH = "llama"` (decoder-only, Phase-1 `LlamaForCausalLM`). |
| `P6_arch2_bart.ipynb` | Rendered notebook, `ARCH = "bart"` (encoder-decoder, Phase-1 v2 `BartForConditionalGeneration`). |
| `P6_arch3_prefixlm.ipynb` | Rendered notebook, `ARCH = "prefixlm"` (prefix-LM `LlamaForCausalLM` with a bidirectional prefix). |
| `P6_arch4_mistral_swa.ipynb` | Rendered notebook, `ARCH = "mistral_swa"` (`MistralForCausalLM`, sliding-window attention + GQA). |
| `run_logs/P6_arch1_llama_after.ipynb` | Executed run `llama_lorma_head`. |
| `run_logs/P6_arch1_llama_prompt_after.ipynb` | Executed run `llama_prompt_head`. |
| `run_logs/P6_arch2_bart_after.ipynb` | Executed run `bart_lorma_head`. |
| `run_logs/P6_arch2_bart_prompt_after.ipynb` | Executed run `bart_prompt_head`. |
| `run_logs/P6_arch3_prefixlm_after.ipynb` | Executed run `prefixlm_lorma_head`. |
| `run_logs/P6_arch4_mistral_swa_after.ipynb` | Executed run `mistral_swa_lorma_head`. |

The four `P6_arch*.ipynb` files match a fresh render of the current template cell for cell. The code cells of each run
log match the rendered notebook of the same architecture, except `METHOD = "prompt"` in the two `*_prompt_after` logs.

Each notebook has 31 cells: a title, a run-name cell, imports and HF login, configuration, then the sections data,
KD targets, tokenizer, long-document selection, model, sanity checks, adaptation method, scoring, training, inference
and submission, held-out measurements, saving, and the all-runs table.

## 3. Data and preprocessing

**Loading** (cell *Data*):
* Source: `hf_hub_download("Exploration-Lab/CS-781-Capstone", "Phase_6_Summ/Train/train.json" | "Phase_6_Summ/Test/test.json", repo_type="dataset")`.
  If that path is missing, the code searches the dataset's file list for a unique match containing `phase_6` / `phase6` /
  `summ`.
* Format: JSON or JSON Lines. A list of records or an `id -> record` dict is accepted.
* Field names: the document field can be any of `document|text|judgement|judgment|doc|source`. The summary field can be
  any of `summary|abstract|headnote|target`. List-valued fields are joined with `\n`.
* Cleaning: a repeated ID is dropped (first one kept) and counted. A training record without a non-empty summary is
  dropped and counted.
* Leakage check: documents with identical text (MD5) in train and test are counted. Nothing is removed.

**Statistics printed by the logs** (identical in all six logs, e.g. `run_logs/P6_arch1_llama_after.ipynb`, cell 6):

```
Train/train.json: list of 7030 (5s)
  !! 2 training records without a summary dropped
Test/test.json: list of 100 (0s)

(words)               n  median     p10     p90      max  empty
train documents    7028    3129    1349    7852   139645      0
train summaries    7028     638     293    1501    27768      0
test documents      100    3171    1503    8562    39695      0
document words / summary words (train): median 4.6  p10 2.5  p90 10.8
documents with identical text in train and test: 0
held-out: 300 training documents (seed 42, id hash eddb1ace9498);  training: 6728 documents
```

**Held-out split** (Phase 6 has no dev set):
* `np.random.default_rng(SEED).permutation(n_train)`. The first `HELDOUT_DOCS` indices (300) are held out and never
  trained on. The rest are training documents.
* `TRAIN_MAX_DOCS` can draw a seeded random subset of the training documents. It was `None` in all logs.
* `heldout_md5` is the first 12 hex characters of the MD5 of the sorted held-out IDs. It is printed and saved in
  `run_config.json`. All six logs print id hash `eddb1ace9498`.

**Tokenisation** (cell *Tokenizer*):
* Uses the Phase-1 `PreTrainedTokenizerFast` from `<hf-user>/cs781-p1-<arch>/best/`.
* Documents and summaries are tokenised whole, without special tokens, and stored as `uint16`.
* An empty text becomes `[<|endoftext|>]`.
* Tokens are cached in `CACHE_DIR/<name>_<tokenizer md5>.npz`, so a second run in the same runtime skips tokenisation.
* Printed IDs:
  * llama, prefixlm and mistral_swa: `vocab 32000 (model rows 32000), eos 0, pad 1, mask None, positions 1024`.
  * bart: `vocab 32001 (model rows 32064), eos 0, pad 1, mask 32000, positions 1024, <mask> after the encoder input: True, decoder start 0`.

Token statistics printed by the logs:

```
(tokens)                median     p90     p99      max   share above
train documents           3660    9102   28115   161917     99.5% SRC_LEN     (bart: 98.7%, SRC_LEN 768)
test documents            3776   10292   33794    45736    100.0% SRC_LEN     (bart: 97.0%)
train summaries            750    1779    5051    32403     76.6% TGT_LEN     (bart: 73.1%, TGT_LEN 512)
summary words per token: 0.847
```

**Long documents to model input** (`DOC_SELECT`, function `select_source`):

At most `SRC_LEN` document tokens reach the model. A document that already fits is used whole by every strategy.
* `head`: the first `SRC_LEN` tokens.
* `head_tail`: `round(SRC_LEN * HEAD_FRAC)` tokens from the start, and the rest from the end.
* `even`: `SRC_LEN // EVEN_PIECE` pieces of `EVEN_PIECE` tokens. The start positions are
  `np.linspace(0, n - EVEN_PIECE, k).round()`, evenly spread over the whole document.
* `importance`: works sentence by sentence.
  1. Sentences are cut after `. ? ! ;` followed by white space, and before every `\n` (regex `(?<=[.?!;])(?=\s)|(?=\n)`).
  2. Each sentence is scored by the dot product of its TF-IDF vector with the TF-IDF vector of its whole document. The
     vectoriser is scikit-learn `TfidfVectorizer(sublinear_tf=True, max_features=TFIDF_MAX_FEATURES, stop_words="english")`,
     fitted on the training documents only. No pretrained model is used.
  3. Sentences are taken best-first while their tokens fit in `SRC_LEN`, then restored to document order.
  4. If no sentence fits, the code falls back to the document start. The fallback count is printed.
  5. The selection is cached in `CACHE_DIR/importance_*.npz`.

There is no chunking or hierarchical summarization in the code. Each document becomes one input of at most `SRC_LEN`
tokens. All six recorded runs used `DOC_SELECT = "head"`.

**Model input layout** (`prompt_tokens`, `target_tokens`, `collate`):

* **Target** = summary tokens cut to `TGT_LEN`. `<|endoftext|>` (id 0) is appended only when the summary was not cut. In
  a `kd` run, the target is the teacher's summary with probability `KD_TEACHER_PROB`.
* **Decoder-only (llama, prefixlm, mistral_swa)**: one sequence `[soft prompt (prompt method only) | selected document
  tokens | SEP_TEXT | target]`.
  * `SEP_TEXT = "\n\nSummary:\n"` is 6 tokens (printed).
  * Training uses right padding. Labels are `-100` everywhere except the target positions.
  * `pfx` = length of document + separator. This is the prefix-LM's bidirectional part.
  * Printed budget: `prompt 0 + document 512 + separator 6 + summary 480 (+1) / generation 480 of 1024`. The prompt runs
    print `prompt 20`.
* **bart**:
  * Encoder input = `[soft prompt | selected document tokens | <mask>]`. The `<mask>` end marker is added because the
    Phase-1 v2 config has `cs781_objective.cont_end_marker = True`.
  * Decoder input = `[decoder_start (=0, <|endoftext|>)] + target[:-1]`. Labels = target.
  * Printed budget: `encoder: prompt 0 + document 768 + <mask> 1 of 1024; decoder: summary 512 (+1) / generation 512 of 1024`.
* Asserts check that every length budget fits in the 1024 Phase-1 positions.
* Batch checks run on random examples:
  * The selected input is at most `SRC_LEN` tokens.
  * A document that fits is unchanged.
  * The supervised labels equal the target exactly.
  * The separator sits right before the target.

**Input coverage** (held-out, printed). This is the ROUGE-1 recall (stemmed unigram counts) of each reference summary
against the whole document and against the decoded selected input:

| Arch / SRC_LEN | Whole document | Selected input (`head`) | Held-out docs longer than SRC_LEN |
|---|---|---|---|
| decoder-only, 512 | 0.9533 | 0.3965 | 99.3% |
| bart, 768 | 0.9533 | 0.5357 | 97.0% |

## 4. Model architecture

**Phase-1 backbones.** Each is loaded with `from_pretrained(<repo>/best)` and cast to fp32. Training and generation then
run under fp16 autocast. The dimensions below come from the Phase-1 template config (`code/phase1`): d_model 768,
1024 positions, tied input/output embeddings, vocab 32000.

| ARCH | Repo | HF class | Blocks / heads / FFN | Params (Phase-6 log) |
|---|---|---|---|---|
| llama | `<hf-user>/cs781-p1-llama` | `LlamaForCausalLM` | 12 blocks, 12 heads (MHA), FFN 2048, causal | 109.5M |
| bart | `<hf-user>/cs781-p1-bart-v2` (v1 is never used) | `BartForConditionalGeneration` | 6 encoder + 6 decoder layers, 12 heads, FFN 3072, `scale_embedding=True`, 32064 embedding rows | 125.4M |
| prefixlm | `<hf-user>/cs781-p1-prefixlm` | `LlamaForCausalLM` | as llama; bidirectional attention over the prefix, causal after it | 109.5M |
| mistral_swa | `<hf-user>/cs781-p1-mistral_swa` | `MistralForCausalLM` | 12 blocks, 12 query heads / 4 KV heads (GQA), sliding window 256, FFN 2048 | 100.1M |

**Head.** No new head is added. The code reuses the Phase-1 LM head (`lm.lm_head`, plus `final_logits_bias` for BART).
`SummModel.forward` runs only the backbone (`lm.model`) and takes `last_hidden_state`. It then applies the LM head
only at supervised positions:
* Decoder-only: `lm_head(h[:, :-1][sel])` with `sel = lab[:, 1:] != -100`. The state at position t predicts target token
  t+1, so the first summary token is predicted from the last separator token.
* bart: the decoder states at label positions go through `lm_head(...) + final_logits_bias`.

**Attention per arch.**
* llama and mistral_swa get a 2-D padding mask (the HF causal / sliding-window masking applies).
* prefixlm gets a 4-D boolean mask from `prefix_mask(att, pfx + P)`. Query i may attend key j if `j <= i` or
  `j < prefix length`. Padding keys are masked, and each query always sees itself. The prefix covers the soft prompt,
  the document and the separator.
* bart uses standard encoder self-attention, decoder causal self-attention and cross-attention.

**Loss.** Token-level cross-entropy, computed in fp32, averaged over all supervised summary tokens of a micro-batch:
`F.cross_entropy(logits.float(), tgt, label_smoothing=LABEL_SMOOTHING)`. For `adalora` the code adds
`ADALORA_ORTH * mean_over_layers(||PᵀP − I||_F + ||QQᵀ − I||_F)`.

**KD student** (`METHOD = "kd"`). `keep_blocks` keeps a subset of the Phase-1 blocks: `STUDENT_LAYERS`, or every 2nd
block (0, 2, 4, ...) when `None`. For BART this is applied to the encoder and the decoder separately. The `layer_idx` of
each kept attention layer is renumbered, and `num_hidden_layers` / `layer_types` (or `encoder_layers` /
`decoder_layers`) are updated in the config.

**Sanity checks** (`model_checks`, `gen_consistency`). Both run after loading, and again after `configure` for `qlora`
and `prompt`. `gen_consistency` runs once more after the adapter merge. The run is stopped (`assert`) if any check fails.
1. Changing the last input token must move the first input token's state only for prefixlm. For bart, changing an
   encoder token must move the decoder states, and the encoder output from `embed()` must equal the encoder output from
   ids.
2. Changing the last target token must not move earlier states.
3. Target logits must not change when a longer document joins the batch.
4. Greedy generation (left padding) for the shortest and longest held-out inputs is re-scored with the training forward
   (right padding). Every generated token must be the arg-max (gap < 0.5 under AMP).

Printed values from the logs:
* Generation gap: `largest logit gap to the arg-max 0.00e+00` in every log.
* prefixlm: `moves the first input token's state by 1.28e-01 (sees right context: True, expected True)`.
* llama and mistral_swa: `0.00e+00 (... False, expected False)`.
* bart: `changing an encoder token moves the decoder states by 3.48e-01`.

## 5. Fine-tuning methods implemented

`METHOD` selects one of the methods below. Every method is implemented inside the template in plain PyTorch (no
`peft`, no `bitsandbytes`). Injection (`inject`) replaces matching `nn.Linear` children of the backbone `lm.model`. The LM head
is never adapted. After injection, an assert checks that the probe logits on two held-out documents change by less than
`1e-3`, so every adapter is the identity at initialisation.

| METHOD | Trainable parameters | Implementation (class) | Knobs |
|---|---|---|---|
| `full` | all | none; every parameter `requires_grad`. The template calls it the full-model reference and notes it is not on the §12.2.1 list. | `LR["full"]=5e-5` |
| `lora` (Hu et al., 2022) | `A ∈ R^{r×d_in}` (Kaiming-uniform), `B ∈ R^{d_out×r}` (zeros) per target | `LoRALinear`: `h = W0x + (α/r)·B(Ax)` | `ADAPT_TARGETS`, `RANK`, `ALPHA` |
| `qlora` (Dettmers et al., 2023) | LoRA `A`, `B` on `ADAPT_TARGETS` | `NF4Linear` + `QLoRALinear`, see the QLoRA notes below | `QLORA_BLOCK`, `QLORA_DOUBLE_QUANT`, `QLORA_MERGE_FOR_INFERENCE`, `RANK`, `ALPHA` |
| `lorma` (Bihany, Patel & Modi, 2025) | `A ∈ R^{r×d_out}`, `B ∈ R^{d_out×r}` per target | `LoRMALinear`: multiplicative update `h = 𝓘(BA)·W0x (+ bias)`, see the LoRMA notes below | `LORMA_VARIANT`, `RANK`, `ALPHA`, `ADAPT_TARGETS` |
| `adalora` (Zhang et al., 2023) | `P ∈ R^{d_out×r0}`, `E ∈ R^{r0}`, `Q ∈ R^{r0×d_in}` | `AdaLoRALinear` + `RankAllocator`, see the AdaLoRA notes below | `ADALORA_INIT_R`, `ADALORA_TARGET_R`, `ADALORA_TINIT`, `ADALORA_TFINAL`, `ADALORA_DELTA_T`, `ADALORA_BETAS`, `ADALORA_ORTH` |
| `houlsby` (Houlsby et al., 2019) | bottleneck `down: d→m`, `up: m→d` (zero init) per wrapped layer, plus layer norms if `ADAPTER_TRAIN_NORMS` | `AdapterLinear`: `y + up(gelu(down(y)))` applied to the output `y` of `HOULSBY_AT` layers, before the residual add | `HOULSBY_AT`, `ADAPTER_DIM`, `ADAPTER_TRAIN_NORMS` |
| `pfeiffer` (Pfeiffer et al., 2021) | same bottleneck, FFN output only (one per block), plus norms | `AdapterLinear` on `PFEIFFER_AT` | `PFEIFFER_AT`, `ADAPTER_DIM`, `ADAPTER_TRAIN_NORMS` |
| `prompt` (Lester et al., 2021) | `PROMPT_LEN × d_model` soft-prompt matrix; the whole model is frozen | `SummModel.prompt`, see the prompt notes below | `PROMPT_LEN` |
| `kd` (sequence-level KD, Kim & Rush, 2016) | all parameters of the smaller student | see section 4 (*KD student*) and the KD notes below | `TEACHER_RUN`, `STUDENT_LAYERS`, `KD_TEACHER_PROB`, `LR["kd"]=1e-4` |

**QLoRA notes.**
* Every backbone `nn.Linear` is frozen and stored as 4-bit NormalFloat codes, packed two per byte, with one absmax per
  `QLORA_BLOCK` weights.
* The NF4 levels are 8 positive and 7 negative N(0,1) quantiles (offset 0.9677083) plus an exact 0, scaled to [-1, 1].
* With double quantisation, the absmax values are mean-centred and stored as int8 in blocks of 256, with one fp32
  scale per block.
* `_NF4Matmul` dequantises in the forward pass and again in the backward pass, computing in fp16.
* No paged optimizer is used.
* Printed checks: the relative error of the first dequantised weight (assert < 0.25), the change in the probe logits,
  and the model size before and after (MB).

**LoRMA notes.**
* `plus` computes `W0x + (α/r)·B(A(W0x))`, with `A` Kaiming-uniform and `B = 0`.
* `pi` computes `𝓘(BA) = gather(BA, rot)`, so row i of `BA` is rotated cyclically by i. The init is `B[:,0] = 1`,
  `A = e₀e₀ᵀ`, which makes `𝓘(BA) = I`, with scale 1.

**AdaLoRA notes.**
* SVD-form update `h = W0x + (α/r0)·P·diag(E)·Q·x`, with `E = 0` at init and `P`, `Q` drawn from N(0, 0.02²).
* Importance per entry: `EMA_β1(|w·g|) · EMA_β2(| |w·g| − EMA(|w·g|) |)`.
* Triplet score: `s(E_i) + mean_j s(Q_ij) + mean_j s(P_ji)`.
* Budget: `INIT_R·n` until `TINIT`, then a cubic decrease to `TARGET_R·n`. The mask is frozen for the last `TFINAL` of
  progress. Every `ADALORA_DELTA_T` steps the lowest-scored triplets get `E = 0`.

**Prompt notes.**
* The vectors are initialised from the embeddings of `PROMPT_LEN` random token IDs (seeded) and placed in front of every
  input. For bart they go in front of the encoder input, through `encoder.embed_tokens`, which includes BART's
  embedding scale.

**KD notes.**
* `TEACHER_RUN` is a finished run in `cs781-p6` that saved `train_gen.json`. It can be any architecture or method.
* The student is this notebook's `ARCH` with fewer blocks.
* The student trains only on the teacher-summarised training documents that are not in its held-out set.
* Each time an example is drawn, its target is the teacher's summary with probability `KD_TEACHER_PROB`, else the
  reference.
* The loss is the same token cross-entropy as other methods. No logits and no temperature are used.
* The teacher's `run_config.json` is read and its arch, method, parameter count and held-out scores are printed. A
  warning is printed if the teacher's `heldout_md5` differs from the student's.

**Not implemented.** Prefix tuning is not implemented; `prompt` is the soft-prompt technique offered.

**`ADAPT_TARGETS` and adapter layer names per arch.**
* `q_proj`/`v_proj` exist in all four archs. In bart they cover encoder self-attention, decoder self-attention and
  decoder cross-attention (`encoder_attn`).
* `HOULSBY_AT` covers the attention output and FFN output: `o_proj`/`down_proj` for llama/mistral and `out_proj`/`fc2`
  for bart. `PFEIFFER_AT` covers `down_proj`/`fc2`.
* The logs print the following layer counts:
  * llama, prefixlm, mistral_swa: `lorma: {'q_proj': 12, 'v_proj': 12}`.
  * bart: `lorma: {'v_proj': 18, 'q_proj': 18}`.

**After training** (`merge_adapters`).
* LoRA, LoRMA and AdaLoRA are merged into plain `nn.Linear` weights. An assert checks that the probe logits change by
  less than `1e-2·max(1, max|logit|)`. The logs print `adapter merge: max |logit change|` = 1.43e-05 (llama), 9.11e-05
  (bart), 1.62e-05 (prefixlm), 1.43e-05 (mistral_swa), and 0.00e+00 for the prompt runs.
* QLoRA merges into dequantised fp32 weights when `QLORA_MERGE_FOR_INFERENCE` is set.
* Bottleneck adapters stay as extra modules. Prompt tuning keeps `PROMPT_LEN` extra positions.

## 6. Training procedure

* **Optimizer**: `torch.optim.AdamW`, betas (0.9, 0.999), `fused=True` on CUDA. `WEIGHT_DECAY` applies only to
  parameters with `dim >= 2`; it is 0 for biases, norms and `E`.
* **Precision**: fp32 master weights, `torch.autocast(float16)` forward, and `torch.amp.GradScaler`. Gradients are
  unscaled before clipping. `GRAD_CKPT` turns on HF activation checkpointing.
* **Batching**:
  * Each epoch uses a random permutation (global NumPy seed `SEED`).
  * Micro-batches have `BATCH_SIZE` examples. The optimizer steps every `GRAD_ACCUM` micro-batches, and also at the
    last micro-batch of an epoch.
  * The loss is divided by `GRAD_ACCUM`.
* **Clipping**: `clip_grad_norm_(trainable, GRAD_CLIP)`.
* **LR schedule**: linear warmup over the first `WARMUP_FRAC` of *progress*, then linear decay to 0 at progress 1.
  * `progress = min(1, max(step / TOTAL_STEPS, train_seconds / (60·MAX_TRAIN_MINUTES)))`, so the schedule (and the
    AdaLoRA budget) also completes if the time cap is hit first.
  * `TOTAL_STEPS = EPOCHS · ceil(n_train / (BATCH_SIZE·GRAD_ACCUM))`. In the logs this is 3 · 421 = 1263.
  * The LR is set at the start of each accumulation group.
* **Time cap**: training stops when the training seconds (evaluation time excluded) reach `MAX_TRAIN_MINUTES`.
  Throughput is printed after step 20.
* **Held-out evaluation** (`evaluate`) runs at the end of every epoch, after every `EVAL_EVERY_MINUTES` of training
  time, and when the time cap is hit. Each evaluation computes:
  * the teacher-forced held-out loss (mean cross-entropy per reference token) on `EVAL_LOSS_DOCS` held-out documents
    (all 300 by default);
  * greedy summaries (`DECODE` with `num_beams=1`) for the first `EVAL_GEN_DOCS` (48) held-out documents, scored with
    ROUGE-1 / ROUGE-L.
* **Model selection**: `SELECT_METRIC`, which is `"loss"` by default (score = −held-out loss), or `"rouge1"` / `"rougeL"`.
  When the score improves, a CPU snapshot of all trainable parameters and AdaLoRA masks is kept.
* **Early stopping**: training stops after `PATIENCE` evaluations without improvement.
* **After training**: the selected snapshot is restored, adapters are merged, `gen_consistency` runs again, and the
  held-out loss on all 300 documents is printed.
* **Checkpoint / resume**: there is no mid-training resume. Fine-tuned weights are saved only when `SAVE_WEIGHTS`
  (`model.pt`). Run files are uploaded to `<hf-user>/cs781-p6/runs/<RUN_NAME>/` at the end.
* **In the six logs**: three evaluations ran, one at each epoch end (steps 421 / 842 / 1263). No `stopping:` line was
  printed. The selected checkpoint was epoch 3, step 1263 in every run.

## 7. Inference and submission

**Decoding** (`generate`). HF `generate` runs with `do_sample=False`, `pad_token_id=1`, `eos_token_id=0`,
`suppress_tokens` (pad, `<mask>` for bart, and IDs from the tokenizer size up to the embedding rows), plus the `DECODE`
dict:
* `num_beams` (with `early_stopping=True` when > 1)
* `max_new_tokens`, `min_new_tokens`
* `no_repeat_ngram_size`, `repetition_penalty`, `length_penalty`

Batching and padding:
* Documents are sorted by input length and generated `GEN_BATCH` at a time.
* Decoder-only: inputs are left-padded and passed as `inputs_embeds`. With `prompt`, the soft prompt is written right
  before each row's input, after the left padding. `generate` returns new tokens only.
* prefixlm: `bidirectional_prefill` wraps `lm.forward`. The prefill call (more than one position, empty KV cache)
  gets a 4-D mask in which every prompt position sees every real prompt token. Later steps are causal over the cache.
  A missing 2-D mask (HF drops an all-ones mask) is treated as all real tokens.
* bart: the encoder runs once with right padding. `generate(encoder_outputs=...)` is called, and the decoder-start
  token is dropped from the output.

**Post-processing** (`summaries`):
1. Cut at the first `<|endoftext|>` and drop pad tokens.
2. `decode(skip_special_tokens=True).strip()`.
3. Optionally cut to `MAX_SUMMARY_WORDS` words.
4. An empty summary is replaced by the first `EMPTY_FALLBACK_WORDS` words of its document (or `"."`). The number of
   replacements is printed. It was 0 in all logs, for both held-out and test.

**Held-out scoring** (cell *Scoring*). This reproduces `evaluate_phase6.py`:
* ROUGE-1 F1 from `rouge_score` with stemming.
* ROUGE-L F1 from the evaluator's bit-parallel LCS over the same stemmed tokens.
* BERTScore F1 from `bert_score.BERTScorer(lang="en", batch_size=8)` (roberta-large, layer 17, no idf, no baseline
  rescale).

The same cell also prints these measurements:
* summary words (whitespace split);
* **repeated trigrams** = 1 − distinct/total word trigrams (lower-cased `\w+`);
* **novel bigrams** = share of summary bigrams not in its own document;
* the same statistics for the held-out references;
* a reference point: the first N words of the selected input, where N = reference word count, scored with ROUGE.

**Submission.** For each entry of `DECODES` (`main` = `DECODE`, then each `EXTRA_DECODES` entry):
1. Held-out summaries for all 300 documents. ROUGE-1, ROUGE-L, BERTScore and the measurements are printed. The
   predictions are written to `heldout_pred_<name>.json` (dict format). `heldout_truth.json` is written once.
2. Test summaries are written to `prediction_<name>.json` in list format `[{"id", "summary"}]`.
3. The evaluator's checks are repeated as asserts: IDs unique, ID set equals the test IDs, and every summary is a
   non-empty string.
4. `submission_<RUN_NAME>.zip` (or `submission_<RUN_NAME>__<name>.zip`) is written with `prediction.json` as its only
   member at the root (asserted), and downloaded through `google.colab.files.download`.

**Local official check** (the command the notebook prints):

```bash
python evaluate_phase6.py --truth heldout_truth.json --prediction heldout_pred_main.json
```

**Saved per run** (`runs/<RUN_NAME>/` in `<hf-user>/cs781-p6`):
* `run_config.json`: all config values, parameter counts, MB, the selected checkpoint, held-out scores for each
  decoding, coverage, by-length table, teacher info and the training history.
* `heldout_truth.json`, `heldout_pred_*.json`, `heldout_details.json` (per-document scores), `prediction_*.json`,
  `submission_*.zip`, `curves.png`.
* `train_gen.json` (when `SAVE_TRAIN_GEN`): summaries generated with `DECODE` for up to `TRAIN_GEN_MAX_DOCS` training
  documents this run trained on. A later `kd` run can use this file as its teacher data.

The last cell lists every run in `cs781-p6` and counts the Phase-1 models and §12.2.1 techniques that have runs.

## 8. Configuration knobs

The first code cell holds `ARCH`, `METHOD`, `DOC_SELECT` and `RUN_NAME`. The configuration cell is headed "all values
below are YOUR CHOICE".

| Name | Template default | Meaning |
|---|---|---|
| `ARCH` | fixed per notebook (`llama` / `bart` / `prefixlm` / `mistral_swa`) | Phase-1 architecture |
| `METHOD` | `"lorma"` | `full` \| `lora` \| `qlora` \| `lorma` \| `adalora` \| `houlsby` \| `pfeiffer` \| `prompt` \| `kd` |
| `DOC_SELECT` | `"head"` | `head` \| `head_tail` \| `even` \| `importance` (section 3) |
| `RUN_NAME` | `f"{ARCH}_{METHOD}_{DOC_SELECT}"` | run folder / zip name; add a suffix when other settings change |
| `DATASET` | `"Exploration-Lab/CS-781-Capstone"` | HF dataset repo |
| `DATA_DIR` | `"Phase_6_Summ"` | phase folder in the dataset (searched for if absent) |
| `P1_REPOS` | `cs781-p1-llama`, `cs781-p1-bart-v2`, `cs781-p1-prefixlm`, `cs781-p1-mistral_swa` (under `<hf-user>`) | Phase-1 checkpoints (`best/`) |
| `P6_REPO` | `<hf-user>/cs781-p6` | private repo for all Phase-6 runs |
| `WORK` / `RUN_DIR` / `CACHE_DIR` | `/content/p6` (else `/tmp/p6`), `WORK/RUN_NAME`, `WORK/cache` | local folders |
| `SEED` | `42` | Python / NumPy / torch seed; held-out split; prompt init |
| `HELDOUT_DOCS` | `300` | training documents held out for validation |
| `TRAIN_MAX_DOCS` | `None` | random subset of training documents (None = all) |
| `SRC_LEN` | `768` (bart) / `512` (others) | document tokens given to the model |
| `TGT_LEN` | `512` (bart) / `480` (others) | summary tokens trained on |
| `SEP_TEXT` | `"\n\nSummary:\n"` | decoder-only separator between document and summary |
| `HEAD_FRAC` | `0.5` | `head_tail`: share of `SRC_LEN` from the start |
| `EVEN_PIECE` | `128` | `even`: tokens per piece |
| `TFIDF_MAX_FEATURES` | `50000` | `importance`: TF-IDF vocabulary size |
| `EPOCHS` | `3` | passes over the training documents |
| `MAX_TRAIN_MINUTES` | `60` | training-time cap (evaluation excluded) |
| `EVAL_EVERY_MINUTES` | `15` | held-out evaluation interval (also at each epoch end) |
| `PATIENCE` | `3` | evaluations without improvement before stopping |
| `BATCH_SIZE` | `4` | examples per forward pass |
| `GRAD_ACCUM` | `4` | micro-batches per optimizer step (effective batch 16) |
| `GRAD_CKPT` | `False` | activation checkpointing |
| `LR` | full `5e-5`; lora / qlora / lorma / adalora / houlsby / pfeiffer `5e-4`; prompt `3e-3`; kd `1e-4` | peak learning rate per method |
| `WEIGHT_DECAY` | `0.01` | AdamW decay on ≥2-D parameters |
| `WARMUP_FRAC` | `0.06` | warmup share of progress |
| `GRAD_CLIP` | `1.0` | max gradient norm |
| `LABEL_SMOOTHING` | `0.0` | cross-entropy label smoothing |
| `SELECT_METRIC` | `"loss"` | checkpoint selection: `loss` \| `rouge1` \| `rougeL` |
| `EVAL_LOSS_DOCS` | `None` | held-out docs for the loss (None = all) |
| `EVAL_GEN_DOCS` | `48` | held-out docs summarised (greedy) for ROUGE at each evaluation |
| `DECODE` | `num_beams=1, max_new_tokens=TGT_LEN, min_new_tokens=0, no_repeat_ngram_size=0, repetition_penalty=1.0, length_penalty=1.0` | decoding for the held-out and test summaries |
| `EXTRA_DECODES` | `{}` | name → overrides of `DECODE`; each entry gets its own held-out scores and zip |
| `GEN_BATCH` | `16` | documents per generation batch |
| `MAX_SUMMARY_WORDS` | `None` | word cut on generated summaries |
| `EMPTY_FALLBACK_WORDS` | `50` | words of the document that replace an empty summary |
| `BERTSCORE` | `True` | compute held-out BERTScore (roberta-large download) |
| `ADAPT_TARGETS` | `["q_proj", "v_proj"]` | linear layers for lora / qlora / lorma / adalora |
| `RANK`, `ALPHA` | `8`, `16` | rank and scale (α/r) for lora / qlora / lorma |
| `QLORA_BLOCK` | `64` | weights per NF4 block |
| `QLORA_DOUBLE_QUANT` | `True` | 8-bit absmax constants |
| `QLORA_MERGE_FOR_INFERENCE` | `True` | dequantise and merge before generation |
| `LORMA_VARIANT` | `"plus"` | `plus` (additive identity) \| `pi` (permutation inflation) |
| `ADALORA_INIT_R`, `ADALORA_TARGET_R` | `12`, `4` | initial / final average rank per layer |
| `ADALORA_TINIT`, `ADALORA_TFINAL` | `0.1`, `0.2` | progress share before pruning / with the budget fixed |
| `ADALORA_DELTA_T` | `10` | steps between rank reallocations |
| `ADALORA_BETAS` | `(0.85, 0.85)` | EMA factors (sensitivity, uncertainty) |
| `ADALORA_ORTH` | `0.1` | orthogonality-penalty weight |
| `HOULSBY_AT` | `["o_proj", "down_proj", "out_proj", "fc2"]` | layers wrapped by Houlsby adapters |
| `PFEIFFER_AT` | `["down_proj", "fc2"]` | layers wrapped by Pfeiffer adapters |
| `ADAPTER_DIM` | `64` | bottleneck size |
| `ADAPTER_TRAIN_NORMS` | `True` | also train layer norms (adapters) |
| `PROMPT_LEN` | `20` | soft-prompt length |
| `TEACHER_RUN` | `"llama_full_head"` | KD teacher run in `P6_REPO` (must have `train_gen.json`) |
| `STUDENT_LAYERS` | `None` | KD student blocks (None = every 2nd block per stack) |
| `KD_TEACHER_PROB` | `1.0` | probability that a KD target is the teacher summary |
| `SAVE_TRAIN_GEN` | `True` | write `train_gen.json` so the run can be a KD teacher |
| `TRAIN_GEN_MAX_DOCS` | `1000` | training documents summarised for `train_gen.json` |
| `SAVE_WEIGHTS` | `False` | upload `model.pt` |

## 9. Runs recorded in `run_logs/`

All six logs ran on a Colab `Tesla T4` with torch `2.11.0+cu130` and transformers `5.22.0` (printed).

**Settings.** The only config edit compared with the template is `METHOD = "prompt"` in the two prompt logs. Every other
value is the template default. The values in the table are therefore the effective values, all with `DOC_SELECT=head`,
`EPOCHS=3`, `BATCH_SIZE=4 × GRAD_ACCUM=4`, `MAX_TRAIN_MINUTES=60`, `SELECT_METRIC=loss`, `RANK=8`, `ALPHA=16` and
`LORMA_VARIANT=plus` (lorma), `PROMPT_LEN=20` (prompt), greedy `DECODE` with no repetition controls, and
`EXTRA_DECODES={}`.

The two `*_prompt_after` logs show execution counts 17-32 and read the tokenised data from cache. They were run in the
same runtime after the lorma log of the same arch.

| File | RUN_NAME | Arch | METHOD | Differs from template | LR | SRC_LEN / TGT_LEN / max_new_tokens |
|---|---|---|---|---|---|---|
| `P6_arch1_llama_after.ipynb` | `llama_lorma_head` | llama | lorma | none | 5e-4 | 512 / 480 / 480 |
| `P6_arch2_bart_after.ipynb` | `bart_lorma_head` | bart | lorma | none | 5e-4 | 768 / 512 / 512 |
| `P6_arch3_prefixlm_after.ipynb` | `prefixlm_lorma_head` | prefixlm | lorma | none | 5e-4 | 512 / 480 / 480 |
| `P6_arch4_mistral_swa_after.ipynb` | `mistral_swa_lorma_head` | mistral_swa | lorma | none | 5e-4 | 512 / 480 / 480 |
| `P6_arch1_llama_prompt_after.ipynb` | `llama_prompt_head` | llama | prompt | `METHOD="prompt"` | 3e-3 | 512 / 480 / 480 |
| `P6_arch2_bart_prompt_after.ipynb` | `bart_prompt_head` | bart | prompt | `METHOD="prompt"` | 3e-3 | 768 / 512 / 512 |

**Results.** All numbers are copied from the logs. Held-out = the 300 held-out training documents (id hash
`eddb1ace9498`), `main` decoding. "Train min" is training time, with wall time (including evaluations) in brackets.

| RUN_NAME | Trainable / total params | Params / MB for inference | Train min (wall) | Peak GPU train | Held-out loss | **BERTScore F1** | ROUGE-1 | ROUGE-L | Held-out gen: s (docs/s), peak GPU | Test gen s | `train_gen` s (1000 docs) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `llama_lorma_head` | 294,912 / 109,824,768 (0.27%) | 109,529,856 / 418 | 21.7 (23.9) | 3.00 GB | 2.5581 | 0.7719 | 0.2325 | 0.1860 | 207 (1.45), 1.95 GB | 74 | 689 |
| `bart_lorma_head` | 442,368 / 125,884,416 (0.35%) | 125,442,048 / 479 | 14.2 (15.5) | 2.76 GB | 2.7777 | 0.7693 | 0.2383 | 0.1909 | 101 (2.98), 1.35 GB | 37 | 331 |
| `prefixlm_lorma_head` | 294,912 / 109,824,768 (0.27%) | 109,529,856 / 418 | 22.8 (25.0) | 3.00 GB | 2.7938 | 0.7643 | 0.2182 | 0.1817 | 210 (1.43), 1.95 GB | 75 | 697 |
| `mistral_swa_lorma_head` | 196,608 / 100,289,280 (0.20%) | 100,092,672 / 382 | 22.1 (24.2) | 2.88 GB | 2.8143 | 0.7622 | 0.2176 | 0.1832 | 186 (1.61), 1.10 GB | 68 | 619 |
| `llama_prompt_head` | 15,360 / 109,545,216 (0.01%) | 109,545,216 / 418 | 21.2 (23.4) | 3.00 GB | 2.6636 | 0.7657 | 0.2302 | 0.1855 | 211 (1.42), 1.97 GB | 75 | 703 |
| `bart_prompt_head` | 15,360 / 125,457,408 (0.01%) | 125,457,408 / 479 | 12.9 (14.0) | 2.73 GB | 3.0208 | 0.7677 | 0.2388 | 0.1935 | 100 (3.00), 1.40 GB | 37 | 332 |

**Held-out evaluations during training.** Each cell is train loss / held-out loss / ROUGE-1 / ROUGE-L, with ROUGE
computed greedily on 48 docs. The training minutes at each evaluation are in brackets.

| RUN_NAME | throughput (ex/s) | epoch 1, step 421 | epoch 2, step 842 | epoch 3, step 1263 |
|---|---|---|---|---|
| `llama_lorma_head` | 15.85 | 2.6228 / 2.5775 / 0.2101 / 0.1722 (7.2) | 2.5629 / 2.5620 / 0.2084 / 0.1743 (14.5) | 2.5468 / 2.5581 / 0.2085 / 0.1744 (21.7) |
| `bart_lorma_head` | 21.91 | 2.8903 / 2.8140 / 0.2204 / 0.1751 (4.7) | 2.7907 / 2.7860 / 0.2241 / 0.1843 (9.4) | 2.7666 / 2.7777 / 0.2159 / 0.1754 (14.2) |
| `prefixlm_lorma_head` | 15.83 | 2.8582 / 2.8169 / 0.1994 / 0.1649 (7.4) | 2.7958 / 2.7983 / 0.2040 / 0.1713 (15.1) | 2.7791 / 2.7938 / 0.2084 / 0.1759 (22.8) |
| `mistral_swa_lorma_head` | 13.77 | 2.8665 / 2.8285 / 0.2030 / 0.1742 (7.5) | 2.8208 / 2.8169 / 0.2083 / 0.1756 (14.8) | 2.8093 / 2.8143 / 0.2061 / 0.1712 (22.1) |
| `llama_prompt_head` | 15.83 | 2.6909 / 2.6710 / 0.2297 / 0.1839 (7.1) | 2.6703 / 2.6657 / 0.2210 / 0.1790 (14.2) | 2.6671 / 2.6636 / 0.2265 / 0.1818 (21.2) |
| `bart_prompt_head` | 21.81 | 3.1446 / 3.0779 / 0.2129 / 0.1821 (4.3) | 3.0527 / 3.0318 / 0.2206 / 0.1869 (8.6) | 3.0264 / 3.0208 / 0.2142 / 0.1834 (12.9) |

**Held-out summary measurements** (main decoding, printed).
* The held-out references average 866 words (median 659), with repeated trigrams 0.099 and novel bigrams 0.228.
* Reference-point ROUGE-1 / ROUGE-L (first N words of the selected input, N = reference word count): 0.4289 / 0.2115
  for the decoder-only runs (SRC_LEN 512) and 0.4969 / 0.2406 for bart (SRC_LEN 768).

| RUN_NAME | Mean words (held-out) | Mean words (test) | Repeated trigrams | Novel bigrams | Generated/reference words: median, p10, p90 | Empty replaced |
|---|---|---|---|---|---|---|
| `llama_lorma_head` | 411 | 414 | 0.911 | 0.296 | 0.62, 0.29, 1.53 | 0 |
| `bart_lorma_head` | 449 | 452 | 0.911 | 0.390 | 0.68, 0.29, 1.61 | 0 |
| `prefixlm_lorma_head` | 413 | 418 | 0.937 | 0.332 | 0.64, 0.29, 1.47 | 0 |
| `mistral_swa_lorma_head` | 424 | 424 | 0.947 | 0.496 | 0.64, 0.31, 1.52 | 0 |
| `llama_prompt_head` | 416 | 407 | 0.929 | 0.342 | 0.63, 0.29, 1.49 | 0 |
| `bart_prompt_head` | 439 | 447 | 0.912 | 0.335 | 0.67, 0.27, 1.47 | 0 |

<details>
<summary>Held-out scores by document length (token quartiles; 75 documents each), as printed</summary>

The quartiles are 412-2458, 2458-3679, 3679-5669 and 5669-98411 tokens. Mean reference words per quartile are 455,
636, 779 and 1597. Each cell is ROUGE-1 / ROUGE-L / BERTScore / mean generated words.

| RUN_NAME | 412-2458 | 2458-3679 | 3679-5669 | 5669-98411 |
|---|---|---|---|---|
| `llama_lorma_head` | 0.2407 / 0.1938 / 0.7777 / 413 | 0.2438 / 0.1989 / 0.7736 / 409 | 0.2388 / 0.1855 / 0.7701 / 409 | 0.2065 / 0.1660 / 0.7662 / 412 |
| `bart_lorma_head` | 0.2487 / 0.2017 / 0.7755 / 440 | 0.2466 / 0.1969 / 0.7687 / 453 | 0.2467 / 0.1958 / 0.7683 / 456 | 0.2114 / 0.1691 / 0.7646 / 446 |
| `prefixlm_lorma_head` | 0.2066 / 0.1792 / 0.7672 / 410 | 0.2352 / 0.1946 / 0.7662 / 418 | 0.2222 / 0.1832 / 0.7634 / 406 | 0.2087 / 0.1700 / 0.7602 / 420 |
| `mistral_swa_lorma_head` | 0.2174 / 0.1892 / 0.7648 / 419 | 0.2393 / 0.1996 / 0.7628 / 428 | 0.2204 / 0.1834 / 0.7605 / 427 | 0.1932 / 0.1607 / 0.7605 / 421 |
| `llama_prompt_head` | 0.2324 / 0.1901 / 0.7685 / 418 | 0.2503 / 0.2010 / 0.7685 / 420 | 0.2300 / 0.1855 / 0.7642 / 408 | 0.2080 / 0.1653 / 0.7616 / 417 |
| `bart_prompt_head` | 0.2224 / 0.1847 / 0.7750 / 413 | 0.2506 / 0.2062 / 0.7669 / 447 | 0.2511 / 0.2005 / 0.7650 / 445 | 0.2311 / 0.1824 / 0.7639 / 453 |

</details>

**Requirement count printed by the last cell** of the last-run logs (`P6_arch1_llama_prompt_after.ipynb`,
`P6_arch2_bart_prompt_after.ipynb`), verbatim:

```
Phase-1 models with runs: ['bart', 'llama', 'mistral_swa', 'prefixlm']  (missing: [])
§12.2.1 techniques with runs (besides KD): ['lorma', 'prompt tuning'] (2)  LoRMA: True  KD: False;  held-out sets: 1 distinct (scores are comparable only within the same held-out set)
```

No `kd`, `lora`, `qlora`, `adalora`, `houlsby`, `pfeiffer` or `full` run is recorded in `run_logs/`.
TODO(student): add the logs of any further Phase-6 runs (e.g. the KD run and a third §12.2.1 technique) if they exist
outside this folder.

## 10. Codabench submissions

Local folders carry the last five digits of their Codabench ID (`submission_<RUN_NAME><ID>`). Each folder's
`prediction.json` is byte-identical (MD5) to the zip downloaded from Colab. All on 2026-11-03 (IST).

| Codabench ID | Submitted | Submission | Arch | Method / key settings | BERTScore F1 |
|---|---|---|---|---|---|
| 1030611 | 20:57 | `submission_bart_lorma_head` (**leaderboard entry**) | bart | lorma (plus, r=8, α=16, q/v), head 768, greedy 512 | 0.7662 |
| 1030797 | 22:15 | `submission_mistral_swa_lorma_head` | mistral_swa | lorma (plus, r=8, α=16, q/v), head 512, greedy 480 | 0.7631 |
| 1030802 | 22:17 | `submission_prefixlm_lorma_head` | prefixlm | lorma (plus, r=8, α=16, q/v), head 512, greedy 480 | 0.7592 |
| 1030867 | 22:31 | `submission_llama_lorma_head` (**highest score**) | llama | lorma (plus, r=8, α=16, q/v), head 512, greedy 480; 0.295M of 109.8M trainable; batch 4 × 4; best held-out loss at step 1263 (epoch 3) | **0.7686** |
| 1030868 | 22:32 | `submission_bart_prompt_head` | bart | prompt (20 tokens), head 768, greedy 512 | 0.7636 |
| 1030870 | 22:33 | `submission_llama_prompt_head` | llama | prompt (20 tokens), head 512, greedy 480 | 0.7658 |

The submission shown on the leaderboard (1030611) is not the highest-scoring one (1030867).

## 11. How to run

1. **Build the notebooks** (optional; the rendered notebooks are already in this folder):
   ```bash
   cd code/phase6
   python build_p6_notebooks.py            # all four
   python build_p6_notebooks.py bart       # only P6_arch2_bart.ipynb
   ```
   This overwrites `P6_arch*.ipynb` in this folder.
2. **Open the notebook in Colab** for the architecture you want (`P6_arch{1..4}_<arch>.ipynb`). Then set *Runtime →
   Change runtime type → T4 GPU*. Kaggle (`kaggle_secrets`) and an `HF_TOKEN` environment variable also work. A CPU run
   needs `CS781_ALLOW_CPU=1`.
3. **Add the secret** `HF`: the same Hugging Face token as Phases 1-5. It must be able to read `<hf-user>/cs781-p1-*`
   and the gated `Exploration-Lab/CS-781-Capstone` dataset, and write `<hf-user>/cs781-p6`.
4. **Edit the first code cell**: `METHOD` and `DOC_SELECT`. `RUN_NAME` follows from them; add a suffix if you change
   other settings, so runs do not overwrite each other. Check the *Configuration* cell.
5. **Run all.** `rouge-score` and `bert-score` are pip-installed if missing. The notebook then:
   1. downloads data and the Phase-1 checkpoint;
   2. runs the sanity checks;
   3. trains with held-out evaluations;
   4. merges adapters;
   5. generates held-out and test summaries for each decoding setting.
6. **Outputs:**
   * Local: `/content/p6/<RUN_NAME>/`.
   * Hub: `<hf-user>/cs781-p6/runs/<RUN_NAME>/` (uploaded at the end).
   * `submission_<RUN_NAME>.zip` (plus `submission_<RUN_NAME>__<name>.zip` per `EXTRA_DECODES` entry) is downloaded to
     the browser automatically. Upload it to Codabench as is; it contains only `prediction.json`.
7. **Another experiment in the same runtime**: change the first cell and *Run all* again. Tokenised data is reused
   from `/content/p6/cache`, and the model is reloaded from the Phase-1 checkpoint.
8. **Knowledge distillation ordering:**
   1. Finish a teacher run first. It can be any arch or method, with `SAVE_TRAIN_GEN = True` (the default), so that
      `runs/<teacher>/train_gen.json` exists in `cs781-p6`. Each of the six recorded runs printed
      `train_gen.json: 1000 training documents summarised`.
   2. In the student notebook (any arch), set `METHOD = "kd"` and `TEACHER_RUN = "<teacher RUN_NAME>"`. Optionally set
      `STUDENT_LAYERS` and `KD_TEACHER_PROB`.
   3. If `TEACHER_RUN` is not a finished run, the config cell raises an error that lists the finished runs. Use the
      same `SEED` / `HELDOUT_DOCS` as the teacher; otherwise a warning about unclean held-out scores is printed.
9. **Local evaluation with the official script** on the held-out files saved by the run:
   ```bash
   python evaluate_phase6.py --truth heldout_truth.json --prediction heldout_pred_main.json
   ```
