# Phase 4 — Court Judgment Prediction (CJPE)

Fine-tunes each of the four Phase-1 legal language models (`llama`, `bart`, `prefixlm`, `mistral_swa`) to predict the
binary outcome (0 / 1) of a long Indian court judgment. All code lives in one percent-format template that is rendered
into one Colab notebook per architecture.

> This README describes the code and copies numbers verbatim from the executed notebooks in `run_logs/`. It contains no
> result analysis; that is left to the student's report.

---

## 1. Task

Capstone §10 (Phase 4, CJPE), formulated as binary classification.

| | |
|---|---|
| Input | one record per case: `{"id": "case_001", "text": "..."}`; `train.json` / `dev.json` records also carry the binary label |
| Output | a JSON list with exactly one object per test case, `[{"id": "case_001", "pred": 1}, {"id": "case_002", "pred": 0}]`; `pred` is an integer, 0 or 1; IDs are those of `test.json` |
| Submission | `submission.zip` containing only `prediction.json` (this code names the zip `submission_<RUN_NAME>.zip`) |
| Primary metric | Macro-F1 (over labels 0 and 1) |
| Secondary metrics | Precision, Recall (of label 1, `average="binary"`, `pos_label=1`, as in `evaluate_phase4.py`) |

Course constraints quoted in the template header (§1, §10.2): at least 6 submissions (the capstone document later lowered this to 4 for every phase); all four Phase-1 models used;
at least 4 substantially different experiments, of which at least 3 use different techniques from the §10.2.1 list
(full, partial / layer-wise, layer freezing / unfreezing, BitFit, LoRMA, AdaLoRA, adapters, KD), LoRMA compulsory, and
at least 1 knowledge-distillation experiment. Every experiment starts from the Phase-1 checkpoint.

## 2. Files in this folder

| File | Description |
|---|---|
| `phase4_cjpe_template.py` | The single source of the Phase-4 pipeline (percent format, `# %%` cells, placeholders `{{ARCH}}`, `{{TAG}}`, `{{FAMILY}}`, `{{MODEL}}`). |
| `build_p4_notebooks.py` | Fills the placeholders for each architecture and writes `P4_arch{1..4}_<arch>.ipynb`; `python build_p4_notebooks.py bart` builds only one. |
| `build_nb.py` | Generic converter: percent-format `.py` -> `.ipynb` (markdown cells from `# %% [markdown]`, code cells from `# %%`). |
| `P4_arch1_llama.ipynb` | Generated notebook, Arch 1, decoder-only (`LlamaModel`, causal attention by default). |
| `P4_arch2_bart.ipynb` | Generated notebook, Arch 2, encoder-decoder (`cs781-p1-bart-v2`, encoder only by default). |
| `P4_arch3_prefixlm.ipynb` | Generated notebook, Arch 3, prefix-LM hybrid (`LlamaModel`, bidirectional attention inside a chunk by default). |
| `P4_arch4_mistral_swa.ipynb` | Generated notebook, Arch 4, sliding-window attention + GQA (`MistralModel`, causal by default). |
| `run_logs/P4_arch1_llama_after.ipynb` | Executed notebook: run `llama_lorma_even_mean`. |
| `run_logs/P4_arch2_bart_after.ipynb` | Executed notebook: run `bart_lorma_even_mean`. |
| `run_logs/P4_arch2_bart_kd_after.ipynb` | Executed notebook: run `bart_kd_even_mean` (student; teacher `llama_lorma_even_mean`). |
| `run_logs/P4_arch3_prefixlm_after.ipynb` | Executed notebook: run `prefixlm_lorma_even_mean`. |

The four generated notebooks (31 cells each) are byte-identical to what `build_p4_notebooks.py` produces from the
current template. There is no executed `mistral_swa` notebook in `run_logs/` (see §9).

Official evaluator (course-provided, not in this folder): `evaluate_phase4.py --truth <truth.json> --prediction <pred.json>`.

## 3. Data and preprocessing

**Source.** Hugging Face dataset `Exploration-Lab/CS-781-Capstone` (gated), files
`Phase_4_CJPE/Train/train.json`, `Phase_4_CJPE/Train/dev.json`, `Phase_4_CJPE/Test/test.json`, fetched with
`hf_hub_download`. Files may be a JSON list, a dict `id -> record`, or JSON Lines. The label is read from `label`
(or `gold`) and must be `0` / `1`; a `text` given as a list is joined with newlines.

**De-duplication.** A repeated `id` inside a file is dropped (first occurrence kept) and counted; documents with
identical text across splits are counted by MD5 hash (measurement only, nothing removed). Output of the data cell,
identical in all four run logs:

```
Train/train.json: list of 37387 ...
  !! 5082 repeated ids dropped (first kept; 0 of them with a different label)
Train/dev.json: list of 1988 ...
  !! 994 repeated ids dropped (first kept; 0 of them with a different label)
Test/test.json: list of 1517 ...

split    docs  label 0  label 1 chars median       p90       max  empty
train   32305    18920    13385        12120     38371   2492701      0
dev       994      497      497        16700     40429    260198      0
test     1517        -        -        17320     42859    233000      0
documents with identical text in two splits: train&dev 0  train&test 0  dev&test 0
```

**Splits.** No held-out split is carved from `train.json`. Training uses `train.json` (optionally a label-stratified
random subset of `TRAIN_MAX_DOCS` documents, seed `SEED`; default `None` = all documents). Model selection and all dev
numbers use the official `dev.json`. Test predictions are made for `test.json`. For `METHOD = "kd"` the training set is
restricted to the documents the teacher scored (§5).

**Tokenisation.** The Phase-1 tokenizer of the architecture (`PreTrainedTokenizerFast` from `<P1 repo>/best/`) encodes
each whole document without special tokens; ids are stored as `uint16` (asserted `vocab <= 65535`). An empty document
becomes a single EOS token. Tokenised splits are cached in `CACHE_DIR` as `<split>_<md5(tokenizer.json)[:12]>.npz` and
reused by later runs in the same runtime.

**Chunking.** For a document of `n` tokens, chunk starts are `0, CHUNK_STRIDE, 2·CHUNK_STRIDE, ...` (while a full
`CHUNK_LEN` window fits) plus one final chunk ending exactly at token `n`, so every token is in at least one chunk
(asserted for every document). With the defaults `CHUNK_LEN = CHUNK_STRIDE = 512` chunks do not overlap except the last
one. A document with at most `MAX_CHUNKS` (training) / `MAX_CHUNKS_PRED` (prediction) chunks keeps all of them;
otherwise `CHUNK_SELECT` picks that many, kept in document order:

| `CHUNK_SELECT` | chunks kept (out of `m`, keep `k`) |
|---|---|
| `first` | chunks `0 .. k-1` |
| `last` | chunks `m-k .. m-1` |
| `first_last` | `ceil(k/2)` from the start, the rest from the end |
| `even` | `round(linspace(0, m-1, k))` (evenly spaced over the document) |
| `random` | a fresh random subset at each training visit (`TRAIN_RNG`, seed `SEED`); `even` at prediction |

Token / chunk statistics printed by the chunk cell (llama, bart and prefixlm logs print the same table):

```
split  tokens/doc median     p90      max chunks/doc median   max > MAX_CHUNKS_PRED tokens seen at prediction
train               2479    7840   497640                 5   972             28.9%                     62.8%
dev                 3806    9311    60714                 8   119             45.5%                     63.3%
test                3955    9689    56924                 8   112             47.3%                     62.2%
32305 training documents -> 4039 optimizer steps per epoch;  coverage and alignment checks passed
```

The KD log (10,000-document training subset; dev / test rows as above) prints:

```
train               2467    7682   497640                 5   972             28.5%                     63.8%
10000 training documents -> 1250 optimizer steps per epoch;  coverage and alignment checks passed
```

**Batching.** A batch of `B` documents becomes one `(Q, L)` tensor holding all their selected chunks as rows (right
padded with `PAD_ID`), plus per-row document index, slot (chunk order) and relative-position bin. Built-in checks: 20
random training batches and one dev batch are pooled on the token ids themselves to verify that each document gets back
its own chunks in order and its own label.

**Labels.** Integer class 0 / 1, used directly as cross-entropy targets (2 logits). Optional class weights from
training label counts (`CLASS_WEIGHTS`): `inverse` = `N / (2·N_c)`, `inverse_sqrt` = its square root.

## 4. Model architecture

Pipeline: Phase-1 backbone -> token states of each chunk -> chunk vector -> document vector -> head -> 2 logits.

**Backbone per architecture** (weights from `<hf-user>/cs781-p1-<arch>/best/`, cast to float32):

| ARCH | Loaded class | Module kept | Token states used | Attention inside a chunk (`ATTN` default) |
|---|---|---|---|---|
| `llama` | `LlamaForCausalLM` | `.model` (`LlamaModel`) | `last_hidden_state` | `causal` (as pretrained) |
| `prefixlm` | `LlamaForCausalLM` | `.model` (`LlamaModel`) | `last_hidden_state` | `bidirectional` (the whole chunk is the prefix) |
| `mistral_swa` | `MistralForCausalLM` | `.model` (`MistralModel`) | `last_hidden_state` | `causal` (sliding window) |
| `bart` (`cs781-p1-bart-v2`) | `BartForConditionalGeneration` | `.model.encoder` (`BART_MODE = "encoder"`) or `.model` (`"decoder"`) | encoder `last_hidden_state`, or decoder `last_hidden_state` with the chunk fed to both encoder and decoder | n/a |

`ATTN = "bidirectional"` for a decoder-only model passes a 4-D boolean mask `(Q, 1, L, L)` that lets every token attend
to every non-pad token; for `mistral_swa` the mask is additionally limited to the band `|i - j| < sliding_window`. The
LM heads are not used. `CHUNK_LEN` must not exceed the checkpoint's `max_position_embeddings` (asserted; the template
comment notes Phase-1 positions = 1024 and that the BART encoder was pretrained on <= 768 tokens).

Phase-1 model sizes are set in `code/phase1/phase1_arch_template.py` (`D_MODEL, N_LAYER, N_HEAD, D_FF = 768, 12, 12,
2048`; `mistral_swa`: 4 KV heads, window 256; `bart`: 6 encoder + 6 decoder layers, FFN 3072). The Phase-4 code reads
the values from each checkpoint's config. Phase-4 logs report 12 adapted `q_proj` / `v_proj` layers each for llama and
prefixlm and 6 each for the BART encoder.

**Chunk vector** `c_k ∈ R^H` (`POOL`): `mean` = mean of the token states over real (non-pad) tokens; `first` = first
token state; `last` = state of the last real token. Chunks are encoded independently. Optional (`CHUNK_POS = True`):
`c_k += E_pos[bucket_k]`, a learned `nn.Embedding(POS_BUCKETS, H)` initialised to zero, where
`bucket_k = min(POS_BUCKETS-1, floor(POS_BUCKETS · (start_k + len_k/2) / n))`.

**Document vector** `v` (`DOC_POOL`, over the document's real chunks):

| `DOC_POOL` | Formula | Dim |
|---|---|---|
| `mean` | `v = (1/K) Σ_k c_k` | H |
| `max` | element-wise max over `c_k` | H |
| `attn` | `a = softmax_k(wᵀ tanh(W c_k))`, `v = Σ_k a_k c_k`; `W: H -> ATT_HIDDEN`, `w: ATT_HIDDEN -> 1` (no bias) (hierarchical attention pooling, Yang et al., 2016) | H |
| `gru` | bidirectional `nn.GRU(H, GRU_HIDDEN)` over the chunk sequence (packed by chunk count); `v` = concatenated final forward and backward states | 2·GRU_HIDDEN |

**Head** (`HEAD`): `linear` = `Linear(D, 2)`; `mlp` = `Linear(D, HEAD_HIDDEN) -> tanh -> Dropout(DROPOUT) ->
Linear(HEAD_HIDDEN, 2)`. Input is `Dropout(DROPOUT)(v)`. Pooling and head run in float32 (autocast disabled).
`p(1) = softmax(logits)[1]`.

**Loss.** `L = CE(logits, y; class_w)`; for `kd` `L = KD_ALPHA·CE + (1 - KD_ALPHA)·T²·KL(softmax(t/T) ‖ softmax(s/T))`
(implemented as `F.kl_div(log_softmax(s/T), softmax(t/T), reduction="batchmean")`, `T = KD_T`, `t` = teacher
scores, `s` = student logits); for `adalora` `+ ADALORA_ORTH · mean_layers(‖PᵀP − I‖_F + ‖QQᵀ − I‖_F)`.

**Parameter counts printed by the logs** (CJPE model = backbone + pooling + head):
llama 109.5M, prefixlm 109.5M, bart (encoder mode) 67.9M, bart KD student 46.7M (`kept blocks [[0, 2, 4]]`).

**Sanity checks run before training** (`model_checks`): changing the last token of a 16-token chunk must change the
first token's state only for bidirectional attention / BART (threshold 1e-4); a document's logits must not change when
a longer document joins the batch (`allclose`, atol 1e-4). Printed lines (log file in the comment line above each):

```
# P4_arch1_llama_after.ipynb
llama (attn=causal, bart_mode=encoder): changing a chunk's last token moves its first token's state by 0.00e+00, its own by 7.39e+00 -> sees right context: False (expected False); logits padding-invariant: True (max diff 5.2e-08)
# P4_arch2_bart_after.ipynb
bart (attn=n/a, bart_mode=encoder): changing a chunk's last token moves its first token's state by 1.15e-01, its own by 4.45e+00 -> sees right context: True (expected True); logits padding-invariant: True (max diff 2.4e-08)
# P4_arch2_bart_kd_after.ipynb (3-block student)
bart (attn=n/a, bart_mode=encoder): changing a chunk's last token moves its first token's state by 1.57e-01, its own by 3.02e+00 -> sees right context: True (expected True); logits padding-invariant: True (max diff 8.9e-08)
# P4_arch3_prefixlm_after.ipynb
prefixlm (attn=bidirectional, bart_mode=encoder): changing a chunk's last token moves its first token's state by 1.53e+00, its own by 5.71e+00 -> sees right context: True (expected True); logits padding-invariant: True (max diff 6.0e-08)
```

## 5. Fine-tuning methods implemented

All methods are implemented in the template itself (no `peft` library). In every method the head, the document pooling
module and the chunk-position embedding are trainable. Every method except `full`, `partial`, `unfreeze` and `kd` is
checked to leave the model output unchanged at initialisation (max |logit change| < 1e-3, printed as
`output change at init`).

| `METHOD` | Trainable parameters | Implementation | Knobs |
|---|---|---|---|
| `full` | all | `requires_grad = True` everywhere | `LR`, `LAYER_LR_DECAY` |
| `partial` | top `PARTIAL_TOP` blocks + final norm(s) | rest of the backbone frozen; for BART in decoder mode the block order is encoder blocks then decoder blocks | `PARTIAL_TOP` |
| `unfreeze` | head first, then blocks from the top | gradual unfreezing (Howard & Ruder, 2018): groups = [top block + final norms], next block down, ..., [embeddings if `UNFREEZE_EMBED`]; at each optimizer step `k = min(G, int(progress·G / UNFREEZE_FRAC))` groups are trainable, so all are trainable from `UNFREEZE_FRAC` of training; all groups are in the optimizer from the start | `UNFREEZE_FRAC`, `UNFREEZE_EMBED` |
| `bitfit` | every backbone parameter named `*.bias` | Ben Zaken et al., 2022; with `BITFIT_ADD_BIAS` each bias-free `nn.Linear` of the backbone (llama / mistral) gets a zero-initialised bias | `BITFIT_ADD_BIAS` |
| `lorma` (compulsory) | `A ∈ R^{r×d}`, `B ∈ R^{d×r}` per adapted layer (`d` = out features) | `LoRMALinear` (Bihany, Patel & Modi, 2025), multiplicative update `h = 𝓘(BA)·W0 x`. `plus`: `h = W0x + (α/r)·B(A(W0x))`, `A` Kaiming-uniform, `B = 0`. `pi`: `𝓘(BA)[i, j] = (BA)[i, (j − i) mod d]`, scale 1, init `B[:, 0] = 1`, `A = e₀e₀ᵀ` (so `𝓘(BA) = I`). Base weight and bias frozen; bias added after the product. Merged after training: `W = W0 + (α/r)·B A W0` (`plus`) or `𝓘(BA)·W0` (`pi`) | `ADAPT_TARGETS`, `RANK`, `ALPHA`, `LORMA_VARIANT` |
| `adalora` | `P ∈ R^{out×r0}`, `E ∈ R^{r0}`, `Q ∈ R^{r0×in}` per adapted layer | `AdaLoRALinear` (Zhang et al., 2023): `h = W0x + (α/r0)·P diag(E) Q x`, `P, Q ~ N(0, 0.02²)`, `E = 0`; orthogonality penalty (see Loss). `RankAllocator`: per entry, sensitivity `s = \|w·∂L/∂w\|`, `Ī = EMA_β1(s)`, `Ū = EMA_β2(\|s − Ī\|)`, score `Ī·Ū`; triplet score = `score(E_i) + mean_j score(Q_ij) + mean_j score(P_ji)`. Budget `b(p)`: `r0·n` for `p < TINIT`, `rT·n` for `p ≥ 1 − TFINAL`, else `rT·n + (r0·n − rT·n)·(1 − (p − TINIT)/(1 − TFINAL − TINIT))³`. Every `ADALORA_DELTA_T` steps the triplets outside the top-`b` scores get `E = 0`; in the final phase the mask is fixed once and re-applied after every step. Merged after training | `ADAPT_TARGETS`, `ADALORA_INIT_R`, `ADALORA_TARGET_R`, `ALPHA`, `ADALORA_TINIT`, `ADALORA_TFINAL`, `ADALORA_DELTA_T`, `ADALORA_BETAS`, `ADALORA_ORTH` |
| `adapter` | adapter weights (+ all LayerNorm / RMSNorm parameters if `ADAPTER_TRAIN_NORMS`) | `AdapterLinear` (Houlsby et al., 2019): after each `nn.Linear` named in `ADAPTER_AT` (`o_proj`, `down_proj` for llama / mistral; `out_proj`, `fc2` for BART), `y + U·gelu(D·y)`, `D: d -> ADAPTER_DIM`, `U: ADAPTER_DIM -> d` zero-initialised; applied before the residual addition; stays as an extra module at inference | `ADAPTER_AT`, `ADAPTER_DIM`, `ADAPTER_TRAIN_NORMS` |
| `kd` | all parameters of the student | see below | `TEACHER_RUN`, `STUDENT_LAYERS`, `KD_T`, `KD_ALPHA`, `LR` |

**Knowledge distillation (`kd`).**
- *Teacher*: any finished run in `<hf-user>/cs781-p4` (any architecture, any method) that saved
  `runs/<TEACHER_RUN>/train_scores.json`. That file holds, for up to `TRAIN_SCORES_MAX_DOCS` (default 10,000; a random
  subset, seed `SEED`) of the training documents the teacher run trained on, the log-probabilities of labels 0 and 1,
  computed like test predictions (`MAX_CHUNKS_PRED`, `CHUNK_SELECT` with `random -> even`). If `TEACHER_RUN` is not a
  finished run the notebook raises `RuntimeError` listing the finished runs.
- *Student*: this notebook's `ARCH`, Phase-1 weights, keeping only the blocks in `STUDENT_LAYERS` (default `None` =
  every 2nd block, `0, 2, 4, ...`, applied to each stack; `layer_idx` and `num_hidden_layers` / `layer_types` updated).
  All student parameters are trained.
- *Data*: the student trains only on the documents present in the teacher's `train_scores.json`.
- *Loss*: `KD_ALPHA · CE(gold) + (1 − KD_ALPHA) · KD_T² · KL(softmax(teacher/KD_T) ‖ softmax(student/KD_T))`.
- The run in `run_logs/`: teacher `llama_lorma_even_mean` (printed: `teacher llama_lorma_even_mean: llama/lorma, 109.5M
  params, dev macro-F1 0.6012, dev inference 64.6s;  student trains on the 10000 of 32305 training documents the teacher
  scored`), student `bart`, kept blocks `[[0, 2, 4]]`.

**Merging.** After training, LoRMA and AdaLoRA layers are merged into plain `nn.Linear` weights; the maximum logit
change on two probe documents is printed and asserted `< 1e-2 · max(1, max|logit|)`. Printed values:
llama `1.79e-07`, bart `1.12e-07`, prefixlm `1.19e-07`, bart KD `0.00e+00` (no adapters).

## 6. Training procedure

- **Precision**: weights in float32; forward under `torch.autocast(float16)` on CUDA with `GradScaler`; pooling / head
  in float32. Optional activation checkpointing (`GRAD_CKPT`).
- **Optimizer**: `AdamW(betas=(0.9, 0.999), fused=True on CUDA)`. Parameter groups by (LR, weight decay): head /
  document pooling / chunk-position embedding at `HEAD_LR`; backbone (or adapter) parameters at
  `LR · LAYER_LR_DECAY^(NB − depth)` (depth = block index + 1, final norms = `NB`, embeddings = 0). Weight decay
  `WEIGHT_DECAY` on tensors with ≥ 2 dims, 0 otherwise. Gradient clipping at `GRAD_CLIP` (global norm).
- **Batching**: `BATCH_DOCS` documents per forward (up to `MAX_CHUNKS` chunks each), `GRAD_ACCUM` forwards per
  optimizer step (effective batch `BATCH_DOCS × GRAD_ACCUM` documents). Training order: `np.random.permutation` per epoch.
- **Time cap and schedule**: training stops after `EPOCHS` passes or when the training time (forward / backward /
  step time; dev evaluations excluded) reaches `MAX_TRAIN_MINUTES`. Progress
  `p = min(1, max(step / TOTAL_STEPS, train_seconds / (60 · MAX_TRAIN_MINUTES)))`, `TOTAL_STEPS = EPOCHS ·
  ceil(N_train / (BATCH_DOCS · GRAD_ACCUM))`. LR multiplier: linear warm-up to 1 over `WARMUP_FRAC` of progress, then
  linear decay to 0 at `p = 1`. The unfreezing and AdaLoRA budget schedules use the same `p`.
- **Evaluation / model selection**: dev macro-F1 at threshold 0.5, computed every `EVAL_EVERY_MINUTES` training minutes,
  at the end of every epoch and at the time cap. The state dict with the highest dev macro-F1 is kept in CPU memory.
- **Early stopping**: after `PATIENCE` consecutive evaluations without a dev macro-F1 improvement.
- **Checkpoint / resume**: none across sessions. Every run starts from the Phase-1 checkpoint; the selected state dict
  lives only in memory and is written to the Hub as `model.pt` only when `SAVE_WEIGHTS = True` (default `False`). Only
  the tokenisation cache is reused between runs in the same runtime.
- **Logged measurements**: throughput after 20 optimizer steps; peak GPU memory (`torch.cuda.max_memory_allocated`) for
  training and, separately, for dev inference; dev and test inference seconds.

## 7. Inference and submission

1. Load the selected state dict, merge LoRMA / AdaLoRA (§5).
2. Dev and test prediction: one forward per document over `MAX_CHUNKS_PRED` chunks chosen by `CHUNK_SELECT`
   (`random -> even`), `PRED_BATCH_DOCS` documents per forward, float16 autocast; `p(1) = softmax(logits)[1]`.
3. Decision: `pred = 1` if `p(1) >= THR`. `THR = THRESHOLD` (default 0.5), or with `THRESHOLD = "dev"` the value
   maximising dev macro-F1 among 0.5 and the midpoints between consecutive distinct dev `p(1)` values (the dev score is
   then measured on the data the threshold was fitted to; the dev score at 0.5 is printed alongside).
4. Dev report: macro-F1 / P / R, per-class table, confusion matrix, macro-F1 / FP / FN by dev-length quartile, and for
   `DOC_POOL = "attn"` the mean chunk attention weight by relative position bin.
5. `prediction.json` = `[{"id": <test id>, "pred": 0|1}, ...]` in `test.json` order. Asserted before writing (same
   checks as the official evaluator): non-empty list, no duplicate IDs, ID set equal to the test IDs, `id` is `str`,
   `pred` is an `int` in {0, 1}. The zip `submission_<RUN_NAME>.zip` is written and re-opened to assert that it
   contains exactly `prediction.json`.
6. Run files in `RUN_DIR = /content/p4/<RUN_NAME>/`: `dev_truth.json`, `dev_pred.json` (official formats),
   `dev_details.json` (per dev document: gold, `p1`, pred, `n_tokens`, `n_chunks`, attention weights), `prediction.json`,
   `test_probs.json`, `submission_<RUN_NAME>.zip`, `train_scores.json` (if `SAVE_TRAIN_SCORES`), `run_config.json`
   (all config values, parameter counts, times, memory, dev scores, eval history, teacher info), `curves.png` (train
   loss and dev F1 / P / R vs training minutes, dev `p(1)` histogram by gold label), `model.pt` (if `SAVE_WEIGHTS`).
   The folder is uploaded to `<hf-user>/cs781-p4` at `runs/<RUN_NAME>/` (3 retries); in Colab the zip is downloaded.
7. Local evaluation with the official script on the dev files:
   `python evaluate_phase4.py --truth dev_truth.json --prediction dev_pred.json`. The notebook's own scorer
   (`cjpe_scores`) uses the same `sklearn` calls (`labels=[0, 1]`, `zero_division=0`).
8. The last cell lists every run saved in `cs781-p4` (dev F1 / P / R, parameters, train minutes, dev inference
   seconds) and counts which Phase-1 models and §10.2.1 techniques have runs.

## 8. Configuration knobs

First code cell and the *Configuration* cell ("all values below are YOUR CHOICE"). Template defaults:

| Name | Default | Meaning |
|---|---|---|
| `ARCH` | set by the builder | `llama` / `bart` / `prefixlm` / `mistral_swa`; fixed per notebook |
| `METHOD` | `"lorma"` | `full`, `partial`, `unfreeze`, `bitfit`, `lorma`, `adalora`, `adapter`, `kd` |
| `CHUNK_SELECT` | `"even"` | `first`, `last`, `first_last`, `even`, `random` (§3) |
| `DOC_POOL` | `"mean"` | `mean`, `max`, `attn`, `gru` (§4) |
| `RUN_NAME` | `f"{ARCH}_{METHOD}_{CHUNK_SELECT}_{DOC_POOL}"` | run folder / Hub path / zip name |
| `DATASET` | `"Exploration-Lab/CS-781-Capstone"` | HF dataset |
| `P1_REPOS` | `cs781-p1-{llama, bart-v2, prefixlm, mistral_swa}` | Phase-1 checkpoints (`best/`) |
| `P4_REPO` | `f"{HF_USER}/cs781-p4"` | private repo for run outputs |
| `WORK`, `RUN_DIR`, `CACHE_DIR` | `/content/p4`, `WORK/RUN_NAME`, `WORK/cache` | local directories |
| `SEED` | `42` | Python / NumPy / torch seeds, subset sampling, `random` chunk selection |
| `TRAIN_MAX_DOCS` | `None` | label-stratified training subset size; `None` = all |
| `CHUNK_LEN` | `512` | tokens per chunk |
| `CHUNK_STRIDE` | `512` | tokens between chunk starts (< `CHUNK_LEN` = overlap) |
| `MAX_CHUNKS` | `8` | chunks per document in training |
| `MAX_CHUNKS_PRED` | `8` | chunks per document at prediction (dev / test / teacher scores) |
| `POOL` | `"mean"` | token states -> chunk vector: `mean`, `first`, `last` |
| `CHUNK_POS` | `False` | add learned chunk relative-position embedding |
| `POS_BUCKETS` | `8` | number of position bins |
| `ATT_HIDDEN` | `256` | hidden size of the `attn` scorer |
| `GRU_HIDDEN` | `256` | hidden size per direction of the `gru` pool |
| `HEAD` | `"linear"` | `linear` or `mlp` |
| `HEAD_HIDDEN` | `256` | hidden size of the `mlp` head |
| `ATTN` | llama `causal`, prefixlm `bidirectional`, mistral_swa `causal`, bart `n/a` | attention inside a chunk for decoder-only backbones |
| `BART_MODE` | `"encoder"` | `encoder` (encoder states) or `decoder` (chunk to encoder and decoder, decoder states) |
| `EPOCHS` | `2` | passes over the training documents |
| `MAX_TRAIN_MINUTES` | `60` | training-time cap (dev evaluations excluded) |
| `EVAL_EVERY_MINUTES` | `12` | dev evaluation interval in training minutes |
| `PATIENCE` | `3` | evaluations without improvement before stopping |
| `BATCH_DOCS` | `2` | documents per training forward |
| `GRAD_ACCUM` | `4` | forwards per optimizer step |
| `PRED_BATCH_DOCS` | `8` | documents per prediction forward |
| `GRAD_CKPT` | `False` | activation checkpointing |
| `LR` | full `2e-5`, partial `5e-5`, unfreeze `5e-5`, bitfit `1e-3`, lorma `5e-4`, adalora `5e-4`, adapter `5e-4`, kd `5e-5` | backbone / adapter learning rate (per method) |
| `HEAD_LR` | `1e-3` | LR of head, document pooling, position embedding |
| `LAYER_LR_DECAY` | `1.0` | per-block LR decay below the top (1.0 = off) |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay (≥ 2-D tensors) |
| `WARMUP_FRAC` | `0.06` | warm-up fraction of training progress |
| `GRAD_CLIP` | `1.0` | gradient-norm clip |
| `DROPOUT` | `0.1` | dropout on the document vector (and inside `mlp`) |
| `CLASS_WEIGHTS` | `"none"` | `none`, `inverse`, `inverse_sqrt` |
| `THRESHOLD` | `0.5` | decision threshold, or `"dev"` (fitted on dev) |
| `PARTIAL_TOP` | `4` | `partial`: number of top blocks trained |
| `UNFREEZE_FRAC` | `0.5` | `unfreeze`: fraction of training at which all groups are trainable |
| `UNFREEZE_EMBED` | `False` | `unfreeze`: embeddings join as the last group |
| `BITFIT_ADD_BIAS` | `True` | `bitfit`: add zero biases to bias-free linear layers |
| `ADAPT_TARGETS` | `["q_proj", "v_proj"]` | `lorma` / `adalora`: target `nn.Linear` names |
| `RANK`, `ALPHA` | `8`, `16` | LoRMA rank and scale `α/r` (AdaLoRA uses `ALPHA/ADALORA_INIT_R`) |
| `LORMA_VARIANT` | `"plus"` | `plus` (additive-identity inflation) or `pi` (permutation inflation) |
| `ADALORA_INIT_R`, `ADALORA_TARGET_R` | `12`, `4` | initial / final average rank per layer |
| `ADALORA_TINIT`, `ADALORA_TFINAL` | `0.1`, `0.2` | fraction before pruning / with final budget fixed |
| `ADALORA_DELTA_T` | `10` | optimizer steps between rank reallocations |
| `ADALORA_BETAS` | `(0.85, 0.85)` | EMA factors for sensitivity / uncertainty |
| `ADALORA_ORTH` | `0.1` | orthogonality-penalty weight |
| `ADAPTER_AT` | `["o_proj", "down_proj", "out_proj", "fc2"]` | `adapter`: projections followed by an adapter |
| `ADAPTER_DIM` | `64` | adapter bottleneck size |
| `ADAPTER_TRAIN_NORMS` | `True` | `adapter`: also train layer norms |
| `TEACHER_RUN` | `"llama_full_even_mean"` | `kd`: finished teacher run in `cs781-p4` |
| `STUDENT_LAYERS` | `None` | `kd`: kept blocks (`None` = every 2nd) |
| `KD_T` | `2.0` | distillation temperature |
| `KD_ALPHA` | `0.5` | weight of the task loss |
| `SAVE_TRAIN_SCORES` | `True` | write `train_scores.json` (teacher targets for later `kd` runs) |
| `TRAIN_SCORES_MAX_DOCS` | `10000` | max training documents scored (`None` = all seen) |
| `SAVE_WEIGHTS` | `False` | upload `model.pt` |

## 9. Runs recorded in `run_logs/`

Environment printed by every log: `2.11.0+cu130 Tesla T4`, `transformers 5.20.1`.
The code of `P4_arch1_llama_after`, `P4_arch2_bart_after` and `P4_arch3_prefixlm_after` is identical to the template
(all defaults). `P4_arch2_bart_kd_after` differs only in `METHOD = "kd"` and `TEACHER_RUN = "llama_lorma_even_mean"`.

**Configuration**

| File | RUN_NAME | Arch | Method | Settings (all others = template defaults) |
|---|---|---|---|---|
| `P4_arch1_llama_after.ipynb` | `llama_lorma_even_mean` | llama (causal) | lorma | defaults: `plus`, r 8, α 16, `q_proj`/`v_proj`, LR 5e-4, `even`, `mean`, 512×8 chunks, batch 2×4, 60 min cap |
| `P4_arch2_bart_after.ipynb` | `bart_lorma_even_mean` | bart (encoder) | lorma | defaults (as above) |
| `P4_arch2_bart_kd_after.ipynb` | `bart_kd_even_mean` | bart (encoder), student blocks `[[0, 2, 4]]` | kd | `METHOD = "kd"`, `TEACHER_RUN = "llama_lorma_even_mean"`; LR 5e-5 (kd default), `KD_T` 2.0, `KD_ALPHA` 0.5; 10,000 training docs (label counts `{0: 5898, 1: 4102}`) |
| `P4_arch3_prefixlm_after.ipynb` | `prefixlm_lorma_even_mean` | prefixlm (bidirectional) | lorma | defaults (as above) |

**Results** (verbatim from the final cells; dev = official `dev.json`, 994 documents, threshold 0.5 in all runs)

| RUN_NAME | Dev macro-F1 | Dev P | Dev R | Params total / trainable / final | Train min (wall) | Docs seen | Stop | Selected eval | Peak GPU train / dev-infer (GB) | Dev infer (s) | Test infer (s) | Test preds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llama_lorma_even_mean` | 0.6012 | 0.6760 | 0.4366 | 109,826,306 / 296,450 / 109,531,394 | 60.0 (65.7) | 27064 | time cap | epoch 1 step 3383 | 3.50 / 1.32 | 64.6 | 100.6 | `{0: 1046, 1: 471}` |
| `bart_lorma_even_mean` | 0.5999 | 0.6647 | 0.4467 | 68,090,882 / 148,994 / 67,943,426 | 56.5 (59.3) | 32305 | 2 epochs done | epoch 2 step 8078 | 1.59 / 1.10 | 25.1 | 38.5 | `{0: 1008, 1: 509}` |
| `bart_kd_even_mean` | 0.5920 | 0.6371 | 0.4628 | 46,679,810 / 46,679,810 / 46,679,810 | 10.6 (11.1) | 10000 | 2 epochs done | epoch 1 step 1250 | 1.73 / 1.43 | 12.3 | 19.0 | `{0: 959, 1: 558}` |
| `prefixlm_lorma_even_mean` | 0.6649 | 0.6608 | 0.6781 | 109,826,306 / 296,450 / 109,531,394 | 60.0 (65.9) | 25336 | time cap | epoch 1 step 1904 | 3.61 / 1.32 | 66.6 | 102.4 | `{0: 682, 1: 835}` |

"Params total" is counted after adapter injection, "final" after merging. Throughput printed after 20 steps
(documents/s): llama 7.93, bart 19.72, bart KD 32.49, prefixlm 7.16. `train_scores.json` was written for 10000
documents in every run (llama 522 s, bart 199 s, bart KD 98 s, prefixlm 523 s).

<details><summary>Dev per-class tables and confusion matrices (verbatim)</summary>

| RUN_NAME | Class 0 P / R / F1 (pred) | Class 1 P / R / F1 (pred) | TN / FP / FN / TP |
|---|---|---|---|
| `llama_lorma_even_mean` | 0.5840 / 0.7907 / 0.6718 (673) | 0.6760 / 0.4366 / 0.5306 (321) | 393 / 104 / 280 / 217 |
| `bart_lorma_even_mean` | 0.5833 / 0.7746 / 0.6655 (660) | 0.6647 / 0.4467 / 0.5343 (334) | 385 / 112 / 275 / 222 |
| `bart_kd_even_mean` | 0.5782 / 0.7364 / 0.6478 (633) | 0.6371 / 0.4628 / 0.5361 (361) | 366 / 131 / 267 / 230 |
| `prefixlm_lorma_even_mean` | 0.6694 / 0.6519 / 0.6606 (484) | 0.6608 / 0.6781 / 0.6693 (510) | 324 / 173 / 160 / 337 |

Gold counts per class: 497 / 497. Printed baseline: `macro-F1 of predicting one label for every document: all-0 0.3333  all-1 0.3333`.

By dev document length (token quartiles), macro-F1 / FP / FN:

| Tokens | Docs | llama | bart | bart KD | prefixlm |
|---|---|---|---|---|---|
| 208–2420 | 249 | 0.6453 / 27 / 61 | 0.6057 / 34 / 64 | 0.5944 / 40 / 61 | 0.6653 / 40 / 43 |
| 2420–3806 | 248 | 0.5837 / 28 / 72 | 0.5871 / 31 / 69 | 0.6123 / 34 / 61 | 0.6774 / 40 / 40 |
| 3806–6108 | 248 | 0.5852 / 26 / 72 | 0.5817 / 27 / 72 | 0.5495 / 32 / 75 | 0.6652 / 43 / 40 |
| 6108–60714 | 249 | 0.5795 / 23 / 75 | 0.6155 / 20 / 70 | 0.5981 / 25 / 70 | 0.6506 / 50 / 37 |

</details>

<details><summary>Dev evaluations during training (verbatim log lines)</summary>

```
llama_lorma_even_mean
epoch 1 step 689 (12.0 train min): loss 0.6989  dev macro-F1 0.5763 (P 0.5915 R 0.5070, predicted 1: 426/994)
epoch 1 step 1362 (24.0 train min): loss 0.6802  dev macro-F1 0.5964 (P 0.6095 R 0.5433, predicted 1: 443/994)
epoch 1 step 2041 (36.0 train min): loss 0.6637  dev macro-F1 0.5116 (P 0.7226 R 0.2254, predicted 1: 155/994)
epoch 1 step 2709 (48.0 train min): loss 0.6463  dev macro-F1 0.6004 (P 0.6525 R 0.4648, predicted 1: 354/994)
epoch 1 step 3383 (60.0 train min): loss 0.6340  dev macro-F1 0.6012 (P 0.6760 R 0.4366, predicted 1: 321/994)
stopping: time cap

bart_lorma_even_mean
epoch 1 step 1730 (12.0 train min): loss 0.6704  dev macro-F1 0.4158 (P 0.6866 R 0.0926, predicted 1: 67/994)
epoch 1 step 3442 (24.0 train min): loss 0.6538  dev macro-F1 0.5870 (P 0.6062 R 0.5111, predicted 1: 419/994)
epoch 1 step 4039 (28.2 train min): loss 0.6478  dev macro-F1 0.5817 (P 0.6189 R 0.4608, predicted 1: 370/994)
epoch 2 step 5746 (40.2 train min): loss 0.6253  dev macro-F1 0.5698 (P 0.6475 R 0.3843, predicted 1: 295/994)
epoch 2 step 7459 (52.2 train min): loss 0.6220  dev macro-F1 0.5917 (P 0.6625 R 0.4266, predicted 1: 320/994)
epoch 2 step 8078 (56.5 train min): loss 0.6267  dev macro-F1 0.5999 (P 0.6647 R 0.4467, predicted 1: 334/994)

bart_kd_even_mean
epoch 1 step 1250 (5.3 train min): loss 0.3534  dev macro-F1 0.5920 (P 0.6371 R 0.4628, predicted 1: 361/994)
epoch 2 step 2500 (10.6 train min): loss 0.2976  dev macro-F1 0.5037 (P 0.7464 R 0.2072, predicted 1: 138/994)

prefixlm_lorma_even_mean
epoch 1 step 633 (12.0 train min): loss 0.6947  dev macro-F1 0.5157 (P 0.6703 R 0.2495, predicted 1: 185/994)
epoch 1 step 1268 (24.0 train min): loss 0.6623  dev macro-F1 0.4646 (P 0.5284 R 0.9537, predicted 1: 897/994)
epoch 1 step 1904 (36.0 train min): loss 0.6447  dev macro-F1 0.6649 (P 0.6608 R 0.6781, predicted 1: 510/994)
epoch 1 step 2532 (48.0 train min): loss 0.6183  dev macro-F1 0.6274 (P 0.6020 R 0.8310, predicted 1: 686/994)
epoch 1 step 3167 (60.0 train min): loss 0.5995  dev macro-F1 0.6153 (P 0.5930 R 0.8531, predicted 1: 715/994)
stopping: time cap
```

</details>

**`mistral_swa`.** No executed `mistral_swa` notebook is in `run_logs/`. The run-summary cell of
`P4_arch2_bart_kd_after.ipynb` (which lists every run in `cs781-p4`) prints this row for it:

```
run                                      arch        method   select     pool   params  trainable  dev F1       P       R train min dev inf s
mistral_swa_lorma_even_mean              mistral_swa lorma    even       mean   100.1M     0.198M  0.6113  0.6188  0.5815      60.0      67.3
```

Its configuration, peak memory, test inference time and other details: TODO(student).

The same cell prints: `Phase-1 models with runs: ['bart', 'llama', 'mistral_swa', 'prefixlm']  (missing: [])` and
`§10.2.1 techniques with runs: ['kd', 'lorma']  (non-KD: 1)  LoRMA: True  KD: True`.

## 10. Codabench submissions

Each scored submission has a local folder whose name ends with its Codabench ID (`submission_<RUN_NAME><ID>`; the KD
folder uses the last four digits, `...5228`). The folder's `prediction.json` is byte-identical (MD5) to the zip
downloaded from Colab, except for prefixlm, whose zip was not kept. All on 2026-10-20 (IST):

| Codabench ID | Submitted | Submission | Arch | Method / key settings | Macro-F1 |
|---|---|---|---|---|---|
| 1005130 | 16:10 | `submission_mistral_swa_lorma_even_mean` | mistral_swa | lorma, `even`, `mean` (other settings TODO(student)) | 0.6034 |
| 1005143 | 16:15 | unknown | - | stuck in "Submitted", never scored | - |
| 1005146 | 16:17 | `submission_prefixlm_lorma_even_mean` (**leaderboard entry**) | prefixlm | lorma (plus, r 8, α 16, `q_proj`/`v_proj`), `even`, `mean`, 8 × 512-token chunks, LR 5e-4, 2 docs × 4 accumulation, threshold 0.5; best dev evaluation at step 1904 | **0.6371** |
| 1005150 | 16:19 | `submission_bart_lorma_even_mean` | bart v2 (encoder) | lorma (plus, r 8), `even`, `mean`, template defaults | 0.6067 |
| 1005160 | 16:26 | `submission_llama_lorma_even_mean` | llama | lorma (plus, r 8), `even`, `mean`, template defaults | 0.5908 |
| 1005228 | 17:11 | `submission_bart_kd_even_mean` | bart student (blocks 0, 2, 4) | kd, teacher `llama_lorma_even_mean`, `KD_T` 2.0, `KD_ALPHA` 0.5 | 0.6017 |

## 11. How to run

1. Build the notebooks (from this folder):
   ```bash
   python build_p4_notebooks.py          # all four: P4_arch1_llama.ipynb ... P4_arch4_mistral_swa.ipynb
   python build_p4_notebooks.py bart     # only P4_arch2_bart.ipynb
   ```
2. Open the notebook in Google Colab; *Runtime -> Change runtime type -> T4 GPU* (the notebook asserts a GPU unless the
   environment variable `CS781_ALLOW_CPU` is set).
3. Add a Colab secret named `HF` holding the same Hugging Face token as Phases 1–3 (read access to the
   `cs781-p1-*` repos and the gated course dataset, write access for `cs781-p4`). A Kaggle secret `HF` or the
   environment variable `HF_TOKEN` are also accepted.
4. The Phase-1 notebooks must have finished: the code downloads `<hf-user>/cs781-p1-<arch>/best/`
   (`cs781-p1-bart-v2` for BART).
5. In the first code cell set `METHOD`, `CHUNK_SELECT`, `DOC_POOL` (`RUN_NAME` follows; add a suffix when other
   settings change). Adjust the *Configuration* cell if needed.
6. *Runtime -> Run all*. Outputs go to `/content/p4/<RUN_NAME>/` and are uploaded to the private repo
   `<hf-user>/cs781-p4` under `runs/<RUN_NAME>/`; `submission_<RUN_NAME>.zip` is downloaded by the browser.
   Upload that zip to Codabench.
7. Another experiment in the same runtime: change the first cell and *Run all* again (tokenised documents are reused
   from `/content/p4/cache`).
8. Knowledge distillation ordering:
   1. Run a teacher notebook with any non-`kd` method and `SAVE_TRAIN_SCORES = True` (default) to the end, so that
      `runs/<teacher RUN_NAME>/train_scores.json` is uploaded.
   2. In the student notebook (any architecture) set `METHOD = "kd"` and `TEACHER_RUN = "<teacher RUN_NAME>"`, then
      *Run all*. If the teacher run is not found, the notebook stops and lists the finished runs.
   3. In `run_logs/`: teacher `P4_arch1_llama_after.ipynb` (`llama_lorma_even_mean`) -> student
      `P4_arch2_bart_kd_after.ipynb` (`bart_kd_even_mean`).
9. Local check against the official evaluator, using the dev files of a run:
   ```bash
   python evaluate_phase4.py --truth dev_truth.json --prediction dev_pred.json
   ```
