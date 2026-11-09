# Phase 5 — Legal Statute Identification (LSI)

Multi-label classification of long Indian case documents into statute IDs, starting from the four Phase-1
language models (`llama`, `bart`, `prefixlm`, `mistral_swa`). One percent-format template is rendered into one Colab
notebook per architecture; each run fine-tunes a Phase-1 checkpoint with one adaptation method and writes a
`submission_<RUN_NAME>.zip`.

---

## 1. Task

| | |
|---|---|
| Input | one legal case document: `{"id": "case_001", "text": "..."}` (train / dev records also carry `"labels": [1, 7, 23]`) |
| Output | a **set** of statute IDs per document (multi-label) |
| Submission | `submission.zip` containing a single file `prediction.json`: a JSON list with exactly one object per test document, `[{"id": "case_001", "pred": [1, 7, 23]}, {"id": "case_002", "pred": [5, 81]}]`; `pred` = unique, non-negative integers that are valid statute IDs (order carries no meaning) |
| Primary metric | **Macro-F1** over statute IDs |
| Secondary metrics | Micro-F1, Subset Accuracy |
| Official evaluator | `evaluate_phase5.py --truth <truth.json> --prediction <prediction.json>`: label dimension = largest gold ID + 1; sklearn `f1_score(average="macro"/"micro", zero_division=0)` and `accuracy_score` on the multi-hot matrices; a predicted ID `>=` the label dimension, a duplicate ID or an ID mismatch makes the evaluation fail |

Course requirements (capstone §11.2): at least 4 substantially different experiments; at least 3 with different
fine-tuning / adaptation techniques from the §11.2.1 list (full, partial / layer-wise, QLoRA, LoRMA, AdaLoRA, adapters,
prefix tuning, prompt tuning, KD); LoRMA compulsory; at least 1 knowledge-distillation experiment from a larger or
stronger teacher into a smaller student; every experiment starts from the Phase-1 model; all four Phase-1 models used.

## 2. Files in this folder

| File | Content |
|---|---|
| `phase5_lsi_template.py` | the single source of the phase: percent-format (`# %%`) notebook template with placeholders `{{ARCH}}`, `{{TAG}}`, `{{FAMILY}}`, `{{MODEL}}` |
| `build_p5_notebooks.py` | renders the template into the four notebooks below (`python build_p5_notebooks.py [arch ...]`) |
| `build_nb.py` | converter percent-format `.py` -> `.ipynb` (used by the builder) |
| `P5_arch1_llama.ipynb` | generated notebook, `ARCH = "llama"` (decoder-only) |
| `P5_arch2_bart.ipynb` | generated notebook, `ARCH = "bart"` (encoder-decoder, Phase-1 v2 checkpoint) |
| `P5_arch3_prefixlm.ipynb` | generated notebook, `ARCH = "prefixlm"` (prefix-LM hybrid) |
| `P5_arch4_mistral_swa.ipynb` | generated notebook, `ARCH = "mistral_swa"` (sliding-window attention + GQA) |
| `run_logs/P5_arch1_llama_after.ipynb` | executed Colab notebook (with outputs) of run `llama_lorma_even_mean_bce` |
| `run_logs/P5_arch2_bart_after.ipynb` | executed notebook of run `bart_lorma_even_mean_bce` |
| `run_logs/P5_arch2_bart_kd_after.ipynb` | executed notebook of run `bart_kd_even_mean_bce` (distillation student) |
| `run_logs/P5_arch4_mistral_swa_after.ipynb` | executed notebook of run `mistral_swa_lorma_even_mean_bce` |

The four generated notebooks are identical to a fresh rendering of the current template. The code cells of the
run logs are identical to the generated notebooks except for the first-cell / config edits listed in §9.

## 3. Data and preprocessing

**Source.** Gated HF dataset `Exploration-Lab/CS-781-Capstone`, folder `Phase_5_LSI` (`DATA_DIR`):
`Train/train.json`, `Train/dev.json`, `Test/test.json`. If a path is not found, the code searches the dataset's file
list for a unique file with the same suffix and a `phase_5` / `phase5` / `lsi` path component. Files may be a JSON list,
a dict `id -> record`, or JSON Lines; the label field is read from `labels` (fallbacks `label`, `gold`, `statutes`).

**Splits.** `train.json` is used for training, `dev.json` as the held-out split for model selection, threshold fitting
and local evaluation (no additional held-out split is carved out). `TRAIN_MAX_DOCS` (default `None`) optionally
trains on a seeded random subset of `train.json`. A `kd` run trains only on the training documents the teacher scored.

**Cleaning.** A repeated document ID inside a file is dropped (first kept, counted); a statute repeated inside one
document's label list is kept once (counted); documents with identical text in two splits are counted (MD5 of the
text) but not removed. The text itself is not altered (no lower-casing, no removal of the anonymisation placeholders
such as `<SECTION>`, `<ACT>`, `<ENTITY>` visible in the printed first records).

**Dataset statistics printed by the run logs** (identical in all four logs):

| split | docs | chars median | chars p90 | chars max | statutes/doc mean | statutes/doc max |
|---|---|---|---|---|---|---|
| train | 42750 | 2717 | 16789 | 535964 | 3.70 | 31 |
| dev | 10181 | 3418 | 17900 | 442434 | 3.92 | 28 |
| test | 13019 | 3368 | 17277 | 820502 | – | – |

- `!! 10 repeated statute IDs inside label lists dropped` (train), `!! 4 ...` (dev); no repeated document IDs printed.
- `documents with identical text in two splits: train&dev 81  train&test 108  dev&test 61`
- `statute IDs: 100 distinct in train, 100 in dev, largest 99 -> 100 model outputs;  IDs below that in neither split: 0;  dev IDs without a training document: []`
- `statutes by number of training documents: 0: 0  1-9: 0  10-49: 0  50-199: 0  200+: 100`
- `most frequent statutes in train (ID: documents, share): 5: 11730 (27.4%)  34: 10380 (24.3%)  97: 7880 (18.4%)  43: 7728 (18.1%)  39: 7120 (16.7%)  12: 4324 (10.1%)  13: 4136 (9.7%)  10: 4134 (9.7%)  77: 4046 (9.5%)  94: 3835 (9.0%)`
- `distinct statute sets in train: 16191`; most frequent co-occurring pairs: `5,34: 3644  12,13: 3511  43,97: 3385  5,43: 3271  13,14: 2831  5,39: 2828  12,14: 2672  34,39: 2596  5,97: 2424  34,35: 2174`
- dev: `39866 (document, statute) pairs, 100 statutes with dev documents of 100 in the dev label dimension`

**Tokenisation.** The Phase-1 tokenizer (`PreTrainedTokenizerFast` from `<P1 repo>/best/`) encodes each whole
document once, without special tokens; ids are stored as `uint16`; an empty document becomes one `<|endoftext|>`
token. Tokenised splits are cached in `/content/p5/cache/<split>_<tokenizer md5>.npz`, so later runs in the same
runtime skip tokenisation (the KD log shows `10000 from cache`).

Token / chunk statistics printed by the llama, bart and mistral_swa logs (identical; `CHUNK_LEN = CHUNK_STRIDE = 512`,
`MAX_CHUNKS_PRED = 8`):

| split | tokens/doc median | p90 | max | chunks/doc median | max | docs with > `MAX_CHUNKS_PRED` chunks | tokens seen at prediction |
|---|---|---|---|---|---|---|---|
| train | 638 | 4158 | 139870 | 2 | 274 | 10.3% | 75.7% |
| dev | 800 | 4489 | 115504 | 2 | 226 | 11.8% | 74.9% |
| test | 791 | 4274 | 220544 | 2 | 431 | 10.7% | 75.0% |

(KD log, 10000-document training subset: train median 632, p90 4144, max 122127, max chunks 239, 10.2%, 77.1%.)

**Chunking (long documents).** Chunk starts are `0, CHUNK_STRIDE, 2·CHUNK_STRIDE, …` plus one last chunk ending exactly
at the document end, so every token lies in at least one chunk (`CHUNK_STRIDE < CHUNK_LEN` gives overlapping chunks).
A document with more than `MAX_CHUNKS` (training) / `MAX_CHUNKS_PRED` (prediction) chunks is represented by the chunks
`CHUNK_SELECT` picks, kept in document order:

| `CHUNK_SELECT` | chunks kept |
|---|---|
| `first` / `last` | the first / last *k* |
| `first_last` | ⌈k/2⌉ from the start, the rest from the end |
| `even` (default) | *k* evenly spaced indices `linspace(0, m-1, k).round()` over the whole document |
| `random` | a fresh random subset each time a document is trained on (`TRAIN_RNG`, seed `SEED`); `even` at prediction |
| `keyword` | the *k* chunks with most tokens that decode to a word of `KEYWORDS` (`section`, `act`, `ipc`, `crpc`, …; ties to the earlier chunk); keywords that are not a single token are reported |

A batch stacks the selected chunks of `BATCH_DOCS` documents as rows of one `(Q, L)` tensor, with per-row document
index, slot and relative-position bucket. Built-in checks (assertions, run before training): chunks cover every token;
pooling + regrouping returns each document's own chunks in order (verified by pooling the token ids); each target row
equals the document's statute set.

**Label space and encoding.** `NUM_LABELS = largest statute ID in train ∪ dev + 1` (= 100 in the logs); targets are
multi-hot rows of length `NUM_LABELS`. `PRED_LABELS = "train"` (default) restricts predictions to IDs with at least one
training document (`ALLOWED`; 100 of 100 in the logs), `"all"` allows every ID `0..NUM_LABELS-1`. The dev label
dimension used for scoring is `largest dev gold ID + 1` (= 100), as in the official evaluator.

## 4. Model architecture

**Backbone per architecture** (Phase-1 checkpoint `best/` from `P1_REPOS[ARCH]`, loaded in fp32; the LM head is not
used):

| `ARCH` | Phase-1 repo | class loaded | module used | attention inside a chunk (`ATTN`) |
|---|---|---|---|---|
| `llama` | `<hf-user>/cs781-p1-llama` | `LlamaForCausalLM` | `.model` (`LlamaModel`), `last_hidden_state` | `causal` (default) or `bidirectional` (full 4-D mask) |
| `prefixlm` | `<hf-user>/cs781-p1-prefixlm` | `LlamaForCausalLM` | `.model`, `last_hidden_state` | `bidirectional` (default: whole chunk as the bidirectional prefix) or `causal` |
| `mistral_swa` | `<hf-user>/cs781-p1-mistral_swa` | `MistralForCausalLM` | `.model` (`MistralModel`), `last_hidden_state` | `causal` sliding window (default) or `bidirectional` (±`sliding_window` band) |
| `bart` | `<hf-user>/cs781-p1-bart-v2` | `BartForConditionalGeneration` | `BART_MODE = "encoder"` (default): `.model.encoder` states; `"decoder"`: chunk fed to encoder **and** decoder, decoder `last_hidden_state` | n/a |

Hidden size *H* is the Phase-1 model's (`D_MODEL = 768` in the Phase-1 template). `CHUNK_LEN (+ PROMPT_LEN)` must not
exceed the model's `max_position_embeddings` (asserted). A `model_checks` cell verifies, by perturbing a chunk's last
token, that the first token sees right context only for bidirectional attention / the BART encoder, and that a
document's logits do not change when a longer document joins the batch (padding invariance).

**Head (chunk -> document -> statute logits).**

1. Chunk vector `c_k ∈ R^H`: token states of chunk *k* pooled by `POOL` — `mean` over real tokens (default), `first`
   token, or `last` real token. Chunks are encoded independently. Pooling and everything after it run in fp32.
2. Optional `CHUNK_POS`: add a learned embedding (zero-initialised) of the chunk centre's relative position in the
   document (`POS_BUCKETS = 8` bins).
3. `DOC_POOL` (chunks -> document), masked over each document's real chunks:
   - `mean` (default) / `max`: element-wise mean / max over chunks -> `v ∈ R^H`.
   - `attn`: `a_k = softmax_k(wᵀ tanh(W c_k))`, `W: H -> ATT_HIDDEN`, `v = Σ_k a_k c_k` (hierarchical attention pooling).
   - `gru`: bidirectional GRU (`GRU_HIDDEN` per direction) over the chunk sequence; `v` = concatenated final states (2·`GRU_HIDDEN`).
   - `label_attn`: per statute *l* a query `u_l`; `a_lk = softmax_k(u_lᵀ c_k)`, `v_l = Σ_k a_lk c_k`, logit `z_l = w_lᵀ dropout(v_l) + b_l` (label-wise attention over chunks; no separate `HEAD`).
   - `chunk_max`: `HEAD` scores every chunk; `z_l = max_k z_lk`.
4. `HEAD` on `dropout(v)`: `linear` (`Linear(D, NUM_LABELS)`, default) or `mlp` (`Linear(D, HEAD_HIDDEN) -> tanh -> dropout -> Linear(HEAD_HIDDEN, NUM_LABELS)`).
5. Optional `LABEL_INTERACT`: `z <- z + W·sigmoid(z)` with a learned `NUM_LABELS × NUM_LABELS` matrix `W`, zero at init.
6. Output: one logit per statute ID; probability `p_l = sigmoid(z_l)` (binary relevance, independent sigmoids).

In the four logged runs (template defaults): `POOL = mean`, `DOC_POOL = mean`, `HEAD = linear`, no `CHUNK_POS`, no
`LABEL_INTERACT`, i.e. `z = Linear_{H→100}(dropout(mean_k mean_t h_{k,t}))`.

**Loss (`LOSS`).** One sigmoid output per statute, averaged over all (document, statute) entries of the batch:

- `bce` (default): `BCEWithLogits(z, y, pos_weight=w)`.
- `focal`: the BCE term × `(1 − p_t)^FOCAL_GAMMA`, `p_t = y·p + (1−y)(1−p)`.
- `asl` (asymmetric loss, Ridnik et al. 2021): `−[ y·w·log σ(z)·(1−p)^γ⁺ + (1−y)·log(1−p_m)·p_m^γ⁻ ]`,
  `p_m = max(p − ASL_CLIP, 0)`, `γ⁺ = ASL_GAMMA_POS`, `γ⁻ = ASL_GAMMA_NEG`.
- Per-statute positive weights `w_l` (`POS_WEIGHT`): `none` (all 1, default); `inverse`: `(n − c_l)/c_l`;
  `inverse_sqrt`: its square root; capped at `POS_WEIGHT_MAX`; `c_l` = training documents of statute *l* (statutes
  without training documents keep weight 1).
- KD adds a soft term (see §5); AdaLoRA adds `ADALORA_ORTH ×` mean orthogonality penalty.

**Rare-statute handling options.** `POS_WEIGHT` (above); `SAMPLE_POWER > 0`: each epoch draws `n` training documents
with replacement, probability ∝ (training count of the document's rarest statute)^−`SAMPLE_POWER`; `focal` / `asl`
losses; per-statute thresholds (`dev_per_label`); `MIN_PRED`. All are off in the template defaults and in the logged
runs. The dev report groups statutes by training frequency (`0`, `1-9`, `10-49`, `50-199`, `200+`).

## 5. Fine-tuning methods implemented

`METHOD` selects one; the head (and pooling / label attention / position embedding / interaction layer) is always
trained. All methods are implemented in plain PyTorch in the template (no `peft`, no `bitsandbytes`). The template
cites the capstone §11.2.1 list; `unfreeze` is a layer-wise variant of the partial entry. **Prefix tuning is not
implemented** (template note: the BART encoder has no key/value input to attach prefixes to); **BitFit is not offered**
(not on the Phase-5 list).

| `METHOD` | §11.2.1 entry | trainable parameters | implementation | knobs |
|---|---|---|---|---|
| `full` | Full fine-tuning | every parameter | — | `LR` (2e-5), `LAYER_LR_DECAY` |
| `partial` | Partial / layer-wise | top `PARTIAL_TOP` blocks + final norm(s) + head | `requires_grad` masks; block order is bottom -> top (BART decoder mode: encoder blocks, then decoder blocks) | `PARTIAL_TOP` (4), `LR` (5e-5), `LAYER_LR_DECAY` |
| `unfreeze` | Partial / layer-wise | head first; then one block at a time from the top (top group includes final norms) at equal intervals until all are trainable at `UNFREEZE_FRAC` of training; embeddings last if `UNFREEZE_EMBED` | gradual unfreezing (Howard & Ruder 2018); all candidate params are in the optimizer from the start | `UNFREEZE_FRAC` (0.5), `UNFREEZE_EMBED`, `LR` (5e-5) |
| `qlora` | QLoRA | LoRA `A`, `B` on `ADAPT_TARGETS` + head | own `NF4Linear`: **every** `nn.Linear` of the backbone frozen and stored as 4-bit NF4 codes (two per byte; 16 levels at N(0,1) quantiles scaled to [−1,1]; one absmax per `QLORA_BLOCK` weights); with `QLORA_DOUBLE_QUANT` the absmax values are stored as int8 (symmetric, blocks of 256, after subtracting their mean); custom autograd function dequantises in forward and again in backward, computing in fp16; LoRA `h = W_nf4·x + (α/r)·B(A x)`, `B = 0` at init; no paged optimizer | `ADAPT_TARGETS` (`q_proj`, `v_proj`), `RANK` (8), `ALPHA` (16), `QLORA_BLOCK` (64), `QLORA_DOUBLE_QUANT` (True), `LR` (5e-4) |
| `lorma` | LoRMA (compulsory) | `A ∈ R^{r×d_out}`, `B ∈ R^{d_out×r}` per adapted layer + head | own `LoRMALinear`, multiplicative update `h = 𝓘(BA)·W0·x` (+ bias). `plus`: `𝓘(BA) = I + (α/r)·BA`, computed as `W0x + (α/r)·B(A(W0x))`, `A` Kaiming-init, `B = 0`. `pi`: `𝓘(BA)` = `BA` with row *i* rotated cyclically by *i*, init first column of `B` = 1 and `A = e₀e₀ᵀ` (gives `I`). Merged into plain `nn.Linear` weights after training (checked numerically) | `LORMA_VARIANT` (`plus`), `ADAPT_TARGETS`, `RANK` (8), `ALPHA` (16), `LR` (5e-4) |
| `adalora` | AdaLoRA | `P`, `E`, `Q` per adapted layer + head | own `AdaLoRALinear`: `h = W0x + (α/r₀)·P·diag(E)·Q·x`, `E = 0` at init, orthogonality penalty `‖PᵀP−I‖_F + ‖QQᵀ−I‖_F`; `RankAllocator`: importance = EMA(\|w·∂L/∂w\|) × EMA of its deviation (`ADALORA_BETAS`), triplet score = s(E_i) + mean s(Q_i·) + mean s(P_·i); every `ADALORA_DELTA_T` steps the lowest-scored triplets get `E = 0`; total budget `INIT_R·n` until `TINIT`, cubic decrease, `TARGET_R·n` fixed for the last `TFINAL`; merged after training | `ADALORA_INIT_R` (12), `ADALORA_TARGET_R` (4), `ADALORA_TINIT` (0.1), `ADALORA_TFINAL` (0.2), `ADALORA_DELTA_T` (10), `ADALORA_BETAS` (0.85, 0.85), `ADALORA_ORTH` (0.1), `ALPHA`, `LR` (5e-4) |
| `adapter` | Adapters | bottleneck `down`/`up` + (optionally) all layer norms + head | own `AdapterLinear` (Houlsby et al. 2019): `y + U·gelu(D·y)` on the outputs of the projections in `ADAPTER_AT`, before the residual addition, `U = 0` at init; stays as extra modules at inference | `ADAPTER_AT` (`o_proj`, `down_proj`, `out_proj`, `fc2`), `ADAPTER_DIM` (64), `ADAPTER_TRAIN_NORMS` (True), `LR` (5e-4) |
| `prompt` | Prompt tuning | `PROMPT_LEN × H` soft tokens + head | `PROMPT_LEN` learned vectors in input-embedding space (initialised from embeddings of random vocabulary tokens, seed `SEED`) prepended to every chunk; their output positions are removed before pooling; backbone frozen; BART: encoder mode only | `PROMPT_LEN` (20), `LR` (3e-3) |
| `kd` | Knowledge distillation | every parameter of the student | see below | `TEACHER_RUN`, `STUDENT_LAYERS`, `KD_T` (2.0), `KD_ALPHA` (0.5), `LR` (5e-5) |

Every adapter-style method asserts that the model output is unchanged at initialisation (`output change at init`
< 1e-3; for `qlora` measured against the already-quantised model); after training, LoRMA / AdaLoRA are merged and the probe-logit change is printed (`adapter merge: max |logit
change|`). Parameter counts for `qlora` count NF4 buffers as the weights they represent.

**Knowledge distillation (`METHOD = "kd"`).**

- *Teacher*: any finished Phase-5 run in `<hf-user>/cs781-p5` (`TEACHER_RUN`, any architecture) that saved
  `runs/<TEACHER_RUN>/train_scores.npz` — per training document it trained on (at most `TRAIN_SCORES_MAX_DOCS = 10000`,
  seeded random subset), its logit for every statute ID (`float16`, clipped to ±60000), computed like test predictions.
  The teacher's `run_config.json` supplies its arch / method / parameter count / dev macro-F1 for the log. The run
  aborts with a list of finished runs if `TEACHER_RUN` has no `train_scores.npz`; teacher and student must have the same
  `NUM_LABELS` (asserted).
- *Student*: this notebook's `ARCH`, initialised from the Phase-1 checkpoint, with only the blocks in `STUDENT_LAYERS`
  kept (`None` = every 2nd block, applied per stack; `layer_idx` and `num_hidden_layers` / `layer_types` updated).
  The student trains on exactly the training documents the teacher scored.
- *Loss*: `KD_ALPHA · task_loss + (1 − KD_ALPHA) · T² · BCEWithLogits(input = z_student / T, target = sigmoid(z_teacher / T))`
  with `T = KD_T`, averaged over all (document, statute) entries (per-statute binary cross-entropy between the
  temperature-scaled teacher probability and the student's temperature-scaled logit); `task_loss` = `LOSS` on the gold
  labels.

## 6. Training procedure

- **Optimizer**: AdamW, betas (0.9, 0.999), `fused=True` on CUDA. Parameter groups by (learning rate, weight decay):
  head-side parameters use `HEAD_LR` (1e-3); backbone / adapter parameters use `LR[METHOD]` ×
  `LAYER_LR_DECAY^(NB − depth)` (depth = block index from the bottom, embeddings lowest; `1.0` = off); weight decay
  `WEIGHT_DECAY` (0.01) on tensors with ≥2 dims, 0 on others.
- **Schedule**: progress `p = max(step / TOTAL_STEPS, train_seconds / (60 · MAX_TRAIN_MINUTES))` with
  `TOTAL_STEPS = EPOCHS · ⌈n_train / (BATCH_DOCS · GRAD_ACCUM)⌉`; LR multiplier = linear warmup to 1 over the first
  `WARMUP_FRAC` (0.06) of progress, then linear decay to 0 at `p = 1`. The unfreezing and AdaLoRA schedules use the same
  progress, so they complete by the time cap.
- **Batching**: `BATCH_DOCS = 2` documents per forward pass (each up to `MAX_CHUNKS = 8` chunks of `CHUNK_LEN = 512`),
  `GRAD_ACCUM = 4` -> 8 documents per optimizer step; each epoch is a fresh permutation of the training documents (or a
  weighted draw with replacement when `SAMPLE_POWER > 0`).
- **Precision**: weights fp32; forward under `torch.autocast(float16)` with `GradScaler`; chunk pooling, document
  pooling, head and loss in fp32. Gradient clipping at `GRAD_CLIP = 1.0` (after unscaling). Optional activation
  checkpointing `GRAD_CKPT`.
- **Time cap**: training stops when training time (dev evaluations excluded) reaches `MAX_TRAIN_MINUTES = 60`, checked
  after every optimizer step; `EPOCHS = 2` is the other limit. A throughput line is printed after 20 steps.
- **Evaluation and model selection**: dev evaluation every `EVAL_EVERY_MINUTES = 12` training minutes, at the end of
  every epoch and at the time cap. Selection metric: **dev macro-F1** computed with `SELECT_THRESHOLD` (0.5, or
  `"dev_global"`). A strictly higher dev macro-F1 replaces the kept state (`best_state`, a CPU copy of the
  `state_dict`).
- **Early stopping**: after `PATIENCE = 3` consecutive evaluations without a macro-F1 increase.
- **Checkpoints / resume**: no intermediate checkpoints and no resume; the kept state lives in RAM and is reloaded at
  the end. Every run starts from the Phase-1 checkpoint; only the tokenisation cache is reused within a runtime.
  Weights are uploaded only if `SAVE_WEIGHTS = True` (`model.pt`).
- **Seeds**: `SEED = 42` for `random`, `numpy`, `torch`; the same seed drives the optional training subset, the
  `random` chunk selection RNG and the `train_scores.npz` subset.

## 7. Inference and submission

1. Load the kept state, merge LoRMA / AdaLoRA into the base weights, check the merge on two probe documents.
2. `predict_logits`: one fp16-autocast forward pass per document over its `MAX_CHUNKS_PRED` chunks chosen by
   `CHUNK_SELECT` (`random` -> `even`), `PRED_BATCH_DOCS = 8` documents per pass; probabilities = sigmoid of the logits.
3. **Thresholding** (`decide`): statute *l* predicted when `p_l >= threshold`, only for `ALLOWED` IDs. `THRESHOLD`:
   - a number (default `0.5`);
   - `"dev_global"`: the value in 0.01…0.99 (step 0.01) that maximises dev macro-F1;
   - `"dev_per_label"`: for each statute with dev positives, the score cut (between distinct dev scores) that
     maximises that statute's dev F1; other statutes use the `dev_global` value.
   Thresholds are fitted at the kept checkpoint on dev, without `MIN_PRED` / `MAX_PRED` (template note: the dev score is
   then measured on the data the threshold was fitted to). Afterwards `MIN_PRED` (each document gets at least its
   *k* highest-scoring statutes) and `MAX_PRED` (at most *k*, by score) apply. The dev table always prints threshold 0.5,
   `dev_global` and `dev_per_label`; the submission uses `THRESHOLD`.
4. **Dev scoring** follows `evaluate_phase5.py` (label dimension = largest dev gold ID + 1, macro / micro F1 with
   `zero_division=0`, subset accuracy) and is asserted equal to an sklearn re-computation of the official formula;
   predicted IDs above the dev label dimension are counted as `rejected_ids` and reported.
5. **Dev report** (printed and saved): threshold variants; threshold-free mean average precision per statute;
   per-training-frequency bins; per number of gold statutes; per dev length quartile; most missed / over-predicted
   statutes and (missed -> predicted) pairs; calibration bins and mean BCE; chunk attention by position (`attn`,
   `label_attn` only).
6. **Submission**: `prediction.json = [{"id": <test id>, "pred": [sorted unique int IDs]}, ...]` in test-file order,
   checked like the official evaluator (non-empty list, unique IDs, same ID set as `test.json`, unique `int` labels in
   `ALLOWED`), zipped as `submission_<RUN_NAME>.zip` containing only `prediction.json` (asserted), and downloaded in Colab.
7. **Run artefacts** in `/content/p5/<RUN_NAME>/`, uploaded to `<hf-user>/cs781-p5` under `runs/<RUN_NAME>/`:
   `prediction.json`, `submission_<RUN_NAME>.zip`, `test_probs.npz`, `dev_truth.json` + `dev_pred.json` (official
   format), `dev_details.json`, `dev_label_report.json`, `dev_probs.npz`, `train_scores.npz` (KD targets, if
   `SAVE_TRAIN_SCORES`), `run_config.json` (all config values, parameter counts, times, memory, dev scores, history),
   `curves.png`, optional `model.pt`.
8. **Local evaluation**: `python evaluate_phase5.py --truth dev_truth.json --prediction dev_pred.json` in the run folder.
9. A final cell lists all runs saved in `cs781-p5` (any architecture) and counts the Phase-1 models and §11.2.1
   techniques that have runs.

## 8. Configuration knobs

First code cell (the notebook's main switches) and the *Configuration* cell ("all values below are YOUR CHOICE";
the template calls the filled-in numbers placeholders that run end to end on a T4, not tuned values).

| Name | Template default | Meaning |
|---|---|---|
| `ARCH` | per notebook | `llama` / `bart` / `prefixlm` / `mistral_swa` (fixed by the builder) |
| `METHOD` | `"lorma"` | `full`, `partial`, `unfreeze`, `qlora`, `lorma`, `adalora`, `adapter`, `prompt`, `kd` |
| `CHUNK_SELECT` | `"even"` | chunk selection for long documents (§3) |
| `DOC_POOL` | `"mean"` | chunks -> document: `mean`, `max`, `attn`, `gru`, `label_attn`, `chunk_max` |
| `LOSS` | `"bce"` | `bce`, `focal`, `asl` |
| `RUN_NAME` | `f"{ARCH}_{METHOD}_{CHUNK_SELECT}_{DOC_POOL}_{LOSS}"` | run folder / Hub path; add a suffix when other settings change |
| `DATASET` | `"Exploration-Lab/CS-781-Capstone"` | HF dataset repo |
| `DATA_DIR` | `"Phase_5_LSI"` | phase folder in the dataset (searched for if missing) |
| `P1_REPOS` | `cs781-p1-{llama, bart-v2, prefixlm, mistral_swa}` | Phase-1 checkpoints (`best/`) |
| `P5_REPO` | `f"{HF_USER}/cs781-p5"` | private repo for all Phase-5 runs |
| `WORK`, `RUN_DIR`, `CACHE_DIR` | `/content/p5`, `/content/p5/<RUN_NAME>`, `/content/p5/cache` | local folders (`/tmp/p5` outside Colab) |
| `SEED` | `42` | global seed |
| `TRAIN_MAX_DOCS` | `None` | train on a seeded random subset of this size |
| `CHUNK_LEN` | `512` | tokens per chunk |
| `CHUNK_STRIDE` | `512` | tokens between chunk starts (< `CHUNK_LEN` = overlap) |
| `MAX_CHUNKS` | `8` | chunks per document in training |
| `MAX_CHUNKS_PRED` | `8` | chunks per document at prediction |
| `KEYWORDS` | 16 words (`section`, `act`, `ipc`, `crpc`, …) | `CHUNK_SELECT = "keyword"` vocabulary |
| `POOL` | `"mean"` | token states -> chunk vector: `mean` / `first` / `last` |
| `CHUNK_POS` | `False` | learned chunk relative-position embedding |
| `POS_BUCKETS` | `8` | number of position bins |
| `ATT_HIDDEN` | `256` | hidden size of the `attn` scorer |
| `GRU_HIDDEN` | `256` | `gru` hidden size per direction |
| `HEAD` | `"linear"` | `linear` / `mlp` |
| `HEAD_HIDDEN` | `256` | `mlp` hidden size |
| `LABEL_INTERACT` | `False` | statute–statute interaction layer |
| `ATTN` | `causal` (llama, mistral_swa), `bidirectional` (prefixlm), `n/a` (bart) | attention inside a chunk |
| `BART_MODE` | `"encoder"` | bart: encoder states, or encoder + decoder states (`"decoder"`) |
| `EPOCHS` | `2` | passes over the training documents |
| `MAX_TRAIN_MINUTES` | `60` | training-time cap (evaluations excluded) |
| `EVAL_EVERY_MINUTES` | `12` | dev evaluation interval in training minutes |
| `PATIENCE` | `3` | evaluations without macro-F1 increase before stopping |
| `BATCH_DOCS` | `2` | documents per forward pass |
| `GRAD_ACCUM` | `4` | forward passes per optimizer step |
| `PRED_BATCH_DOCS` | `8` | documents per forward pass at prediction |
| `GRAD_CKPT` | `False` | activation checkpointing |
| `LR` | full 2e-5, partial 5e-5, unfreeze 5e-5, qlora 5e-4, lorma 5e-4, adalora 5e-4, adapter 5e-4, prompt 3e-3, kd 5e-5 | backbone / adapter learning rate per method |
| `HEAD_LR` | `1e-3` | head, pooling, label attention, position embedding, interaction layer |
| `LAYER_LR_DECAY` | `1.0` | per-block LR decay below the top (1.0 = off) |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay (matrices only) |
| `WARMUP_FRAC` | `0.06` | warmup fraction of progress |
| `GRAD_CLIP` | `1.0` | gradient-norm clip |
| `DROPOUT` | `0.1` | dropout on document / per-statute / chunk vectors and inside the `mlp` head |
| `SAMPLE_POWER` | `0.0` | oversampling exponent by rarest-statute count (0 = off) |
| `POS_WEIGHT` | `"none"` | `none` / `inverse` / `inverse_sqrt` positive weights |
| `POS_WEIGHT_MAX` | `100.0` | cap on positive weights |
| `FOCAL_GAMMA` | `2.0` | focal-loss exponent |
| `ASL_GAMMA_POS`, `ASL_GAMMA_NEG`, `ASL_CLIP` | `0.0`, `4.0`, `0.05` | asymmetric-loss parameters |
| `THRESHOLD` | `0.5` | number, `"dev_global"` or `"dev_per_label"` (decision for dev report and submission) |
| `SELECT_THRESHOLD` | `0.5` | threshold for model selection during training (number or `"dev_global"`) |
| `MIN_PRED` | `0` | minimum statutes per document (top-scoring), 0 = off |
| `MAX_PRED` | `None` | maximum statutes per document (top-scoring) |
| `PRED_LABELS` | `"train"` | predictable IDs: `train` (≥1 training doc) or `all` |
| `PARTIAL_TOP` | `4` | `partial`: number of top blocks trained |
| `UNFREEZE_FRAC` | `0.5` | `unfreeze`: fraction of training at which all blocks are trainable |
| `UNFREEZE_EMBED` | `False` | `unfreeze`: embeddings join as the last group |
| `ADAPT_TARGETS` | `["q_proj", "v_proj"]` | `qlora` / `lorma` / `adalora` target linear layers |
| `RANK`, `ALPHA` | `8`, `16` | LoRA / LoRMA rank and scale α/r |
| `QLORA_BLOCK` | `64` | NF4 quantisation block size |
| `QLORA_DOUBLE_QUANT` | `True` | 8-bit absmax constants |
| `LORMA_VARIANT` | `"plus"` | `plus` (additive-identity) / `pi` (permutation inflation) |
| `ADALORA_INIT_R`, `ADALORA_TARGET_R` | `12`, `4` | initial / final average rank per adapted layer |
| `ADALORA_TINIT`, `ADALORA_TFINAL` | `0.1`, `0.2` | fraction before pruning / with final budget fixed |
| `ADALORA_DELTA_T` | `10` | steps between rank reallocations |
| `ADALORA_BETAS` | `(0.85, 0.85)` | EMA factors of sensitivity / uncertainty |
| `ADALORA_ORTH` | `0.1` | orthogonality-regulariser weight |
| `ADAPTER_AT` | `["o_proj", "down_proj", "out_proj", "fc2"]` | projections followed by an adapter |
| `ADAPTER_DIM` | `64` | adapter bottleneck size |
| `ADAPTER_TRAIN_NORMS` | `True` | train layer norms with adapters |
| `PROMPT_LEN` | `20` | soft-prompt length |
| `TEACHER_RUN` | `"llama_full_even_mean_bce"` | KD teacher run in `cs781-p5` |
| `STUDENT_LAYERS` | `None` | KD student blocks (`None` = every 2nd) |
| `KD_T` | `2.0` | distillation temperature |
| `KD_ALPHA` | `0.5` | weight of the task loss in KD |
| `SAVE_TRAIN_SCORES` | `True` | save `train_scores.npz` (teacher targets for later KD runs) |
| `TRAIN_SCORES_MAX_DOCS` | `10000` | documents scored for `train_scores.npz` (`None` = all seen) |
| `SAVE_WEIGHTS` | `False` | upload `model.pt` |

## 9. Runs recorded in `run_logs/`

Environment printed by every log: `2.11.0+cu130 Tesla T4`, `transformers 5.20.2`.

**Settings.** All four runs use the template defaults except where listed: `CHUNK_SELECT = even`, `DOC_POOL = mean`,
`LOSS = bce`, `POOL = mean`, `HEAD = linear`, `CHUNK_LEN = CHUNK_STRIDE = 512`, `MAX_CHUNKS = MAX_CHUNKS_PRED = 8`,
`BATCH_DOCS × GRAD_ACCUM = 2 × 4`, `EPOCHS = 2`, `MAX_TRAIN_MINUTES = 60`, `HEAD_LR = 1e-3`, `DROPOUT = 0.1`,
`POS_WEIGHT = none`, `SAMPLE_POWER = 0`, `THRESHOLD = SELECT_THRESHOLD = 0.5`, `PRED_LABELS = train`.

| File | `RUN_NAME` | Arch | Method | Changes from template defaults | LR | Method settings |
|---|---|---|---|---|---|---|
| `P5_arch1_llama_after.ipynb` | `llama_lorma_even_mean_bce` | llama (`ATTN = causal`) | lorma | none | 5e-4 | `plus`, r = 8, α = 16, `q_proj`/`v_proj` (`{'q_proj': 12, 'v_proj': 12}`) |
| `P5_arch2_bart_after.ipynb` | `bart_lorma_even_mean_bce` | bart (`BART_MODE = encoder`) | lorma | none | 5e-4 | `plus`, r = 8, α = 16 (`{'v_proj': 6, 'q_proj': 6}`) |
| `P5_arch2_bart_kd_after.ipynb` | `bart_kd_even_mean_bce` | bart (`BART_MODE = encoder`) | kd | `METHOD = "kd"`, `TEACHER_RUN = "bart_lorma_even_mean_bce"` | 5e-5 | student `kept blocks [[0, 2, 4]]`, `KD_T = 2.0`, `KD_ALPHA = 0.5`, 10000 teacher-scored training documents |
| `P5_arch4_mistral_swa_after.ipynb` | `mistral_swa_lorma_even_mean_bce` | mistral_swa (`ATTN = causal`) | lorma | none | 5e-4 | `plus`, r = 8, α = 16 (`{'q_proj': 12, 'v_proj': 12}`) |

KD teacher line printed: `teacher bart_lorma_even_mean_bce: bart/lorma, 68.0M params, dev macro-F1 0.1591, dev inference 130.3s;  student trains on the 10000 of 42750 training documents the teacher scored`.

**Size, time and memory** (from the final JSON print and log lines):

| Run | params_total | params_trainable | params_final (after merge) | model MB | train min (wall) | stop | docs/s | peak GPU train / infer (GB) |
|---|---|---|---|---|---|---|---|---|
| `llama_lorma_even_mean_bce` | 109901668 | 371812 (0.34%) | 109606756 | 418 | 60.0 (95.3) | `time cap` | 12.67 | 3.53 / 1.13 |
| `bart_lorma_even_mean_bce` | 68166244 | 224356 (0.33%) | 68018788 | 259 | 43.3 (53.2) | `EPOCHS` reached (no stop message) | 33.74 | 1.59 / 0.93 |
| `bart_kd_even_mean_bce` | 46755172 | 46755172 (100.00%) | 46755172 | 178 | 6.8 (9.0) | `EPOCHS` reached (no stop message) | 49.28 | 1.75 / 1.27 |
| `mistral_swa_lorma_even_mean_bce` | 100366180 | 273508 (0.27%) | 100169572 | 382 | 60.0 (91.6) | `time cap` | 13.19 | 3.48 / 1.07 |

(`model MB` as printed in the runs table of the last cell; `params_total` includes the unmerged adapter parameters.)

**Dev metrics** (10181 dev documents; the submission used `THRESHOLD = 0.5`; the `dev_global` / `dev_per_label`
rows are fitted on dev and printed for reference):

| Run | kept checkpoint | threshold | macro-F1 | micro-F1 | subset acc | statutes/doc | no statute |
|---|---|---|---|---|---|---|---|
| llama lorma | epoch 2, step 5656 | **0.5 (submitted)** | 0.1542 | 0.3149 | 0.0495 | 1.16 | 3817 |
| | | dev_global (0.12) | 0.3040 | 0.4232 | 0.0046 | 7.70 | 38 |
| | | dev_per_label | 0.3405 | 0.4354 | 0.0193 | 5.48 | 338 |
| bart lorma | epoch 2, step 10688 | **0.5 (submitted)** | 0.1591 | 0.3201 | 0.0553 | 1.19 | 3645 |
| | | dev_global (0.14) | 0.3135 | 0.4408 | 0.0086 | 6.78 | 110 |
| | | dev_per_label | 0.3483 | 0.4395 | 0.0196 | 5.74 | 305 |
| bart kd | epoch 2, step 2500 | **0.5 (submitted)** | 0.1101 | 0.2778 | 0.0436 | 0.96 | 4248 |
| | | dev_global (0.08) | 0.2692 | 0.3780 | 0.0009 | 10.51 | 12 |
| | | dev_per_label | 0.3229 | 0.4125 | 0.0129 | 6.45 | 326 |
| mistral_swa lorma | epoch 2, step 5856 | **0.5 (submitted)** | 0.1454 | 0.2967 | 0.0447 | 1.08 | 4039 |
| | | dev_global (0.13) | 0.2945 | 0.4236 | 0.0069 | 7.33 | 53 |
| | | dev_per_label | 0.3318 | 0.4311 | 0.0207 | 5.59 | 314 |

Dev gold: 3.92 statutes/doc. Every run printed `(identical to the evaluate_phase5.py computation)` and
`rejected_ids: 0`.

| Run | dev mean AP (100 statutes) | dev mean BCE | dev inference (s) | test inference (s) | test statutes/doc | test docs with no statute | distinct test IDs predicted | `train_scores.npz` time (s) |
|---|---|---|---|---|---|---|---|---|
| llama lorma | 0.2791 | 0.1050 | 335.4 | 419.4 | 1.13 | 5082 | 72 | 302 |
| bart lorma | 0.2881 | 0.1042 | 130.3 | 163.0 | 1.19 | 4745 | 79 | 118 |
| bart kd | 0.2573 | 0.1064 | 67.4 | 84.0 | 0.96 | 5612 | 51 | 60 |
| mistral_swa lorma | 0.2677 | 0.1066 | 304.1 | 380.2 | 1.08 | 5223 | 78 | 273 |

<details>
<summary>Dev evaluation history printed during training (threshold 0.5)</summary>

```
llama_lorma_even_mean_bce
epoch 1 step 1144 (12.0 train min): loss 0.14275  dev macro-F1 0.0641  micro-F1 0.1707  subset acc 0.0252
epoch 1 step 2280 (24.0 train min): loss 0.10642  dev macro-F1 0.1003  micro-F1 0.2217  subset acc 0.0389
epoch 1 step 3412 (36.0 train min): loss 0.10360  dev macro-F1 0.1331  micro-F1 0.2861  subset acc 0.0421
epoch 1 step 4532 (48.0 train min): loss 0.10117  dev macro-F1 0.1406  micro-F1 0.2884  subset acc 0.0472
epoch 1 step 5344 (56.7 train min): loss 0.09951  dev macro-F1 0.1455  micro-F1 0.3051  subset acc 0.0495
epoch 2 step 5656 (60.0 train min): loss 0.09717  dev macro-F1 0.1542  micro-F1 0.3149  subset acc 0.0494

bart_lorma_even_mean_bce
epoch 1 step 3005 (12.0 train min): loss 0.13336  dev macro-F1 0.0957  micro-F1 0.2217  subset acc 0.0367
epoch 1 step 5344 (21.5 train min): loss 0.10312  dev macro-F1 0.1510  micro-F1 0.3043  subset acc 0.0490
epoch 2 step 8291 (33.5 train min): loss 0.09866  dev macro-F1 0.1575  micro-F1 0.3002  subset acc 0.0529
epoch 2 step 10688 (43.3 train min): loss 0.09611  dev macro-F1 0.1592  micro-F1 0.3201  subset acc 0.0554

bart_kd_even_mean_bce   (loss = KD_ALPHA * task loss + (1 - KD_ALPHA) * soft term)
epoch 1 step 1250 (3.4 train min): loss 0.69175  dev macro-F1 0.0638  micro-F1 0.2175  subset acc 0.0317
epoch 2 step 2500 (6.8 train min): loss 0.63009  dev macro-F1 0.1101  micro-F1 0.2778  subset acc 0.0436

mistral_swa_lorma_even_mean_bce
epoch 1 step 1181 (12.0 train min): loss 0.14371  dev macro-F1 0.0850  micro-F1 0.2210  subset acc 0.0342
epoch 1 step 2355 (24.0 train min): loss 0.10733  dev macro-F1 0.1040  micro-F1 0.2417  subset acc 0.0369
epoch 1 step 3520 (36.0 train min): loss 0.10542  dev macro-F1 0.1104  micro-F1 0.2631  subset acc 0.0362
epoch 1 step 4678 (48.0 train min): loss 0.10223  dev macro-F1 0.1280  micro-F1 0.2747  subset acc 0.0427
epoch 1 step 5344 (54.8 train min): loss 0.10139  dev macro-F1 0.1323  micro-F1 0.2843  subset acc 0.0449
epoch 2 step 5856 (60.0 train min): loss 0.10002  dev macro-F1 0.1454  micro-F1 0.2968  subset acc 0.0447
```
</details>

Other printed lines: reference rows `no statute for any document: macro-F1 0.0000` and `the most frequent training
statute for every document: macro-F1 0.0045  micro-F1 0.1173  subset acc 0.0028`; in the KD run (10000 training
documents) the training-frequency bins are `50-199: 54 statutes` and `200+: 46 statutes`. The per-statute,
per-cardinality, per-length, most-missed and calibration tables are in each log's dev-report cell and in
`run_config.json` / `dev_label_report.json` of the run folder on the Hub.

`prefixlm`: a local file `submission_prefixlm_lorma_even_mean_bce` exists (run name pattern
`prefixlm` / `lorma` / `even` / `mean` / `bce`), but there is no executed notebook for it in `run_logs/`; its settings
and dev numbers: TODO(student). The last cell of the mistral_swa log (the latest log) prints
`Phase-1 models with runs: ['bart', 'llama', 'mistral_swa']  (missing: ['prefixlm'])` and
`§11.2.1 techniques with runs: ['kd', 'lorma']  (non-KD: 1)  LoRMA: True  KD: True`; runs made after it: TODO(student).

## 10. Codabench submissions

Local folders carry the last four digits of their Codabench ID (`submission_<RUN_NAME><ID>`). Each folder's
`prediction.json` is byte-identical (MD5) to the zip downloaded from Colab. The zips were built on 2026-10-25; the
Codabench entries are dated 2026-10-28 (IST).

| Codabench ID | Submitted | Submission | Arch | Method / key settings | Macro-F1 | Status |
|---|---|---|---|---|---|---|
| 1016023 | 19:11 | `submission_llama_lorma_even_mean_bce` | llama | LoRMA `plus` r=8 on q/v, even chunks ×8, mean/mean pooling, BCE, threshold 0.5 | 0.1543 | finished |
| 1016027 | 19:12 | `submission_mistral_swa_lorma_even_mean_bce` | mistral_swa | LoRMA `plus` r=8 on q/v, even chunks ×8, mean/mean pooling, BCE, threshold 0.5 | (0.1487 shown) | **failed** |
| 1016033 | 19:14 | unknown (no folder carries 6033) | - | - | - | never evaluated ("Submitted") |
| 1016037 | 19:15 | `submission_prefixlm_lorma_even_mean_bce` | prefixlm | LoRMA, even, mean, BCE (other settings TODO(student)) | 0.1544 | finished |
| 1016039 | 19:17 | `submission_bart_kd_even_mean_bce` | bart (3-block encoder student) | KD from `bart_lorma_even_mean_bce`, T=2, α=0.5, 10000 docs, threshold 0.5 | - | never evaluated ("Preparing") |
| 1016048 | 19:21 | `submission_bart_lorma_even_mean_bce` (**inferred**; leaderboard entry) | bart (v2, encoder) | LoRMA `plus` r=8 on q/v, even chunks ×8, mean/mean pooling, BCE, threshold 0.5; 0.224M of 68.2M trainable; best dev at step 10688 | **0.1651** | finished |

1016048 is attributed to BART LoRMA because it is the only Phase-5 run folder without an ID; 1016033 may be an earlier
upload of the same file. TODO(student): confirm.

## 11. How to run

1. **Render the notebooks** (only needed after editing the template):
   `python build_p5_notebooks.py` (all four) or `python build_p5_notebooks.py bart` (one architecture).
2. **Open** `P5_arch<i>_<arch>.ipynb` in Google Colab; *Runtime -> Change runtime type -> T4 GPU*.
3. **Secret**: add `HF` (key icon) = the same Hugging Face token as Phases 1–4 (reads `<hf-user>/cs781-p1-*` and the
   gated dataset, writes `<hf-user>/cs781-p5`). Outside Colab, Kaggle secret `HF` or env `HF_TOKEN` is used; CPU runs
   need `CS781_ALLOW_CPU=1`.
4. **Choose the experiment** in the first code cell (`METHOD`, `CHUNK_SELECT`, `DOC_POOL`, `LOSS`; `RUN_NAME` follows)
   and check the *Configuration* cell (add a `RUN_NAME` suffix when other settings change).
5. **Runtime -> Run all.** The notebook downloads data and the Phase-1 checkpoint, tokenises (cached for later runs in
   the same runtime), trains, prints the dev report, writes the test predictions and uploads the run.
6. **Outputs**: `/content/p5/<RUN_NAME>/` and `<hf-user>/cs781-p5/runs/<RUN_NAME>/`; the browser downloads
   `submission_<RUN_NAME>.zip` (contains only `prediction.json`), which is the file to upload to Codabench.
7. **Local check**: `python evaluate_phase5.py --truth dev_truth.json --prediction dev_pred.json` in the run folder.
8. **Knowledge distillation (order matters)**: first finish a teacher run with any `METHOD` except `kd` and
   `SAVE_TRAIN_SCORES = True` (default) so that `runs/<teacher>/train_scores.npz` exists in `cs781-p5`; then, in the
   student's notebook (any architecture), set `METHOD = "kd"` and `TEACHER_RUN = "<teacher RUN_NAME>"` and optionally
   `STUDENT_LAYERS`, `KD_T`, `KD_ALPHA`. If the teacher run is not found, the notebook stops and lists the finished
   runs. The logged KD run used teacher `bart_lorma_even_mean_bce` and student `bart` (every 2nd encoder block).
9. Another experiment on the same account: change the first cell and *Run all* again; each run starts from the Phase-1
   checkpoint. The last cell lists all runs in `cs781-p5` and counts architectures / techniques covered.
