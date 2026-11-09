# Phase 2 - Legal Named Entity Recognition (L-NER)

Token-classification fine-tuning of the four Phase-1 legal language models (llama, bart, prefixlm, mistral_swa) to
predict character-exact legal entity spans in Indian Supreme Court judgments. All code lives in one percent-format
template that is rendered into one Colab notebook per architecture.

This README is descriptive only: it documents what the code does and copies the numbers that the executed
notebooks in `run_logs/` printed. It contains no analysis of those numbers.

---

## 1. Task

Course spec: capstone section 8 (Phase 2, L-NER).

- **Input**: legal documents keyed by document ID, e.g. `{"1128835": "REPORTABLE IN THE SUPREME COURT OF INDIA ..."}`.
- **Output** (`prediction.json`, zipped as `submission.zip`): for every test document ID, a list of
  `[start_character, end_character, label]` (an optional 4th field `entity_text` is accepted and ignored by the
  scorer). This code writes the 3-field form:

  ```json
  {"1128835": [[18, 40, "COURT"], [114, 162, "CASENO"], [213, 229, "RESP"]]}
  ```

- **Labels (12)**: `APP` (appellant), `RESP` (respondent), `A.COUNSEL` (appellant's counsel), `R.COUNSEL`
  (respondent's counsel), `JUDGE`, `WIT` (witness), `AUTH` (authority), `COURT`, `STAT` (statute), `PREC`
  (precedent), `DATE`, `CASENO` (case number).
- **Matching**: strict - a predicted entity counts only if start offset, end offset and label all equal a gold
  annotation.
- **Primary metric**: Strict Macro-F1 (per-label F1 averaged over the 12 labels).
  **Secondary metrics**: Strict Macro-Precision, Strict Macro-Recall.
- **Official evaluator**: `evaluate_phase2.py --truth truth.json --prediction prediction.json --labels labels.txt`
  (rejects duplicate JSON keys, ID mismatches, non-integer offsets, `start >= end`, unknown labels and duplicate
  spans).
- **Course requirements cited in the template header** (capstone section 1 and 8.2): at least 6 submissions (the capstone document later lowered this to 4 for every phase); all
  four Phase-1 models used; at least 3 different fine-tuning / PEFT techniques with LoRMA compulsory; at least one
  knowledge-distillation run; required runs must not differ only in hyperparameters.

## 2. Files in this folder

| File | What it is |
|---|---|
| `phase2_lner_template.py` | The single source of the pipeline: percent-format Python (`# %%` code cells, `# %% [markdown]` cells) with placeholders `{{ARCH}}`, `{{TAG}}`, `{{FAMILY}}`, `{{MODEL}}`. Renders to 35 cells (18 code, 17 markdown). |
| `build_p2_notebooks.py` | Fills the placeholders for each architecture and writes `P2_arch{1..4}_<arch>.ipynb`. `python build_p2_notebooks.py [arch ...]` builds all four or only the named ones. |
| `build_nb.py` | Generic converter from a percent-format `.py` file to an `.ipynb` (used by the builder). |
| `P2_arch1_llama.ipynb` | Generated notebook, `ARCH = "llama"` (Arch 1, decoder-only). |
| `P2_arch2_bart.ipynb` | Generated notebook, `ARCH = "bart"` (Arch 2, encoder-decoder). |
| `P2_arch3_prefixlm.ipynb` | Generated notebook, `ARCH = "prefixlm"` (Arch 3, prefix-LM hybrid). |
| `P2_arch4_mistral_swa.ipynb` | Generated notebook, `ARCH = "mistral_swa"` (Arch 4, sliding-window attention + GQA). |
| `bart_p2_diagnostic_cell.py` | Optional, read-only diagnostic cell for the Phase-1 BART checkpoint (see below). Not part of the generated notebooks. |
| `run_logs/` | Notebooks executed on Colab with their outputs: `P2_arch1_llama_after.ipynb`, `P2_arch3_prefixlm_after.ipynb`, `P2_arch4_mistral_swa_after.ipynb`. |

**Notebook / template revision note.** `P2_arch2_bart.ipynb` is identical to what the current template renders.
`P2_arch1_llama.ipynb`, `P2_arch3_prefixlm.ipynb` and `P2_arch4_mistral_swa.ipynb` (and the three run logs) were
rendered from an earlier template revision that loads the Phase-1 checkpoint as `f"{HF_USER}/cs781-p1-{ARCH}"`
instead of through the `P1_REPOS` map; for these three architectures both resolve to the same repository name. The
current template adds `P1_REPOS`, which maps `bart` to `cs781-p1-bart-v2` (the comment in the template states that the
v1 repo `cs781-p1-bart` collapsed and must not be used). Re-running `python build_p2_notebooks.py` regenerates all
four notebooks from the current template.

### `bart_p2_diagnostic_cell.py`

A cell to paste into the BART notebook after the "Tokenizer, character <-> token alignment, BIO tags" cell has run
(it uses `TOK`, `TRAIN_IDS`, `VOCAB`, `p1_dir`, `DEVICE`). It loads a fresh fp32 copy of the Phase-1 BART checkpoint
and changes nothing. On the first 128 tokens of the longest training document (and a copy with token 40 replaced by
`(id + 1) mod VOCAB`) it prints:

1. Mean / max `|gain|` and mean `|bias|` of every LayerNorm (encoder `layernorm_embedding`, each encoder layer's
   `self_attn_layer_norm` and `final_layer_norm`, decoder `layernorm_embedding`, each decoder layer's
   `final_layer_norm`); the gains are initialised to 1.0.
2. For every encoder layer output (0 = embeddings + `layernorm_embedding`): mean `|h|`, the change at position 40,
   and the largest change at positions < 40.
3. The largest `|input|` reaching each encoder LayerNorm (via forward pre-hooks), printed next to the fp16 limit
   6.55e+04.
4. For `BART_MODE = "decoder"` (document fed to encoder and decoder): mean `|h|` and the change at position 40,
   positions > 40 and positions < 40.

No executed output of this cell is stored in this repository (TODO(student): paste its output if it was run).

## 3. Data and preprocessing

### Loading

- Dataset: Hugging Face dataset `Exploration-Lab/CS-781-Capstone` (gated; read with the `HF` token), folder
  `Phase_2_LNER/`: `Train/fold1-text.json`, `Train/fold1-gold.json`, `Train/fold2-text.json`,
  `Train/fold2-gold.json`, `Train/labels.txt`, `Test/fold3-text.json`.
- The first record of each file is printed. The parsers accept a dict `{id: text}` or a list of records for texts,
  and `[start, end, label(, text)]` lists or dicts with `start/end/label` keys for gold entities.
- `labels.txt` must contain 12 labels (asserted). Document IDs must not repeat across folds (asserted).

Statistics printed by all three run logs (identical in each):

| Item | Value |
|---|---|
| fold1 | 35 documents, 4120 entities, gold for 35 docs |
| fold2 | 35 documents, 4273 entities, gold for 35 docs |
| fold3 (test) | 35 documents |
| Labels (file order) | `['APP', 'RESP', 'A.COUNSEL', 'R.COUNSEL', 'JUDGE', 'WIT', 'AUTH', 'COURT', 'STAT', 'PREC', 'DATE', 'CASENO']` |
| Document length (chars, fold1+fold2) | min 10443, median 26620, max 114044 |
| Entities per label (fold1+fold2) | APP 407, RESP 381, A.COUNSEL 195, R.COUNSEL 184, JUDGE 245, WIT 187, AUTH 1003, COURT 1013, STAT 1260, PREC 1183, DATE 1575, CASENO 760 |
| Tokens per document (fold1+fold2) | median 5533, max 23825, total 460,409 |

### Offset convention and gold sanity checks

- The gold records carry the entity text, so the code tests both conventions. Printed: `entity text given for 8393
  spans: text[s:e] matches 8393, text[s:e+1] matches 17 -> END_INCLUSIVE = False`. Internally all spans are
  end-exclusive (`text[start:end]`); predictions are written back in the detected gold convention (end-exclusive
  here, `off = 0`).
- Gold spans with an unknown label or outside `0 <= s < e <= len(text)` are dropped; printed: `invalid gold spans
  dropped: 0`, `spans whose entity text != document slice: 0`.
- Overlapping / nested gold spans cannot be expressed in BIO; `non_overlapping()` keeps the earliest-starting (then
  longest) span of each overlapping group. Printed: `overlapping / nested gold spans that BIO cannot represent: 0`.

### Train / validation split

No official dev split exists, so fold1 and fold2 are pooled (70 documents), the sorted document IDs are shuffled with
`random.Random(SEED)` and the first `round(VAL_FRAC * 70)` documents become the validation set. With `SEED = 42`,
`VAL_FRAC = 0.15` the logs print `train 60 docs  val 10 docs  (val from Counter({'fold1': 7, 'fold2': 3}))`. The split
depends only on `SEED` and `VAL_FRAC`, so it is identical for every run that keeps them; the validation IDs are saved
in `run_config.json` (`val_ids`).

### Tokenisation and character <-> token alignment

- The tokenizer is the Phase-1 tokenizer saved in the checkpoint's `best/` folder, loaded with
  `PreTrainedTokenizerFast`; the template comment describes it as byte-level BPE shared by all four Phase-1 models.
  `model_max_length` is set to 1e9 so whole documents are tokenised without truncation.
- Each document is tokenised **once, as a whole**, with `add_special_tokens=False` (no BOS/EOS) and
  `return_offsets_mapping=True`. The fast tokenizer's offset mapping gives, for every token, its
  `[start, end)` range in the Python string (character indices, not byte indices).
- Byte-level BPE tokens carry their leading whitespace (e.g. `" Court"`), so each token range is **trimmed**: leading
  and trailing whitespace characters are removed (`str.isspace`). A token that consists only of whitespace gets an
  empty trimmed range (`s == e`). Both the trimmed ranges and the raw ranges are kept.
- Windows (below) are slices of this document-level token sequence, so every token keeps its document-level
  character offsets; no re-tokenisation happens per window.

### Label encoding (BIO, 25 tags)

- Tag set `TAGS = ["O"] + [B-L, I-L for L in LABELS]` = 25 tags (O, B-APP, I-APP, B-RESP, ..., I-CASENO).
- A character -> entity-index array (`owner`) is built from the non-overlapping gold spans. A token's tag comes from
  the gold span that covers its **first visible character** (trimmed start): `O` if none, `I-L` if that span is the
  same span as the previous visible token's, otherwise `B-L`.
- Whitespace-only tokens never start an entity: they get `I-L` if their raw start character lies inside the same span
  as the previous visible token, else `O`; they do not update the "previous span".

### Span decoding (tags -> character spans)

`decode(tags, trimmed_ranges)`:

- whitespace-only tokens are skipped (transparent);
- `O` closes the open span;
- `I-L` after an open span of the same label extends it (`end = max(end, token_end)`);
- otherwise the open span is closed and a new span `[token_start, token_end, L]` is opened if the tag is `B-L`, or
  if it is `I-L` and `DECODE = "lenient"`. With `DECODE = "strict"` an `I-L` that does not continue a same-label span
  is dropped (as are the `I-L` tokens following it);
- duplicates are removed and spans are sorted.

Span start = trimmed start of the first token, span end = trimmed end of the last token, so predicted spans never
begin or end with whitespace.

### Oracle check

Gold spans -> BIO tags -> `decode` -> official strict metric, run before any training. The template states that
anything below 1.0 "is lost before any model is trained (span boundaries inside a token, overlapping spans)". Printed
by all three run logs:

```
ORACLE strict macro-F1 0.9863  (P 0.9863  R 0.9863)
  per-label oracle recall: {'APP': 0.9828, 'RESP': 0.9816, 'A.COUNSEL': 0.9949, 'R.COUNSEL': 1.0, 'JUDGE': 1.0, 'WIT': 1.0, 'AUTH': 0.986, 'COURT': 0.9822, 'STAT': 0.9802, 'PREC': 0.929, 'DATE': 1.0, 'CASENO': 0.9987}
  lost PREC      'State of M.P. vs. Kashiram & Ors.'  tokens: ['of', 'State', 'of', 'M', '.', 'P', '.', 'vs', '.', 'Kashi', 'ram', '&', 'Ors', '.[', '1', '],']
  lost AUTH      'Railway Board'  tokens: ['of', 'Railway', 'Boards', 'decision']
  lost AUTH      'State Government'  tokens: ['the', 'State', 'Governments', 'and']
  lost STAT      'Constitution'  tokens: ['the', 'Constitutional']
```

(excerpt of `P2_arch1_llama_after.ipynb`; the full list of 8 printed examples is in each log).

### Long-document windowing

- `windows(n, WIN, stride)`: a document of `n <= WIN` tokens is one window; otherwise windows start at
  `0, stride, 2*stride, ...` and a final window `[n - WIN, n)` is appended if the last regular window does not reach
  the end.
- **Training**: `WIN = 512`, `STRIDE_TRAIN = 256`. Every token of every training window is labelled and contributes
  to the loss (overlapping regions are trained twice). Printed: `1476 training windows of <= 512 tokens -> 93
  steps/epoch`.
- **Prediction**: `WIN = 512`, `STRIDE_PRED = 128`. Per token, the outputs of all windows covering it are averaged
  (linear head: softmax probabilities; CRF head: emission scores), then tags are chosen for the whole document
  (argmax, or one Viterbi pass over the document for the CRF).
- Batches are right-padded with the tokenizer's pad ID; padded positions have attention mask 0 and label -100.

### Class imbalance

The only class-balancing knob is `O_WEIGHT`: the cross-entropy weight of tag `O` (all 24 entity tags weigh 1). It
applies to the linear head only (the CRF loss is unweighted). Template default and value in all three run logs: 1.0.
Model selection uses macro-F1 over the 12 labels (Section 6).

## 4. Model architecture

`Tagger(arch, attn, bart_mode, layers, head, crf_constraints)` = Phase-1 backbone + dropout + linear layer (+ optional
CRF). The Phase-1 weights are loaded from `<hf-user>/cs781-p1-<arch>/best/` (bart: `cs781-p1-bart-v2/best/` in the
current template) with `snapshot_download(..., allow_patterns=["best/*"])` and cast to fp32. The language-model head
is discarded; the backbone's final hidden states (`last_hidden_state`, size `H`) feed the tagger head.

| ARCH | Class loaded | Backbone module used | Default `ATTN` | Tagger parameters printed (linear head) |
|---|---|---|---|---|
| `llama` | `LlamaForCausalLM` | `.model` (`LlamaModel`) | `causal` | `llama tagger (linear head): 109.5M parameters` (`params_total` 109549081) |
| `bart` | `BartForConditionalGeneration` | `.model.encoder` (`BART_MODE="encoder"`) or `.model` (`BART_MODE="decoder"`) | `n/a` | not in run_logs (TODO(student)) |
| `prefixlm` | `LlamaForCausalLM` | `.model` (`LlamaModel`) | `bidirectional` | `prefixlm tagger (linear head): 109.5M parameters` (`params_total` 109549081) |
| `mistral_swa` | `MistralForCausalLM` | `.model` (`MistralModel`) | `causal` (checkpoint `sliding_window`) | `mistral_swa tagger (linear head): 100.1M parameters` (`params_total` 100111897) |

The printed totals equal exactly 32,000 x 768 (token embeddings) + 12 blocks with d_model 768 and FFN 2048
(mistral_swa: 4 KV heads of dimension 64) + final norm + head (768 x 25 + 25), i.e. the Phase-1 template defaults in
`code/phase1/phase1_arch_template.py` (the Phase-2 code itself reads `hidden_size` / `d_model` from the checkpoint
config).

### Attention during fine-tuning (`ATTN`, `BART_MODE`)

- Decoder-only backbones, `ATTN = "causal"`: the 2-D padding mask is passed to the HF model, which applies its own
  causal mask (mistral_swa: causal sliding-window mask of the checkpoint's `sliding_window`).
- `ATTN = "bidirectional"`: a 4-D boolean mask `(B, 1, L, L)` is built from the padding mask (every real token
  attends to every real token); for mistral_swa it is further restricted to the band `|i - j| < sliding_window`.
- BART, `BART_MODE = "encoder"`: encoder `last_hidden_state`. `BART_MODE = "decoder"`: the window is fed to the
  encoder and also as `decoder_input_ids` (unshifted) and the decoder's `last_hidden_state` is used. The template
  notes that the Phase-1 BART decoder only saw positions < `TGT_LEN = 256` and the encoder up to 768.

**Attention / padding sanity check** (runs once before training): token 40 of a 64-token input is changed; outputs at
positions < 40 may only move if the model sees right context; appending 8 pad tokens must not change real-token
outputs. A mismatch raises an error, except for BART, where the numbers are printed (plus encoder-only numbers) and
training continues. Printed by the run logs:

| Run log | Output |
|---|---|
| llama | `changing token 40 moves outputs at positions <40 by 0.00e+00, at 40 by 3.06e+00 -> sees right context: False (expected False); padding-invariant: True` |
| prefixlm | `changing token 40 moves outputs at positions <40 by 7.17e-01, at 40 by 2.05e+00 -> sees right context: True (expected True); padding-invariant: True` |
| mistral_swa | `changing token 40 moves outputs at positions <40 by 0.00e+00, at 40 by 2.46e+00 -> sees right context: False (expected False); padding-invariant: True` |

### Heads and losses (`HEAD`)

- `HEAD = "linear"`: logits `z_t = W * Dropout(h_t) + b`, `W` in R^(25 x H), dropout `DROPOUT` (only before the
  head; the backbone's own dropout settings come from the Phase-1 config). Loss: class-weighted cross-entropy over
  real tokens (`ignore_index = -100`, weights 1 except `O_WEIGHT` for `O`; PyTorch weighted-mean reduction).
- `HEAD = "crf"` (linear-chain CRF, Lafferty et al. 2001, own implementation): the linear outputs are emission scores
  `e_t`; learned parameters `start` (25), `end` (25), `trans` (25 x 25). Sequence score
  `s(y) = start[y_1] + sum_t e_t[y_t] + sum_t trans[y_(t-1), y_t] + end[y_L]`. Loss = `(log Z - s(y_gold))`, summed
  over the batch and divided by the number of real tokens (forward algorithm with `logsumexp`). Decoding: Viterbi in
  float64 on CPU over the window-averaged emissions of the whole document.
  `CRF_BIO_CONSTRAINTS = True` sets the transition score of invalid BIO transitions (`O -> I-X`, `B-Y/I-Y -> I-X` for
  `Y != X`) to -1e4; the start-of-sequence transition is left unconstrained because windows can begin inside an
  entity.
- KD loss: see Section 5.

## 5. Fine-tuning methods implemented

`METHOD` selects which parameters train; `HEAD` (linear / crf) is independent of it. The head (and CRF parameters)
are trainable in every method. No PEFT library is used: LoRA, LoRMA, partial, BitFit and KD are implemented in the
template. These correspond to the capstone section 8.2.1 list entries Full fine-tuning, Partial/layer-wise
fine-tuning, LoRA, LoRMA, BitFit and Knowledge Distillation (Prefix tuning, Prompt tuning and Houlsby/Pfeiffer
adapters are not implemented).

| `METHOD` | Trainable parameters | Implementation | Knobs (default) | Default `LR` |
|---|---|---|---|---|
| `full` | all backbone parameters + head | `requires_grad = True` everywhere | - | 5e-5 |
| `partial` | head + the top `PARTIAL_TRAIN_TOP` transformer blocks + the final norm `backbone.norm` if the backbone has one; embeddings and lower blocks frozen | `block_stacks()` lists blocks in forward order (bart decoder mode: encoder blocks then decoder blocks) and unfreezes the last N | `PARTIAL_TRAIN_TOP = 4` | 1e-4 |
| `bitfit` | head + every backbone parameter whose name ends in `.bias` or contains `norm` | name filter over `named_parameters()` | - | 1e-3 |
| `lora` | head + LoRA `A`, `B` on every `nn.Linear` whose attribute name is in `ADAPT_TARGETS` | own `LoRALinear` wrapper | `ADAPT_TARGETS = ["q_proj", "v_proj"]`, `RANK = 8`, `ALPHA = 16` | 5e-4 |
| `lorma` | head + LoRMA `A`, `B` on the same layers | own `LoRMALinear` wrapper | `RANK`, `ALPHA`, `LORMA_VARIANT = "plus"` | 5e-4 |
| `kd` | all parameters of a reduced-depth student | student = this notebook's `ARCH` with a subset of Phase-1 blocks; frozen teacher from a previous run | `TEACHER_RUN = "llama_full"`, `STUDENT_LAYERS = None`, `KD_T = 2.0`, `KD_ALPHA = 0.5` | 1e-4 |

### LoRA (Hu et al., 2021)

`y = W0 x + (alpha / r) * B (A x)`, `A` in R^(r x d_in) (Kaiming-uniform init), `B` in R^(d_out x r) (zero init), `W0`
frozen. Trainable count per adapted layer: `r * (d_in + d_out)`. After training the adapter is merged:
`W = W0 + (alpha / r) * B A`.

### LoRMA (Bihany, Patel & Modi, 2025) - multiplicative update `y = I(BA) * W0 x`

`A` in R^(r x d_out), `B` in R^(d_out x r) (they act on the output space of `W0`); trainable count per adapted layer:
`2 * r * d_out`.

- `LORMA_VARIANT = "plus"` (additive-identity rank inflation): `I(BA) = I + (alpha / r) * BA`, computed as
  `W0 x + (alpha / r) * B (A (W0 x))` without forming `BA`; `A` Kaiming-uniform, `B = 0` at init. Merged:
  `W = W0 + (alpha / r) * B (A W0)`.
- `LORMA_VARIANT = "pi"` (permutation-based inflation): `I(BA)[i, j] = (BA)[i, (j - i) mod d]` (row `i` of `BA`
  rotated cyclically by `i`); scale fixed to 1 (`ALPHA` unused). Init: `B` Kaiming-uniform with first column set to 1,
  `A = e0 e0^T`, which gives `I(BA) = I`. Merged: `W = I(BA) W0`.
- A bias (present only in BART linear layers) is added after the multiplicative transform.

For both LoRA and LoRMA the code (a) checks on two training windows that the injected model's output equals the
original at initialisation (asserts max change < 1e-3; prints `output change at init`), (b) merges every adapter
into a plain `nn.Linear` after training (no inference overhead), and (c) checks the merge numerically on up to four
validation windows (asserts max `|logit change|` < 1e-2 x max(1, max `|logit|`); prints `adapter merge: max |logit
change|`). With the template defaults on llama (12 blocks, d = 768, `q_proj` + `v_proj`, r = 8) the formulas give
294,912 adapter parameters + 19,225 head (+ 675 CRF) - computed from the code, not printed by any run log.

### Knowledge distillation (`METHOD = "kd"`)

- **Teacher**: any finished Phase-2 run in `<hf-user>/cs781-p2` named by `TEACHER_RUN` (any architecture; the template
  states all four Phase-1 models share one tokenizer, so token positions line up). The notebook downloads
  `runs/<TEACHER_RUN>/run_config.json` (arch, attn, bart_mode, layers, head, CRF constraints, tag list, val IDs) and
  `runs/<TEACHER_RUN>/model.pt` (merged weights, written when the teacher run had `SAVE_WEIGHTS = True`), rebuilds the
  teacher `Tagger`, freezes it and evaluates it on the validation set (prints teacher params, val macro-F1, stored
  val macro-F1, val inference seconds). It asserts the tag sets match and prints a warning if the teacher used a
  different validation split.
- **Student**: this notebook's `ARCH`, initialised from Phase-1 weights but keeping only the blocks in
  `STUDENT_LAYERS` (default `None` = every 2nd block, indices 0, 2, 4, ...; for bart applied to each stack that the
  backbone contains). All student parameters train.
- **Loss**: `KD_ALPHA * L_task + (1 - KD_ALPHA) * T^2 * KL(softmax(z_teacher / T) || softmax(z_student / T))`, with
  `T = KD_T`, computed over the real (non-padding) tokens of each batch (`batchmean` over those tokens). `L_task` is
  the cross-entropy or CRF loss above. The softmax of the per-token emissions is distilled for both head types.

## 6. Training procedure

- **Optimizer**: AdamW, betas (0.9, 0.999), `fused=True` on CUDA. Three parameter groups: head (+CRF) at `HEAD_LR`
  with `WEIGHT_DECAY`; backbone / adapter parameters with >= 2 dimensions at `LR` with `WEIGHT_DECAY`; backbone
  1-D parameters (biases, norms) at `LR` with weight decay 0.
- **Schedule**: linear warm-up over `WARMUP_FRAC x total_steps`, then linear decay to 0 at
  `total_steps = EPOCHS x ceil(#train windows / BATCH)`; applied as a multiplier to each group's base LR every step.
- **Batch**: `BATCH` windows per step, no gradient accumulation; window order reshuffled every epoch
  (`np.random.permutation`, seeded with `SEED`).
- **Precision**: weights in fp32; forward in fp16 autocast; loss in fp32; `torch.amp.GradScaler`; gradient clipping
  to `GRAD_CLIP` (L2 norm over trainable parameters).
- **Evaluation and model selection**: after every epoch the full validation documents are predicted (stride
  `STRIDE_PRED`) and scored with `strict_scores()`, which reimplements the official per-label strict P/R/F1 and the
  macro average over the 12 labels. When validation strict macro-F1 is strictly higher than the best so far, a CPU copy
  of the full `state_dict` is kept. After training, the best state is reloaded.
- **Stopping**: `EPOCHS` maximum; early stop after `PATIENCE` consecutive epochs without improvement; hard time cap
  `MAX_TRAIN_MINUTES` (wall clock since training start, checked after every step; the partial epoch is still
  evaluated, and the best epoch so far is kept).
- **Reported timing / memory**: `train_minutes` sums the training part of each epoch (validation prediction
  excluded); the per-epoch time printed in the log includes validation. `peak_gpu_gb` is
  `torch.cuda.max_memory_allocated()` after a reset just before training.
- **Checkpoint / resume**: there is no intermediate checkpointing or resume; every run starts from the Phase-1
  checkpoint. The final (best, merged) `state_dict` is saved as `model.pt` when `SAVE_WEIGHTS = True`.

## 7. Inference and submission

1. Best state reloaded, adapters merged (Section 5), validation re-predicted and reported: strict macro-F1 / P / R,
   inference seconds, and a per-label table (P, R, F1, gold count, predicted count).
2. **Error breakdown on validation (counts only)**: each non-exact predicted span is put in one category - right span
   / wrong label, right label / wrong boundary (overlaps a gold span), wrong boundary and label, spurious (overlaps no
   gold span); gold spans with no overlapping prediction are counted as missed. The 8 most frequent (gold -> predicted)
   label confusions are printed.
3. **Test**: the 35 fold3 documents are predicted with the same windowing (`STRIDE_PRED`, batch `2 x BATCH`, fp16
   autocast) and decoding (`DECODE`, argmax or Viterbi). No thresholds or post-processing rules beyond `decode`.
4. **prediction.json**: `{doc_id: [[start, end - off, label], ...]}` for every test ID, sorted, de-duplicated, spans
   with `end - off <= start` removed (`off = 1` only if the gold convention were end-inclusive; 0 here).
5. **Validation of the file** with the same checks as the official evaluator: ID set equals the test IDs; integer
   offsets with `0 <= start < end`; label in the 12 labels; no duplicate span. Printed: number of documents, entities,
   seconds, entities per label.
6. **Zip**: `submission_<RUN_NAME>.zip` containing only `prediction.json` at the archive root (asserted), written to
   `RUN_DIR`, and downloaded automatically in Colab.
7. **Run upload** to the private repo `<hf-user>/cs781-p2` under `runs/<RUN_NAME>/` (3 upload attempts, commit message
   contains the val F1): `prediction.json`, `submission_<RUN_NAME>.zip`, `run_config.json` (all config values,
   parameter counts, timings, oracle F1, val metrics, per-label metrics, error breakdown, teacher info, epoch
   history), `val_pred.json`, `val_gold.json`, `curves.png` (train loss and val macro-F1 per epoch) and, if
   `SAVE_WEIGHTS`, `model.pt`.
8. **Local evaluation with the official script**: `val_gold.json` / `val_pred.json` use the
   `[start, end, label]` list format, so they can be scored with
   `python evaluate_phase2.py --truth val_gold.json --prediction val_pred.json --labels labels.txt`
   (`labels.txt` from `Phase_2_LNER/Train/`). Note these two files are written in the internal end-exclusive
   convention (identical to the gold convention here, `END_INCLUSIVE = False`).

## 8. Configuration knobs

Cell 1 (`METHOD`, `HEAD`, `RUN_NAME`) and the cell under "Configuration - all values below are YOUR CHOICE".

| Variable | Template default | Meaning |
|---|---|---|
| `ARCH` | set by the builder | Phase-1 architecture of this notebook (fixed). |
| `METHOD` | `"full"` | `full`, `partial`, `bitfit`, `lora`, `lorma`, `kd` (Section 5). |
| `HEAD` | `"linear"` | `linear` (per-token softmax) or `crf` (linear emissions + linear-chain CRF). |
| `RUN_NAME` | `f"{ARCH}_{METHOD}"` + `_<HEAD>` if not linear | Unique run ID; names the run folder, Hub folder and zip. |
| `DATASET` | `"Exploration-Lab/CS-781-Capstone"` | HF dataset repo. |
| `P1_REPOS` / `P1_REPO` | `cs781-p1-{llama, bart-v2, prefixlm, mistral_swa}` | Phase-1 checkpoint repo per architecture (`best/`). |
| `P2_REPO` | `f"{HF_USER}/cs781-p2"` | Private repo receiving all Phase-2 runs (`runs/<RUN_NAME>/`); also the KD teacher source. |
| `WORK`, `RUN_DIR` | `/content/p2` (else `/tmp/p2`), `WORK/RUN_NAME` | Local output folders. |
| `SEED` | 42 | Seed for Python, NumPy, torch and the train/val split. |
| `VAL_FRAC` | 0.15 | Fraction of fold1+fold2 documents held out for validation. |
| `WIN` | 512 | Window length in tokens (training and prediction). |
| `STRIDE_TRAIN` | 256 | Window stride during training. |
| `STRIDE_PRED` | 128 | Window stride during prediction (overlaps averaged per token). |
| `ATTN` | llama `causal`, prefixlm `bidirectional`, mistral_swa `causal`, bart `n/a` | Attention mask used by decoder-only backbones during fine-tuning. |
| `BART_MODE` | `"encoder"` | bart only: encoder states, or encoder + decoder (decoder states). |
| `EPOCHS` | 10 | Maximum epochs (also sets the LR schedule length). |
| `PATIENCE` | 3 | Early stopping: epochs without val macro-F1 improvement. |
| `MAX_TRAIN_MINUTES` | 60 | Hard training-time cap (minutes). |
| `BATCH` | 16 | Windows per optimisation step (prediction uses 2 x BATCH). |
| `LR` | full 5e-5, partial 1e-4, bitfit 1e-3, lora 5e-4, lorma 5e-4, kd 1e-4 | Peak LR of backbone / adapter parameters. |
| `HEAD_LR` | 1e-3 | Peak LR of the head (and CRF). |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay (not applied to 1-D backbone parameters). |
| `WARMUP_FRAC` | 0.06 | Fraction of total steps used for linear warm-up. |
| `GRAD_CLIP` | 1.0 | Gradient-norm clipping threshold. |
| `DROPOUT` | 0.1 | Dropout before the token-classification head. |
| `O_WEIGHT` | 1.0 | Cross-entropy weight of tag `O` (linear head only). |
| `CRF_BIO_CONSTRAINTS` | `False` | CRF only: forbid invalid BIO transitions. |
| `PARTIAL_TRAIN_TOP` | 4 | `partial`: number of top transformer blocks trained. |
| `ADAPT_TARGETS` | `["q_proj", "v_proj"]` | `lora` / `lorma`: attribute names of the `nn.Linear` layers that get adapters. |
| `RANK`, `ALPHA` | 8, 16 | `lora` / `lorma` rank and scaling (`alpha / r`). |
| `LORMA_VARIANT` | `"plus"` | `plus` (additive-identity rank inflation) or `pi` (permutation inflation). |
| `TEACHER_RUN` | `"llama_full"` | `kd`: finished run in `P2_REPO` used as teacher. |
| `STUDENT_LAYERS` | `None` | `kd`: kept Phase-1 block indices (`None` = every 2nd block). |
| `KD_T` | 2.0 | `kd`: softmax temperature. |
| `KD_ALPHA` | 0.5 | `kd`: weight of the gold-label loss (KL term gets `1 - KD_ALPHA`). |
| `DECODE` | `"lenient"` | Span decoding: `lenient` (an `I-X` can open an entity) or `strict` (only `B-X` opens one). |
| `SAVE_WEIGHTS` | `True` | Save merged `model.pt` to the Hub (needed for a KD teacher). |

## 9. Runs recorded in `run_logs/`

Environment printed by all three logs: `2.11.0+cu130 Tesla T4`, `transformers 5.18.0`.

### Configuration and summary

All three runs use the template defaults for every knob except the one listed (diffed against the generated
notebooks).

| File | RUN_NAME | Arch | METHOD / HEAD | Config differing from template | Best epoch | Val strict macro-F1 / P / R | Trainable / total params | Train min | Peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|
| `P2_arch1_llama_after.ipynb` | `llama_full` | llama | full / linear | `DECODE = "strict"` | 4 | 0.3903 / 0.4423 / 0.3681 | 109549081 / 109549081 | 4.928696858882904 | 5.87561984 |
| `P2_arch3_prefixlm_after.ipynb` | `prefixlm_full` | prefixlm | full / linear | none | 2 | 0.3383 / 0.3004 / 0.4111 | 109549081 / 109549081 | 3.872544248898824 | 5.979428864 |
| `P2_arch4_mistral_swa_after.ipynb` | `mistral_swa_full` | mistral_swa | full / linear | none | 4 | 0.3206 / 0.2859 / 0.3768 | 100111897 / 100111897 | 4.929499351978302 | 5.825288192 |

Settings in force in these three runs: `LR = 5e-5`, `HEAD_LR = 1e-3`, `BATCH = 16` (no accumulation), `EPOCHS = 10`,
`PATIENCE = 3`, `MAX_TRAIN_MINUTES = 60`, `WIN = 512`, `STRIDE_TRAIN = 256`, `STRIDE_PRED = 128`, `DROPOUT = 0.1`,
`WEIGHT_DECAY = 0.01`, `WARMUP_FRAC = 0.06`, `GRAD_CLIP = 1.0`, `O_WEIGHT = 1.0`, `ATTN` = arch default
(llama causal, prefixlm bidirectional, mistral_swa causal), `SEED = 42`, `VAL_FRAC = 0.15`.

Inference times and test output printed by the logs:

| RUN_NAME | Final val prediction | Test prediction | Test entities written | Stop message |
|---|---|---|---|---|
| `llama_full` | 5.1s | 17.0s | 35 documents, 3639 entities | `stopping: no improvement for 3 epochs` |
| `prefixlm_full` | 5.7s | 18.9s | 35 documents, 5535 entities | `stopping: no improvement for 3 epochs` |
| `mistral_swa_full` | 5.1s | 17.1s | 35 documents, 5141 entities | `stopping: no improvement for 3 epochs` |

All three print `adapter merge: max |logit change| = 0.00e+00` (no adapters in `full`) and
`oracle_F1 0.986284164100113`.

### Per-epoch history (as printed)

| Epoch | llama_full loss / val F1 / P / R | prefixlm_full loss / val F1 / P / R | mistral_swa_full loss / val F1 / P / R |
|---|---|---|---|
| 1 | 0.5669 / 0.3231 / 0.3817 / 0.2978 (46s) | 0.5695 / 0.2596 / 0.2425 / 0.2865 (49s) | 0.6189 / 0.2422 / 0.2205 / 0.2769 (45s) |
| 2 | 0.0498 / 0.3618 / 0.4037 / 0.3533 (47s) | 0.0520 / 0.3383 / 0.3004 / 0.4111 (51s) | 0.0497 / 0.2925 / 0.2597 / 0.3526 (46s) |
| 3 | 0.0185 / 0.3744 / 0.4291 / 0.3450 (48s) | 0.0199 / 0.3004 / 0.2850 / 0.3296 (53s) | 0.0190 / 0.2888 / 0.2614 / 0.3328 (47s) |
| 4 | 0.0076 / 0.3903 / 0.4423 / 0.3681 (48s) | 0.0078 / 0.3317 / 0.3053 / 0.3799 (54s) | 0.0076 / 0.3206 / 0.2859 / 0.3768 (48s) |
| 5 | 0.0027 / 0.3806 / 0.4373 / 0.3512 (48s) | 0.0029 / 0.3185 / 0.2976 / 0.3511 (54s) | 0.0029 / 0.3121 / 0.2810 / 0.3590 (48s) |
| 6 | 0.0016 / 0.3744 / 0.4417 / 0.3388 (48s) | - | 0.0013 / 0.3115 / 0.2819 / 0.3562 (48s) |
| 7 | 0.0011 / 0.3754 / 0.4453 / 0.3386 (48s) | - | 0.0010 / 0.3146 / 0.2904 / 0.3508 (48s) |

### Per-label validation results of the selected epoch (P / R / F1; gold, predicted counts)

| Label | Gold | llama_full P / R / F1 (pred) | prefixlm_full P / R / F1 (pred) | mistral_swa_full P / R / F1 (pred) |
|---|---|---|---|---|
| APP | 78 | 0.2105 / 0.1026 / 0.1379 (38) | 0.1011 / 0.2436 / 0.1429 (188) | 0.0625 / 0.0641 / 0.0633 (80) |
| RESP | 58 | 0.1250 / 0.1552 / 0.1385 (72) | 0.0621 / 0.1724 / 0.0913 (161) | 0.1333 / 0.2759 / 0.1798 (120) |
| A.COUNSEL | 53 | 0.5806 / 0.3396 / 0.4286 (31) | 0.0303 / 0.0189 / 0.0233 (33) | 0.2299 / 0.3774 / 0.2857 (87) |
| R.COUNSEL | 53 | 0.3684 / 0.3962 / 0.3818 (57) | 0.2500 / 0.5472 / 0.3432 (116) | 0.2468 / 0.3585 / 0.2923 (77) |
| JUDGE | 48 | 0.4219 / 0.5625 / 0.4821 (64) | 0.3103 / 0.5625 / 0.4000 (87) | 0.3140 / 0.5625 / 0.4030 (86) |
| WIT | 28 | 0.0385 / 0.0357 / 0.0370 (26) | 0.0000 / 0.0000 / 0.0000 (37) | 0.0000 / 0.0000 / 0.0000 (35) |
| AUTH | 91 | 0.2581 / 0.0879 / 0.1311 (31) | 0.2925 / 0.3407 / 0.3147 (106) | 0.0722 / 0.0769 / 0.0745 (97) |
| COURT | 148 | 0.9048 / 0.8986 / 0.9017 (147) | 0.8618 / 0.8851 / 0.8733 (152) | 0.8405 / 0.9257 / 0.8810 (163) |
| STAT | 214 | 0.6321 / 0.3131 / 0.4187 (106) | 0.4246 / 0.5000 / 0.4592 (252) | 0.3348 / 0.3598 / 0.3468 (230) |
| PREC | 210 | 0.3041 / 0.2810 / 0.2921 (194) | 0.1801 / 0.3190 / 0.2302 (372) | 0.1207 / 0.2333 / 0.1591 (406) |
| DATE | 324 | 0.9359 / 0.9012 / 0.9182 (312) | 0.8912 / 0.9352 / 0.9127 (340) | 0.8169 / 0.8673 / 0.8413 (344) |
| CASENO | 169 | 0.5273 / 0.3432 / 0.4158 (110) | 0.2006 / 0.4083 / 0.2690 (344) | 0.2591 / 0.4201 / 0.3205 (274) |

### Validation error breakdown (counts printed by the logs)

| Category | llama_full | prefixlm_full | mistral_swa_full |
|---|---|---|---|
| exact | 701 | 794 | 709 |
| right label, wrong boundary | 190 | 590 | 656 |
| wrong boundary and label | 80 | 232 | 196 |
| right span, wrong label | 33 | 58 | 27 |
| spurious | 184 | 514 | 411 |
| missed (no overlapping prediction) | 479 | 131 | 224 |

Most frequent label confusions (gold -> predicted), verbatim:

- llama_full: `[(('A.COUNSEL', 'R.COUNSEL'), 10), (('RESP', 'PREC'), 4), (('APP', 'PREC'), 4), (('RESP', 'APP'), 3), (('PREC', 'COURT'), 2), (('APP', 'RESP'), 2), (('A.COUNSEL', 'JUDGE'), 1), (('PREC', 'JUDGE'), 1)]`
- prefixlm_full: `[(('A.COUNSEL', 'R.COUNSEL'), 30), (('R.COUNSEL', 'A.COUNSEL'), 10), (('PREC', 'APP'), 6), (('RESP', 'APP'), 5), (('APP', 'RESP'), 2), (('PREC', 'COURT'), 2), (('AUTH', 'APP'), 1), (('PREC', 'JUDGE'), 1)]`
- mistral_swa_full: `[(('R.COUNSEL', 'A.COUNSEL'), 7), (('RESP', 'PREC'), 4), (('APP', 'PREC'), 3), (('APP', 'RESP'), 2), (('A.COUNSEL', 'R.COUNSEL'), 2), (('DATE', 'CASENO'), 1), (('PREC', 'COURT'), 1), (('CASENO', 'PREC'), 1)]`

Test entities per label printed by the logs:

- llama_full: `{'CASENO': 300, 'APP': 129, 'RESP': 103, 'JUDGE': 92, 'DATE': 722, 'COURT': 526, 'AUTH': 228, 'STAT': 683, 'PREC': 597, 'A.COUNSEL': 133, 'R.COUNSEL': 69, 'WIT': 57}`
- prefixlm_full: `{'COURT': 622, 'CASENO': 522, 'APP': 349, 'RESP': 254, 'JUDGE': 168, 'DATE': 795, 'AUTH': 440, 'STAT': 987, 'PREC': 994, 'WIT': 114, 'A.COUNSEL': 62, 'R.COUNSEL': 228}`
- mistral_swa_full: `{'CASENO': 477, 'APP': 200, 'RESP': 182, 'JUDGE': 156, 'DATE': 763, 'COURT': 587, 'PREC': 1123, 'AUTH': 321, 'STAT': 978, 'A.COUNSEL': 154, 'R.COUNSEL': 103, 'WIT': 97}`

### Runs without a log in this repository

- `llama_lorma_crf` (known only from the local submission folder name `submission_llama_lorma_crf`; by the `RUN_NAME`
  pattern: `ARCH = "llama"`, `METHOD = "lorma"`, `HEAD = "crf"`). Its other settings (`LORMA_VARIANT`, `RANK`, `ALPHA`,
  `ADAPT_TARGETS`, `CRF_BIO_CONSTRAINTS`, `DECODE`, `LR`, ...), parameter counts and metrics: TODO(student) - they are
  stored in `<hf-user>/cs781-p2/runs/llama_lorma_crf/run_config.json`.
- `bart` (no run log). During Phase 2 (deadline 2026-10-11 23:59 IST) only the collapsed BART v1 checkpoint
  (`cs781-p1-bart`) existed; v2 was trained on 2026-10-14/15. So every Phase-2 BART run fine-tuned v1. Two BART prediction
  files were downloaded from Colab (not copied into this repository): `submission_bart_full.zip` (2026-10-11 11:24 IST)
  and `submission_bart_full (1).zip` (2026-10-14 21:04 IST, after the deadline). See section 10 for whether they were
  submitted.
- No run log or local submission folder for a `kd` run is present here: TODO(student) if such runs exist.

## 10. Codabench submissions

Five Phase-2 submissions were made, all on 2026-10-11 (deadline 23:59 IST). They were uploaded as `submission.zip`
without a record of which run produced each one. Six Phase-2 prediction files were built before the deadline; times
are the file time inside each zip, converted from Colab's UTC to IST. Each file's span count was compared with the
test-set count printed by its notebook (`run_logs/`):

| File (local name) | Built (IST) | Spans | Notes |
|---|---|---|---|
| `submission_llama_full` (local copy saved 11:24) | 11:18 | 5264 | first llama full fine-tuning run; no run log kept |
| `submission_bart_full.zip` | 11:23 | 39,953 | BART **v1** backbone (the only BART checkpoint at the time). Median span 6 characters; 187 spans are at least 1,000 characters long (longest 20,497) |
| `submission_prefixlm_full` (local copy saved 13:26) | 12:01 | 5535 | same count as printed by `P2_arch3_prefixlm_after.ipynb` (`DECODE = "lenient"`) |
| `submission_llama_full.zip` | 13:37 | 3639 | same count as printed by `P2_arch1_llama_after.ipynb` (`DECODE = "strict"`) |
| `submission_mistral_swa_full.zip` (local copy saved 13:42) | 13:39 | 5141 | same count as printed by `P2_arch4_mistral_swa_after.ipynb` (`DECODE = "lenient"`) |
| `submission_llama_lorma_crf.zip` (local copy saved 16:59) | 14:07 | 3288 | no run log kept |

A seventh file, `submission_bart_full (1).zip` (2026-10-14, after the deadline, BART v1), has 0 spans.

The local copies of four files were saved 1-2 minutes before an upload (11:24 -> 11:28, 13:26 -> 13:27,
13:42 -> 13:44, 16:59 -> 16:59). Using that pattern:

| Codabench ID | Submitted (IST) | Run | Key settings | Strict Macro-F1 | Attribution |
|---|---|---|---|---|---|
| 976438 | 11:28 | llama full, first run | `full`, linear head; `DECODE` not recorded | 0.3957 | inferred. The BART v1 file, built in the same minutes, is the only other candidate. TODO(student): confirm |
| 976728 | 13:27 | prefixlm full | `full`, linear head, bidirectional attention, `DECODE = "lenient"` | 0.3893 | local copy saved 13:26 |
| 976758 | 13:40 | **llama full, strict** (leaderboard entry) | `full`, linear head, causal, `DECODE = "strict"`, LR 5e-5, head LR 1e-3, batch 16, window 512, best epoch 4 | **0.4399** | inferred: 13:40 / 13:44 are the llama-strict and mistral_swa files, ordered by the pattern above. TODO(student): confirm |
| 976769 | 13:44 | mistral_swa full | `full`, linear head, causal SWA, `DECODE = "lenient"` | 0.3886 | inferred (see above) |
| 977284 | 16:59 | llama LoRMA + CRF | `lorma`, CRF head; other settings in `<hf-user>/cs781-p2/runs/llama_lorma_crf/run_config.json` | 0.4038 | the only file built after 13:39 before the deadline; local copy saved 16:59 |

No BART run and no KD run was submitted in Phase 2 by this attribution.

---

## 11. How to run

1. **Build the notebooks** (any machine with Python 3; no extra packages needed):
   ```bash
   cd code/phase2
   python build_p2_notebooks.py              # all four: P2_arch1_llama.ipynb ... P2_arch4_mistral_swa.ipynb
   python build_p2_notebooks.py bart         # only P2_arch2_bart.ipynb
   ```
2. **Open** `P2_arch<N>_<arch>.ipynb` in Google Colab and select *Runtime -> Change runtime type -> T4 GPU* (the
   notebook asserts a GPU unless the environment variable `CS781_ALLOW_CPU` is set).
3. **Secret**: add a Colab secret named `HF` holding the same Hugging Face token used in Phase 1 (it must read the
   private `cs781-p1-*` repos and the gated course dataset, and write `cs781-p2`). On Kaggle the secret `HF` is read
   via `kaggle_secrets`; elsewhere the `HF_TOKEN` environment variable is used.
4. **Choose the experiment** in the first code cell (`METHOD`, `HEAD`; `RUN_NAME` is derived from them and must be
   unique per experiment), then review the *Configuration* cell (all values are the student's choice).
5. **Run all.** The notebook downloads the data and the Phase-1 checkpoint, prints the gold / oracle / attention
   checks, trains with early stopping, reports validation metrics, writes the test predictions and uploads the run.
6. **Outputs**: local `/content/p2/<RUN_NAME>/` (`prediction.json`, `submission_<RUN_NAME>.zip`, `run_config.json`,
   `val_pred.json`, `val_gold.json`, `curves.png`, `model.pt`), mirrored to `<hf-user>/cs781-p2/runs/<RUN_NAME>/`.
   The zip is downloaded automatically in Colab and is the file to upload to Codabench.
7. **Another experiment** on the same account: change `METHOD` / `HEAD` (and thus `RUN_NAME`) and *Run all* again;
   every run starts from the Phase-1 checkpoint.
8. **Knowledge distillation ordering**: first finish the teacher run (any architecture, `SAVE_WEIGHTS = True`, so
   `runs/<teacher>/model.pt` and `run_config.json` exist in `cs781-p2`). Then, in the student's notebook, set
   `METHOD = "kd"`, `TEACHER_RUN = "<teacher RUN_NAME>"` and optionally `STUDENT_LAYERS`, `KD_T`, `KD_ALPHA`, and run
   all. Keep `SEED` / `VAL_FRAC` equal to the teacher's so both use the same validation split (the notebook warns
   otherwise).
9. **Local scoring with the official evaluator** (optional): download `val_gold.json` and `val_pred.json` of a run and
   run `python evaluate_phase2.py --truth val_gold.json --prediction val_pred.json --labels labels.txt`.
10. **BART checkpoint diagnostic** (optional): in `P2_arch2_bart.ipynb`, after the tokenizer cell has run, paste the
    contents of `bart_p2_diagnostic_cell.py` into a new cell and run it.
