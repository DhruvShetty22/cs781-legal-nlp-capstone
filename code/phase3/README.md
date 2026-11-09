# Phase 3 — Rhetorical Role Prediction (RR)

Sentence-level rhetorical-role labelling of Indian legal judgments from two domains, Corporate Law (CL) and
Information Technology (IT), starting from the four Phase-1 language models (`llama`, `bart` (v2), `prefixlm`,
`mistral_swa`). All code is one percent-format template that is rendered into one Colab notebook per architecture.

---

## 1. Task

| | |
|---|---|
| Input | A document = `id` + ordered list of sentences (`text`). Train/dev documents also carry an ordered `labels` list of the same length. |
| Output | Exactly one label per sentence, from the 13 labels of `label_vocab.json`: `Fact`, `Issue`, `ArgumentPetitioner`, `ArgumentRespondent`, `Statute`, `Dissent`, `PrecedentReliedUpon`, `PrecedentNotReliedUpon`, `PrecedentOverruled`, `RulingByLowerCourt`, `RatioOfTheDecision`, `RulingByPresentCourt`, `None` |
| Primary metric | Macro-F1 (capstone §9.4) |
| Secondary metrics | Macro Precision, Macro Recall |

Submission schema (capstone §9.3.5, §6.3): one `prediction.json` at the root of a zip archive, mapping every official
test document ID of `CL_test.json` **and** `IT_test.json` to its ordered label list, with
`len(prediction[doc]) == len(test_document["text"])`:

```json
{
  "document_001": ["Fact", "Issue", "ArgumentPetitioner", "RulingByPresentCourt"],
  "...": ["..."]
}
```

The official scorer (`evaluate_phase3.py`) merges the CL and IT references, strips a trailing `.txt` from IDs,
requires the prediction ID set to equal the reference ID set and the per-document lengths to match, pools all
sentences of all documents, and computes sklearn macro F1 / precision / recall over the 13 labels with
`zero_division=0`.

Phase-3 rules restated in the notebook header (capstone §1, §9.2): ≥ 6 submissions; all four Phase-1 models used;
≥ 4 substantially different experiments, of which ≥ 3 use different techniques from the §9.2.1 list (LoRA, QLoRA,
AdaLoRA, rsLoRA, LoRMA, Prefix Tuning, Prompt Tuning, IA³, KD), LoRMA compulsory, ≥ 1 knowledge-distillation run.

## 2. Files in this folder

| File | Description |
|---|---|
| `phase3_rr_template.py` | Source of the notebook (percent format, `# %%` / `# %% [markdown]` cells, 31 cells). Placeholders `{{ARCH}}`, `{{TAG}}`, `{{FAMILY}}`, `{{MODEL}}`. |
| `build_p3_notebooks.py` | Fills the placeholders per architecture and writes `P3_arch{1..4}_<arch>.ipynb`; optional arch arguments restrict the output (`python build_p3_notebooks.py bart`). |
| `build_nb.py` | Generic percent-format `.py` → `.ipynb` converter used by the builder. |
| `P3_arch1_llama.ipynb` | Generated notebook, `ARCH = "llama"` (no outputs). |
| `P3_arch2_bart.ipynb` | Generated notebook, `ARCH = "bart"` (Phase-1 v2 checkpoint). |
| `P3_arch3_prefixlm.ipynb` | Generated notebook, `ARCH = "prefixlm"`. |
| `P3_arch4_mistral_swa.ipynb` | Generated notebook, `ARCH = "mistral_swa"`. |
| `run_logs/P3_arch1_llama_after.ipynb` | Executed on Colab (Tesla T4) with outputs: run `llama_lorma_window`. |
| `run_logs/P3_arch3_prefixlm_after.ipynb` | Executed on Colab (Tesla T4) with outputs: run `prefixlm_lorma_window`. |
| `run_logs/P3_arch4_mistral_swa_after.ipynb` | Executed on Colab (Tesla T4) with outputs: run `mistral_swa_lorma_window`. |

Running `python build_p3_notebooks.py` on the current template reproduces the four generated notebooks byte for byte.

## 3. Data and preprocessing

**Source.** `hf_hub_download` from the gated dataset repo `Exploration-Lab/CS-781-Capstone`, folder `Phase_3_RR/`:
`Train/CL_train.json`, `Train/IT_train.json`, `Train/CL_dev.json`, `Train/IT_dev.json`, `Train/label_vocab.json`,
`Test/CL_test.json`, `Test/IT_test.json`. Each file is read as JSON (JSON Lines as a fallback). `as_docs` accepts a
list of `{id, text, labels}` records, a dict `id -> record`, or a dict `id -> list of sentences` (the test files
are the last form: the run logs print `Test/CL_test.json: dict of 5`).

**Parsing checks.** A document ID may appear in only one of the six files; for train/dev documents
`len(labels) == len(text)` and every label must be in `label_vocab.json`. The vocabulary is read with the same rules
as the official scorer (list, or `label -> id` dict sorted by id, or `id -> label` dict) and must contain 13 unique
labels; the label index order used everywhere is the order printed by the logs (the order in §1 above).

**Splits.** No additional held-out split is built. Training uses the `<domain>_train.json` documents of
`TRAIN_DOMAINS` (default `["CL", "IT"]`, i.e. one model trained on both domains together, with no domain indicator in
the input). Model selection always uses `CL_dev` + `IT_dev` pooled; test predictions always cover `CL_test` + `IT_test`.

**Statistics printed by the run logs** (identical output in all three logs; quoted from
`run_logs/P3_arch1_llama_after.ipynb`, data cell):

```
domain split  docs sentences  empty sent/doc med
CL     train    40     10593      0          120
IT     train    40      6252      0          126
CL     dev       5      1396      0          184
IT     dev       5       746      0          120
CL     test      5      1339      0          114
IT     test      5       858      0          159
training on ['CL', 'IT']: 80 docs; dev 10 docs; test 10 docs
```

```
sentences per label (counts):
label                     CL train    CL dev  IT train    IT dev
Fact                          2979       249      1119       195
Issue                           17         3        90        12
ArgumentPetitioner             976       344       637        56
ArgumentRespondent             996       236       352        36
Statute                        367        58       732        62
Dissent                        284         0         0         0
PrecedentReliedUpon           1305       142       640       112
PrecedentNotReliedUpon         209         0       360        50
PrecedentOverruled               0         0         3         0
RulingByLowerCourt             438        56       436        46
RatioOfTheDecision            2856       288      1445       115
RulingByPresentCourt           113        18       194        31
None                            53         2       244        31
```

The scoring cell also prints the macro-F1 obtained when the dev gold labels are scored against themselves (with
`zero_division=0`, a label with no gold sentence scores 0):

```
attainable dev macro-F1 (gold as prediction): all 0.8462  CL 0.7692  IT 0.8462   labels absent from dev: ['Dissent', 'PrecedentOverruled']
```

**Tokenisation.** The Phase-1 tokenizer of the architecture (`PreTrainedTokenizerFast` from `<P1 repo>/best/`;
the Phase-1 llama log prints `vocab 32000 eos 0 pad 1`). Every sentence is encoded as `" " + sentence.strip()` with
`add_special_tokens=False` and truncated to its first `MAX_SENT_TOKENS` (128) tokens; a sentence whose encoding is
empty is replaced by the single EOS token so that it still receives a vector (the logs print 0 empty sentences in
every split). Printed by all three logs:

```
tokenised 21184 sentences in 2s; tokens/sentence: median 33  p90 73  p99 169  max 1186;  truncated at 128: 1.90%
```

**Sentence windows (`sent_windows`).** A window is a run of consecutive sentences `[a, b)` of one document. Starting
at `a`, sentences are appended while the window has fewer than `WIN_SENTS` (24) sentences and, for
`CONTEXT = "window"`, while the summed token count stays ≤ `WIN_TOKENS` (512); a window always holds at least one
sentence and a sentence is never split. The next window starts at `a + max(1, min(stride, b - a))` with
`stride = STRIDE_TRAIN` (12) for training and `STRIDE_PRED` (6) for dev, test and teacher-score prediction; the
last window of a document ends at its last sentence.

* `CONTEXT = "window"`: the token ids of the window's sentences are concatenated (no separator or special tokens) into
  one sequence; each sentence's tokens can attend to the other sentences of the window as allowed by the attention
  mask (§4).
* `CONTEXT = "none"`: every sentence is its own sequence; the window only groups up to `WIN_SENTS` sentences for the
  optional BiLSTM / CRF.

Every training window is one training example; with `STRIDE_TRAIN < WIN_SENTS` a sentence can occur in several
training windows. Printed by all three logs:

```
1550 training windows (sentences/window: median 12, max 24)  -> 194 steps/epoch;  coverage and alignment checks passed
```

**Batch layout (`collate`).** `ids`/`att` `(Q, L)` (one row per window, or per sentence for `CONTEXT="none"`),
`pool` `(Q, K, L)` token weights of the K sentences of each row, `take` `(B, S)` index of each window sentence in the
flattened `(Q·K)` sentence vectors (`-1` = padding), `y` `(B, S)` label ids (`-100` = padding / unlabelled), plus
`tsc`/`tmask` teacher scores for distillation.

**Checks before training.** (1) Every train/dev/test sentence is covered by at least one window and windows respect
`WIN_TOKENS`. (2) Alignment: for 20 random training batches and one dev batch, the token *ids* are pooled with the
same `pool`/`take` tensors and compared with each sentence's own token ids (mean / first / last), and the batch
labels are compared with the document labels.

**Mapping back to the sentence lists.** Prediction accumulates each window's per-sentence scores into a per-document
array with one row per input sentence (row index = sentence position), divides by the number of windows that
contained the sentence, and decodes one label per row. The output list therefore has the length and order of the
document's `text` list; this is asserted before `prediction.json` is written (§7).

## 4. Model architecture

Phase-1 backbones (sizes from the Phase-1 configuration cell, `code/phase1/phase1_arch_template.py`; the same values
appear in the Phase-1 run logs): d_model 768, 12 blocks,
12 heads, FFN 2048, 1024 positions for `llama` / `prefixlm` / `mistral_swa`; `mistral_swa` additionally 4 key/value
heads (GQA) and sliding window 256; `bart` 6 encoder + 6 decoder blocks, FFN 3072, 1024 positions. Weights are loaded
in float32 from `<hf-user>/cs781-p1-<arch>/best/` (`bart`: `cs781-p1-bart-v2`); the LM head is not used.

| `ARCH` | Loaded class → module used | Default attention (`ATTN` / `BART_MODE`) | Printed size of the RR model |
|---|---|---|---|
| `llama` | `LlamaForCausalLM` → `.model` (`LlamaModel`) | `causal`: 2-D padding mask passed, the model applies its causal mask | `109.5M parameters` |
| `prefixlm` | `LlamaForCausalLM` → `.model` | `bidirectional`: 4-D boolean mask `(B,1,L,L)`, every non-pad key visible to every query | `109.5M parameters` |
| `mistral_swa` | `MistralForCausalLM` → `.model` (`MistralModel`) | `causal`: the model's causal sliding-window mask; the `bidirectional` option uses the band `abs(i − j) < sliding_window` | `100.1M parameters` |
| `bart` | `BartForConditionalGeneration` → `.model.encoder` | `encoder`: encoder states; `decoder` option: the same ids go to encoder and decoder (`decoder_input_ids = input_ids`), decoder states used | no run log — TODO(student) |

`ATTN` can be switched to `"bidirectional"` for `llama` / `mistral_swa` and to `"causal"` for `prefixlm`.

**Context sanity check** (cell "Sanity checks"): changing the first token of the last sentence of a two-sentence window
and measuring the change of the first sentence's vector; also checks that extra padding in the batch leaves the
vectors unchanged. Printed:

```
llama (CONTEXT=window, attn=causal, bart_mode=encoder): changing the last sentence moves the first sentence's vector by 0.00e+00, its own by 3.04e-01 -> first sentence sees right context: False (expected False); padding-invariant: True
prefixlm (CONTEXT=window, attn=bidirectional, bart_mode=encoder): changing the last sentence moves the first sentence's vector by 6.95e-01, its own by 2.19e-01 -> first sentence sees right context: True (expected True); padding-invariant: True
mistral_swa (CONTEXT=window, attn=causal, bart_mode=encoder): changing the last sentence moves the first sentence's vector by 0.00e+00, its own by 2.24e-01 -> first sentence sees right context: False (expected False); padding-invariant: True
```

**Sentence representation.** With `h_i ∈ R^768` the backbone's `last_hidden_state` (soft-prompt positions removed),
the vector of sentence *k* is `v_k = Σ_i P[k,i] · h_i`, where `P[k,i] = 1/|s_k|` over the tokens of sentence *k*
(`POOL="mean"`, default), or a one-hot on its first (`"first"`) or last (`"last"`) token.

**Sentence-level context (optional).** `SENT_LSTM > 0` adds a bidirectional `nn.LSTM(768 → SENT_LSTM)` over the
sentence vectors of each window (packed by window length); its output has `2·SENT_LSTM` features.

**Head.** `z_k = W · Dropout(v_k) + b`, `W ∈ R^{13×d}` (`d` = 768, or `2·SENT_LSTM`), dropout rate `DROPOUT`.

* `HEAD = "linear"`: per-sentence softmax; training loss = cross-entropy over all non-padding sentences of the batch,
  optionally weighted per label (`CLASS_WEIGHTS`: `inverse` `w_c = N / (13 · n_c)`, `inverse_sqrt` = its square
  root, `w_c = 1` for labels with `n_c = 0`; `n_c` = training-sentence count).
* `HEAD = "crf"`: linear-chain CRF (Lafferty et al., 2001) with learned start (13), end (13) and transition (13×13)
  scores, zero-initialised; the head outputs are emission scores. Loss = negative log-likelihood of the gold label
  sequence of each window (forward algorithm), summed over windows and divided by the number of sentences
  (`CLASS_WEIGHTS` is not applied). Prediction runs Viterbi (float64) over the window-averaged emissions of the whole
  document.

## 5. Fine-tuning methods implemented

All adapters are implemented in this notebook (plain PyTorch, no `peft`). The head (and CRF / BiLSTM when enabled) is
trained in every method. For `lora`, `rslora`, `lorma`, `ia3` the notebook asserts that the injected modules leave the
model output unchanged at initialisation (max |Δ| < 1e-3) and, after training, merges them into plain `nn.Linear`
weights and asserts that the scores change by < 1e-2 · max(1, max |score|).

| `METHOD` | Trainable parameters | Implementation | Knobs (default) | Default `LR` |
|---|---|---|---|---|
| `lora` | `A ∈ R^{r×d_in}`, `B ∈ R^{d_out×r}` in each `nn.Linear` named in `ADAPT_TARGETS` | `LoRALinear`: `h = W0·x + (α/r)·B(A·x)`; `A` Kaiming-uniform (a=√5), `B = 0`; merge `W = W0 + (α/r)·B·A` | `ADAPT_TARGETS` (`q_proj`, `v_proj`), `RANK` 8, `ALPHA` 16 | 5e-4 |
| `rslora` | as `lora` | `LoRALinear(rank_stabilised=True)`: scale `α/√r` | as `lora` | 5e-4 |
| `lorma` | `A ∈ R^{r×d_out}`, `B ∈ R^{d_out×r}` per targeted layer | `LoRMALinear` (Bihany, Patel & Modi, 2025), multiplicative `h = 𝓘(BA)·W0·x (+ bias)`. `plus`: `𝓘(BA) = I + (α/r)·BA`, computed as `W0x + (α/r)·B(A(W0x))`, `A` Kaiming-uniform, `B = 0`; merge `W = W0 + (α/r)·B(A·W0)`. `pi`: `𝓘(BA)[i, j] = (BA)[i, (j−i) mod d]` (row *i* rotated by *i*), scale 1, init `B` Kaiming-uniform with `B[:,0] = 1`, `A = e₀e₀ᵀ` (gives `𝓘 = I`); merge `W = 𝓘(BA)·W0`. The bias (BART only) is added after the transform | `ADAPT_TARGETS`, `RANK` 8, `ALPHA` 16, `LORMA_VARIANT` `"plus"` | 5e-4 |
| `ia3` | one vector per targeted layer, initialised to 1 | `IA3Linear` (Liu et al., 2022): outputs rescaled `l ⊙ (W·x + b)` for `IA3_OUT`; inputs rescaled `W·(l ⊙ x) + b` for `IA3_IN`; merged into `W` (and `b`) | `IA3_OUT` (`k_proj`, `v_proj`), `IA3_IN` (`down_proj` for llama/mistral, `fc2` for bart) | 3e-3 |
| `prompt` | `PROMPT_LEN × d_model` soft-prompt matrix; backbone frozen | Prompt tuning (Lester et al., 2021): initialised from the input embeddings of `PROMPT_LEN` random token ids in `[2, VOCAB)` (generator seeded with `SEED`), prepended via `inputs_embeds`, attention mask extended with ones, prompt positions dropped before pooling; not merged (adds `PROMPT_LEN` positions at inference); `bart` only with `BART_MODE="encoder"` | `PROMPT_LEN` 20 | 3e-3 |
| `kd` | every parameter of a reduced-depth student | Student = this notebook's `ARCH` with a subset of the Phase-1 blocks (`STUDENT_LAYERS = None` → blocks 0, 2, 4, … of every stack; `layer_idx` and the config's layer count updated). Teacher = `train_scores.json` of the finished run `TEACHER_RUN` in `<hf-user>/cs781-p3` (any architecture; its label order is asserted equal and the score shape `(n_sentences, 13)` per document is asserted). Loss: `KD_ALPHA · L_task + (1 − KD_ALPHA) · T² · KL(softmax(t/T) ‖ softmax(z/T))` over sentences that have teacher scores (`F.kl_div`, `batchmean`); documents without teacher scores get the task loss only | `TEACHER_RUN` `"llama_full_window"`, `STUDENT_LAYERS` None, `KD_T` 2.0, `KD_ALPHA` 0.5 | 1e-4 |
| `full` | every backbone parameter | Full fine-tuning; marked in the notebook as a reference point, not a §9.2.1 technique | — | 5e-5 |

Teacher scores (`train_scores.json`, written by every run): per training sentence, `log(max(p, 1e-8))` of the
window-averaged softmax for a linear-head run, or the window-averaged CRF emissions for a CRF run, rounded to 4
decimals; computed with the prediction windows (`STRIDE_PRED`). The `kd` cell prints the teacher's arch / method /
context, parameter count, dev macro-F1 and dev inference time next to the student's parameter count.

§9.2.1 techniques **not** implemented in this notebook: QLoRA, AdaLoRA, Prefix Tuning.

Parameter counts printed by the run logs (all three runs `METHOD = "lorma"`, `LORMA_VARIANT = "plus"`, r = 8):

```
llama:       lorma: adapted layers {'q_proj': 12, 'v_proj': 12}; output change at init 0.00e+00
             METHOD=lorma: trainable 0.305M of 109.8M (0.28%)
prefixlm:    lorma: adapted layers {'q_proj': 12, 'v_proj': 12}; output change at init 0.00e+00
             METHOD=lorma: trainable 0.305M of 109.8M (0.28%)
mistral_swa: lorma: adapted layers {'q_proj': 12, 'v_proj': 12}; output change at init 0.00e+00
             METHOD=lorma: trainable 0.207M of 100.3M (0.21%)
```

## 6. Training procedure

* **Optimiser:** `torch.optim.AdamW`, betas (0.9, 0.999), `fused=True` on CUDA. Three parameter groups: head / CRF /
  BiLSTM (`HEAD_LR` 1e-3, weight decay `WEIGHT_DECAY` 0.01); other trainable tensors with ≥ 2 dims (`LR`, weight
  decay 0.01); other trainable tensors with < 2 dims, e.g. IA³ vectors and norms (`LR`, weight decay 0).
* **Learning-rate schedule:** per optimiser step *s*, `lr = base_lr · m(s)` with linear warm-up
  `m(s) = (s+1)/W` for `s < W` and linear decay `m(s) = max(0, (T−s)/(T−W))` afterwards, where
  `T = EPOCHS · steps_per_epoch` and `W = max(1, int(WARMUP_FRAC · T))`. With the logged 194 steps/epoch and
  `EPOCHS = 10`: `T = 1940`, `W = 116`. Early stopping ends training before `T` is reached.
* **Batching:** `BATCH` = 8 windows per optimiser step, no gradient accumulation; training windows shuffled each
  epoch with `np.random.permutation` (seeds: `random`, `numpy`, `torch` set to `SEED` = 42).
* **Precision:** float32 weights, forward pass under `torch.autocast(float16)`, `GradScaler`; loss computed in
  float32. Gradient clipping at `GRAD_CLIP` = 1.0 (global norm over trainable parameters).
* **Model selection:** after every epoch the dev documents are predicted (§7) and scored; the state with the highest
  pooled `CL_dev + IT_dev` macro-F1 (strictly greater) is copied to CPU memory. Per-domain dev macro-F1 is printed
  alongside.
* **Stopping:** at most `EPOCHS` (10) epochs; stop after `PATIENCE` (3) consecutive epochs without a new best; stop
  when the wall-clock time since the start of training (dev evaluations included) exceeds `MAX_TRAIN_MINUTES` (60) —
  checked after every step, followed by a final dev evaluation of the partial epoch.
* **Reported times / memory:** `train_minutes` = sum of the training-loop time of each epoch (dev evaluation
  excluded); the per-epoch seconds printed after each epoch include the dev evaluation. `peak_gpu_gb` =
  `torch.cuda.max_memory_allocated()` since the start of training (training and per-epoch dev evaluation).
* **Checkpoint / resume:** none. The best state lives in memory; nothing is uploaded until the save cell at the end
  of the notebook, so an interrupted run leaves no files in `cs781-p3`.

## 7. Inference and submission

1. The best state is reloaded and adapters are merged (`adapter merge: max |score change| = …` is printed).
2. `predict_scores` builds windows with `STRIDE_PRED` = 6, batches `2·BATCH` = 16 windows, runs under float16
   autocast, and averages each sentence's scores over all windows containing it — softmax probabilities for the
   linear head (decision: `argmax`), raw emissions for the CRF (decision: Viterbi over the whole document). No
   thresholds or label-prior adjustments are applied.
3. **Dev report:** pooled and per-domain macro P / R / F1, a per-label table (P, R, F1, gold count, predicted count,
   CL F1, IT F1), the 13×13 confusion counts and the ten most frequent gold→predicted confusions. The dev gold and
   predictions are written in the official format (`dev_truth_CL.json`, `dev_truth_IT.json`, `dev_pred.json`,
   `labels.json`) and the matching `evaluate_phase3.py` command is printed. The notebook's `rr_scores` uses the same
   computation as the official scorer (all sentences pooled, sklearn macro averages over the 13 labels,
   `zero_division=0`).
4. **Test:** `prediction = {doc_id: [label, …]}` for all test IDs, checked with the scorer's rules (IDs unique after
   `.txt` normalisation, ID set equal to the test IDs, one label per sentence, every label in the vocabulary). The
   predicted label counts per domain are printed. `prediction.json` is written and zipped as
   `submission_<RUN_NAME>.zip` with `prediction.json` as its only member at the archive root (asserted).
5. **Save cell:** writes `train_scores.json` (§5), `run_config.json` (run name, arch, method, context, head, attention,
   Phase-1 repo, kept blocks, labels, training IDs, all config values, parameter counts, train minutes, peak memory,
   best epoch, dev/test/train prediction seconds, dev pooled/per-domain scores, per-label scores, confusion matrix,
   attainable F1, teacher summary, epoch history), `curves.png` (train loss, dev macro-F1 per epoch, row-normalised
   dev confusion) and, if `SAVE_WEIGHTS`, `model.pt`; uploads the run folder to `<hf-user>/cs781-p3` at
   `runs/<RUN_NAME>/` (3 attempts) and triggers a browser download of the zip on Colab.
6. **Last cell:** lists every run in `cs781-p3` (arch, method, context, head, LSTM size, trainable parameters, dev
   F1 pooled / CL / IT, train minutes) and counts which Phase-1 models and which §9.2.1 techniques have runs.

Local evaluation with the course scorer, from the files of a run folder:

```bash
python evaluate_phase3.py --truth-cl dev_truth_CL.json --truth-it dev_truth_IT.json \
                          --prediction dev_pred.json --labels labels.json
```

## 8. Configuration knobs

First code cell and the cell under "Configuration — all values below are YOUR CHOICE":

| Name | Template default | Meaning |
|---|---|---|
| `ARCH` | set by the builder | Phase-1 architecture of this notebook (`llama` / `bart` / `prefixlm` / `mistral_swa`). |
| `METHOD` | `"lorma"` | `lora`, `rslora`, `lorma`, `ia3`, `prompt`, `kd`, `full` (§5). |
| `CONTEXT` | `"window"` | `none`: each sentence encoded alone; `window`: consecutive sentences share one sequence. |
| `HEAD` | `"linear"` | `linear` (per-sentence softmax) or `crf` (linear-chain CRF, Viterbi per document). |
| `SENT_LSTM` | `0` | 0 = off; else hidden size of a BiLSTM over sentence vectors before the head. |
| `RUN_NAME` | `f"{ARCH}_{METHOD}_{CONTEXT}"` + `"_crf"` if CRF + `"_lstm<N>"` if LSTM | Run folder / submission name. |
| `DATASET` | `"Exploration-Lab/CS-781-Capstone"` | HF dataset repo (folder `Phase_3_RR/`). |
| `P1_REPOS` | `<hf-user>/cs781-p1-{llama, bart-v2, prefixlm, mistral_swa}` | Phase-1 checkpoint repo per arch (`best/`). |
| `P3_REPO` | `<hf-user>/cs781-p3` | Private repo receiving `runs/<RUN_NAME>/`. |
| `WORK`, `RUN_DIR` | `/content/p3` (else `/tmp/p3`), `<WORK>/<RUN_NAME>` | Local output folder. |
| `SEED` | `42` | Seed for `random`, `numpy`, `torch`, prompt initialisation. |
| `TRAIN_DOMAINS` | `["CL", "IT"]` | Domains whose `_train.json` documents are trained on. |
| `MAX_SENT_TOKENS` | `128` | Per-sentence token cap (first tokens kept). |
| `WIN_SENTS` | `24` | Max sentences per window. |
| `WIN_TOKENS` | `512` | Token budget of a window (`CONTEXT="window"`). |
| `STRIDE_TRAIN` | `12` | Sentences between window starts, training. |
| `STRIDE_PRED` | `6` | Sentences between window starts, prediction (scores averaged over windows). |
| `POOL` | `"mean"` | Token states → sentence vector: `mean` / `first` / `last`. |
| `ATTN` | llama `causal`, prefixlm `bidirectional`, mistral_swa `causal`, bart `n/a` | Attention mask of decoder-only backbones. |
| `BART_MODE` | `"encoder"` | bart: encoder states, or `decoder` (input to encoder and decoder, decoder states). |
| `EPOCHS` | `10` | Max epochs. |
| `PATIENCE` | `3` | Epochs without dev macro-F1 improvement before stopping. |
| `MAX_TRAIN_MINUTES` | `60` | Training time cap. |
| `BATCH` | `8` | Windows per optimiser step. |
| `LR` | full 5e-5, lora 5e-4, rslora 5e-4, lorma 5e-4, ia3 3e-3, prompt 3e-3, kd 1e-4 | Learning rate of backbone / adapter / prompt parameters. |
| `HEAD_LR` | `1e-3` | Learning rate of head, CRF, BiLSTM. |
| `WEIGHT_DECAY` | `0.01` | AdamW weight decay (0 for < 2-D non-head tensors). |
| `WARMUP_FRAC` | `0.06` | Fraction of `EPOCHS·steps_per_epoch` used for linear warm-up. |
| `GRAD_CLIP` | `1.0` | Gradient-norm clip. |
| `DROPOUT` | `0.1` | Dropout on sentence vectors before the head. |
| `CLASS_WEIGHTS` | `"none"` | `none` / `inverse` / `inverse_sqrt` per-label CE weights (linear head only). |
| `ADAPT_TARGETS` | `["q_proj", "v_proj"]` | `nn.Linear` names receiving LoRA / rsLoRA / LoRMA. |
| `RANK`, `ALPHA` | `8`, `16` | Adapter rank and scale numerator. |
| `LORMA_VARIANT` | `"plus"` | `plus` (additive-identity) or `pi` (permutation-based) rank inflation. |
| `IA3_OUT` | `["k_proj", "v_proj"]` | Layers whose outputs IA³ rescales. |
| `IA3_IN` | `["down_proj", "fc2"]` | Layers whose inputs IA³ rescales. |
| `PROMPT_LEN` | `20` | Soft-prompt length. |
| `TEACHER_RUN` | `"llama_full_window"` | Finished run in `cs781-p3` whose `train_scores.json` is the KD target. |
| `STUDENT_LAYERS` | `None` | KD student blocks (`None` = every 2nd block per stack, or an explicit index list). |
| `KD_T` | `2.0` | Distillation temperature. |
| `KD_ALPHA` | `0.5` | Weight of the task loss (1 − `KD_ALPHA` on the KL term). |
| `SAVE_WEIGHTS` | `False` | Also save the merged `model.pt` into the run folder / Hub. |

## 9. Runs recorded in `run_logs/`

All three notebooks were executed on Colab with a Tesla T4 (`2.11.0+cu130 Tesla T4`, `transformers 5.18.0`). Their
first code cell and configuration cell carry the template defaults (`METHOD="lorma"`, `CONTEXT="window"`,
`HEAD="linear"`, `SENT_LSTM=0`, all other knobs as in §8). The only source difference from the current generated
notebooks: the logged configuration cell lacks the `finished_runs()` / `TEACHER_RUN` pre-check, which was added to the
template afterwards (it only acts when `METHOD == "kd"`).

**Configuration of each logged run**

| Run log | `RUN_NAME` | Arch | Method | Attention | Config values differing from template defaults |
|---|---|---|---|---|---|
| `P3_arch1_llama_after.ipynb` | `llama_lorma_window` | llama | lorma (plus, r 8, α 16, `q_proj`/`v_proj`), LR 5e-4, linear head, window context | causal | none |
| `P3_arch3_prefixlm_after.ipynb` | `prefixlm_lorma_window` | prefixlm | as above | bidirectional | none |
| `P3_arch4_mistral_swa_after.ipynb` | `mistral_swa_lorma_window` | mistral_swa | as above | causal (sliding window) | none |

**Numbers printed at the end of each run** (final values are those printed after reloading the best epoch and merging
the adapters; values copied from the logs)

| `RUN_NAME` | Epochs run / best epoch | Best-epoch dev macro-F1 (training loop) | Final dev macro-F1 (pooled) | Dev macro-P | Dev macro-R | CL dev F1 | IT dev F1 | Trainable / total params | `params_final` (merged) | Train minutes | Peak GPU (GB) | Dev / test predict time |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `llama_lorma_window` | 7 / 4 | 0.4027 | 0.40272073945397496 | 0.4527590872286159 | 0.3874604977971971 | 0.29106308902066796 | 0.42364544131911397 | 304,909 / 109,834,765 (0.28%) | 109,539,853 | 5.326108761628469 | 2.140122624 | 3.9 s / 4.0 s |
| `prefixlm_lorma_window` | 9 / 6 | 0.4035 | 0.4032943149413008 | 0.4588272965256405 | 0.3829044006305715 | 0.342916440206718 | 0.40825324756743975 | 304,909 / 109,834,765 (0.28%) | 109,539,853 | 6.532926881313324 | 2.142219776 | 3.6 s / 3.7 s |
| `mistral_swa_lorma_window` | 5 / 2 | 0.4072 | 0.4072761198463404 | 0.4346589971657316 | 0.4047083710909852 | 0.3207913032278321 | 0.43968491434431284 | 206,605 / 100,299,277 (0.21%) | 100,102,669 | 3.281213653087616 | 2.03520768 | 3.3 s / 3.4 s |

Dev sentences scored: 2142 pooled (CL 1396, IT 746). Test: `10 documents, 2197 sentences` in every log.
Adapter-merge checks printed: llama `6.44e-06`, prefixlm `1.24e-05`, mistral_swa `3.22e-06`.

**Per-epoch training output** (verbatim):

`llama_lorma_window` (`run_logs/P3_arch1_llama_after.ipynb`; preceded by `class weights:` 1.0 for all 13 labels):

```
epoch 1: loss 1.6971  dev macro-F1 0.3577 (P 0.4465 R 0.3431)  CL 0.2278  IT 0.4033  (45s)
epoch 2: loss 1.1282  dev macro-F1 0.3960 (P 0.4442 R 0.3998)  CL 0.2959  IT 0.4381  (50s)
epoch 3: loss 0.8974  dev macro-F1 0.3873 (P 0.4323 R 0.3772)  CL 0.2996  IT 0.4229  (51s)
epoch 4: loss 0.6796  dev macro-F1 0.4027 (P 0.4529 R 0.3875)  CL 0.2911  IT 0.4235  (50s)
epoch 5: loss 0.5008  dev macro-F1 0.3914 (P 0.4189 R 0.3874)  CL 0.3118  IT 0.4027  (52s)
epoch 6: loss 0.3595  dev macro-F1 0.3900 (P 0.4345 R 0.3751)  CL 0.3122  IT 0.3990  (50s)
epoch 7: loss 0.2507  dev macro-F1 0.3790 (P 0.4219 R 0.3687)  CL 0.2693  IT 0.3978  (50s)
stopping: no improvement for 3 epochs
best epoch 4: dev macro-F1 0.4027;  train time 5.3 min;  peak GPU memory 2.14 GB
```

`prefixlm_lorma_window` (`run_logs/P3_arch3_prefixlm_after.ipynb`; preceded by `class weights:` 1.0 for all 13 labels):

```
epoch 1: loss 1.6862  dev macro-F1 0.3251 (P 0.4072 R 0.3233)  CL 0.2363  IT 0.3490  (45s)
epoch 2: loss 1.0799  dev macro-F1 0.3936 (P 0.4298 R 0.3921)  CL 0.3286  IT 0.3962  (48s)
epoch 3: loss 0.8250  dev macro-F1 0.3871 (P 0.4422 R 0.3798)  CL 0.3122  IT 0.3945  (48s)
epoch 4: loss 0.6149  dev macro-F1 0.4002 (P 0.4498 R 0.3876)  CL 0.3323  IT 0.4023  (48s)
epoch 5: loss 0.4578  dev macro-F1 0.4004 (P 0.4457 R 0.3876)  CL 0.3335  IT 0.4003  (48s)
epoch 6: loss 0.3384  dev macro-F1 0.4035 (P 0.4590 R 0.3832)  CL 0.3434  IT 0.4083  (47s)
epoch 7: loss 0.2522  dev macro-F1 0.3959 (P 0.4412 R 0.3832)  CL 0.3345  IT 0.3954  (48s)
epoch 8: loss 0.1868  dev macro-F1 0.3892 (P 0.4269 R 0.3812)  CL 0.3346  IT 0.3953  (48s)
epoch 9: loss 0.1414  dev macro-F1 0.3850 (P 0.4284 R 0.3748)  CL 0.3305  IT 0.3852  (47s)
stopping: no improvement for 3 epochs
best epoch 6: dev macro-F1 0.4035;  train time 6.5 min;  peak GPU memory 2.14 GB
```

`mistral_swa_lorma_window` (`run_logs/P3_arch4_mistral_swa_after.ipynb`; preceded by `class weights:` 1.0 for all 13 labels):

```
epoch 1: loss 1.7170  dev macro-F1 0.3665 (P 0.4572 R 0.3513)  CL 0.2341  IT 0.4026  (44s)
epoch 2: loss 1.1572  dev macro-F1 0.4072 (P 0.4346 R 0.4044)  CL 0.3208  IT 0.4389  (41s)
epoch 3: loss 0.9575  dev macro-F1 0.3796 (P 0.4447 R 0.3704)  CL 0.2623  IT 0.4009  (43s)
epoch 4: loss 0.7731  dev macro-F1 0.3802 (P 0.4824 R 0.3620)  CL 0.2612  IT 0.4143  (43s)
epoch 5: loss 0.6268  dev macro-F1 0.3994 (P 0.4689 R 0.3782)  CL 0.2883  IT 0.4039  (43s)
stopping: no improvement for 3 epochs
best epoch 2: dev macro-F1 0.4072;  train time 3.3 min;  peak GPU memory 2.04 GB
```

<details>
<summary>Per-label dev tables, confusion counts and test label counts (verbatim from the logs)</summary>

**`llama_lorma_window`** (`run_logs/P3_arch1_llama_after.ipynb`)

```
adapter merge: max |score change| = 6.44e-06
DEV macro-F1 0.4027  P 0.4528  R 0.3875   CL F1 0.2911  IT F1 0.4236   (3.9s)

label                          P       R      F1   gold   pred    CL F1    IT F1
Fact                      0.7233  0.5946  0.6527    444    365   0.5826   0.7450
Issue                     0.7778  0.4667  0.5833     15      9   0.0000   0.7000
ArgumentPetitioner        0.6061  0.2500  0.3540    400    165   0.3529   0.3596
ArgumentRespondent        0.4621  0.4485  0.4552    272    264   0.4684   0.3548
Statute                   0.7946  0.7417  0.7672    120    112   0.7731   0.7611
Dissent                   0.0000  0.0000  0.0000      0      3   0.0000   0.0000
PrecedentReliedUpon       0.4667  0.5512  0.5054    254    300   0.4669   0.5468
PrecedentNotReliedUpon    0.4333  0.2600  0.3250     50     30   0.0000   0.3291
PrecedentOverruled        0.0000  0.0000  0.0000      0      0   0.0000   0.0000
RulingByLowerCourt        0.3377  0.2549  0.2905    102     77   0.1266   0.4200
RatioOfTheDecision        0.4382  0.8189  0.5709    403    753   0.5967   0.5031
RulingByPresentCourt      0.4651  0.4082  0.4348     49     43   0.4167   0.4545
None                      0.3810  0.2424  0.2963     33     21   0.0000   0.3333

confusion counts (rows gold, columns predicted):
           F     I    AP    AR     S     D   PRU  PNRU    PO  RBLC  ROTD  RBPC     N
     F   264     0    26     6    10     3     7     0     0    22    94     9     3
     I     1     7     0     1     0     0     2     0     0     0     3     0     1
    AP    21     1   100   124     5     0    29     8     0    11   100     0     1
    AR     9     0    30   122     0     0    35     4     0     0    72     0     0
     S     1     0     0     0    89     0     8     0     0     0    21     0     1
     D     0     0     0     0     0     0     0     0     0     0     0     0     0
   PRU    39     0     3     1     0     0   140     2     0    12    57     0     0
  PNRU     6     0     0     4     0     0    15    13     0     2     7     2     1
    PO     0     0     0     0     0     0     0     0     0     0     0     0     0
  RBLC    13     0     1     0     0     0     6     1     0    26    46     9     0
  ROTD     8     1     4     5     7     0    39     1     0     4   330     2     2
  RBPC     0     0     0     0     1     0     2     1     0     0    21    20     4
     N     3     0     1     1     0     0    17     0     0     0     2     1     8
most frequent confusions (gold -> predicted): [('ArgumentPetitioner', 'ArgumentRespondent', 124), ('ArgumentPetitioner', 'RatioOfTheDecision', 100), ('Fact', 'RatioOfTheDecision', 94), ('ArgumentRespondent', 'RatioOfTheDecision', 72), ('PrecedentReliedUpon', 'RatioOfTheDecision', 57), ('RulingByLowerCourt', 'RatioOfTheDecision', 46), ('RatioOfTheDecision', 'PrecedentReliedUpon', 39), ('PrecedentReliedUpon', 'Fact', 39), ('ArgumentRespondent', 'PrecedentReliedUpon', 35), ('ArgumentRespondent', 'ArgumentPetitioner', 30)]

official check: python evaluate_phase3.py --truth-cl dev_truth_CL.json --truth-it dev_truth_IT.json --prediction dev_pred.json --labels labels.json   (files in /content/p3/llama_lorma_window)

test: 10 documents, 2197 sentences  (4.0s)
  CL predicted labels: {'Fact': 228, 'RulingByLowerCourt': 19, 'ArgumentPetitioner': 113, 'RatioOfTheDecision': 577, 'Statute': 71, 'RulingByPresentCourt': 21, 'PrecedentReliedUpon': 220, 'Issue': 2, 'PrecedentNotReliedUpon': 2, 'None': 7, 'ArgumentRespondent': 79}
  IT predicted labels: {'None': 20, 'Issue': 6, 'Fact': 124, 'RulingByLowerCourt': 75, 'ArgumentPetitioner': 44, 'Statute': 103, 'RatioOfTheDecision': 216, 'RulingByPresentCourt': 24, 'ArgumentRespondent': 56, 'PrecedentReliedUpon': 185, 'PrecedentNotReliedUpon': 5}
wrote /content/p3/llama_lorma_window/submission_llama_lorma_window.zip
```

**`prefixlm_lorma_window`** (`run_logs/P3_arch3_prefixlm_after.ipynb`)

```
adapter merge: max |score change| = 1.24e-05
DEV macro-F1 0.4033  P 0.4588  R 0.3829   CL F1 0.3429  IT F1 0.4083   (3.6s)

label                          P       R      F1   gold   pred    CL F1    IT F1
Fact                      0.5907  0.7185  0.6484    444    540   0.5709   0.7438
Issue                     0.7500  0.4000  0.5217     15      8   0.5000   0.5263
ArgumentPetitioner        0.5478  0.4875  0.5159    400    356   0.5077   0.5636
ArgumentRespondent        0.5489  0.2684  0.3605    272    133   0.3778   0.2222
Statute                   0.7949  0.7750  0.7848    120    117   0.8000   0.7679
Dissent                   0.0000  0.0000  0.0000      0      0   0.0000   0.0000
PrecedentReliedUpon       0.4299  0.5315  0.4754    254    314   0.4373   0.5214
PrecedentNotReliedUpon    0.4130  0.3800  0.3958     50     46   0.0000   0.4471
PrecedentOverruled        0.0000  0.0000  0.0000      0      0   0.0000   0.0000
RulingByLowerCourt        0.2381  0.1471  0.1818    102     63   0.1136   0.2597
RatioOfTheDecision        0.5192  0.6700  0.5850    403    520   0.6242   0.4867
RulingByPresentCourt      0.5938  0.3878  0.4691     49     32   0.5263   0.4186
None                      0.5385  0.2121  0.3043     33     13   0.0000   0.3500

confusion counts (rows gold, columns predicted):
           F     I    AP    AR     S     D   PRU  PNRU    PO  RBLC  ROTD  RBPC     N
     F   319     0    34     1     5     0    18     0     0     4    56     5     2
     I     4     6     0     0     0     0     1     0     0     1     2     0     1
    AP    58     0   195    52     5     0    20     8     0    18    44     0     0
    AR    25     0   103    73     1     0    33    12     0     6    19     0     0
     S     1     0     0     0    93     0     8     1     0     2    15     0     0
     D     0     0     0     0     0     0     0     0     0     0     0     0     0
   PRU    59     0     4     2     1     0   135     3     0     1    49     0     0
  PNRU     8     1     2     0     0     0    12    19     0     3     3     2     0
    PO     0     0     0     0     0     0     0     0     0     0     0     0     0
  RBLC    36     0     2     0     2     0     6     1     0    15    35     3     2
  ROTD    25     0    15     4     7     0    70     1     0     9   270     2     0
  RBPC     2     0     0     0     3     0     1     0     0     3    20    19     1
     N     3     1     1     1     0     0    10     1     0     1     7     1     7
most frequent confusions (gold -> predicted): [('ArgumentRespondent', 'ArgumentPetitioner', 103), ('RatioOfTheDecision', 'PrecedentReliedUpon', 70), ('PrecedentReliedUpon', 'Fact', 59), ('ArgumentPetitioner', 'Fact', 58), ('Fact', 'RatioOfTheDecision', 56), ('ArgumentPetitioner', 'ArgumentRespondent', 52), ('PrecedentReliedUpon', 'RatioOfTheDecision', 49), ('ArgumentPetitioner', 'RatioOfTheDecision', 44), ('RulingByLowerCourt', 'Fact', 36), ('RulingByLowerCourt', 'RatioOfTheDecision', 35)]

official check: python evaluate_phase3.py --truth-cl dev_truth_CL.json --truth-it dev_truth_IT.json --prediction dev_pred.json --labels labels.json   (files in /content/p3/prefixlm_lorma_window)

test: 10 documents, 2197 sentences  (3.7s)
  CL predicted labels: {'Fact': 244, 'ArgumentPetitioner': 211, 'RulingByLowerCourt': 22, 'Statute': 64, 'PrecedentReliedUpon': 280, 'RatioOfTheDecision': 430, 'RulingByPresentCourt': 21, 'Dissent': 14, 'PrecedentNotReliedUpon': 18, 'ArgumentRespondent': 24, 'Issue': 3, 'None': 8}
  IT predicted labels: {'Fact': 180, 'ArgumentPetitioner': 86, 'RulingByLowerCourt': 41, 'Statute': 93, 'Issue': 8, 'RatioOfTheDecision': 214, 'RulingByPresentCourt': 17, 'None': 14, 'PrecedentReliedUpon': 193, 'ArgumentRespondent': 7, 'PrecedentNotReliedUpon': 5}
wrote /content/p3/prefixlm_lorma_window/submission_prefixlm_lorma_window.zip
```

**`mistral_swa_lorma_window`** (`run_logs/P3_arch4_mistral_swa_after.ipynb`)

```
adapter merge: max |score change| = 3.22e-06
DEV macro-F1 0.4073  P 0.4347  R 0.4047   CL F1 0.3208  IT F1 0.4397   (3.3s)

label                          P       R      F1   gold   pred    CL F1    IT F1
Fact                      0.5632  0.7523  0.6442    444    593   0.5710   0.7500
Issue                     0.7500  0.6000  0.6667     15     12   0.5000   0.6957
ArgumentPetitioner        0.5388  0.2950  0.3813    400    219   0.3872   0.3448
ArgumentRespondent        0.4337  0.3971  0.4146    272    249   0.4089   0.4507
Statute                   0.6855  0.9083  0.7814    120    159   0.7801   0.7826
Dissent                   0.0000  0.0000  0.0000      0      0   0.0000   0.0000
PrecedentReliedUpon       0.4857  0.4685  0.4770    254    245   0.3600   0.5944
PrecedentNotReliedUpon    0.4000  0.2800  0.3294     50     35   0.0000   0.3415
PrecedentOverruled        0.0000  0.0000  0.0000      0      0   0.0000   0.0000
RulingByLowerCourt        0.3958  0.1863  0.2533    102     48   0.1449   0.3457
RatioOfTheDecision        0.5382  0.6824  0.6018    403    511   0.6015   0.6025
RulingByPresentCourt      0.4151  0.4490  0.4314     49     53   0.4167   0.4444
None                      0.4444  0.2424  0.3137     33     18   0.0000   0.3636

confusion counts (rows gold, columns predicted):
           F     I    AP    AR     S     D   PRU  PNRU    PO  RBLC  ROTD  RBPC     N
     F   334     0    22    20     8     0     3     0     0     8    34    13     2
     I     3     9     0     0     0     0     0     0     0     0     2     0     1
    AP    59     0   118   109    12     0    23     9     0     1    67     2     0
    AR    20     0    61   108     3     0    32     8     0     0    39     1     0
     S     1     0     0     0   109     0     2     0     0     0     7     0     1
     D     0     0     0     0     0     0     0     0     0     0     0     0     0
   PRU    70     0     5     2     4     0   119     2     0     6    44     0     2
  PNRU    11     2     0     0     0     0    12    14     0     7     1     2     1
    PO     0     0     0     0     0     0     0     0     0     0     0     0     0
  RBLC    39     0     2     0     0     0     6     0     0    19    29     7     0
  ROTD    54     1     9    10    16     0    28     0     0     3   275     5     2
  RBPC     0     0     1     0     6     0     3     0     0     4    12    22     1
     N     2     0     1     0     1     0    17     2     0     0     1     1     8
most frequent confusions (gold -> predicted): [('ArgumentPetitioner', 'ArgumentRespondent', 109), ('PrecedentReliedUpon', 'Fact', 70), ('ArgumentPetitioner', 'RatioOfTheDecision', 67), ('ArgumentRespondent', 'ArgumentPetitioner', 61), ('ArgumentPetitioner', 'Fact', 59), ('RatioOfTheDecision', 'Fact', 54), ('PrecedentReliedUpon', 'RatioOfTheDecision', 44), ('RulingByLowerCourt', 'Fact', 39), ('ArgumentRespondent', 'RatioOfTheDecision', 39), ('Fact', 'RatioOfTheDecision', 34)]

official check: python evaluate_phase3.py --truth-cl dev_truth_CL.json --truth-it dev_truth_IT.json --prediction dev_pred.json --labels labels.json   (files in /content/p3/mistral_swa_lorma_window)

test: 10 documents, 2197 sentences  (3.4s)
  CL predicted labels: {'Fact': 279, 'ArgumentPetitioner': 100, 'RulingByLowerCourt': 20, 'RatioOfTheDecision': 505, 'Issue': 7, 'Statute': 76, 'RulingByPresentCourt': 21, 'ArgumentRespondent': 117, 'PrecedentReliedUpon': 197, 'None': 12, 'PrecedentNotReliedUpon': 5}
  IT predicted labels: {'Fact': 184, 'RulingByLowerCourt': 39, 'Statute': 137, 'Issue': 9, 'RatioOfTheDecision': 184, 'RulingByPresentCourt': 35, 'None': 19, 'ArgumentPetitioner': 14, 'ArgumentRespondent': 80, 'PrecedentNotReliedUpon': 8, 'PrecedentReliedUpon': 149}
wrote /content/p3/mistral_swa_lorma_window/submission_mistral_swa_lorma_window.zip
```

</details>

**No run log in this folder** for Arch 2 (`bart`) or for any `kd` run. The configurations and dev numbers of the bart
runs behind the submissions in §10 are in `<hf-user>/cs781-p3/runs/<RUN_NAME>/run_config.json`: TODO(student).

## 10. Codabench submissions

Three zips uploaded on 2026-10-16 were saved locally under their Codabench IDs
(`submission/phase3_upload/submission9941xx.zip`), and the BART KD folder carries its ID
(`submission_bart_kd_window994217`). Their `prediction.json` files are byte-identical (MD5) to the run folders below,
so these attributions are exact. The fourth zip of the same batch (`phase3_upload/submission.zip`) is the BART LoRMA
file. The 2026-10-15 entries have the same scores as the 2026-10-16 re-uploads of the same files.

| Codabench ID | Submitted (IST) | Run (local folder) | Method / key settings | Macro-F1 | Attribution |
|---|---|---|---|---|---|
| 989630 | 10-15 14:58 | - | cancelled | - | - |
| 989634 | 10-15 14:59 | **prefixlm** (`submission_prefixlm_lorma_window`), leaderboard entry | lorma plus, r 8, α 16, `q_proj`/`v_proj`; window context; linear head; bidirectional (run log above) | **0.4373** | same score as 994188; folder saved 14:58 |
| 989644 | 10-15 15:06 | mistral_swa (`submission_mistral_swa_lorma_window`) | lorma plus, r 8, α 16; window context; linear head; causal sliding window (run log above) | 0.4083 | same score as 994190; zip downloaded 15:00 |
| 989831 | 10-15 16:19 | - | cancelled | - | - |
| 994155 | 10-16 22:57 | - | failed | - | - |
| 994175 | 10-16 23:03 | - | failed | - | - |
| 994188 | 10-16 23:08 | prefixlm (re-upload of 989634's file) | as 989634 | 0.4373 | `submission994188.zip`, MD5 match |
| 994190 | 10-16 23:10 | mistral_swa (re-upload of 989644's file) | as 989644 | 0.4083 | `submission994190.zip`, MD5 match |
| 994192 | 10-16 23:10 | llama (`submission_llama_lorma_window`) | lorma plus, r 8, α 16, `q_proj`/`v_proj`; window context; linear head; causal (run log above) | 0.4118 | `submission994192.zip`, MD5 match |
| 994194 | 10-16 23:11 | bart v2 (`submission_bart_lorma_window`) | lorma, window context, linear head (from the run name; other settings: TODO(student)) | 0.3765 | fourth zip of the batch, MD5 match |
| 994217 | 10-16 23:20 | bart v2 student (`submission_bart_kd_window`) | kd, window context, linear head (from the run name); `TEACHER_RUN`: TODO(student) | 0.3697 | folder named with the ID, MD5 match with the downloaded zip |

Notes:
* The `prediction.json` label counts of the local `llama`, `prefixlm` and `mistral_swa` submission files equal the
  per-domain test label counts printed by the corresponding run logs (summed over CL + IT). All five files listed
  above contain the 10 test documents and 2197 labels.
* `submission_bart_phase1_retrained` (dated 2026-10-15) was **not** produced by this template: its `prediction.json`
  holds 1000 entries keyed `"1"…"1000"` with `{"text": …}` values (the Phase-1 continuation format), whereas this
  notebook always writes `{doc_id: [labels]}` and names its zip `submission_<arch>_<method>_<context>…`. Whether and
  to which Codabench phase it was submitted: TODO(student).
* Phase-3 Codabench totals: 11 submissions; 7 finished (5 distinct files, because 994188 and 994190 re-upload the
  files of 989634 and 989644), 2 failed, 2 cancelled. The course minimum is 4 submissions per phase (updated rule).

## 11. How to run

1. Open Google Colab and set *Runtime → Change runtime type → T4 GPU*.
2. Add a Colab secret named `HF` holding the same Hugging Face token as Phases 1–2 (read access to the
   `cs781-p1-*` repos and the gated course dataset, write access for `cs781-p3`). Outside Colab the notebook reads a
   Kaggle secret `HF` or the environment variable `HF_TOKEN`; CPU execution needs `CS781_ALLOW_CPU=1`.
3. Generate the notebooks (already present in this folder):
   ```bash
   python build_p3_notebooks.py            # all four
   python build_p3_notebooks.py bart       # only P3_arch2_bart.ipynb
   ```
4. Upload `P3_arch<N>_<arch>.ipynb` to Colab. In the first code cell choose `METHOD`, `CONTEXT`, `HEAD`,
   `SENT_LSTM` (`RUN_NAME` follows); adjust the configuration cell if needed.
5. *Runtime → Run all*. Every run starts from the Phase-1 checkpoint. Outputs: `/content/p3/<RUN_NAME>/` locally and
   `<hf-user>/cs781-p3/runs/<RUN_NAME>/` on the Hub (uploaded only when the notebook reaches the save cell).
6. The submission zip `submission_<RUN_NAME>.zip` (containing only `prediction.json`) is downloaded by the browser;
   upload it to the Codabench Phase-3 page.
7. Knowledge distillation: first let a non-`kd` run (the teacher, any architecture) finish, so that its
   `train_scores.json` is in `cs781-p3`; then, in the student's notebook, set `METHOD = "kd"` and
   `TEACHER_RUN = "<teacher RUN_NAME>"`. The configuration cell raises an error listing the finished runs if the teacher
   is not found.
8. Local scoring: download `dev_truth_CL.json`, `dev_truth_IT.json`, `dev_pred.json`, `labels.json` from the run
   folder and run `evaluate_phase3.py` (command in §7).

Dependencies: `torch`, `transformers`, `huggingface_hub`, `scikit-learn`, `numpy`, `matplotlib` (pre-installed on
Colab; the logs show torch 2.11.0+cu130 and transformers 5.18.0).
