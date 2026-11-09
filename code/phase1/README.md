# Phase 1: Legal language model pretraining

Four language models are trained from random initialisation on the Phase-1 legal corpus: a decoder-only Transformer
(`llama`), an encoder-decoder Transformer (`bart`, retrained as `bart-v2`), a prefix-LM hybrid (`prefixlm`) and a
sliding-window / grouped-query-attention Transformer (`mistral_swa`). Each model's `best/` checkpoint is the
starting point for Phases 2-7.

Everything in this file describes what the code does and copies numbers verbatim from the executed notebooks in
`run_logs/`. Items that cannot be read from the code or the logs are marked `TODO(student)`.

---

## 1. Task

| | |
|---|---|
| Input | one 512-word legal-text prompt per test ID (`Phase_1_Pretraining/Test/`; the code reads `test.json`, which has 1000 prompts according to every run log) |
| Output | a 128-word continuation per ID. The text must not repeat the prompt |
| Submission | `submission.zip` with `prediction.json` at the root of the archive |
| Primary metric | BERTScore F1 |
| Secondary metrics | ROUGE-L F1, BLEU-4 |
| Constraints (spec §7.5-7.6) | trained from scratch on the Phase-1 corpus. Size must be 50M-300M parameters. No pretrained LMs or weights, no commercial APIs |

`prediction.json` schema (same shape as the test file):

```json
{
  "1": {"text": "Generated 128-word continuation ..."},
  "2": {"text": "Generated 128-word continuation ..."}
}
```

The official checker (`evaluate_phase1.py`) rejects duplicate JSON keys, empty or non-string `text`, and any ID
mismatch between truth and prediction. It computes `bert_score.score(..., lang="en")` (BERTScore's default English
model, which the logs show as `roberta-large`), ROUGE-L F1 on whitespace tokens (LCS), and corpus BLEU-4 on whitespace
tokens with a brevity penalty.

---

## 2. Files in this folder

| Path | What it is |
|---|---|
| `initial_pipeline/00_data_prep.py` / `.ipynb` | Streams the 4 corpus files, samples documents, holds out validation documents, builds a local test set shaped like the official one, trains a 32k byte-level BPE tokenizer, writes per-source `uint16` `.bin` token files and uploads them to the private dataset repo `<hf-user>/cs781-p1-data`. **Both code generations use this script** |
| `initial_pipeline/01_train.py` / `.ipynb` | First-generation trainer: one notebook, `ARCH` is set by hand on each Colab account |
| `initial_pipeline/02_generate.py` / `.ipynb` | First-generation generator: loads `best/` from the Hub, compares decoding configs on the local held-out set and writes `submission.zip` |
| `phase1_arch_template.py` | Second-generation, percent-format (`# %%`) template. Part A trains and Part B generates and evaluates. Placeholders such as `{{ARCH}}` are filled in per architecture |
| `build_arch_notebooks.py` | Renders the template into `P1_arch{1..4}_<arch>.ipynb` (`python build_arch_notebooks.py [arch ...]`). Contains the `ARCHS` table and `BART_NOTE` |
| `build_nb.py` | Converts a percent-format `.py` into an `.ipynb` (also used to make the `initial_pipeline/*.ipynb` files) |
| `P1_arch1_llama.ipynb`, `P1_arch3_prefixlm.ipynb`, `P1_arch4_mistral_swa.ipynb` | Per-arch notebooks rendered from an **earlier version** of the template (see 2.1) |
| `P1_arch2_bart.ipynb` | Per-arch notebook rendered from the **current** template (BART v2: text infilling + `<mask>`, repo `cs781-p1-bart-v2`) |
| `bart_diagnostic_cell.py` | Standalone Colab cell that checks the v1 BART checkpoint (`cs781-p1-bart`) for prompt dependence and fp16 effects (see 7.4) |
| `tools/compare_submissions.py` | Stand-alone, standard-library-only viewer. It puts the prompt ending, the reference and each submission's continuation side by side (see 7.5) |
| `run_logs/` | Notebooks executed on Colab, saved with their outputs (see §9) |

### 2.1 Which code generation produced the run logs and the Phase-1 models

There are two code generations. The run logs come from the **second** one.

| | Generation (a): `initial_pipeline/` | Generation (b): template + per-arch notebooks |
|---|---|---|
| Data | `00_data_prep` | `00_data_prep` (the template's header requires it: "`00_data_prep.ipynb` has finished once") |
| Training | `01_train`, `ARCH` edited per account | `P1_arch{N}_<arch>.ipynb` Part A, `ARCH` fixed per notebook |
| `MIX` (sc/hc/trib/acts) | 0.40 / 0.30 / 0.20 / 0.10 | 0.35 / 0.35 / 0.15 / 0.15 |
| `TRAIN_HOURS` | 5.5 | 3 |
| Hub uploads | direct `api.upload_folder` | `hub_push` (3 retries) and `super_squash_history` after each `latest/` push (`SQUASH_HISTORY = True`) |
| Generation | separate `02_generate` notebook | Part B of the same notebook. It uses the session's `best/` if present, otherwise downloads it |
| BART | continuation objective only | v1 (earlier template): continuation only. v2 (current template): continuation + text infilling |

Evidence:

* A cell-by-cell diff shows that the code in `run_logs/P1_arch1_llama_after_run.ipynb` and
  `run_logs/P1_arch3_prefixlm_after_run.ipynb` is identical to the committed `P1_arch1_llama.ipynb` and
  `P1_arch3_prefixlm.ipynb`. The code in `run_logs/P1_arch2_bart_after_run.ipynb` (v1) is the same as
  `P1_arch1_llama.ipynb` except for the header and `ARCH`. The code in `run_logs/P1_arch2_bart_v2_after.ipynb` is
  identical to the committed `P1_arch2_bart.ipynb`, which in turn equals a fresh rendering of the current
  `phase1_arch_template.py`.
* The config cells in all four logs have `MIX = {"sc": 0.35, "hc": 0.35, "trib": 0.15, "acts": 0.15}` and
  `TRAIN_HOURS = 3`. Those are the generation (b) values. The logs also contain `hub_push` / `SQUASH_HISTORY`, which
  `01_train.py` does not have.
* The llama, bart (v1) and prefixlm logs print `No checkpoint found, starting fresh` at the resume step. So no
  `latest/state.pt` existed in `cs781-p1-{llama,bart,prefixlm}` when those runs started. Whether generation (a)'s
  `01_train` / `02_generate` was ever executed is `TODO(student)`.
* File modification times in the student's working folder (outside this repo; the copies here all carry the copy
  time):
  * `P1_arch{1,3,4}` notebooks: 2026-10-06 13:24
  * llama / bart-v1 / prefixlm run logs: 2026-10-06 16:53-17:10
  * `compare_submissions.py`: 2026-10-06 17:20
  * `bart_diagnostic_cell.py`: 2026-10-06 18:24
  * `phase1_arch_template.py`, `build_arch_notebooks.py` and `P1_arch2_bart.ipynb`: 2026-10-14 16:36-16:38

  The Phase-1 deadline in the spec is 2026-10-06 23:59.

Phase-1 checkpoints that later phases load, all produced by generation (b):

| Arch | Repo (`best/`) | Notebook that trained it | Run log |
|---|---|---|---|
| llama | `<hf-user>/cs781-p1-llama` | `P1_arch1_llama.ipynb` (Oct-6 rendering) | `P1_arch1_llama_after_run.ipynb` |
| bart v1 | `<hf-user>/cs781-p1-bart` | Oct-6 rendering of `P1_arch2_bart.ipynb` (not committed; its code is in the run log) | `P1_arch2_bart_after_run.ipynb` |
| bart v2 | `<hf-user>/cs781-p1-bart-v2` | `P1_arch2_bart.ipynb` (current template) | `P1_arch2_bart_v2_after.ipynb` (resumed session only) |
| prefixlm | `<hf-user>/cs781-p1-prefixlm` | `P1_arch3_prefixlm.ipynb` (Oct-6 rendering) | `P1_arch3_prefixlm_after_run.ipynb` |
| mistral_swa | `<hf-user>/cs781-p1-mistral_swa` | `TODO(student)`: presumably `P1_arch4_mistral_swa.ipynb` | none locally (`TODO(student)`) |

`BART_NOTE` in `build_arch_notebooks.py` states that later phases must load BART from `cs781-p1-bart-v2`.

Running `python build_arch_notebooks.py` with no argument regenerates all four notebooks from the current template.
The llama / prefixlm / mistral_swa notebooks would then differ from the committed (Oct-6) ones in these ways:

* the `REPO_SUFFIX` variable
* per-micro-batch token accounting
* `distinct_outputs` in the held-out results
* the distinct-prediction assertion before the zip is written
* the `suppress_tokens` list
* the BART-only cells

For these three archs the model, objective and training-loop arithmetic are unchanged.

---

## 3. Data and preprocessing (`initial_pipeline/00_data_prep.py`)

**Source.** HF dataset `Exploration-Lab/CS-781-Capstone`, files `Phase_1_Pretraining/Train/train_{supreme_court,
high_courts,tribunals,acts}.json` (source keys `sc`, `hc`, `trib`, `acts`). Test file:
`Phase_1_Pretraining/Test/test.json`.

**Streaming and sampling.**
* The script fetches each file over HTTP and parses it with `ijson`. It asserts the `yajl2` backend and detects
  whether the file is a JSON list or a dict from the first 2000 bytes.
* Each non-empty `text` record is kept with probability `keep_p = min(1, target_gb·1e9 / file_size)`. The
  `target_gb` values in the file are sc 1.30, hc 1.80, trib 1.00, acts 0.35.
* Each kept document goes to validation with probability `VAL_DOC_FRAC = 0.01`, otherwise to train (`random.seed(1234)`).
* A stream error keeps the partial data.

The values actually used to build `cs781-p1-data` (its `meta.json`/`sampling_stats.json`) are not in this repo:
`TODO(student)`.

**Local held-out test set.**
* Built from validation documents with `HELDOUT_PER_SOURCE = 100` examples per source.
* Documents shorter than 512+128+10 words are skipped.
* A random word offset is drawn (`random.Random(1234)`). The prompt is the next 512 words: the original text span is
  kept if the official prompts contain newlines, otherwise it is single-space joined. The reference is the following
  128 words.
* Keys are `<source>_<n>`. Files: `heldout_prompts.json` and `heldout_truth.json`.
* Every run log prints `held-out: 373  test: 1000`.

**Tokenizer.**
* Byte-level BPE (`tokenizers`), `vocab_size = 32000`, `min_frequency = 2`, `add_prefix_space = False`, byte-level
  initial alphabet.
* Special tokens: `<|endoftext|>` (id 0; also used as BOS/UNK) and `<pad>` (id 1).
* Trained on train documents from every source until 0.4e9 characters in total (`TOKENIZER_TRAIN_GB`), with each
  source's budget proportional to its character share.
* The script prints tokens/word for this tokenizer and for GPT-2 on 200 test prompts. That output is not in the repo:
  `TODO(student)`.
* Run logs print `vocab 32000 eos 0 pad 1`. BART v2 prints `vocab 32001 eos 0 pad 1 mask 32000`.
* Part B prints `tokens per word on references: 1.213` (first 100 held-out references, all logs).

**Token files.** Each document is encoded, `<|endoftext|>` is appended, and the ids are concatenated into
`<src>_{train,val}.bin` (`uint16`). Token counts printed by every run log:

| Source | train tokens | val tokens | `MIX` weight |
|---|---|---|---|
| sc (Supreme Court) | 252.5M | 2.70M | 0.35 |
| hc (High Courts) | 396.6M | 3.67M | 0.35 |
| trib (Tribunals) | 250.3M | 2.78M | 0.15 |
| acts (Acts) | 75.0M | 1.14M | 0.15 |

The four printed train counts sum to 974.4M tokens.

**Window sampling and mixing (training notebooks).**
* Each training window chooses a source with probability `MIX[s]` (normalised). It then takes a uniform random start
  offset in that source's train `.bin`.
* Windows are contiguous token runs and may cross `<|endoftext|>` document boundaries. No per-document packing or
  padding is used.
* Window length: `CTX = 1024` for llama / prefixlm / mistral_swa. For BART it is `SRC_MAX + TGT_LEN = 768 + 256`, plus
  `DENOISE_LEN = 512` windows for BART-v2 denoising.

**Validation windows.**
* Fixed per source: `EVAL_BATCHES_PER_SOURCE = 10` batches × `MICRO_BS = 8` windows from the val `.bin`, drawn with
  `np.random.default_rng(0)`.
* BART v2 adds 2 × 8 clean 512-token windows per source, corrupted once with `default_rng(1)`.

---

## 4. Model architecture

All models are Hugging Face classes built from a config with random initialisation. `build_model()` asserts
`50e6 <= n_params <= 300e6`. Every config has `bos = eos = <|endoftext|>` (id 0) and `pad = 1`. It also stores
`config.cs781_arch`, and for BART v2 `config.cs781_objective` (the denoising settings, `mask_id = 32000`,
`cont_end_marker = True`).

| | llama | prefixlm | mistral_swa | bart v1 | bart v2 |
|---|---|---|---|---|---|
| Class | `LlamaForCausalLM` | `LlamaForCausalLM` + 4-D prefix mask | `MistralForCausalLM` | `BartForConditionalGeneration` | `BartForConditionalGeneration` |
| Layers | 12 decoder | 12 decoder | 12 decoder | 6 encoder + 6 decoder | 6 encoder + 6 decoder |
| d_model | 768 | 768 | 768 | 768 | 768 |
| Attention heads | 12 | 12 | 12 | 12 (enc and dec) | 12 (enc and dec) |
| KV heads | 12 (MHA) | 12 (MHA) | 4 (GQA, head dim 64) | 12 | 12 |
| FFN | 2048, gated (gate/up/down) | 2048, gated | 2048, gated | 3072 (fc1/fc2) | 3072 (fc1/fc2) |
| Activation | SiLU (SwiGLU) | SiLU (SwiGLU) | SiLU (SwiGLU) | GELU | GELU |
| Norm | RMSNorm, pre-norm, eps 1e-6 | RMSNorm | RMSNorm | LayerNorm (post-LN) + `layernorm_embedding` | same |
| Positions | RoPE (config default base) | RoPE | RoPE | learned absolute (1026 × 768 per stack) | same |
| Embedding vocab | 32000 | 32000 | 32000 | 32000 | 32064 (32001 rounded up to a multiple of 64) |
| Max positions / context | 1024 | 1024 | 1024 | 1024 (training: encoder ≤ 768, decoder 256) | 1024 (encoder ≤ 769 incl. `<mask>`. Denoising: encoder ≤ 512, decoder 512) |
| Attention pattern | causal | bidirectional on prefix, causal after | causal, `sliding_window = 256` | bidirectional encoder, causal decoder + cross-attn | same |
| Tied embeddings | yes | yes | yes | yes (encoder, decoder, LM head shared) | yes |
| Biases in linear layers | no | no | no | yes | yes |
| Embedding scaling | none | none | none | `scale_embedding=True` (×√768) | same |
| Dropout | `attention_dropout = 0.0` | 0.0 | 0.0 | `dropout = attention_dropout = activation_dropout = 0.0` | same |
| **Parameters** | **109,529,856** (log) | **109,529,856** (log) | **100,092,672** (computed, see below) | **125,392,896** (log) | **125,442,048** (log) |

Parameter counts:
* llama, prefixlm and both BARTs: the `params` field of `cs781_info.json` printed in Part B of each run log (also
  printed as `109.5M` / `125.4M`).
* mistral_swa: no Phase-1 log exists locally. The number is computed from the config as
  12·(2·768·768 + 2·768·256 + 3·768·2048 + 2·768) + 32000·768 + 768 = 100,092,672. The Phase-6 run log
  (`code/phase6/run_logs/P6_arch4_mistral_swa_after.ipynb`) prints `"params_final": 100092672` for its merged
  mistral_swa model.
* For reference, the llama formula is 12·(4·768² + 3·768·2048 + 2·768) + 32000·768 + 768 = 109,529,856.

All counts are within the 50M-300M limit. No knowledge-distillation or compression step is implemented in the Phase-1
code.

---

## 5. Pretraining objectives (Phase 1 has no fine-tuning methods)

Phase 1 trains every parameter from random initialisation. The training objective depends on the architecture
(`compute_loss` / `train_loss` in the template):

| Arch | Training loss | Validation loss (`compute_loss(..., fixed=True)`) |
|---|---|---|
| llama, mistral_swa | next-token cross-entropy on every position of a 1024-token window (`labels = input_ids`; HF shifts internally) | same, on the fixed val windows |
| prefixlm | prefix length `P ~ randint[PREFIX_MIN=64, CTX-128=896)`. Mask `allowed[q,k] = (k <= q) or (k < P)`. `labels[:, :P] = -100`, so loss covers tokens `P..1023` | `P = L // 2 = 512` |
| bart v1 | continuation only: from a 1024 window, `S ~ randint[128, 768]`. Encoder input = tokens `[768-S, 768)`, labels = tokens `[768, 1024)`. Decoder input = labels shifted right with start token `<\|endoftext\|>` | `S = 768` |
| bart v2 | per micro-batch: with probability `P_DENOISE = 0.5` a **text-infilling** batch, otherwise a **continuation** batch (as v1, plus a `<mask>` token appended to the encoder input when `CONT_END_MARKER = True`) | continuation with `S = 768` (+`<mask>`). The denoising loss and encoder-use measurements are reported separately |

**BART v2 text infilling.**
* Window of `DENOISE_LEN = 512` clean tokens.
* Spans of length `max(1, Poisson(SPAN_LAMBDA = 3))` are placed at uniform random starts until at least
  `MASK_RATIO = 0.3` of positions are covered. Overlapping spans merge.
* Each maximal masked run is replaced by one `<mask>` (id 32000, added with `tokenizer.add_special_tokens`, so all BPE
  ids are unchanged). The corrupted sequences are right-padded and get an attention mask.
* The decoder input is `[<mask>] + clean[:-1]` and the labels are the 512 clean tokens.
* Continuation batches start the decoder from `<|endoftext|>`.

**Sanity checks run before training.**
* prefixlm:
  * an all-ones prefix mask equals the default causal path;
  * changing token `P-1` changes position 0 (bidirectional prefix);
  * changing token 50 leaves positions < 50 unchanged (causal suffix).

  Logged: `prefix-LM mask checks passed`.
* bart v2:
  * the `labels` loss equals a manual loss with decoder input `[<|endoftext|>] + labels[:-1]`;
  * changing one encoder token changes decoder logits;
  * infilling keeps unmasked tokens in order with one `<mask>` per span.

  Logged (random-init model): `bart checks passed: label shift ok, encoder->decoder max |delta logit| 6.07e-02,
  infilling masked 30.5% of 512 tokens in 43 spans -> encoder length 399`.

**BART v2 encoder-use monitor (`bart_monitor`, at every evaluation).**
* `denoise`: validation denoising loss.
* `cont_first32` and `cont_first32_shuffled`: continuation loss on the first 32 target tokens with the window's own
  prompt, and with the prompt of another window in the batch (`src.roll(1, 0)`). `enc gain` = shuffled − own.
* `distinct greedy outputs`: number of distinct 32-token greedy continuations for 8 validation prompts (2 per source).
* The loop prints
  `!! all prompts gave the same continuation: the decoder is ignoring the encoder (the v1 failure)` when the distinct
  count is 1.

---

## 6. Training procedure (Part A of the template; identical in `01_train.py` except where noted in 2.1)

| Item | Setting |
|---|---|
| Optimizer | `torch.optim.AdamW(fused=True)`, `betas = (0.9, 0.95)`, weight decay 0.1 on parameters with `dim >= 2`, 0 on the rest |
| LR schedule | linear warm-up for 300 steps to `LR = 6e-4` (`LR·(step+1)/300`), then cosine to `MIN_LR_FRAC·LR = 6e-5` at `max_steps` |
| Time budget | after 30 optimizer steps, `max_steps = int(TRAIN_HOURS·3600·steps_per_sec·0.93)` (7% reserved for eval and checkpoints). Run logs: `TRAIN_HOURS = 3` |
| Batch | `MICRO_BS = 8` windows × `GRAD_ACCUM = 8` = 64 windows per optimizer step (65,536 tokens for 1024-token windows) |
| Precision | `torch.autocast("cuda", float16)` + `GradScaler`. Gradient-norm clip 1.0 |
| Token counter | generation (a) and Oct-6 notebooks: `MICRO_BS·GRAD_ACCUM·WINDOW` per step (BART v1 counts 1024 per window). Current template: tokens actually used per micro-batch (BART: `S + 256`, or 512 for denoising) |
| Logging | every 25 steps: loss, lr, grad-norm, tok/s, tokens (BART v2 also per-objective `cont`/`den` means) |
| Evaluation | every `EVAL_EVERY = 250` steps and at the last step, in fp16 autocast, on the fixed val windows. Per-source mean loss and `mean` = unweighted mean of the 4 sources |
| Model selection | `best/` is pushed whenever `mean` val loss is lower than the best so far (`save_pretrained` + tokenizer + `cs781_info.json` with step, val_loss, arch, params, tokens_seen and the objective for BART v2) |
| Early stopping | none. Training runs to `max_steps` |
| Checkpoint / resume | `latest/state.pt` (model, optimizer, scaler, state incl. `max_steps` and history) every `CKPT_MINUTES = 30` and at the end. Re-running the notebook resumes from it. The data RNG is re-seeded with `SEED + step` (`SEED = 42`) |
| Hub storage | template: `super_squash_history` after each `latest/` push |
| Plot | `loss_<arch>.png` (train loss and per-source val loss vs tokens; BART v2 adds an encoder-use panel), uploaded to the model repo |
| Hardware / software (logs) | `Tesla T4`, torch `2.11.0+cu130`, transformers `5.18.0` |

---

## 7. Inference and submission (Part B of the template / `02_generate.py`)

### 7.1 Model loading
* Loads `best/` (the session copy if present, otherwise from the Hub) in fp16.
* Prompts: llama / prefixlm / mistral_swa keep the last `max_position_embeddings - MAX_NEW_TOKENS = 1024 - 224 = 800`
  tokens (logs print `max prompt tokens 800`). BART keeps the last `SRC_MAX = 768` tokens, and v2 appends `<mask>`
  when `cs781_objective.cont_end_marker` is set.
* Padding: decoder-only batches are left-padded and BART encoder batches are right-padded.

### 7.2 Decoding
* `max_new_tokens = min_new_tokens = MAX_NEW_TOKENS = 224`, so EOS cannot end generation early.
* Suppressed tokens: `<pad>`, plus `<mask>` and ids ≥ `len(tokenizer)` (current template).
* llama, mistral_swa and bart use `model.generate`. prefixlm uses its own loop (`prefixlm_generate`):
  * the prompt is encoded with a 4-D mask that lets every prompt token see every non-pad prompt token (padding queries
    see only themselves), with position ids taken from the attention-mask cumsum;
  * decoding then continues causally with the KV cache;
  * the repetition-penalty and no-repeat-n-gram processors are applied, EOS and PAD are set to −inf, and sampling uses
    temperature if `do_sample`.
* The new tokens are decoded with `skip_special_tokens`, split on whitespace and truncated to `TARGET_WORDS = 128`
  words. An empty output is replaced by `"the"`.

`DECODE_CONFIGS` (template default, used unchanged in all run logs):

| Name | Settings |
|---|---|
| `greedy_rp1.2` | `do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=0` |
| `greedy_ngram4` | `do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=4` |

### 7.3 Local evaluation and submission
1. Each config generates for the 373 held-out prompts. `HELDOUT_LIMIT = 400`: only if the held-out file had more than
   400 examples would the code take an equal number per source. The notebook scores the outputs with a copy of the
   official metric code:
   * BERTScore `lang="en"`, batch 32, on GPU. A BERTScore exception is caught, and then only ROUGE-L / BLEU-4 are
     reported;
   * ROUGE-L F1 and BLEU-4;
   * per-source BERTScore `bs_<src>`;
   * `distinct_outputs` (current template).

   Files `heldout_pred_<config>.json`, `heldout_truth.json` and `heldout_scores.json` are written so that
   `evaluate_phase1.py --truth heldout_truth.json --prediction heldout_pred_<config>.json` can be run on them.
2. `CHOSEN` = the config with the highest held-out `bertscore_f1` (falls back to `rougeL_f1`). The comment says it may
   be set by hand.
3. The notebook generates the 1000 test continuations and asserts:
   * the ID set equals the test IDs;
   * every `text` is a non-empty string;
   * (current template) more than half of the predictions are distinct.

   It then prints words-per-prediction min/mean/max.
4. It writes `prediction.json` and `submission.zip`, and asserts that `namelist() == ["prediction.json"]`. The folder
   `/content/p1_train/gen_<RUN_TAG>/` (with `submission_record.json`) is uploaded to
   `<model repo>/submissions/<RUN_TAG>/`, and the zip is downloaded in Colab.

### 7.4 `bart_diagnostic_cell.py`
A standalone Colab cell that needs only the `HF` secret.
* Loads `cs781-p1-bart/best` (v1) and the held-out files, and takes 24 held-out keys spread over the file
  (`list(P)[::len(P)//24]`).
* Prompts: the last 768 tokens, right-padded. Labels: the first 64 tokens of each reference, cut to the shortest.
* It prints:
  * **C**: fp32 per-position loss on the true continuation with the right prompt vs a mismatched prompt (batch
    rolled by 1), for the first 8 tokens and for all tokens;
  * **B**: the number of distinct greedy outputs (60 tokens, `no_repeat_ngram_size=4`) for 6 prompts in fp32, and in
    fp16 on GPU;
  * **A** (GPU only): whether the fp16 encoder output is finite, and its max |h|;
  * three prompt endings with their fp32 outputs.
* Its output is not stored in this repo: `TODO(student)`.

### 7.5 `tools/compare_submissions.py`
```
python3 tools/compare_submissions.py LABEL=FILE [LABEL=FILE ...] [--prompts test.json|heldout_prompts.json]
        [--truth heldout_truth.json] [--ids ID ... | --n 12 | --all] [--seed 0] [--prompt-words 60]
        [--out comparison.html] [--text]
```
* `FILE` is a `submission.zip` (it reads `prediction.json`) or a prediction `.json`.
* For each selected ID (12 random IDs by default, seed 0) it shows the last `--prompt-words` words of the prompt, the
  reference if `--truth` is given, and every submission's text with its word count.
* Output is a self-contained `comparison.html`, or terminal text with `--text`.

---

## 8. Configuration knobs ("YOUR CHOICE")

Template config cell (`phase1_arch_template.py`). Defaults are the values used in every run log.

| Variable | Default | Meaning |
|---|---|---|
| `DATA_REPO` | `<hf-user>/cs781-p1-data` | private dataset repo written by `00_data_prep` |
| `REPO_SUFFIX` | `""` (`"-v2"` for bart) | suffix of the model repo `cs781-p1-<arch><suffix>` |
| `MIX` | `{"sc": .35, "hc": .35, "trib": .15, "acts": .15}` | sampling probability of each source per training window (sources with weight 0 are dropped) |
| `TRAIN_HOURS` | 3 | wall-clock training budget that sets `max_steps` |
| `CKPT_MINUTES` | 30 | interval for pushing `latest/` |
| `EVAL_EVERY` | 250 | optimizer steps between evaluations |
| `EVAL_BATCHES_PER_SOURCE` | 10 | val batches (of `MICRO_BS` windows) per source |
| `LR`, `MIN_LR_FRAC`, `WARMUP_STEPS` | 6e-4, 0.1, 300 | peak LR, final LR fraction, linear warm-up steps |
| `WEIGHT_DECAY`, `BETAS`, `GRAD_CLIP` | 0.1, (0.9, 0.95), 1.0 | AdamW settings and gradient-norm clip |
| `MICRO_BS`, `GRAD_ACCUM` | 8, 8 | windows per micro-batch, micro-batches per step |
| `DROPOUT` | 0.0 | attention dropout (Llama/Mistral). BART dropout and attention dropout |
| `SEED` | 42 | Python / NumPy / torch seed. The data RNG is `SEED + step` |
| `CTX` | 1024 | window / max positions for llama, prefixlm, mistral_swa |
| `PREFIX_MIN` | 64 | prefixlm: minimum prefix length (max is `CTX - 128`) |
| `SRC_MIN`, `SRC_MAX` | 128, 768 | bart: encoder prompt length range in training (`SRC_MAX` also at inference) |
| `TGT_LEN` | 256 | bart: continuation tokens per training example |
| `D_MODEL`, `N_LAYER`, `N_HEAD`, `D_FF` | 768, 12, 12, 2048 | width, decoder layers, heads, FFN (llama / prefixlm / mistral_swa). BART uses `D_MODEL` and `N_HEAD` |
| `N_KV_HEAD_SWA`, `SWA_WINDOW` | 4, 256 | mistral_swa: KV heads, sliding-window size |
| `BART_ENC_LAYERS`, `BART_DEC_LAYERS`, `BART_FF` | 6, 6, 3072 | bart depth and FFN |
| `P_DENOISE` | 0.5 | bart v2: probability that a micro-batch is text infilling |
| `DENOISE_LEN` | 512 | bart v2: tokens per denoising window |
| `MASK_RATIO` | 0.3 | bart v2: fraction of a denoising window covered by masked spans |
| `SPAN_LAMBDA` | 3.0 | bart v2: Poisson mean of span length |
| `CONT_END_MARKER` | True | bart v2: append `<mask>` after the prompt (training and inference) |
| `SQUASH_HISTORY` (optimiser cell) | True | squash the model repo history after each `latest/` push |
| `RUN_TAG` (Part B) | `"v1"` | name of `gen_<RUN_TAG>/` and `submissions/<RUN_TAG>/` |
| `DECODE_CONFIGS` (Part B) | see 7.2 | decoding configs compared on held-out |
| `HELDOUT_LIMIT` | 400 | maximum held-out examples scored per config |
| `TARGET_WORDS` | 128 | words kept per continuation |
| `MAX_NEW_TOKENS` | 224 | tokens generated per prompt |
| `BATCH` | 32 | generation batch size |
| `CHOSEN` | argmax held-out BERTScore | decoding config used for the test set |

`00_data_prep.py` configuration:

| Variable | Default | Meaning |
|---|---|---|
| `SOURCES` | sc 1.30, hc 1.80, trib 1.00, acts 0.35 (GB) | target GB of raw JSON kept per source |
| `VAL_DOC_FRAC` | 0.01 | fraction of kept documents held out |
| `HELDOUT_PER_SOURCE` | 100 | local test examples per source |
| `PROMPT_WORDS`, `CONT_WORDS` | 512, 128 | local test prompt and reference lengths |
| `VOCAB_SIZE` | 32000 | BPE vocabulary (must be ≤ 65535 for `uint16`) |
| `TOKENIZER_TRAIN_GB` | 0.4 | characters (×1e9) used to learn merges |
| `SEED` | 1234 | sampling, split and held-out seed |

`01_train.py` has the same knobs as the template minus the BART-v2 ones and `REPO_SUFFIX`, with `MIX = {sc .40, hc .30,
trib .20, acts .10}` and `TRAIN_HOURS = 5.5`. `02_generate.py` has `ARCH`, `RUN_TAG`, `DECODE_CONFIGS`,
`HELDOUT_LIMIT`, `TARGET_WORDS`, `MAX_NEW_TOKENS` and `BATCH`, with the same defaults as Part B.

---

## 9. Runs recorded in `run_logs/`

Phase 1 has no `RUN_NAME`. A run is identified by its model repo and the Part-B `RUN_TAG`. In all four logs the config
cell and `DECODE_CONFIGS` equal the template defaults (BART v1 has no v2 knobs). Train minutes (`train_seconds`) and
peak GPU memory are **not printed** by the code.

### 9.1 Training

| Run log | Repo / RUN_TAG | Arch | Session | Throughput printed at step 30 | `max_steps` | `tokens_seen` | Final EVAL (sc / hc / trib / acts / mean) | best val | Params |
|---|---|---|---|---|---|---|---|---|---|
| `P1_arch1_llama_after_run.ipynb` | `cs781-p1-llama` / v1 | llama | fresh | 14,333 tok/s, `~144M tokens` | 2196 | 143,917,056 | step 2196: 2.8047 / 2.0209 / 2.4273 / 2.2253 / 2.3695 | 2.3695422977209093 | 109,529,856 |
| `P1_arch2_bart_after_run.ipynb` | `cs781-p1-bart` / v1 | bart v1 | fresh | 44,006 tok/s, `~442M tokens` | 6744 | 441,974,784 | step 6744: 3.1185 / 2.2166 / 2.6675 / 2.5464 / 2.6373 | 2.6372566401958464 | 125,392,896 |
| `P1_arch2_bart_v2_after.ipynb` | `cs781-p1-bart-v2` / v1 | bart v2 | `RESUMED from step 4145 / 5872` (steps 0-4145 not in this log) | not in log | 5872 | 228,682,120 | step 5872: 3.0634 / 2.1899 / 2.6328 / 2.4815 / 2.5919 | 2.5918909221887585 | 125,442,048 |
| `P1_arch3_prefixlm_after_run.ipynb` | `cs781-p1-prefixlm` / v1 | prefixlm | fresh | 13,020 tok/s, `~131M tokens` | 1995 | 130,744,320 | step 1995: 3.0014 / 2.1882 / 2.5192 / 2.4137 / 2.5306 | 2.5306235373020174 | 109,529,856 |
| (none) | `cs781-p1-mistral_swa` | mistral_swa | `TODO(student)` | | | | | | 100,092,672 (computed) |

Each architecture's val loss is the quantity defined in §5 (`fixed=True`). In every log the `cs781_info.json` printed
in Part B has `step` equal to `max_steps`, so `best/` is the final-step checkpoint (the log prints
`-> new best pushed` at the final evaluation).

BART v2 encoder-use line at the final evaluation (verbatim):
`BART step 5872: denoise 1.2204  continuation first-32 loss: own prompt 2.8543, other prompt 3.5813 (enc gain +0.7271)  distinct greedy outputs 8/8`.
The printed values at steps 4250 to 5750:

| step | denoise | own prompt | other prompt | enc gain | distinct |
|---|---|---|---|---|---|
| 4250 | 1.6366 | 2.9403 | 3.6376 | +0.6973 | 8/8 |
| 4500 | 1.5368 | 2.9096 | 3.6033 | +0.6938 | 8/8 |
| 4750 | 1.4647 | 2.9108 | 3.5950 | +0.6841 | 8/8 |
| 5000 | 1.3599 | 2.8868 | 3.6156 | +0.7287 | 8/8 |
| 5250 | 1.3078 | 2.8784 | 3.5780 | +0.6997 | 8/8 |
| 5500 | 1.2602 | 2.8653 | 3.5788 | +0.7135 | 8/8 |
| 5750 | 1.2266 | 2.8609 | 3.5791 | +0.7182 | 8/8 |

### 9.2 Held-out (373 examples) and test generation

| Run log | Decode config | BERTScore F1 | ROUGE-L F1 | BLEU-4 | bs_acts | bs_hc | bs_sc | bs_trib | distinct | time |
|---|---|---|---|---|---|---|---|---|---|---|
| llama | greedy_rp1.2 | 0.8224 | 0.1829 | 0.0548 | 0.814 | 0.8305 | 0.8201 | 0.8226 | n/p | 60s |
| llama | greedy_ngram4 | 0.8309 | 0.1995 | 0.0509 | 0.8291 | 0.8362 | 0.8272 | 0.8305 | n/p | 60s |
| bart v1 | greedy_rp1.2 | 0.8047 | 0.1589 | 0.0136 | 0.7899 | 0.8114 | 0.8048 | 0.8086 | n/p | 26s |
| bart v1 | greedy_ngram4 | 0.8047 | 0.1697 | 0.009 | 0.7928 | 0.8119 | 0.805 | 0.8061 | n/p | 26s |
| bart v2 | greedy_rp1.2 | 0.8259 | 0.1755 | 0.04 | 0.8214 | 0.8291 | 0.8279 | 0.8241 | 372 of 373 | 26s |
| bart v2 | greedy_ngram4 | 0.826 | 0.1891 | 0.0458 | 0.8233 | 0.8299 | 0.8243 | 0.8258 | 373 of 373 | 26s |
| prefixlm | greedy_rp1.2 | 0.8182 | 0.173 | 0.0373 | 0.8091 | 0.8225 | 0.8197 | 0.8193 | n/p | 63s |
| prefixlm | greedy_ngram4 | 0.8272 | 0.1904 | 0.0387 | 0.8253 | 0.8311 | 0.8281 | 0.8239 | n/p | 62s |

n/p = not printed (the Oct-6 notebooks do not compute `distinct_outputs`).

| Run log | `CHOSEN` | Test time (1000 prompts) | Words per prediction (min / mean / max) | Distinct predictions | Output |
|---|---|---|---|---|---|
| llama | greedy_ngram4 | 155s | 78 / 127.6 / 128 | n/p | `/content/p1_train/gen_v1/submission.zip` |
| bart v1 | greedy_ngram4 | 68s | 128 / 128.0 / 128 | n/p | `/content/p1_train/gen_v1/submission.zip` |
| bart v2 | greedy_ngram4 | 69s | 1 / 127.4 / 128 | 999/1000 | `/content/p1_train/gen_v1/submission.zip` |
| prefixlm | greedy_ngram4 | 156s | 5 / 127.0 / 128 | n/p | `/content/p1_train/gen_v1/submission.zip` |

Statements about the v1 BART run that exist in the code (verbatim, from `BART_NOTE` in `build_arch_notebooks.py`):
"The v1 BART run (`cs781-p1-bart`) collapsed: the decoder ignored the encoder and wrote the same continuation for
every prompt." The v1 run log does not print a distinct-output count. The output of `bart_diagnostic_cell.py` is not in
this repo (`TODO(student)`).

---

## 10. Codabench submissions

Five Phase-1 submissions were made (deadline 2026-10-06 23:59 IST). They were uploaded as `submission.zip` without a
record of which model produced each one. The table matches every Codabench entry to the zip downloaded from Colab a few
minutes earlier and to the executed notebooks in `run_logs/`. "Words" is min / mean / max words per prediction counted
from the zip; the llama, bart v1 and prefixlm notebooks print the same line (`words per prediction: ...`) after
generating the test set. All four 6 Oct runs used greedy decoding with `no_repeat_ngram_size = 4` (`greedy_ngram4`,
chosen by the notebook from the held-out comparison).

| Codabench ID | Submitted (IST) | Run | Settings | BERTScore F1 | Evidence for the attribution |
|---|---|---|---|---|---|
| 946235 | 2026-09-26 15:08 | starter-code output (inferred; not our model) | course-provided pre-generated output | 0.8010 | The only Phase-1 prediction file on this machine from before 6 Oct is the starter kit's `submission.zip` (downloaded 24 Sep, byte-identical to `Starter Code/submission.zip`). Starter-code predictions do not count toward the minimum. Confirmed by the student. |
| 964531 | 2026-10-06 16:45 | **llama** (leaderboard entry) | step 2196, `greedy_ngram4` | **0.8294** | zip built 16:42; words 78 / 127.6 / 128, as printed in `P1_arch1_llama_after_run.ipynb` |
| 964554 | 2026-10-06 17:01 | bart **v1** (collapsed) | step 6744, `greedy_ngram4` | 0.8048 | zip built 16:57; all 1000 predictions are one identical 128-word text, which also appears in `P1_arch2_bart_after_run.ipynb` |
| 964576 | 2026-10-06 17:09 | prefixlm | step 1995, `greedy_ngram4` | 0.8269 | zip built 17:06; words 5 / 127.0 / 128, as printed in `P1_arch3_prefixlm_after_run.ipynb` |
| 965161 | 2026-10-06 21:32 | mistral_swa (inferred by elimination) | no executed notebook kept | 0.8265 | zip built 21:28; words 63 / 127.3 / 128 match none of the saved logs; its first 8 words equal llama's for 42 / 1000 prompts (prefixlm vs llama: 45 / 1000), so it is not a re-decode of the llama model with the same settings. Confirmed by the student. |

**BART v2 was never submitted in Phase 1.** The v1 collapse was fixed only after the Phase-1 and Phase-2 deadlines
(v2 template built 2026-10-14, `P1_arch2_bart_v2_after.ipynb` saved 2026-10-15). The Phase-1 leaderboard therefore
never saw a working BART, and the v2 numbers in section 9 are local held-out scores only. A Phase-1-format prediction
file from v2 exists locally (`submission_bart_phase1_retrained`, 2026-10-15) but there is no Codabench entry for it.

---

## 11. How to run

1. **Data (once, CPU runtime is enough).** Open `initial_pipeline/00_data_prep.ipynb` in Colab and add the secret
   `HF` = a Hugging Face **write** token (notebook access on). Edit the `SOURCES` target GB and other knobs if wanted,
   then Run all. It creates the private dataset repo `<hf-user>/cs781-p1-data` with the `.bin` files, `meta.json`,
   `tokenizer/` and `heldout_{prompts,truth}.json`.
2. **Render notebooks.** `python build_arch_notebooks.py` (all four) or `python build_arch_notebooks.py bart` (only
   one). This overwrites `P1_arch{N}_<arch>.ipynb` in this folder.
3. **Train (one Colab account per architecture, in parallel).**
   * Runtime: **T4 GPU**. Add the same `HF` secret.
   * Edit the "Configuration — all values below are YOUR CHOICE" cell, then Run all.
   * Part A trains for `TRAIN_HOURS`, pushing `latest/` every 30 min and `best/` on each new best mean val loss to
     `<hf-user>/cs781-p1-<arch>` (`-v2` for bart).
   * If Colab disconnects, reconnect and Run all again: it resumes from `latest/`. If training had already finished,
     it goes straight to Part B.
4. **Generate and submit.** Part B runs after training. It scores `DECODE_CONFIGS` on the held-out set, generates the
   test set with `CHOSEN`, and writes `/content/p1_train/gen_<RUN_TAG>/{prediction.json, submission.zip,
   heldout_pred_*.json, heldout_truth.json, heldout_scores.json, submission_record.json}`. It uploads them to
   `submissions/<RUN_TAG>/` in the model repo. Upload `submission.zip` to Codabench.
5. **Another submission from the same checkpoint** (e.g. a different decoding config): run the cells up to and
   including *Configuration*, change `RUN_TAG` / `DECODE_CONFIGS` / `CHOSEN`, and run Part B.
6. **Official check of a held-out prediction:**
   `python evaluate_phase1.py --truth heldout_truth.json --prediction heldout_pred_<config>.json` (files in
   `gen_<RUN_TAG>/`).
7. **Side-by-side reading:**
   `python3 tools/compare_submissions.py llama=.../submission.zip bart=.../submission.zip --prompts test.json`
   (or with `--truth heldout_truth.json` for held-out predictions).
8. **BART v1 diagnostic:** paste `bart_diagnostic_cell.py` into a fresh Colab notebook (GPU or CPU) with the `HF`
   secret and run it.
9. **Generation (a), for reference:** `00_data_prep` → `01_train` (set `ARCH`, Run all on each account) →
   `02_generate` (set `ARCH`, `RUN_TAG`). It writes to the same `cs781-p1-<arch>` repos (no `-v2` suffix).

Phase 1 has no knowledge-distillation step, so there is no teacher/student ordering.
