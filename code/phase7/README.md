# Phase 7: Prior Case Retrieval (PCR)

A bi-encoder built on the student's own Phase-1 language models: every query case and every candidate case is
encoded into one or more vectors, the candidate pool of the same split is ranked by similarity, and the top 7
candidate IDs are submitted. One percent-format template renders into four per-architecture Colab notebooks.

> All numbers in this README are copied from the executed notebooks in `run_logs/`. Nothing here interprets them.
> Gaps are marked `TODO(student)`.

---

## 1. Task

Course spec: capstone §13 (Phase 7, 3–7 November 2026).

- **Input:** a query case (IndianKanoon case ID, `text` = the judgment split into sentences) plus the candidate pool
  of the same split. Each split has its own pool: `train.json` goes with `train_candidates.json`, `dev.json` with
  `dev_candidates.json`, and `test.json` with `test_candidates.json`. Train and dev queries include
  `relevant_candidates`. Test queries do not.
- **Output:** `prediction.json`, a JSON object that maps each test query ID to a list of **exactly 7 unique** candidate
  IDs from `test_candidates.json`, ordered from highest to lowest predicted relevance. Every test query must be
  present. The file is submitted zipped as `submission.zip` containing only `prediction.json`.

  ```json
  { "0000021652": ["0000590455", "0000916840", "0001481813", "0001907199", "0000000987", "0000001566", "0000002886"],
    "0000030721": ["0000004354", "..."] }
  ```
- **Primary metric:** Micro-F1@7. **Secondary metrics:** Micro-Precision@7 and Micro-Recall@7. Only the first 7
  candidates count. They are compared against the complete gold set, which is neither truncated nor padded:
  $TP = \sum_q |G_q \cap P_q|$, $FP = \sum_q |P_q \setminus G_q|$, $FN = \sum_q |G_q \setminus P_q|$,
  $F1 = 2TP / (2TP + FP + FN)$. These are the definitions in the official `evaluate_phase7.py`.
- **Experiment requirements (§13.2):** at least 4 substantially different experiments. At least 3 of them must use
  different §13.2.1 techniques (full, partial / layer-wise, QLoRA, LoRMA, prefix tuning, prompt tuning, IA³, KD).
  **LoRMA is compulsory.** At least 1 experiment must be knowledge distillation from a larger or stronger teacher into
  a smaller student. Every experiment starts from the Phase-1 model.

## 2. Files in this folder

| File | Contents |
|---|---|
| `phase7_pcr_template.py` | The single source of the Phase-7 code: a percent-format script with `# %%` cells and `{{ARCH}}`, `{{TAG}}`, `{{FAMILY}}`, `{{MODEL}}` placeholders. |
| `build_p7_notebooks.py` | Fills the placeholders for each architecture and writes `P7_arch{1..4}_<arch>.ipynb`. `python build_p7_notebooks.py [arch ...]` renders all four notebooks or only the named ones. |
| `build_nb.py` | Generic converter from a percent-format `.py` file to `.ipynb` (`# %%` becomes a code cell, `# %% [markdown]` becomes a markdown cell). |
| `P7_arch1_llama.ipynb` | Rendered notebook for Arch 1, decoder-only (Phase-1 `LlamaForCausalLM` body). |
| `P7_arch2_bart.ipynb` | Rendered notebook for Arch 2, encoder-decoder (Phase-1 v2 `BartForConditionalGeneration`, `cs781-p1-bart-v2`). |
| `P7_arch3_prefixlm.ipynb` | Rendered notebook for Arch 3, prefix-LM hybrid (Phase-1 prefix-LM `LlamaModel`). |
| `P7_arch4_mistral_swa.ipynb` | Rendered notebook for Arch 4, sliding-window attention + GQA (Phase-1 `MistralForCausalLM` body). |
| `run_logs/P7_arch1_llama_after.ipynb` | Executed Colab notebook with outputs: run `llama_lorma_mean_infonce_random`. |
| `run_logs/P7_arch2_bart_after.ipynb` | Executed Colab notebook with outputs: run `bart_lorma_mean_infonce_random`. |
| `run_logs/P7_arch3_prefixlm_after.ipynb` | Executed Colab notebook with outputs: run `prefixlm_lorma_mean_infonce_random`. |

The four `P7_arch*.ipynb` files match a fresh render of the current template cell for cell (checked by
re-rendering and diffing). The code cells of the three run logs match the corresponding rendered notebooks.

## 3. Data and preprocessing

**Source.** Gated HF dataset `Exploration-Lab/CS-781-Capstone`, folder `Phase_7_PCR/`, files
`{Train,Dev,Test}/{train,dev,test}.json` and `{split}_candidates.json`.

- If the expected path is missing, `fetch()` searches the dataset for a single file with the same name under a
  folder containing `phase_7`, `phase7`, `phase 7` or `pcr`.
- JSON and JSON Lines are both accepted (`read_json_any`).
- Query files can be `{query ID: {text, relevant_candidates}}` or a list of records with an `id` / `query_id` / `qid`
  field. Candidate files can be `{candidate ID: text or {text}}` or a list of records.
- `text` (a list of sentences) is joined with `"\n"`.
- IDs are kept as the exact strings in the files. A warning is printed for IDs that are not JSON strings.
- Repeated IDs are dropped (the first one is kept). Blank relevant IDs are ignored, and a relevant ID repeated within
  one list is kept once. Both rules follow `evaluate_phase7.py`.
- The raw first record of each file is printed.

**Splits.** No extra held-out split is built:
- **train** is used for training.
- **dev** (labelled) is used for model selection and local evaluation.
- **test** is ranked for the submission.

**Dataset statistics** below are printed by the data cell of all three run logs (identical in each).

| split | queries | pool | relevant / query (mean) | min | max | queries with 0 relevant | relevant not in pool | queries in own pool | self relevant |
|---|---|---|---|---|---|---|---|---|---|
| train | 827 | 4320 | 6.42 | 0 | 52 | 10 | 0 | 827 | 0 |
| dev | 118 | 1023 | 8.25 | 1 | 92 | 0 | 0 | 118 | 0 |
| test | 237 | 1727 | – | – | – | – | – | 237 | – |

| documents (characters) | median | p90 | max | empty |
|---|---|---|---|---|
| train queries | 25893 | 72937 | 717260 | 0 |
| train candidates | 20848 | 71744 | 2425986 | 0 |
| dev queries | 26363 | 88437 | 619579 | 0 |
| dev candidates | 23207 | 97249 | 2425986 | 0 |
| test queries | 26411 | 82525 | 529766 | 0 |
| test candidates | 22627 | 78000 | 2425986 | 0 |

The data cell also prints:

```
shared IDs train/dev: candidates 436  queries 0  train queries in dev pool 124
shared IDs train/test: candidates 740  queries 0  train queries in test pool 269
shared IDs dev/test: candidates 220  queries 0  dev queries in test pool 43
train: 4107 distinct relevant candidates; most often relevant (ID: queries): 0000608874: 14  0001857950: 10  0000424874: 10  0000525017: 9  0001281050: 8  0001377254: 8  0001949249: 8  0000135147: 8
training queries with >= 1 relevant candidate in the training pool: 817 of 827;  training pairs: 5308
```

**Training pairs.** Each training pair is a (query, relevant candidate in the training pool), with the query's own ID
excluded. Queries with no such candidate are dropped, which leaves 817 queries and 5308 pairs.

**Tokenisation.**
- Each document is tokenised whole with the Phase-1 tokenizer (`PreTrainedTokenizerFast` from `best/`), with no
  special tokens.
- Token IDs are stored as `uint16`. An empty document becomes a single EOS token.
- Tokenised documents are cached in `/content/p7/cache/{split}_{q|c}_{md5(tokenizer.json)[:12]}.npz` and reused by
  later runs in the same runtime. The BART log shows every document "from cache".

**Chunking (long documents).**
- Chunk starts are `0, CHUNK_STRIDE, 2·CHUNK_STRIDE, …`. A final chunk ending exactly at the document end is added
  when needed, so every token is in at least one chunk.
- The defaults (`CHUNK_LEN = CHUNK_STRIDE = 512`) give non-overlapping chunks, except that the last chunk may
  overlap the one before it.
- `CHUNK_SELECT` keeps at most `MAX_CHUNKS` chunks (training, default 4) or `MAX_CHUNKS_PRED` chunks (encoding for
  ranking, default 8), in document order.
- Selection modes: `first`, `last`, `first_last`, `even` (`np.linspace(0, m-1, k).round()`), or `random`. `random`
  applies in training only. Encoding uses `even` instead.
- Chunks are right-padded with `PAD_ID` and carry an attention mask.

**Checks performed by the code:**
- Every token of every document is covered by its chunks.
- Pooling the token IDs themselves and regrouping returns each document's own chunks in order.

Both checks print `coverage and alignment checks passed`.

**Token and chunk statistics** (Phase-1 tokenizer, `CHUNK_LEN = 512`, `MAX_CHUNKS_PRED = 8`). The same values are
printed in all three logs.

| documents | tokens median | p90 | max | chunks median | chunks max | > MAX_CHUNKS_PRED | tokens seen when encoding |
|---|---|---|---|---|---|---|---|
| train queries | 5735 | 15949 | 149936 | 12 | 293 | 66.9% | 40.7% |
| train candidates | 4616 | 15584 | 518603 | 10 | 1013 | 55.5% | 40.1% |
| dev queries | 5713 | 19759 | 132964 | 12 | 260 | 65.3% | 33.6% |
| dev candidates | 5146 | 21147 | 518603 | 11 | 1013 | 60.2% | 30.2% |
| test queries | 5924 | 17679 | 114375 | 12 | 224 | 65.0% | 40.2% |
| test candidates | 4962 | 17287 | 518603 | 10 | 1013 | 58.8% | 37.0% |

**BM25 (built in, lexical, no pretrained weights).**
- Word-level Okapi BM25 implemented with scikit-learn `CountVectorizer`: lowercase, scikit-learn English stop-word
  list when `BM25_STOPWORDS`, default token pattern (words of at least 2 characters).
- Vocabulary, document frequencies and average length are computed from each split's own candidate pool.
- $\mathrm{IDF}(w) = \log(1 + (N - df + 0.5)/(df + 0.5))$. Term weight $= \mathrm{IDF}\cdot tf\cdot(k_1+1) /
  (tf + k_1(1 - b + b\cdot dl/\overline{dl}))$, with $k_1 = 1.2$ and $b = 0.75$.
- The query is its set of distinct words.
- Printed vocabularies: train 98217, dev 50923, test 64335 words.
- BM25 serves as a dev reference row, a hard-negative source, and an optional reranking / fusion input (off by
  default).

## 4. Model architecture

A shared bi-encoder: queries and candidates go through the same network.
`PCRModel(arch, ATTN, BART_MODE, layers)` works as follows.

| Arch | Loaded from | Backbone used | Token states | Default attention (`ATTN`) |
|---|---|---|---|---|
| `llama` | `<hf-user>/cs781-p1-llama/best/`, `LlamaForCausalLM` | `full.model` (`LlamaModel`) | `last_hidden_state` | `causal` |
| `prefixlm` | `<hf-user>/cs781-p1-prefixlm/best/`, `LlamaForCausalLM` | `full.model` (`LlamaModel`) | `last_hidden_state` | `bidirectional` (whole chunk as prefix) |
| `mistral_swa` | `<hf-user>/cs781-p1-mistral_swa/best/`, `MistralForCausalLM` | `full.model` (`MistralModel`) | `last_hidden_state` | `causal` (sliding window from the config) |
| `bart` | `<hf-user>/cs781-p1-bart-v2/best/`, `BartForConditionalGeneration` | `BART_MODE="encoder"`: `full.model.encoder`. `"decoder"`: `full.model`, with the chunk fed to both the encoder and the decoder (`decoder_input_ids = ids`) | encoder (or decoder) `last_hidden_state` | n/a (encoder is bidirectional) |

- Weights are loaded in float32 (`.float()`). The forward pass runs under fp16 autocast on the GPU. Pooling,
  projection and similarity run in float32.
- **Bidirectional mode for decoder-only models:** a 4-D boolean mask `(Q, 1, L, L)` that masks padding keys only. For
  `mistral_swa` it is additionally limited to the band $|i-j| <$ `sliding_window`.
- **Activation checkpointing** (`GRAD_CKPT = True`) is enabled on the backbone with `use_reentrant=False`.
- **Dropout:** Phase 7 has no dropout setting. Dropout in train mode is whatever the Phase-1 config stores (the Phase-1
  template default is `DROPOUT = 0.0`; `TODO(student)`: confirm for the trained checkpoints).
- **Phase-1 sizes** (from the Phase-1 template defaults, see `code/phase1`): $d_{model} = 768$, 12 blocks and 12 heads
  for llama / prefixlm / mistral_swa (mistral_swa: 4 KV heads, window 256), and 6 + 6 layers for BART. The Phase-7
  logs print a vector dim of 768. LoRMA wraps 12 `q_proj` + 12 `v_proj` for llama and prefixlm, and 6 + 6 for the
  BART encoder.
- **Parameter counts printed by the logs:**
  - llama and prefixlm: retrieval encoder `109.5M parameters`.
  - bart (encoder mode): `67.9M parameters`.

**Head and pooling.**
1. **Chunk vector** (`POOL`): mean of the token states over real tokens (default), or the first token, or the last
   real token.
2. **Document representation** (`REPR`):
   - `doc` gives one vector per document, pooling the chunk vectors with `DOC_POOL`:
     - `mean` (masked mean, default).
     - `max` (element-wise max).
     - `attn`: score of chunk $k$ = $w^\top \tanh(W c_k)$ with $W: H \to$ `ATT_HIDDEN` $=256$, softmax over the
       document's chunks, weighted sum (Yang et al., 2016).
     - `gru`: a bidirectional GRU (`GRU_HIDDEN` $=256$ per direction) over the chunk vectors in document order. The
       document vector is the concatenation of the two final states (dim 512).
   - `chunks` keeps one vector per chunk (multi-vector). `DOC_POOL` is unused.
3. **Projection** (`PROJ`): `none` (default), `linear` ($D \to$ `PROJ_DIM` $=256$), or `mlp` (Linear $D \to D$,
   GELU, Linear $D \to$ `PROJ_DIM`).
4. **Similarity** (`SIM`): `cos` (L2-normalise, then dot product; default) or `dot`. With `REPR = "chunks"` the score
   is max-sim: for each query chunk, the maximum over the candidate's chunks, then the mean over the query's chunks.
5. **Logits** = similarity / `TAU` (default 0.05). They are used by the losses, by distillation, and in the saved
   `train_scores.npz`.

**Training losses** (`LOSS`). They are computed on a micro-batch score matrix $S$ (rows = distinct queries, columns =
distinct candidates). A column counts as relevant to a row if it is in that query's full gold list. A column equal to
the query's own ID is masked to $-10^9$.
- `infonce` (default): one term per (query, positive) pair. Cross-entropy over all columns of the row, with the query's
  *other* relevant columns masked to $-10^9$:
  $\mathcal{L} = -\log \frac{e^{S_{q,c^+}}}{\sum_{c \notin G_q\setminus\{c^+\}} e^{S_{q,c}}}$.
- `listwise` (ListNet): target $t_{q,\cdot}$ uniform over the row's relevant columns;
  $\mathcal{L} = -\sum_c t_{q,c}\log\mathrm{softmax}(S_{q,\cdot})_c$, averaged over rows with at least one relevant
  column.
- `pairwise` (RankNet logistic): mean of $\mathrm{softplus}(S_{q,k} - S_{q,j})$ over all relevant $j$ and
  non-relevant, non-self $k$ in the row.
- `kd` adds a distillation term (see §5).

**Built-in sanity checks** (`model_checks`, run after model construction and again after `configure` for `qlora` and
`prompt`):
- **Attention direction:** changing a chunk's last token must change the first token's state only for
  bidirectional / BART models.
- **Padding invariance:** a document's vectors must not change when a longer document joins the batch.

Printed results:

```
llama (attn=causal, bart_mode=encoder): changing a chunk's last token moves its first token's state by 0.00e+00, its own by 5.72e+00 -> sees right context: False (expected False); vectors padding-invariant: True (max diff 1.3e-06)
bart (attn=n/a, bart_mode=encoder): changing a chunk's last token moves its first token's state by 1.32e-01, its own by 2.18e+00 -> sees right context: True (expected True); vectors padding-invariant: True (max diff 2.9e-07)
prefixlm (attn=bidirectional, bart_mode=encoder): changing a chunk's last token moves its first token's state by 8.98e-01, its own by 4.55e+00 -> sees right context: True (expected True); vectors padding-invariant: True (max diff 1.3e-06)
```

## 5. Fine-tuning methods implemented

`METHOD` is one of `full | partial | qlora | lorma | prompt | ia3 | kd`. The template maps them to the §13.2.1 list.
Prefix tuning is **not** implemented. All adapters are implemented in plain PyTorch inside the template: no `peft`,
no `bitsandbytes`. The document-pooling (`attn` / `gru`) and projection parameters are always trainable. For
`qlora`, `lorma` and `ia3`, the code asserts that the adapted model's probe vectors equal the unadapted ones at
initialisation (max abs diff below $10^{-3}\cdot\max(1, |ref|)$).

| Method | Trainable parameters | Implementation | Knobs |
|---|---|---|---|
| `full` | every parameter | `requires_grad = True` everywhere | `LR` (2e-5) |
| `partial` | top `PARTIAL_TOP` transformer blocks + final norm module(s) (`norm` / `layer_norm`, if present) + head | other blocks frozen. For BART in decoder mode the block list is encoder blocks followed by decoder blocks. | `PARTIAL_TOP` (4), `LR` (5e-5) |
| `qlora` (Dettmers et al., 2023) | LoRA `A`, `B` on `ADAPT_TARGETS` | Every `nn.Linear` of the backbone is replaced by a frozen `NF4Linear`: 4-bit NormalFloat codes, two per byte. The 16 levels are quantiles of N(0,1) (offset 0.9677083) plus an exact 0, scaled to [-1, 1], with one absmax per `QLORA_BLOCK` weights. With `QLORA_DOUBLE_QUANT`, the absmax values minus their mean are stored as int8 in blocks of 256, each with a float scale. Weights are dequantised on the fly in the forward pass and again in the backward pass (custom autograd function, fp16 compute). LoRA: $h = W_{nf4}x + (\alpha/r)B(Ax)$ with $B = 0$ at init. No paged optimizer. Stays quantised after training. | `ADAPT_TARGETS` (`q_proj`, `v_proj`), `RANK` 8, `ALPHA` 16, `QLORA_BLOCK` 64, `QLORA_DOUBLE_QUANT` True, `LR` (5e-4) |
| `lorma` (Bihany, Patel & Modi, 2025) | `A` ($r\times d_{out}$), `B` ($d_{out}\times r$) per target | Multiplicative update $h = \mathcal{I}(BA)\,W_0x$ (+ bias afterwards). `plus`: $\mathcal{I}(BA) = I + (\alpha/r)BA$, computed as $W_0x + (\alpha/r)B(A(W_0x))$; `A` uses Kaiming-uniform init and $B = 0$. `pi`: $\mathcal{I}(BA)$ rotates row $i$ of $BA$ cyclically by $i$; the init (first column of `B` = 1, $A = e_0e_0^\top$) gives $\mathcal{I} = I$. Merged into plain `nn.Linear` weights after training. | `ADAPT_TARGETS`, `RANK` 8, `ALPHA` 16, `LORMA_VARIANT` (`plus`), `LR` (5e-4) |
| `ia3` (Liu et al., 2022) | one vector per target (init 1) | `IA3_OUT` layers: $l \odot (Wx + b)$. `IA3_IN` layers: $W(l\odot x) + b$. Merged into `nn.Linear` after training. | `IA3_OUT` (`k_proj`, `v_proj`), `IA3_IN` (`down_proj`, `fc2`), `LR` (3e-3) |
| `prompt` (Lester et al., 2021) | `PROMPT_LEN` × $d_{model}$ soft tokens | Prepended to every chunk in input-embedding space. Initialised from the embeddings of random vocabulary IDs (seed `SEED`). Prompt positions are dropped before pooling. Backbone frozen. BART: `BART_MODE = "encoder"` only. | `PROMPT_LEN` 20, `LR` (3e-3) |
| `kd` | every parameter of the student | See below. | `TEACHER_RUN`, `STUDENT_LAYERS`, `KD_T`, `KD_ALPHA`, `NEG_SOURCE="teacher"`, `LR` (5e-5) |

After training, LoRMA and IA³ are merged with a numerical check: max abs change of the probe vectors below
$10^{-2}\cdot\max(1,|before|)$. The logs print `adapter merge: max |vector change| = 7.15e-07` (llama), `1.19e-07`
(bart) and `7.45e-07` (prefixlm).

**Knowledge distillation (`METHOD = "kd"`).**
- **Teacher:** any finished Phase-7 run in `<hf-user>/cs781-p7`, of any architecture, that saved
  `runs/<TEACHER_RUN>/train_scores.npz`. This file holds `query_ids`, `cand_ids`, and `logits` = teacher similarity /
  teacher `TAU` for every training query × every training candidate. The teacher's `run_config.json` is also read
  (arch, method, params, dev F1@7, dev encoding time).
- The code asserts that the teacher's candidate set equals this run's training pool. It reorders the teacher columns
  into this run's pool order and keeps only training queries that have teacher scores.
- **Student:** the notebook's own `ARCH` built from the same Phase-1 checkpoint with fewer blocks. With
  `STUDENT_LAYERS = None`, every 2nd block (indices 0, 2, 4, …) of each stack is kept. Layer indices are renumbered,
  and `num_hidden_layers` / `layer_types` are updated. All student parameters are trained.
- **Loss:** $\mathcal{L} = \alpha\,\mathcal{L}_{LOSS} + (1-\alpha)\,T^2\,\mathrm{KL}(p_t\Vert p_s)$, with
  $p = \mathrm{softmax}(\text{logits}/T)$ over the micro-batch's candidate columns (self masked), $\alpha$ =
  `KD_ALPHA` (0.5) and $T$ = `KD_T` (1.0).
- `NEG_SOURCE = "teacher"` draws hard negatives from the teacher's top `NEG_DEPTH` non-relevant candidates.
- The template default `TEACHER_RUN` is `"llama_full_mean_infonce_random"`. If `TEACHER_RUN` is not a finished run,
  the notebook raises an error listing the finished runs.

## 6. Training procedure

- **Micro-batch:**
  - `PAIRS_PER_BATCH` (4) (query, relevant candidate) pairs, taken in order from a per-epoch permutation of all
    training pairs (`np.random.default_rng(SEED)`).
  - For each pair, `HARD_NEGS` (1) negatives from `NEG_SOURCE` (`NEG_RNG = default_rng(SEED + 1)`); never a relevant
    candidate of that query and never the query itself:
    - `random`: uniform over the training pool.
    - `bm25`: uniform over the top `NEG_DEPTH` (100) non-relevant BM25 candidates.
    - `teacher`: uniform over the teacher's top `NEG_DEPTH`.
    - `none`: in-batch negatives only.
  - Distinct queries (rows) and distinct candidates (columns) are encoded in one forward pass, `MAX_CHUNKS` chunks
    each. Every row is scored against every column (in-batch negatives).
  - Logged: `5308 training pairs -> 1327 micro-batches per epoch, 6635 optimizer steps for 5 epochs; a micro-batch encodes up to 4 queries + 8 candidates x 4 chunks`.
- **Optimiser:** AdamW, betas (0.9, 0.999), `fused=True` on CUDA. Parameter groups by learning rate (`LR` for the
  backbone / adapters, `HEAD_LR` = 1e-3 for doc pooling / projection) and by weight decay (`WEIGHT_DECAY` = 0.01 for
  tensors with dim ≥ 2, 0 otherwise).
- **Per-method LR defaults:** full 2e-5, partial 5e-5, qlora 5e-4, lorma 5e-4, prompt 3e-3, ia3 3e-3, kd 5e-5.
- **Schedule:** linear warmup over the first `WARMUP_FRAC` (0.06) of progress, then linear decay to 0. Progress =
  $\max(\text{step}/\text{TOTAL\_STEPS},\ \text{train seconds}/(60\cdot\text{MAX\_TRAIN\_MINUTES}))$, so the schedule
  finishes by the time cap. The LR is set at the start of each accumulation group.
- **Precision:** fp16 autocast + `torch.amp.GradScaler`, with float32 master weights.
- **Other settings:** gradient clipping at `GRAD_CLIP` = 1.0 (global norm). `GRAD_ACCUM` = 1; in-batch negatives
  always come from one micro-batch.
- **Time cap:** `MAX_TRAIN_MINUTES` = 60 minutes of training time, with dev evaluations excluded. `EPOCHS` = 5 is the
  upper bound.
- **Dev evaluation:**
  - Runs every `EVAL_EVERY_MINUTES` (10) training minutes (except on an epoch's last micro-batch), at the end of every
    epoch, and when the time cap is reached.
  - Each evaluation encodes the whole dev pool and the dev queries and computes Micro-F1@7 of the submitted ranking
    (`final_scores`, so fusion / reranking is included when enabled). The model-only F1 is recorded too.
- **Model selection:** the state dict with the highest dev Micro-F1@7 is copied to CPU (`best_state`). Training stops
  after `PATIENCE` (3) consecutive evaluations without improvement, or at the time cap.
- **Checkpoint / resume:** there is no mid-run checkpoint and no resume. Every run starts from the Phase-1 checkpoint.
  Fine-tuned weights are written (`model.pt`) only if `SAVE_WEIGHTS = True` (default False).
- **Throughput line:** printed after 20 optimizer steps, e.g. `throughput 2.21 pairs/s -> about 7950 pairs in MAX_TRAIN_MINUTES (5308 per epoch)` (llama).

## 7. Inference and submission

1. Load `best_state`, merge LoRMA / IA³ adapters (with the numerical check above), and switch to eval mode.
2. **Encode a split** (`run_split`):
   - Every candidate, then every query, is encoded in batches of `PRED_BATCH_DOCS` (8) documents under fp16
     autocast, with `MAX_CHUNKS_PRED` (8) chunks selected by `CHUNK_SELECT` (`random` becomes `even`).
   - The (queries × pool) similarity matrix is computed on the GPU in blocks of 16 queries × 2048 candidates.
   - Recorded: encoding seconds, candidates/s, queries/s, scoring ms/query, number of index vectors, index size in
     fp16, and peak GPU memory.
3. **Final scores** (`final_scores`):
   - `FUSION_WEIGHT` > 0: $(1-w)\,z(\text{model}) + w\,z(\text{BM25})$, where $z$ is standardised per query.
   - `RERANK_TOPN` = N: the model reorders BM25's top N, and the rest follow in BM25 order.
   - Both are off by default (`FUSION_WEIGHT = 0.0`, `RERANK_TOPN = None`).
   - `EXCLUDE_SELF = True` sets a query's own ID to $-\infty$. Every query is in its own split's pool (table in §3).
   - The ranking is a stable descending argsort, and the top 7 IDs are taken.
4. **Dev report:**
   - `rank_metrics` reproduces `evaluate_phase7.py`'s Micro-P/R/F1@7. The code asserts equality with
     `official_scores` (a copy of the official `score()`) to $10^{-12}$.
   - Additional printed diagnostics: Recall@20/50/100 over all relevant IDs, MRR of the first relevant candidate,
     queries with no hit in the top 7, and breakdowns by number of relevant candidates, by query-length quartile, by
     the rank bin of each relevant candidate, by hub candidates, and by overlap with BM25's top 7.
   - Files written: `dev_truth.json`, `dev_pred.json`, `dev_candidate_ids.json` (official format),
     `dev_details.json` (per query) and `dev_scores.npz`.
5. **Test:**
   - The test pool and queries are ranked and passed through `validate()` with the official rules: every query,
     exactly 7 unique IDs, all from `test_candidates.json`.
   - The cell writes `prediction.json` (indent 1) and `submission_<RUN_NAME>.zip`, which contains only
     `prediction.json` (this is asserted).
6. **Distillation targets:** with `SAVE_TRAIN_SCORES = True` (default), the whole training pool and all training
   queries are encoded and `train_scores.npz` is saved. The model's training-query metrics are printed.
7. **Upload and download:**
   - `run_config.json` (config, params, timings, dev metrics, references, history, breakdowns) and `curves.png` are
     saved.
   - The whole run directory is uploaded to `<hf-user>/cs781-p7/runs/<RUN_NAME>/` (commit message
     `<RUN_NAME> dev F1@7 x.xxxx`, 3 retries).
   - The zip is downloaded through `google.colab.files`.
8. **Run summary:** the last cell tabulates every run in `cs781-p7` and prints which §13.2.1 techniques have runs.

**Local official check** (the command is printed by the notebook):
```bash
python evaluate_phase7.py --truth dev_truth.json --prediction dev_pred.json --candidate-ids dev_candidate_ids.json
```

## 8. Configuration knobs

The first code cell plus the cell under *Configuration — **all values below are YOUR CHOICE***. The template states:
"The numbers filled in are placeholders that make the notebook run end to end on a T4, not tuned values."

| Name | Template default | Meaning |
|---|---|---|
| `ARCH` | per notebook | `llama` / `bart` / `prefixlm` / `mistral_swa` (fixed by the builder) |
| `METHOD` | `"lorma"` | `full`, `partial`, `qlora`, `lorma`, `prompt`, `ia3`, `kd` |
| `REPR` | `"doc"` | `doc` (one vector per document) or `chunks` (one per chunk, max-sim) |
| `DOC_POOL` | `"mean"` | `REPR="doc"`: `mean`, `max`, `attn`, `gru` |
| `LOSS` | `"infonce"` | `infonce`, `listwise`, `pairwise` |
| `NEG_SOURCE` | `"random"` | `random`, `bm25`, `teacher` (kd only), `none` |
| `RUN_NAME` | `f"{ARCH}_{METHOD}_{DOC_POOL or 'maxsim'}_{LOSS}_{NEG_SOURCE}"` | run folder name in `cs781-p7` |
| `DATASET` / `DATA_DIR` | `Exploration-Lab/CS-781-Capstone` / `Phase_7_PCR` | data location |
| `P1_REPOS` | `cs781-p1-{llama,bart-v2,prefixlm,mistral_swa}` | Phase-1 checkpoint per arch (`best/`) |
| `P7_REPO` | `<hf-user>/cs781-p7` | output repo |
| `WORK`, `RUN_DIR`, `CACHE_DIR` | `/content/p7`, `/content/p7/<RUN_NAME>`, `/content/p7/cache` | local paths |
| `SEED` | 42 | Python / NumPy / torch seed; base for chunk / negative / epoch RNGs |
| `CHUNK_LEN` | 512 | tokens per chunk (Phase-1 positions: 1024; BART encoder pretrained on ≤ 768 tokens) |
| `CHUNK_STRIDE` | 512 | tokens between chunk starts (< `CHUNK_LEN` gives overlap) |
| `CHUNK_SELECT` | `"even"` | `first`, `last`, `first_last`, `even`, `random` (random in training only) |
| `MAX_CHUNKS` | 4 | chunks per document in training |
| `MAX_CHUNKS_PRED` | 8 | chunks per document when encoding for ranking |
| `POOL` | `"mean"` | token states to chunk vector: `mean`, `first`, `last` |
| `ATT_HIDDEN` | 256 | `DOC_POOL="attn"` scorer hidden size |
| `GRU_HIDDEN` | 256 | `DOC_POOL="gru"` hidden size per direction |
| `PROJ` / `PROJ_DIM` | `"none"` / 256 | `none`, `linear`, `mlp` projection and its output dim |
| `ATTN` | llama `causal`, prefixlm `bidirectional`, mistral_swa `causal`, bart `n/a` | attention inside a chunk for decoder-only backbones |
| `BART_MODE` | `"encoder"` | `encoder` states, or `decoder` (chunk to encoder and decoder) |
| `SIM` | `"cos"` | `cos` or `dot` |
| `TAU` | 0.05 | logits = similarity / TAU |
| `EXCLUDE_SELF` | True | never rank a query's own ID |
| `RERANK_TOPN` | None | N: model reorders BM25's top N |
| `FUSION_WEIGHT` | 0.0 | weight of z(BM25) in the submitted score |
| `BM25_K1`, `BM25_B` | 1.2, 0.75 | Okapi parameters |
| `BM25_STOPWORDS` | True | drop scikit-learn English stop words |
| `ZERO_SHOT_EVAL` | True | dev score of the untrained model before training |
| `PAIRS_PER_BATCH` | 4 | (query, relevant) pairs per micro-batch |
| `HARD_NEGS` | 1 | extra negatives per pair |
| `NEG_DEPTH` | 100 | top-k non-relevant pool for `bm25` / `teacher` negatives |
| `EPOCHS` | 5 | passes over the training pairs |
| `MAX_TRAIN_MINUTES` | 60 | training-time cap (dev evals excluded) |
| `EVAL_EVERY_MINUTES` | 10 | dev evaluation interval |
| `PATIENCE` | 3 | evaluations without improvement before stopping |
| `GRAD_ACCUM` | 1 | micro-batches per optimizer step |
| `PRED_BATCH_DOCS` | 8 | documents per forward pass when encoding |
| `GRAD_CKPT` | True | activation checkpointing |
| `LR` | full 2e-5, partial 5e-5, qlora 5e-4, lorma 5e-4, prompt 3e-3, ia3 3e-3, kd 5e-5 | backbone / adapter LR |
| `HEAD_LR` | 1e-3 | LR of doc pooling (`attn` / `gru`) and projection |
| `WEIGHT_DECAY` | 0.01 | AdamW weight decay (tensors with dim ≥ 2) |
| `WARMUP_FRAC` | 0.06 | warmup fraction of progress |
| `GRAD_CLIP` | 1.0 | global gradient-norm clip |
| `PARTIAL_TOP` | 4 | partial: top blocks trained |
| `ADAPT_TARGETS` | `["q_proj", "v_proj"]` | qlora / lorma target layers |
| `RANK`, `ALPHA` | 8, 16 | qlora / lorma rank and scale α/r |
| `QLORA_BLOCK` | 64 | weights per NF4 block |
| `QLORA_DOUBLE_QUANT` | True | 8-bit absmax constants |
| `LORMA_VARIANT` | `"plus"` | `plus` (additive identity) or `pi` (permutation) |
| `IA3_OUT` | `["k_proj", "v_proj"]` | ia3 output-rescaled layers |
| `IA3_IN` | `["down_proj", "fc2"]` | ia3 input-rescaled layers |
| `PROMPT_LEN` | 20 | soft tokens per chunk |
| `TEACHER_RUN` | `"llama_full_mean_infonce_random"` | kd teacher run in `cs781-p7` |
| `STUDENT_LAYERS` | None | None = every 2nd block per stack, or an explicit index list |
| `KD_T` | 1.0 | distillation temperature |
| `KD_ALPHA` | 0.5 | weight of the ranking loss (1 − α on the KD term) |
| `SAVE_TRAIN_SCORES` | True | save `train_scores.npz` (teacher file for later kd runs) |
| `SAVE_WEIGHTS` | False | upload fine-tuned weights (`model.pt`) |
| `K_EVAL` | 7 | official cut-off (constant) |

## 9. Runs recorded in `run_logs/`

All three executed notebooks have code cells identical to the rendered template, so **no configuration value differs
from the template defaults** apart from `ARCH`. Their shared settings are:
- `METHOD="lorma"` (`plus`, rank 8, α 16, `q_proj` / `v_proj`), `REPR="doc"`, `DOC_POOL="mean"`, `POOL="mean"`,
  `PROJ="none"`, `SIM="cos"`, `TAU=0.05`.
- `LOSS="infonce"`, `NEG_SOURCE="random"`, `HARD_NEGS=1`, `PAIRS_PER_BATCH=4`.
- `CHUNK_LEN=CHUNK_STRIDE=512`, `MAX_CHUNKS=4`, `MAX_CHUNKS_PRED=8`, `CHUNK_SELECT="even"`.
- `LR=5e-4`, `EPOCHS=5`, `MAX_TRAIN_MINUTES=60`, `PATIENCE=3`, and fusion / reranking off.

Environment printed: `2.14.1+cu130 Tesla T4`, `transformers 5.22.1`.

### Summary

| File | RUN_NAME | Arch (ATTN / mode) | Method | Dev F1@7 | Dev P@7 | Dev R@7 | Trainable / total params | Train min | Peak GPU (train) | Stop |
|---|---|---|---|---|---|---|---|---|---|---|
| `P7_arch1_llama_after.ipynb` | `llama_lorma_mean_infonce_random` | llama (causal) | lorma | 0.3235 | 0.3523 | 0.2991 | 294,912 / 109,824,768 (0.27%) | 60.0 (wall 70.6) | 2.36 GB | time cap |
| `P7_arch2_bart_after.ipynb` | `bart_lorma_mean_infonce_random` | bart (encoder) | lorma | 0.3369 | 0.3668 | 0.3114 | 147,456 / 68,089,344 (0.22%) | 31.4 (wall 33.7) | 1.74 GB | no improvement in 3 evaluations |
| `P7_arch3_prefixlm_after.ipynb` | `prefixlm_lorma_mean_infonce_random` | prefixlm (bidirectional) | lorma | 0.3091 | 0.3366 | 0.2857 | 294,912 / 109,824,768 (0.27%) | 60.0 (wall 70.5) | 2.39 GB | time cap |

Dev values are from the `DEV submitted ranking:` line printed after the adapter merge. For prefixlm, the training
cell's `best:` line prints `dev F1@7 0.3079 at epoch 2 step 1960`, while the post-merge `DEV submitted ranking:` line
prints `F1@7 0.3091`.

### Full dev metrics and efficiency (verbatim)

| Run | R@20 | R@50 | R@100 | MRR | no hit in top 7 | TP / FP / FN | Selected at | Params after merge | Model MB | Dev encode (1023 cand + 118 q) | cand/s | Scoring | Peak GPU (dev encode) | Test encode | `train_scores` encode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| llama | 0.4892 | 0.6300 | 0.7215 | 0.7121 | 18 / 118 | 291 / 535 / 682 | epoch 2 step 1987 | 109.5M | 418 | 85.9 s | 13.4 | 0.03 ms/query | 1.47 GB | 146.1 s | 381 s |
| bart | 0.4872 | 0.6197 | 0.6978 | 0.7449 | 16 / 118 | 303 / 523 / 670 | epoch 1 step 846 | 67.9M | 259 | 31.3 s | 36.8 | 0.02 ms/query | 1.25 GB | 53.4 s | 138 s |
| prefixlm | 0.4758 | 0.6238 | 0.7081 | 0.7085 | 16 / 118 | 278 / 548 / 695 | epoch 2 step 1960 | 109.5M | 418 | 85.7 s | 13.4 | 0.03 ms/query | 1.47 GB | 146.9 s | 380 s |

All three print an index of `1023 vectors x 768 = 1.5 MB in fp16`.

- **Training throughput:** llama `2.21 pairs/s`, bart `5.65 pairs/s`, prefixlm `2.14 pairs/s`.
- **Test lines:**
  - llama: `test: 237 queries x 7 candidates from a pool of 1727; 1012 distinct candidates returned, most often 8 times`.
  - bart: `969 distinct candidates returned, most often 8 times`.
  - prefixlm: `1000 distinct candidates returned, most often 6 times`.

### Dev reference rows (verbatim from the logs)

The BM25, random and oracle rows are identical in all three logs.

| Row | F1@7 | P@7 | R@7 | R@20 | R@50 | R@100 | MRR | no hit in top 7 |
|---|---|---|---|---|---|---|---|---|
| random top 7 (expected) | 0.0074 | 0.0081 | 0.0068 | – | – | – | – | – |
| oracle (as many relevant in-pool IDs as possible) | 0.6181 | 0.6731 | 0.5714 | – | – | – | – | – |
| BM25 alone | 0.1123 | 0.1223 | 0.1038 | 0.1644 | 0.2415 | 0.3422 | 0.3618 | 57 / 118 |
| untrained llama (Phase-1 backbone, adapters at identity) | 0.2624 | 0.2857 | 0.2425 | 0.4162 | 0.5745 | 0.6711 | 0.6279 | 23 / 118 |
| untrained bart encoder | 0.2401 | 0.2615 | 0.2220 | 0.3669 | 0.4985 | 0.6053 | 0.5704 | 30 / 118 |
| untrained prefixlm | 0.2479 | 0.2700 | 0.2292 | 0.4090 | 0.5663 | 0.6691 | 0.5943 | 22 / 118 |

The rows are computed as follows:
- **random:** expected TP = $\sum_q 7|G_q|/(\text{pool}-1)$.
- **oracle:** TP = $\sum_q \min(7, |G_q|)$.
- **BM25:** the code asserts it equals the `evaluate_phase7.py` computation.

Training-split rows:
```
train references -- BM25 alone: F1@7 0.1117  P@7 0.1071  R@7 0.1168  R@20 0.1777  R@50 0.2415  R@100 0.3092  MRR 0.3303  (no hit in top 7: 426 of 827 queries)
llama    train_scores.npz: 827 queries x 4320 candidates (381s);  this model on the training queries: F1@7 0.2110  P@7 0.2023  R@7 0.2206  R@20 0.3830  R@50 0.5482  R@100 0.6564  MRR 0.4815  (no hit in top 7: 239 of 827 queries)
bart     train_scores.npz: 827 queries x 4320 candidates (138s);  this model on the training queries: F1@7 0.2132  P@7 0.2044  R@7 0.2229  R@20 0.3694  R@50 0.5268  R@100 0.6366  MRR 0.4726  (no hit in top 7: 250 of 827 queries)
prefixlm train_scores.npz: 827 queries x 4320 candidates (380s);  this model on the training queries: F1@7 0.2064  P@7 0.1978  R@7 0.2157  R@20 0.3751  R@50 0.5317  R@100 0.6494  MRR 0.4635  (no hit in top 7: 261 of 827 queries)
```

### Dev evaluations during training (verbatim)

**llama_lorma_mean_infonce_random**
```
epoch 1 step 330 (10.0 train min): loss 0.8589  dev F1@7 0.3046  P@7 0.3317  R@7 0.2816  R@100 0.7163  MRR 0.6785  [91s]
epoch 1 step 661 (20.0 train min): loss 0.7976  dev F1@7 0.3068  P@7 0.3341  R@7 0.2837  R@100 0.7081  MRR 0.7235  [91s]
epoch 1 step 991 (30.0 train min): loss 0.7342  dev F1@7 0.3124  P@7 0.3402  R@7 0.2888  R@100 0.7133  MRR 0.7160  [90s]
epoch 1 step 1324 (40.0 train min): loss 0.7473  dev F1@7 0.3168  P@7 0.3450  R@7 0.2929  R@100 0.7133  MRR 0.7170  [90s]
epoch 1 step 1327 (40.1 train min): loss 0.4994  dev F1@7 0.3202  P@7 0.3487  R@7 0.2960  R@100 0.7143  MRR 0.7167  [90s]
epoch 2 step 1659 (50.2 train min): loss 0.6586  dev F1@7 0.3113  P@7 0.3390  R@7 0.2878  R@100 0.7153  MRR 0.7223  [90s]
epoch 2 step 1987 (60.0 train min): loss 0.6292  dev F1@7 0.3235  P@7 0.3523  R@7 0.2991  R@100 0.7215  MRR 0.7121  [90s]
stopping: time cap
best: dev F1@7 0.3235 at epoch 2 step 1987;  train time 60.0 min (wall 70.6 min);  peak GPU memory 2.36 GB
```

**bart_lorma_mean_infonce_random**
```
epoch 1 step 846 (10.0 train min): loss 0.9167  dev F1@7 0.3369  P@7 0.3668  R@7 0.3114  R@100 0.6978  MRR 0.7449  [35s]
epoch 1 step 1327 (15.7 train min): loss 0.7288  dev F1@7 0.3335  P@7 0.3632  R@7 0.3083  R@100 0.7071  MRR 0.7172  [35s]
epoch 2 step 2174 (25.7 train min): loss 0.6786  dev F1@7 0.3346  P@7 0.3644  R@7 0.3094  R@100 0.7112  MRR 0.7032  [35s]
epoch 2 step 2654 (31.4 train min): loss 0.6979  dev F1@7 0.3291  P@7 0.3584  R@7 0.3042  R@100 0.7071  MRR 0.7318  [35s]
stopping: no improvement in 3 evaluations
best: dev F1@7 0.3369 at epoch 1 step 846;  train time 31.4 min (wall 33.7 min);  peak GPU memory 1.74 GB
```

**prefixlm_lorma_mean_infonce_random**
```
epoch 1 step 327 (10.0 train min): loss 0.8959  dev F1@7 0.2924  P@7 0.3184  R@7 0.2703  R@100 0.7050  MRR 0.6505  [90s]
epoch 1 step 653 (20.0 train min): loss 0.8286  dev F1@7 0.3013  P@7 0.3281  R@7 0.2785  R@100 0.6927  MRR 0.6864  [90s]
epoch 1 step 980 (30.0 train min): loss 0.7622  dev F1@7 0.3046  P@7 0.3317  R@7 0.2816  R@100 0.7050  MRR 0.6938  [89s]
epoch 1 step 1309 (40.1 train min): loss 0.8045  dev F1@7 0.3046  P@7 0.3317  R@7 0.2816  R@100 0.7112  MRR 0.7073  [90s]
epoch 1 step 1327 (40.6 train min): loss 0.8757  dev F1@7 0.3057  P@7 0.3329  R@7 0.2826  R@100 0.7102  MRR 0.7100  [90s]
epoch 2 step 1653 (50.6 train min): loss 0.6807  dev F1@7 0.3035  P@7 0.3305  R@7 0.2806  R@100 0.7050  MRR 0.6951  [90s]
epoch 2 step 1960 (60.0 train min): loss 0.6606  dev F1@7 0.3079  P@7 0.3354  R@7 0.2847  R@100 0.7081  MRR 0.7085  [90s]
stopping: time cap
best: dev F1@7 0.3079 at epoch 2 step 1960;  train time 60.0 min (wall 70.5 min);  peak GPU memory 2.39 GB
```

### Dev breakdowns printed by the logs (verbatim values)

**By number of relevant candidates of the dev query** (F1@7 / P@7 / R@7 / queries with no hit):

| relevant | queries | llama | bart | prefixlm |
|---|---|---|---|---|
| 1-2 | 33 | 0.2199 / 0.1342 / 0.6078 / 11 | 0.2057 / 0.1255 / 0.5686 / 11 | 0.2270 / 0.1385 / 0.6275 / 9 |
| 3-5 | 30 | 0.3442 / 0.2762 / 0.4567 / 3 | 0.3858 / 0.3095 / 0.5118 / 2 | 0.3442 / 0.2762 / 0.4567 / 4 |
| 6-10 | 27 | 0.4655 / 0.4815 / 0.4505 / 2 | 0.4706 / 0.4868 / 0.4554 / 3 | 0.4552 / 0.4709 / 0.4406 / 1 |
| 11+ | 28 | 0.2814 / 0.5663 / 0.1872 / 2 | 0.2966 / 0.5969 / 0.1973 / 0 | 0.2510 / 0.5051 / 0.1669 / 2 |

**By dev query length (token quartiles)** (F1@7 / R@100):

| tokens | queries | llama | bart | prefixlm |
|---|---|---|---|---|
| 733–3412 | 30 | 0.2930 / 0.7885 | 0.2739 / 0.7692 | 0.2866 / 0.7596 |
| 3412–5713 | 29 | 0.3846 / 0.8519 | 0.4260 / 0.8444 | 0.3728 / 0.8296 |
| 5713–10113 | 29 | 0.4237 / 0.8475 | 0.4419 / 0.8305 | 0.4146 / 0.8390 |
| 10113–132964 | 30 | 0.2458 / 0.6124 | 0.2571 / 0.5803 | 0.2232 / 0.6024 |

**Rank of the 973 relevant dev candidates:**

| rank bin | 1-7 | 8-20 | 21-50 | 51-100 | >100 | not in pool |
|---|---|---|---|---|---|---|
| llama | 291 | 185 | 137 | 89 | 271 | 0 |
| bart | 303 | 171 | 129 | 76 | 294 | 0 |
| prefixlm | 278 | 185 | 144 | 82 | 284 | 0 |

**Other dev diagnostics:**

| Run | distinct candidates in dev top-7s (of 1023) | mean overlap with BM25's top 7 |
|---|---|---|
| llama | 559 | 0.143 |
| bart | 542 | 0.150 |
| prefixlm | 532 | 0.142 |

The hub-candidate lists (the candidates that appear in the most dev top-7s) are printed in each log's dev-report cell.

### Run table printed by the last cell of `P7_arch1_llama_after.ipynb` (all runs in `cs781-p7` at that time)

```
run                                          arch        method   repr   loss     neg      params  trainable     MB    F1@7     P@7     R@7   R@100 model F1 train min  cand/s
bart_lorma_mean_infonce_random               bart        lorma    mean   infonce  random    67.9M     0.147M    259  0.3369  0.3668  0.3114  0.6978   0.3369      31.4    36.8
llama_lorma_mean_infonce_random              llama       lorma    mean   infonce  random   109.5M     0.295M    418  0.3235  0.3523  0.2991  0.7215   0.3235      60.0    13.4
prefixlm_lorma_mean_infonce_random           prefixlm    lorma    mean   infonce  random   109.5M     0.295M    418  0.3091  0.3366  0.2857  0.7081   0.3091      60.0    13.4

Phase-1 models with runs: ['bart', 'llama', 'prefixlm']  (missing: ['mistral_swa'])
§13.2.1 techniques with runs: ['lorma']  (non-KD: 1)  LoRMA: True  KD: False
```

**Notes on the log files:**
- In `P7_arch2_bart_after.ipynb`, code cells 1–22 have execution counts 52–63 and cells 24–32 have execution counts
  47–51.
- No run log is present for `mistral_swa`, for the `listwise` run, or for any `full` / `partial` / `qlora` / `prompt`
  / `ia3` / `kd` run. `TODO(student)`: add executed notebooks for those runs if they exist.

## 10. Codabench submissions

Local folders carry the last five digits of their Codabench ID (`submission_<RUN_NAME><ID>`; the BART InfoNCE folder
`...random402061` is a typo for 40261). Each folder's `prediction.json` is byte-identical (MD5) to the zip downloaded
from Colab. For runs without a log in `run_logs/`, the settings come only from the `RUN_NAME` pattern
`<arch>_<method>_<doc_pool>_<loss>_<neg_source>`. All on 2026-11-07 (IST).

| Codabench ID | Submitted | Submission | Arch | Method / key settings | Micro-F1@7 |
|---|---|---|---|---|---|
| 1040261 | 18:27 | `submission_bart_lorma_mean_infonce_random` | bart | lorma; doc / mean; infonce; random negatives (template defaults). The later of two BART InfoNCE files (built 14:45 IST; an earlier one built 12:57 was not submitted) | 0.2942 |
| 1040535 | 20:02 | `submission_prefixlm_lorma_mean_infonce_random` | prefixlm | lorma; doc / mean; infonce; random negatives (template defaults) | 0.3030 |
| 1040539 | 20:04 | `submission_llama_lorma_mean_infonce_random` | llama | lorma; doc / mean; infonce; random negatives (template defaults) | 0.3154 |
| 1040543 | 20:08 | `submission_mistral_swa_lorma_mean_infonce_random` | mistral_swa | lorma; doc / mean; infonce; random negatives (other settings: TODO(student), no run log) | 0.3048 |
| 1040545 | 20:09 | `submission_bart_lorma_mean_listwise_random` (**leaderboard entry**) | bart | lorma; doc / mean; listwise (ListNet); random negatives (other settings: TODO(student), no run log; see `<hf-user>/cs781-p7/runs/bart_lorma_mean_listwise_random/run_config.json`) | **0.3166** |

## 11. How to run

1. **Render the notebooks** (optional; rendered copies are committed):
   ```bash
   cd code/phase7
   python build_p7_notebooks.py            # all four
   python build_p7_notebooks.py bart       # only P7_arch2_bart.ipynb
   ```
2. **Upload and set up Colab.** Upload `P7_arch<i>_<arch>.ipynb` to Google Colab and choose *Runtime → Change runtime
   type → T4 GPU*.
3. **Add the HF secret.** Under *Secrets* (key icon), add `HF` = the same Hugging Face token used for Phases 1–6. It
   must be able to read the `cs781-p1-*` repos and the gated course dataset, and to write `cs781-p7`. Kaggle secrets or
   the `HF_TOKEN` environment variable are also accepted. A CPU run requires `CS781_ALLOW_CPU=1`.
4. **Edit the first code cell** (`METHOD`, `REPR`, `DOC_POOL`, `LOSS`, `NEG_SOURCE`; `RUN_NAME` follows) and, if
   needed, the *Configuration* cell. Then use *Runtime → Run all*.
5. **Outputs:**
   - Local: `/content/p7/<RUN_NAME>/` (`prediction.json`, `submission_<RUN_NAME>.zip`, `run_config.json`,
     `dev_*.json`, `dev_scores.npz`, `train_scores.npz`, `curves.png`).
   - Hub: uploaded to `<hf-user>/cs781-p7/runs/<RUN_NAME>/`. The zip is also downloaded by the browser.
6. **Submission zip:** `submission_<RUN_NAME>.zip` contains only `prediction.json` and has already passed the official
   validation rules. Upload it to Codabench as is.
7. **Local dev check:** run `evaluate_phase7.py` on `dev_truth.json` / `dev_pred.json` / `dev_candidate_ids.json`
   (command in §7).
8. **Another experiment in the same runtime:** change the first cell and *Run all* again. Each run starts from the
   Phase-1 checkpoint, and tokenised documents are reused from `/content/p7/cache`.
9. **Knowledge distillation order:**
   1. Run a teacher to completion with any non-`kd` method and `SAVE_TRAIN_SCORES = True` (default). It uploads
      `runs/<teacher RUN_NAME>/train_scores.npz` and `run_config.json`.
   2. In a student notebook (any architecture whose training pool matches), set `METHOD = "kd"`,
      `TEACHER_RUN = "<teacher RUN_NAME>"`, and optionally `NEG_SOURCE = "teacher"`, `STUDENT_LAYERS`, `KD_T`,
      `KD_ALPHA`.
   3. *Run all*. The notebook raises an error listing the finished runs if `TEACHER_RUN` is not one of them.
10. **Requirement check:** the last cell lists every run in `cs781-p7` and which §13.2.1 techniques have runs. Whether
    two runs are "substantially different" is left to the student.

Dependencies used by the notebooks: `torch`, `transformers`, `huggingface_hub`, `numpy`, `scipy`, `scikit-learn`,
`matplotlib`. The versions printed in the run logs are torch `2.14.1+cu130` and transformers `5.22.1`.
