# CS781 Capstone: In-House Legal Intelligence System

This is the code for the CS781 (IIT Kanpur, 2026) capstone by **Dhruv Vivekanand Shetty (230370)**.
It is a seven-phase legal-NLP pipeline over Indian case law. Every model descends from language models pretrained
**from scratch** in Phase 1 on the course corpus. No third-party pretrained weights and no external LLM APIs are used.

The final report (PDF) is in `report/`. Model weights and checkpoints are **not** in this repository, as the course
rules require. They live in private Hugging Face repositories. The code reads them through the `HF` token secret
(see [Running the code](#running-the-code)).

| Phase | Task | Primary metric | Folder |
|---|---|---|---|
| 1 | Legal language-model pretraining (512-word prompt → 128-word continuation) | BERTScore F1 | [`code/phase1`](code/phase1/README.md) |
| 2 | Legal named-entity recognition (character spans, 12 labels) | strict Macro-F1 | [`code/phase2`](code/phase2/README.md) |
| 3 | Rhetorical-role prediction (13 roles per sentence, CL + IT) | Macro-F1 | [`code/phase3`](code/phase3/README.md) |
| 4 | Court judgment prediction (binary) | Macro-F1 | [`code/phase4`](code/phase4/README.md) |
| 5 | Legal statute identification (multi-label, 100 statutes) | Macro-F1 | [`code/phase5`](code/phase5/README.md) |
| 6 | Abstractive summarization of judgments | BERTScore F1 | [`code/phase6`](code/phase6/README.md) |
| 7 | Prior-case retrieval (top-7 from a candidate pool) | Micro-F1@7 | [`code/phase7`](code/phase7/README.md) |

Each phase folder has its own detailed README with these sections: task, files, data and preprocessing, model
architecture, fine-tuning methods, training procedure, inference and submission, every configuration knob, the runs
recorded in `run_logs/`, the Codabench submissions, and how to run.

---

## Repository layout

```
code/
  phaseN/
    README.md                      detailed documentation of the phase and of every submission
    phaseN_<task>_template.py      single source of the phase pipeline (percent-format notebook, "# %%" cells)
    build_pN_notebooks.py          renders the template into one notebook per architecture
    build_nb.py                    percent-format .py -> .ipynb converter used by the builders
    PN_arch1_llama.ipynb           generated notebooks (unexecuted), one per Phase-1 architecture
    PN_arch2_bart.ipynb
    PN_arch3_prefixlm.ipynb
    PN_arch4_mistral_swa.ipynb
    run_logs/                      notebooks as executed on Colab, with all outputs (real configs + local scores)
  phase1/initial_pipeline/         first-generation Phase-1 scripts (data prep is shared by both generations)
  phase1/tools/compare_submissions.py   side-by-side viewer for Phase-1 generations
report/
  CS781-CP-Dhruv-Shetty-230370-report.tex / .pdf, references.bib
```

## Overall design

1. **Phase 1: four foundation models trained from scratch** (Hugging Face model classes built from our own configs,
   random initialisation, d_model 768, shared 32k byte-level BPE tokenizer trained on the corpus):

   | name | family | layers | attention | parameters |
   |---|---|---|---|---|
   | `llama` | decoder-only (RoPE, RMSNorm, SwiGLU) | 12 | causal | 109,529,856 |
   | `prefixlm` | prefix-LM hybrid (llama config, bidirectional prefix mask) | 12 | bidirectional prefix + causal | 109,529,856 |
   | `mistral_swa` | efficient: sliding window 256 + grouped-query attention (4 KV heads) | 12 | causal, windowed | 100,092,672 |
   | `bart` (v2) | encoder-decoder, continuation + text-infilling objective | 6 + 6 | bidirectional encoder | 125,442,048 |

   The first BART run (v1) collapsed: its decoder ignored the encoder and produced the same continuation for every
   prompt. BART was retrained as v2 with a text-infilling objective and a `<mask>` token. **The fix came after the
   Phase-1 and Phase-2 deadlines** (v2 was trained on 14-15 Oct 2026). The Phase-1 leaderboard's only BART entry is
   therefore the collapsed v1, and Phase-2 BART runs fine-tuned v1. Phases 3-7 load BART from v2. See
   [`code/phase1/README.md`](code/phase1/README.md).

2. **Phases 2-7: every task model is initialised from a Phase-1 `best/` checkpoint** and adapted with methods we
   implemented ourselves in plain PyTorch (no `peft`, no `bitsandbytes`):
   - full fine-tuning;
   - partial / gradual unfreezing;
   - BitFit;
   - LoRA and rsLoRA;
   - **LoRMA** (multiplicative low-rank adaptation; compulsory in every phase);
   - QLoRA (own NF4 + double quantisation);
   - AdaLoRA;
   - Houlsby / Pfeiffer adapters;
   - IA³;
   - prompt tuning;
   - knowledge distillation into a student that keeps every second Phase-1 block.

   Which methods exist in which phase is listed in each phase README (section 5).

3. **One template per phase, four notebooks per template.** A phase's whole pipeline is one percent-format Python
   file:
   - data loading;
   - preprocessing;
   - model;
   - every adaptation method;
   - training;
   - local evaluation with the official scorer's logic;
   - `submission.zip` creation and validation.

   `build_pN_notebooks.py` fills in the architecture and writes `PN_arch{1..4}_<arch>.ipynb`. The four notebooks
   ran in parallel on four Colab T4 accounts. All hyperparameters of a run are in one config cell marked
   **YOUR CHOICE**. The run name (`RUN_NAME`, e.g. `llama_lorma_even_mean`) encodes the main settings. Each run
   uploads its config, metrics, predictions and merged weights to `<hf-user>/cs781-pN/runs/<RUN_NAME>/` (private).

4. **Long documents** (Phases 4-7) are cut into 512-token chunks. At most 8 chunks are selected (`CHUNK_SELECT`), each
   chunk is encoded by the backbone and mean-pooled, and the chunk vectors are pooled into a document vector
   (`DOC_POOL`). Phase 6 truncates the source to its head instead.

## Submissions and the best model per phase

The full list of Codabench submissions, with the settings of each one, is in section 10 of each phase README. The
local dev / held-out scores printed by the executed notebooks are in section 9.

| Phase | Best Codabench submission (run name) | Codabench primary score |
|---|---|---|
| 1 | `llama` (Codabench ID 964531, the leaderboard entry; see the attribution table in [`code/phase1/README.md`](code/phase1/README.md#10-codabench-submissions)) | 0.8294 (BERTScore F1) |
| 2 | `llama_full`, strict decoding (Codabench ID 976758, the leaderboard entry; inferred, see [`code/phase2/README.md`](code/phase2/README.md#10-codabench-submissions)) | 0.4399 (strict Macro-F1) |
| 3 | `prefixlm_lorma_window` (Codabench ID 989634, the leaderboard entry; see [`code/phase3/README.md`](code/phase3/README.md#10-codabench-submissions)) | 0.4373 (Macro-F1) |
| 4 | `prefixlm_lorma_even_mean` (Codabench ID 1005146, the leaderboard entry; see [`code/phase4/README.md`](code/phase4/README.md#10-codabench-submissions)) | 0.6371 (Macro-F1) |
| 5 | `bart_lorma_even_mean_bce` (Codabench ID 1016048, the leaderboard entry; inferred, see [`code/phase5/README.md`](code/phase5/README.md#10-codabench-submissions)) | 0.1651 (Macro-F1) |
| 6 | `llama_lorma_head` (Codabench ID 1030867, highest score). The leaderboard shows `bart_lorma_head` (1030611, 0.7662); see [`code/phase6/README.md`](code/phase6/README.md#10-codabench-submissions) | 0.7686 (BERTScore F1) |
| 7 | `bart_lorma_mean_listwise_random` (Codabench ID 1040545, the leaderboard entry; see [`code/phase7/README.md`](code/phase7/README.md#10-codabench-submissions)) | 0.3166 (Micro-F1@7) |

## Running the code

Environment used for every logged run: Google Colab, Tesla T4 (16 GB), `torch 2.11.0+cu130`,
`transformers 5.18.0`. Each notebook installs its few extra packages in its first cell (e.g. `bert-score`,
`ijson`).

1. Create a Hugging Face **write** token. In Colab add it as a secret named `HF` with notebook access on. The notebooks
   resolve your HF username at runtime and create the private repos they need.
2. **Phase 1**: run `code/phase1/initial_pipeline/00_data_prep.ipynb` once on a CPU runtime. It samples the corpus,
   trains the tokenizer and uploads the token files to `<hf-user>/cs781-p1-data`. Then run
   `code/phase1/P1_arch{1..4}_*.ipynb` on GPU runtimes (one per account). Each trains for `TRAIN_HOURS`, pushes
   `latest/` every 30 min (it resumes automatically after a disconnect) and `best/` on every new best validation loss,
   then generates, scores the local held-out set and writes `submission.zip`.
3. **Phases 2-7**: open `code/phaseN/PN_archK_<arch>.ipynb` in Colab and set the **YOUR CHOICE** config cell (method,
   pooling, loss, etc.; the values of every submitted run are listed in the phase README). Then use *Run all*. The
   notebook loads `<hf-user>/cs781-p1-<arch>/best` (`-v2` for BART), trains under a time cap, evaluates on dev or
   held-out data, and writes `submission.zip` with `prediction.json`. KD runs need their teacher run to have finished
   first. The teacher's saved training-set scores are read from `<hf-user>/cs781-pN/runs/<TEACHER_RUN>/`.
4. To regenerate notebooks after editing a template: `cd code/phaseN && python build_pN_notebooks.py [arch]`
   (Phase 1: `python build_arch_notebooks.py [arch]`).

Local scores are computed with the same logic as the official `evaluate_phaseN.py` scripts distributed with the
course. The Phase 5 and 7 notebooks print that their dev numbers are identical to the official computation.

## Citation of methods

The method papers (LoRA, LoRMA, QLoRA, AdaLoRA, rsLoRA, Houlsby and Pfeiffer adapters, IA³, prompt tuning, BitFit,
knowledge distillation, BART, LLaMA, Mistral, GQA, InfoNCE, ListNet, RankNet, BM25, CRF) are cited in the report's
bibliography (`report/references.bib`). The task definitions follow the IL-TUR benchmark (Joshi et al., ACL 2024).
