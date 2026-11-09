# %% [markdown]
# # CS781 Phase 1 — 00 Data Preparation
# Run **once** on ONE Colab account: Runtime -> Change runtime type -> **CPU** (saves GPU quota). Secret `HF` = HF *write* token.
#
# What it does:
# 1. Streams the 4 training files from HF (no full download / no `json.load` of 10 GB files) and samples documents per source.
# 2. Holds out documents for validation + builds a local test set shaped like the official one (512-word prompt → 128-word reference).
# 3. Trains a 32k byte-level BPE tokenizer on the legal corpus.
# 4. Tokenizes every source into its own `uint16` `.bin` file (docs separated by `<|endoftext|>`).
# 5. Uploads everything to a **private** HF dataset repo so all training accounts pull the same data.

# %%
!pip -q install ijson

# %%
import os, json, random, re, time, gc
import numpy as np
import requests, ijson
from huggingface_hub import login, whoami, HfApi, hf_hub_url, hf_hub_download

def get_hf_token():
    try:                                    # Colab: key icon (Secrets) -> name "HF", notebook access ON
        from google.colab import userdata
        return userdata.get("HF")
    except Exception:
        pass
    try:                                    # Kaggle: Add-ons -> Secrets -> "HF"
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF")
    except Exception:
        return os.environ["HF_TOKEN"]

HF_TOKEN = get_hf_token()
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)
HF_USER = whoami(token=HF_TOKEN)["name"]
print("HF user:", HF_USER, "| ijson backend:", ijson.backend)
assert ijson.backend.startswith("yajl2"), "pure-python ijson is far too slow for 10 GB files"

# %% [markdown]
# ## Configuration
# **Data mix = your design decision** (document it in the report). `target_gb` is how many GB of the raw JSON
# to keep per source (documents are sampled uniformly across the whole file with probability `target_gb / file_gb`).
# Rough rule: 1 GB JSON ≈ 220–250M tokens with a 32k legal BPE.

# %%
HF_DATASET_REPO = "Exploration-Lab/CS-781-Capstone"
OUT_REPO = f"{HF_USER}/cs781-p1-data"          # private dataset repo that all accounts read from

SOURCES = {   # name: (file, target_gb)   <-- YOUR CHOICE
    "sc":  ("Phase_1_Pretraining/Train/train_supreme_court.json", 1.30),
    "hc":  ("Phase_1_Pretraining/Train/train_high_courts.json",   1.80),
    "trib":("Phase_1_Pretraining/Train/train_tribunals.json",     1.00),
    "acts":("Phase_1_Pretraining/Train/train_acts.json",          0.35),
}
TEST_FILE = "Phase_1_Pretraining/Test/test.json"

VAL_DOC_FRAC = 0.01        # fraction of sampled docs held out (validation loss + local test set)
HELDOUT_PER_SOURCE = 100   # local "official-like" test examples per source
PROMPT_WORDS, CONT_WORDS = 512, 128

VOCAB_SIZE = 32000         # <= 65535 so tokens fit in uint16
TOKENIZER_TRAIN_GB = 0.4   # amount of text used to learn BPE merges (Colab has ~12 GB RAM)
SEED = 1234

WORK = "/tmp/p1" if os.path.isdir("/kaggle") else "/content/p1"
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)
random.seed(SEED)

# %% [markdown]
# ## Look at the official test prompts (format check)

# %%
test_path = hf_hub_download(HF_DATASET_REPO, TEST_FILE, repo_type="dataset", token=HF_TOKEN)
test_data = json.load(open(test_path, encoding="utf-8"))
print("test examples:", len(test_data))
k0 = next(iter(test_data))
print("record keys:", list(test_data[k0].keys()))
wc = [len(r["text"].split()) for r in test_data.values()]
print("prompt words: min", min(wc), "max", max(wc), "mean", sum(wc) / len(wc))
nl = sum("\n" in r["text"] for r in test_data.values())
print(f"prompts containing newlines: {nl}/{len(test_data)}")
print("---- sample prompt (last 600 chars) ----")
print(test_data[k0]["text"][-600:])

# If the official prompts have NO newlines, build local prompts the same way (single-space joined).
TEST_HAS_NEWLINES = nl > 0

# %% [markdown]
# ## Stream + sample documents per source
# Detects whether a file is a JSON list (`[{"text":...}, ...]`) or a dict (`{"id": {"text":...}}`) from its first bytes.

# %%
def open_stream(filename):
    url = hf_hub_url(HF_DATASET_REPO, filename, repo_type="dataset")
    r = requests.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"}, stream=True, timeout=60)
    r.raise_for_status()
    r.raw.decode_content = True
    return r

def peek_structure(filename):
    url = hf_hub_url(HF_DATASET_REPO, filename, repo_type="dataset")
    r = requests.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}", "Range": "bytes=0-2000"}, timeout=60)
    head = r.content.decode("utf-8", errors="ignore").lstrip()
    print(filename, "->", repr(head[:300]))
    return "list" if head.startswith("[") else "dict"

def file_size(filename):
    info = HfApi().get_paths_info(HF_DATASET_REPO, [filename], repo_type="dataset", token=HF_TOKEN)[0]
    return info.size

def iter_texts(filename):
    structure = peek_structure(filename)
    r = open_stream(filename)
    if structure == "list":
        it = ijson.items(r.raw, "item", buf_size=1 << 20)
    else:
        it = (v for _, v in ijson.kvitems(r.raw, "", buf_size=1 << 20))
    for rec in it:
        text = rec.get("text") if isinstance(rec, dict) else rec
        if isinstance(text, str) and text.strip():
            yield text

# %%
stats = {}
for name, (fname, target_gb) in SOURCES.items():
    size = file_size(fname)
    keep_p = min(1.0, target_gb * 1e9 / size)
    print(f"\n[{name}] size={size/1e9:.2f} GB keep_p={keep_p:.3f}")
    t0 = time.time(); n_train = n_val = 0; chars = 0
    with open(f"{WORK}/{name}_train.jsonl", "w", encoding="utf-8") as ft, \
         open(f"{WORK}/{name}_val.jsonl", "w", encoding="utf-8") as fv:
        try:
            for i, text in enumerate(iter_texts(fname)):
                if random.random() > keep_p:
                    continue
                if random.random() < VAL_DOC_FRAC:
                    fv.write(json.dumps({"text": text}, ensure_ascii=False) + "\n"); n_val += 1
                else:
                    ft.write(json.dumps({"text": text}, ensure_ascii=False) + "\n"); n_train += 1
                chars += len(text)
                if i % 20000 == 0:
                    print(f"  docs seen {i:,} kept {n_train+n_val:,} chars {chars/1e9:.2f}G  {time.time()-t0:.0f}s")
        except Exception as e:  # network drop mid-stream: keep what we have
            print("  STREAM ERROR (keeping partial data):", repr(e))
    stats[name] = {"train_docs": n_train, "val_docs": n_val, "chars": chars, "keep_p": keep_p}
    print(f"  done: {stats[name]}  in {time.time()-t0:.0f}s")

json.dump(stats, open(f"{OUT}/sampling_stats.json", "w"), indent=2)

# %% [markdown]
# ## Build the local official-like test set from held-out docs
# `heldout_prompts.json` has the same format as the official test file, and `heldout_truth.json` is the reference for `evaluate_phase1.py`.

# %%
def make_example(text, rng):
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    need = PROMPT_WORDS + CONT_WORDS
    if len(spans) < need + 10:
        return None
    s = rng.randint(0, len(spans) - need)
    p_start, p_end = spans[s][0], spans[s + PROMPT_WORDS - 1][1]
    prompt = text[p_start:p_end]
    if not TEST_HAS_NEWLINES:
        prompt = " ".join(prompt.split())
    ref = " ".join(text[a:b] for a, b in spans[s + PROMPT_WORDS: s + need])
    return prompt, ref

rng = random.Random(SEED)
prompts, truth = {}, {}
for name in SOURCES:
    n = 0
    for line in open(f"{WORK}/{name}_val.jsonl", encoding="utf-8"):
        ex = make_example(json.loads(line)["text"], rng)
        if ex is None:
            continue
        key = f"{name}_{n}"
        prompts[key] = {"text": ex[0]}; truth[key] = {"text": ex[1]}
        n += 1
        if n >= HELDOUT_PER_SOURCE:
            break
    print(name, "held-out examples:", n)

json.dump(prompts, open(f"{OUT}/heldout_prompts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(truth, open(f"{OUT}/heldout_truth.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# %% [markdown]
# ## Train the 32k byte-level BPE tokenizer
# Trained on an equal-ish slice of every source so the vocabulary covers statutes, judgments and tribunal orders.

# %%
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

SPECIALS = ["<|endoftext|>", "<pad>"]

def tokenizer_corpus():
    total = sum(s["chars"] for s in stats.values())
    for name in SOURCES:
        budget = TOKENIZER_TRAIN_GB * 1e9 * stats[name]["chars"] / total
        used = 0
        for line in open(f"{WORK}/{name}_train.jsonl", encoding="utf-8"):
            t = json.loads(line)["text"]
            yield t
            used += len(t)
            if used >= budget:
                break

t0 = time.time()
bpe = Tokenizer(models.BPE())
bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
bpe.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=SPECIALS,
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=True)
bpe.train_from_iterator(tokenizer_corpus(), trainer=trainer)
print(f"tokenizer trained in {time.time()-t0:.0f}s, vocab={bpe.get_vocab_size()}")

tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|endoftext|>", bos_token="<|endoftext|>",
                              unk_token="<|endoftext|>", pad_token="<pad>")
tok.save_pretrained(f"{OUT}/tokenizer")
EOS_ID, PAD_ID = tok.eos_token_id, tok.pad_token_id
print("eos", EOS_ID, "pad", PAD_ID)

# Tokenizer efficiency on the official test prompts (report material): compare with GPT-2
from transformers import AutoTokenizer
gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
sample = [r["text"] for r in list(test_data.values())[:200]]
words = sum(len(s.split()) for s in sample)
for nm, t in [("legal-bpe-32k", tok), ("gpt2-50k", gpt2)]:
    n = sum(len(t.encode(s, add_special_tokens=False)) for s in sample)
    print(f"{nm}: {n/words:.3f} tokens/word")

# %% [markdown]
# ## Tokenize every source into uint16 .bin files

# %%
def tokenize_file(src_jsonl, dst_bin, batch_docs=512):
    n_tok = 0; t0 = time.time()
    with open(dst_bin, "wb") as fo:
        batch = []
        def flush(batch):
            nonlocal n_tok
            ids = []
            for e in bpe.encode_batch(batch):
                ids.extend(e.ids); ids.append(EOS_ID)
            arr = np.array(ids, dtype=np.uint16)
            fo.write(arr.tobytes()); n_tok += len(arr)
        for line in open(src_jsonl, encoding="utf-8"):
            batch.append(json.loads(line)["text"])
            if len(batch) == batch_docs:
                flush(batch); batch = []
        if batch:
            flush(batch)
    print(f"  {os.path.basename(dst_bin)}: {n_tok/1e6:.1f}M tokens ({time.time()-t0:.0f}s)")
    return n_tok

os.environ["TOKENIZERS_PARALLELISM"] = "true"
token_counts = {}
for name in SOURCES:
    token_counts[f"{name}_train"] = tokenize_file(f"{WORK}/{name}_train.jsonl", f"{OUT}/{name}_train.bin")
    token_counts[f"{name}_val"] = tokenize_file(f"{WORK}/{name}_val.jsonl", f"{OUT}/{name}_val.bin")

meta = {"vocab_size": len(tok),
        "eos_id": EOS_ID, "pad_id": PAD_ID, "dtype": "uint16", "sources": list(SOURCES),
        "token_counts": token_counts, "sampling": stats, "config": {k: v[1] for k, v in SOURCES.items()},
        "test_has_newlines": TEST_HAS_NEWLINES}
json.dump(meta, open(f"{OUT}/meta.json", "w"), indent=2)
print(json.dumps(token_counts, indent=1))
print("TOTAL train tokens: %.1fM" % (sum(v for k, v in token_counts.items() if k.endswith("_train")) / 1e6))

# %% [markdown]
# ## Upload to the private HF dataset repo

# %%
api = HfApi(token=HF_TOKEN)
api.create_repo(OUT_REPO, repo_type="dataset", private=True, exist_ok=True)
api.upload_folder(folder_path=OUT, repo_id=OUT_REPO, repo_type="dataset", commit_message="phase1 data")
print("Uploaded to https://huggingface.co/datasets/" + OUT_REPO)
print("Use THIS SAME HF token as the `HF` secret on every Colab account so they can read it and push checkpoints.")
