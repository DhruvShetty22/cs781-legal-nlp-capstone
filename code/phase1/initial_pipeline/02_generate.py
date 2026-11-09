# %% [markdown]
# # CS781 Phase 1 — 02 Generate, evaluate locally, build submission.zip
# Runtime: **T4 GPU**. Secret `HF` as before. Works for all 4 architectures (reads `cs781_arch` from the model config).
#
# 1. Generates on the local held-out set (`heldout_prompts.json`) and scores it with the official Phase-1 evaluator.
# 2. Generates on the official test prompts, validates IDs, writes `prediction.json` and `submission.zip`.
#
# Can be run while training is still going (it pulls the current `best/` checkpoint).

# %%
!pip -q install bert-score

# %%
ARCH = "llama"          # which model repo to load: "llama" | "bart" | "prefixlm" | "mistral_swa"
RUN_TAG = "v1"          # name for this submission (stored in the HF repo with the scores)

# Decoding configurations to compare on the held-out set — YOUR CHOICE.
DECODE_CONFIGS = {
    "greedy_rp1.2": dict(do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=0),
    "greedy_ngram4": dict(do_sample=False, repetition_penalty=1.0, no_repeat_ngram_size=4),
}
HELDOUT_LIMIT = 400     # how many held-out examples to score per config
TARGET_WORDS = 128
MAX_NEW_TOKENS = 224    # must cover 128 words (check the words/token printout below)
BATCH = 32

# %%
import os, json, re, time, zipfile, math, sys
import numpy as np, torch
from huggingface_hub import login, whoami, HfApi, hf_hub_download, snapshot_download

def get_hf_token():
    try:
        from google.colab import userdata
        return userdata.get("HF")
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF")
    except Exception:
        return os.environ["HF_TOKEN"]

HF_TOKEN = get_hf_token(); os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)
HF_USER = whoami(token=HF_TOKEN)["name"]
api = HfApi(token=HF_TOKEN)
DATA_REPO = f"{HF_USER}/cs781-p1-data"
MODEL_REPO = f"{HF_USER}/cs781-p1-{ARCH}"
OUT = f"/content/p1_gen/{ARCH}_{RUN_TAG}"
os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## Load model + data

# %%
from transformers import AutoConfig, AutoModelForCausalLM, BartForConditionalGeneration, PreTrainedTokenizerFast

mdir = snapshot_download(MODEL_REPO, token=HF_TOKEN, allow_patterns=["best/*"])
mdir = os.path.join(mdir, "best")
info = json.load(open(f"{mdir}/cs781_info.json")); print(info)
cfg = AutoConfig.from_pretrained(mdir)
arch = getattr(cfg, "cs781_arch", ARCH)
tok = PreTrainedTokenizerFast.from_pretrained(mdir)
EOS_ID, PAD_ID = tok.eos_token_id, tok.pad_token_id
device = "cuda"
if arch == "bart":
    model = BartForConditionalGeneration.from_pretrained(mdir, torch_dtype=torch.float16)
    MAX_PROMPT_TOKENS = 768                       # = SRC_MAX used in training
else:
    model = AutoModelForCausalLM.from_pretrained(mdir, torch_dtype=torch.float16)
    MAX_PROMPT_TOKENS = cfg.max_position_embeddings - MAX_NEW_TOKENS
model = model.to(device).eval()
print(arch, f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params, max prompt tokens {MAX_PROMPT_TOKENS}")

data_dir = snapshot_download(DATA_REPO, repo_type="dataset", token=HF_TOKEN,
                             allow_patterns=["heldout_*.json", "meta.json"])
heldout_prompts = json.load(open(f"{data_dir}/heldout_prompts.json", encoding="utf-8"))
heldout_truth = json.load(open(f"{data_dir}/heldout_truth.json", encoding="utf-8"))
keys = list(heldout_prompts)[:HELDOUT_LIMIT] if HELDOUT_LIMIT < len(heldout_prompts) else list(heldout_prompts)
# keep the per-source balance when limiting
if HELDOUT_LIMIT < len(heldout_prompts):
    by_src = {}
    for k in heldout_prompts: by_src.setdefault(k.split("_")[0], []).append(k)
    per = HELDOUT_LIMIT // len(by_src)
    keys = [k for v in by_src.values() for k in v[:per]]

test_path = hf_hub_download("Exploration-Lab/CS-781-Capstone", "Phase_1_Pretraining/Test/test.json",
                            repo_type="dataset", token=HF_TOKEN)
test_data = json.load(open(test_path, encoding="utf-8"))
print("held-out:", len(keys), " test:", len(test_data))

s = [heldout_truth[k]["text"] for k in keys[:100]]
print("tokens per word on references: %.3f" % (sum(len(tok.encode(x)) for x in s) / sum(len(x.split()) for x in s)))

# %% [markdown]
# ## Generation
# Prompts are truncated from the **left** (keep the end of the prompt, which is what the continuation follows) and
# decoder-only batches are **left-padded**. EOS is blocked until `MAX_NEW_TOKENS` so outputs are never cut short.

# %%
from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor, NoRepeatNGramLogitsProcessor

def encode_prompts(prompts):
    ids = [tok.encode(p, add_special_tokens=False)[-MAX_PROMPT_TOKENS:] for p in prompts]
    L = max(len(x) for x in ids)
    if arch == "bart":   # encoder: right padding
        inp = torch.tensor([x + [PAD_ID] * (L - len(x)) for x in ids])
        att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids])
    else:                # decoder: left padding
        inp = torch.tensor([[PAD_ID] * (L - len(x)) + x for x in ids])
        att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in ids])
    return inp.to(device), att.to(device)

@torch.inference_mode()
def prefixlm_generate(inp, att, dc):
    """Prompt = bidirectional prefix (as in training), then causal decoding with the KV cache."""
    B, L = inp.shape
    procs = LogitsProcessorList()
    if dc.get("repetition_penalty", 1.0) != 1.0: procs.append(RepetitionPenaltyLogitsProcessor(dc["repetition_penalty"]))
    if dc.get("no_repeat_ngram_size", 0) > 0: procs.append(NoRepeatNGramLogitsProcessor(dc["no_repeat_ngram_size"]))
    valid = att.bool()
    mask4d = (valid[:, None, None, :] | torch.eye(L, dtype=torch.bool, device=device)[None, None])  # pad queries see themselves
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    out = model(input_ids=inp, attention_mask=mask4d, position_ids=pos, use_cache=True)
    past, logits = out.past_key_values, out.logits[:, -1].float()
    seq, gen, att2d, last_pos = inp, [], att, pos[:, -1]
    for t in range(MAX_NEW_TOKENS):
        scores = procs(seq, logits)
        scores[:, EOS_ID] = -float("inf"); scores[:, PAD_ID] = -float("inf")
        if dc.get("do_sample"):
            probs = torch.softmax(scores / dc.get("temperature", 1.0), -1)
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = scores.argmax(-1, keepdim=True)
        gen.append(nxt); seq = torch.cat([seq, nxt], 1)
        att2d = torch.cat([att2d, torch.ones_like(nxt)], 1); last_pos = last_pos + 1
        out = model(input_ids=nxt, attention_mask=att2d, position_ids=last_pos[:, None], past_key_values=past, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1].float()
    return torch.cat(gen, 1)

@torch.inference_mode()
def generate_batch(prompts, dc):
    inp, att = encode_prompts(prompts)
    if arch == "prefixlm":
        new = prefixlm_generate(inp, att, dc)
    else:
        kw = dict(max_new_tokens=MAX_NEW_TOKENS, min_new_tokens=MAX_NEW_TOKENS, pad_token_id=PAD_ID,
                  eos_token_id=EOS_ID, suppress_tokens=[PAD_ID], **dc)
        out = model.generate(input_ids=inp, attention_mask=att, **kw)
        new = out if arch == "bart" else out[:, inp.shape[1]:]
    texts = []
    for row in new.tolist():
        words = tok.decode(row, skip_special_tokens=True).split()[:TARGET_WORDS]
        texts.append(" ".join(words) if words else "the")   # evaluator rejects empty strings
    return texts

def run(prompt_dict, ids, dc, label):
    preds, t0 = {}, time.time()
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        for k, txt in zip(chunk, generate_batch([prompt_dict[k]["text"] for k in chunk], dc)):
            preds[str(k)] = {"text": txt}
        print(f"\r{label}: {min(i+BATCH, len(ids))}/{len(ids)}  {time.time()-t0:.0f}s", end="")
    print()
    return preds

# %% [markdown]
# ## Official Phase-1 metrics (same code as `evaluate_phase1.py`)

# %%
import bert_score
from collections import Counter

def lcs_length(a, b):
    if not a or not b: return 0
    if len(a) < len(b): a, b = b, a
    masks = {}
    for j, t in enumerate(a): masks[t] = masks.get(t, 0) | (1 << j)
    state = 0
    for t in b:
        m = masks.get(t, 0); x = m | state; y = (state << 1) | 1; state = x & ~(x - y)
    return bin(state).count("1")

def rouge_l_f1(p, r):
    pt, rt = p.split(), r.split()
    if not pt or not rt: return 0.0
    l = lcs_length(pt, rt)
    if not l: return 0.0
    pp, rr = l / len(pt), l / len(rt)
    return 2 * pp * rr / (pp + rr)

def ngrams(t, n): return Counter(tuple(t[i:i+n]) for i in range(len(t)-n+1)) if len(t) >= n else Counter()

def bleu4(preds, refs):
    clipped = [0]*4; totals = [0]*4; pl = rl = 0
    for p, r in zip(preds, refs):
        pl += len(p); rl += len(r)
        for n in range(1, 5):
            pn, rn = ngrams(p, n), ngrams(r, n); totals[n-1] += sum(pn.values())
            for g, c in pn.items(): clipped[n-1] += min(c, rn.get(g, 0))
    prec = [clipped[i]/totals[i] if totals[i] else 0.0 for i in range(4)]
    if any(x <= 0 for x in prec) or pl == 0: return 0.0
    bp = 1.0 if pl > rl else math.exp(1.0 - rl/pl)
    return bp * math.exp(sum(math.log(x) for x in prec) / 4)

def score(pred, truth):
    ids = list(truth)
    P = [pred[i]["text"].strip() for i in ids]; R = [truth[i]["text"].strip() for i in ids]
    _, _, f = bert_score.score(P, R, lang="en", batch_size=32, verbose=False, device="cuda")
    out = {"bertscore_f1": float(f.mean()), "rougeL_f1": float(np.mean([rouge_l_f1(p, r) for p, r in zip(P, R)])),
           "bleu4": bleu4([p.split() for p in P], [r.split() for r in R])}
    for src in sorted({i.split("_")[0] for i in ids}):        # per-source BERTScore (report material)
        idx = [j for j, i in enumerate(ids) if i.startswith(src + "_")]
        out[f"bs_{src}"] = float(f[idx].mean())
    return out

# %% [markdown]
# ## 1) Compare decoding configs on the local held-out set

# %%
truth_sub = {k: heldout_truth[k] for k in keys}
results = {}
for name, dc in DECODE_CONFIGS.items():
    preds = run(heldout_prompts, keys, dc, name)
    results[name] = score(preds, truth_sub)
    print(name, {k: round(v, 4) for k, v in results[name].items()})
    print("  sample:", preds[keys[0]]["text"][:300])
json.dump({"model_info": info, "decode_configs": DECODE_CONFIGS, "heldout": results},
          open(f"{OUT}/heldout_scores.json", "w"), indent=2)

# %% [markdown]
# ## 2) Generate the official test predictions with the chosen config
# Set `CHOSEN` yourself after looking at the table above.

# %%
CHOSEN = max(results, key=lambda n: results[n]["bertscore_f1"])   # or set a name explicitly
print("Using decode config:", CHOSEN, DECODE_CONFIGS[CHOSEN])
test_ids = list(test_data)
predictions = run(test_data, test_ids, DECODE_CONFIGS[CHOSEN], "test")

# validation (same checks as the evaluator / Codabench)
assert set(predictions) == set(map(str, test_data)), "ID mismatch"
assert all(isinstance(v["text"], str) and v["text"].strip() for v in predictions.values()), "empty prediction"
wl = [len(v["text"].split()) for v in predictions.values()]
print(f"words per prediction: min {min(wl)} mean {np.mean(wl):.1f} max {max(wl)}")

pred_path = f"{OUT}/prediction.json"
json.dump(predictions, open(pred_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
zip_path = f"{OUT}/submission.zip"
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    z.write(pred_path, arcname="prediction.json")
with zipfile.ZipFile(zip_path) as z: assert z.namelist() == ["prediction.json"]
print("Ready:", zip_path)

# keep a record of every submission (config + local scores) next to the model
json.dump({"model_info": info, "decode": {CHOSEN: DECODE_CONFIGS[CHOSEN]}, "heldout": results[CHOSEN]},
          open(f"{OUT}/submission_record.json", "w"), indent=2)
api.upload_folder(folder_path=OUT, path_in_repo=f"submissions/{RUN_TAG}", repo_id=MODEL_REPO)

# %%
try:
    from google.colab import files
    files.download(zip_path)
except Exception:
    print("Download manually from", zip_path)
