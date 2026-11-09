# %% [markdown]
# # CS781 Phase 1 — 01 Train (one architecture per Colab account)
# Runtime -> Change runtime type -> **T4 GPU**. Secret `HF` = the same write token used in 00_data_prep.
#
# Set `ARCH` below and **Run all**. Every `CKPT_MINUTES` the full training state is pushed to
# `<hf-user>/cs781-p1-<ARCH>` (folder `latest/`), and the best-validation model to `best/`.
# If Colab disconnects: reconnect and **Run all** again — it resumes from `latest/` automatically.
#
# | ARCH | Family | HF class (random init) | Objective |
# |---|---|---|---|
# | `llama` | Decoder-only | `LlamaForCausalLM` (RoPE, RMSNorm, SwiGLU) | next-token on all tokens |
# | `bart` | Encoder-decoder | `BartForConditionalGeneration` | encoder reads prompt, decoder generates continuation |
# | `prefixlm` | Encoder/decoder hybrid | `LlamaForCausalLM` + prefix (bidirectional) mask | next-token on continuation only |
# | `mistral_swa` | Efficient | `MistralForCausalLM` (sliding-window attention + GQA) | next-token on all tokens |

# %%
ARCH = "llama"   # "llama" | "bart" | "prefixlm" | "mistral_swa"

# %%
import os, json, math, time, shutil, random
import numpy as np
import torch
import torch.nn.functional as F
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

HF_TOKEN = get_hf_token()
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)
HF_USER = whoami(token=HF_TOKEN)["name"]
api = HfApi(token=HF_TOKEN)

assert torch.cuda.is_available(), "Switch the runtime to a GPU"
print(torch.__version__, torch.cuda.get_device_name(0))
import transformers; print("transformers", transformers.__version__)

# %% [markdown]
# ## Configuration — **all values below are YOUR CHOICE** (document them in the report)

# %%
DATA_REPO = f"{HF_USER}/cs781-p1-data"
MODEL_REPO = f"{HF_USER}/cs781-p1-{ARCH}"
WORK = "/content/p1_train" if os.path.isdir("/content") else "/tmp/p1_train"
os.makedirs(WORK, exist_ok=True)

# Data mixing weights per source (sampling probability of each training window)
MIX = {"sc": 0.40, "hc": 0.30, "trib": 0.20, "acts": 0.10}

# Training budget: wall-clock hours of pure training (the LR schedule is fitted to this)
TRAIN_HOURS = 5.5
CKPT_MINUTES = 30
EVAL_EVERY = 250          # optimizer steps
EVAL_BATCHES_PER_SOURCE = 10

# Optimisation
LR = 6e-4
MIN_LR_FRAC = 0.1
WARMUP_STEPS = 300
WEIGHT_DECAY = 0.1
BETAS = (0.9, 0.95)
GRAD_CLIP = 1.0
MICRO_BS = 8
GRAD_ACCUM = 8
DROPOUT = 0.0
SEED = 42

# Sequence lengths
CTX = 1024                 # decoder-only / prefix-LM window
PREFIX_MIN = 64            # prefix-LM: prefix length sampled in [PREFIX_MIN, CTX - 128]
SRC_MIN, SRC_MAX = 128, 768   # bart: encoder length sampled in this range
TGT_LEN = 256              # bart: decoder (continuation) length

# Model sizes
D_MODEL, N_LAYER, N_HEAD, D_FF = 768, 12, 12, 2048
N_KV_HEAD_SWA, SWA_WINDOW = 4, 256                   # mistral_swa only
BART_ENC_LAYERS, BART_DEC_LAYERS, BART_FF = 6, 6, 3072  # bart only

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# %% [markdown]
# ## Data: per-source token memmaps

# %%
data_dir = snapshot_download(DATA_REPO, repo_type="dataset", token=HF_TOKEN,
                             allow_patterns=["*.bin", "meta.json", "tokenizer/*"])
meta = json.load(open(f"{data_dir}/meta.json"))
SOURCES = [s for s in meta["sources"] if MIX.get(s, 0) > 0]
weights = np.array([MIX[s] for s in SOURCES], dtype=np.float64); weights /= weights.sum()
bins = {split: {s: np.memmap(f"{data_dir}/{s}_{split}.bin", dtype=np.uint16, mode="r") for s in SOURCES}
        for split in ("train", "val")}
for s in SOURCES:
    print(f"{s:5s} train {len(bins['train'][s])/1e6:8.1f}M  val {len(bins['val'][s])/1e6:6.2f}M  weight {MIX[s]:.2f}")

from transformers import PreTrainedTokenizerFast
tokenizer = PreTrainedTokenizerFast.from_pretrained(f"{data_dir}/tokenizer")
VOCAB, EOS_ID, PAD_ID = len(tokenizer), tokenizer.eos_token_id, tokenizer.pad_token_id
print("vocab", VOCAB, "eos", EOS_ID, "pad", PAD_ID)

WINDOW = SRC_MAX + TGT_LEN if ARCH == "bart" else CTX

def sample_windows(split, n, length, rng, source=None):
    out = np.empty((n, length), dtype=np.int64)
    srcs = [source] * n if source else rng.choice(SOURCES, size=n, p=weights)
    for i, s in enumerate(srcs):
        arr = bins[split][s]
        j = int(rng.integers(0, len(arr) - length - 1))
        out[i] = arr[j:j + length]
    return torch.from_numpy(out)

# fixed validation windows per source (identical for every run of the same ARCH family)
val_rng = np.random.default_rng(0)
VAL_SET = {s: [sample_windows("val", MICRO_BS, WINDOW, val_rng, source=s) for _ in range(EVAL_BATCHES_PER_SOURCE)]
           for s in SOURCES}

# %% [markdown]
# ## Model (Hugging Face classes, randomly initialised — no pretrained weights)

# %%
from transformers import (LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
                          BartConfig, BartForConditionalGeneration)

def build_model():
    common = dict(vocab_size=VOCAB, bos_token_id=EOS_ID, eos_token_id=EOS_ID, pad_token_id=PAD_ID)
    if ARCH in ("llama", "prefixlm"):
        cfg = LlamaConfig(hidden_size=D_MODEL, intermediate_size=D_FF, num_hidden_layers=N_LAYER,
                          num_attention_heads=N_HEAD, num_key_value_heads=N_HEAD, max_position_embeddings=CTX,
                          attention_dropout=DROPOUT, tie_word_embeddings=True, **common)
        m = LlamaForCausalLM(cfg)
    elif ARCH == "mistral_swa":
        cfg = MistralConfig(hidden_size=D_MODEL, intermediate_size=D_FF, num_hidden_layers=N_LAYER,
                            num_attention_heads=N_HEAD, num_key_value_heads=N_KV_HEAD_SWA,
                            sliding_window=SWA_WINDOW, max_position_embeddings=CTX,
                            attention_dropout=DROPOUT, tie_word_embeddings=True, **common)
        m = MistralForCausalLM(cfg)
    elif ARCH == "bart":
        cfg = BartConfig(d_model=D_MODEL, encoder_layers=BART_ENC_LAYERS, decoder_layers=BART_DEC_LAYERS,
                         encoder_attention_heads=N_HEAD, decoder_attention_heads=N_HEAD,
                         encoder_ffn_dim=BART_FF, decoder_ffn_dim=BART_FF, max_position_embeddings=1024,
                         dropout=DROPOUT, attention_dropout=DROPOUT, activation_dropout=0.0,
                         decoder_start_token_id=EOS_ID, forced_bos_token_id=None, forced_eos_token_id=None,
                         scale_embedding=True, tie_word_embeddings=True, **common)
        m = BartForConditionalGeneration(cfg)
    else:
        raise ValueError(ARCH)
    m.config.cs781_arch = ARCH          # read by the generation notebook
    return m

model = build_model()
n_params = sum(p.numel() for p in model.parameters())
print(f"{ARCH}: {n_params/1e6:.1f}M parameters")
assert 50e6 <= n_params <= 300e6, "Phase-1 rule: 50M-300M parameters"
model = model.cuda()

# %% [markdown]
# ## Objectives
# Prefix-LM: tokens `< P` attend bidirectionally among themselves; tokens `>= P` are causal. Loss only on positions `>= P`
# (predicting prefix tokens would leak, since the prefix sees itself bidirectionally).

# %%
def prefix_mask(B, L, P, device):
    i = torch.arange(L, device=device)
    allowed = (i[None, :] <= i[:, None]) | (i[None, :] < P)      # [query, key]
    return allowed[None, None].expand(B, 1, L, L)                  # bool, True = attend

def compute_loss(batch, rng, fixed=False):
    batch = batch.cuda(non_blocking=True)
    B, L = batch.shape
    if ARCH in ("llama", "mistral_swa"):
        return model(input_ids=batch, labels=batch).loss
    if ARCH == "prefixlm":
        P = L // 2 if fixed else int(rng.integers(PREFIX_MIN, L - 128))
        labels = batch.clone(); labels[:, :P] = -100
        return model(input_ids=batch, attention_mask=prefix_mask(B, L, P, batch.device), labels=labels).loss
    if ARCH == "bart":
        S = SRC_MAX if fixed else int(rng.integers(SRC_MIN, SRC_MAX + 1))
        src = batch[:, SRC_MAX - S:SRC_MAX].contiguous()
        tgt = batch[:, SRC_MAX:SRC_MAX + TGT_LEN].contiguous()
        return model(input_ids=src, labels=tgt).loss

# %% [markdown]
# ## Sanity checks (prefix mask really is bidirectional on the prefix and causal after it)

# %%
if ARCH == "prefixlm":
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        x = sample_windows("val", 2, 64, np.random.default_rng(1)).cuda()
        ref = model(input_ids=x).logits.float()
        a = model(input_ids=x, attention_mask=prefix_mask(2, 64, 0, x.device)).logits.float()
        assert torch.allclose(ref, a, atol=2e-2), "4D mask ignored / misread by this transformers version"
        P = 32; x2 = x.clone(); x2[:, P - 1] = (x2[:, P - 1] + 1) % VOCAB
        b1 = model(input_ids=x, attention_mask=prefix_mask(2, 64, P, x.device)).logits.float()
        b2 = model(input_ids=x2, attention_mask=prefix_mask(2, 64, P, x.device)).logits.float()
        assert (b1[:, 0] - b2[:, 0]).abs().max() > 1e-5, "prefix is not bidirectional"
        x3 = x.clone(); x3[:, 50] = (x3[:, 50] + 1) % VOCAB
        b3 = model(input_ids=x3, attention_mask=prefix_mask(2, 64, P, x.device)).logits.float()
        assert torch.allclose(b1[:, :50], b3[:, :50], atol=2e-2), "suffix is not causal"
    print("prefix-LM mask checks passed")
model.train()

# %% [markdown]
# ## Optimiser, schedule, checkpoint / resume

# %%
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                         {"params": no_decay, "weight_decay": 0.0}], lr=LR, betas=BETAS, fused=True)
scaler = torch.amp.GradScaler("cuda")

def lr_at(step, max_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    prog = min(1.0, (step - WARMUP_STEPS) / max(1, max_steps - WARMUP_STEPS))
    return LR * (MIN_LR_FRAC + (1 - MIN_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * prog)))

state = {"step": 0, "max_steps": None, "best_val": float("inf"), "history": [], "train_seconds": 0.0, "tokens": 0}
api.create_repo(MODEL_REPO, private=True, exist_ok=True)
if api.file_exists(MODEL_REPO, "latest/state.pt"):
    ck = torch.load(hf_hub_download(MODEL_REPO, "latest/state.pt", token=HF_TOKEN, force_download=True),
                    map_location="cuda", weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
    state = ck["state"]
    print(f"RESUMED from step {state['step']} / {state['max_steps']}")
    del ck
else:
    print("No checkpoint found, starting fresh")

def push_latest():
    d = f"{WORK}/latest"; shutil.rmtree(d, ignore_errors=True); os.makedirs(d)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "state": state},
               f"{d}/state.pt")
    json.dump(state["history"], open(f"{d}/history.json", "w"))
    api.upload_folder(folder_path=d, path_in_repo="latest", repo_id=MODEL_REPO, commit_message=f"latest step {state['step']}")

def push_best(val):
    d = f"{WORK}/best"; shutil.rmtree(d, ignore_errors=True)
    model.save_pretrained(d); tokenizer.save_pretrained(d)
    json.dump({"step": state["step"], "val_loss": val, "arch": ARCH, "params": n_params,
               "tokens_seen": state["tokens"]}, open(f"{d}/cs781_info.json", "w"), indent=2)
    api.upload_folder(folder_path=d, path_in_repo="best", repo_id=MODEL_REPO, commit_message=f"best step {state['step']} val {val:.4f}")

@torch.no_grad()
def evaluate():
    model.eval(); out = {}
    for s in SOURCES:
        losses = []
        for b in VAL_SET[s]:
            with torch.autocast("cuda", dtype=torch.float16):
                losses.append(compute_loss(b, None, fixed=True).item())
        out[s] = float(np.mean(losses))
    out["mean"] = float(np.mean([out[s] for s in SOURCES]))
    model.train(); return out

# %% [markdown]
# ## Training loop
# `max_steps` is fixed after measuring throughput on the first steps so the cosine schedule ends at `TRAIN_HOURS`.

# %%
rng = np.random.default_rng(SEED + state["step"])
tokens_per_step = MICRO_BS * GRAD_ACCUM * WINDOW
last_ckpt = time.time(); t_seg = time.time(); seg_steps = 0
print(f"tokens/step {tokens_per_step:,}")

while state["max_steps"] is None or state["step"] < state["max_steps"]:
    step = state["step"]
    lr = lr_at(step, state["max_steps"] or 10**9)
    for g in opt.param_groups: g["lr"] = lr
    t0 = time.time(); loss_acc = 0.0
    for _ in range(GRAD_ACCUM):
        batch = sample_windows("train", MICRO_BS, WINDOW, rng)
        with torch.autocast("cuda", dtype=torch.float16):
            loss = compute_loss(batch, rng) / GRAD_ACCUM
        scaler.scale(loss).backward(); loss_acc += loss.item()
    scaler.unscale_(opt)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP).item()
    scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    state["train_seconds"] += time.time() - t0; state["tokens"] += tokens_per_step; state["step"] += 1; seg_steps += 1

    if state["max_steps"] is None and state["step"] == 30:
        sps = 30 / state["train_seconds"]
        state["max_steps"] = int(TRAIN_HOURS * 3600 * sps * 0.93)   # ~7% reserved for eval/checkpoint overhead
        print(f"throughput {sps*tokens_per_step:,.0f} tok/s -> max_steps {state['max_steps']} "
              f"(~{state['max_steps']*tokens_per_step/1e6:.0f}M tokens)")

    if state["step"] % 25 == 0:
        el = time.time() - t_seg
        print(f"step {state['step']}/{state['max_steps']} loss {loss_acc:.4f} lr {lr:.2e} gnorm {gnorm:.2f} "
              f"{seg_steps*tokens_per_step/el:,.0f} tok/s  tokens {state['tokens']/1e6:.0f}M")
        state["history"].append({"step": state["step"], "loss": loss_acc, "lr": lr, "tokens": state["tokens"]})

    done = state["max_steps"] is not None and state["step"] >= state["max_steps"]
    if state["step"] % EVAL_EVERY == 0 or done:
        ev = evaluate()
        state["history"].append({"step": state["step"], "eval": ev, "tokens": state["tokens"]})
        print(f"  EVAL step {state['step']}: " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items()))
        if ev["mean"] < state["best_val"]:
            state["best_val"] = ev["mean"]; push_best(ev["mean"]); print("  -> new best pushed")
    if time.time() - last_ckpt > CKPT_MINUTES * 60 or done:
        push_latest(); last_ckpt = time.time(); print("  -> checkpoint pushed")

print("TRAINING DONE. best val", state["best_val"])

# %% [markdown]
# ## Loss curves

# %%
import matplotlib.pyplot as plt
h = state["history"]
tr = [(x["tokens"] / 1e6, x["loss"]) for x in h if "loss" in x]
ev = [(x["tokens"] / 1e6, x["eval"]) for x in h if "eval" in x]
plt.figure(figsize=(10, 5))
plt.plot(*zip(*tr), label="train", alpha=0.6)
for s in SOURCES + ["mean"]:
    plt.plot([t for t, _ in ev], [e[s] for _, e in ev], marker="o", label=f"val {s}")
plt.xlabel("tokens seen (M)"); plt.ylabel("loss"); plt.title(f"Phase 1 — {ARCH}"); plt.legend(); plt.grid(True)
plt.savefig(f"{WORK}/loss_{ARCH}.png", dpi=150); plt.show()
api.upload_file(path_or_fileobj=f"{WORK}/loss_{ARCH}.png", path_in_repo=f"loss_{ARCH}.png", repo_id=MODEL_REPO)
