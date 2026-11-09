# Standalone arch2 (BART) diagnostic -- paste into a fresh Colab notebook (GPU or CPU runtime); needs only the HF secret.
# Answers: does the trained model use the prompt at all, and does fp16 generation change anything?
!pip -q install -U transformers
import os, json, torch
import torch.nn.functional as F
from huggingface_hub import login, whoami, snapshot_download
from transformers import BartForConditionalGeneration, PreTrainedTokenizerFast

try:
    from google.colab import userdata; HF_TOKEN = userdata.get("HF")
except Exception:
    HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN); user = whoami(token=HF_TOKEN)["name"]
mdir = os.path.join(snapshot_download(f"{user}/cs781-p1-bart", token=HF_TOKEN, allow_patterns=["best/*"]), "best")
ddir = snapshot_download(f"{user}/cs781-p1-data", repo_type="dataset", token=HF_TOKEN, allow_patterns=["heldout_*.json"])
P = json.load(open(f"{ddir}/heldout_prompts.json")); T = json.load(open(f"{ddir}/heldout_truth.json"))
ks = list(P)[::max(1, len(P) // 24)][:24]                       # spread across the 4 sources
tok = PreTrainedTokenizerFast.from_pretrained(mdir)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m32 = BartForConditionalGeneration.from_pretrained(mdir).to(dev).eval()

def encode(texts):                                              # same as Part B: last 768 tokens, right-padded
    ids = [tok.encode(t, add_special_tokens=False)[-768:] for t in texts]
    L = max(map(len, ids))
    inp = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in ids], device=dev)
    att = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in ids], device=dev)
    return inp, att

inp, att = encode([P[k]["text"] for k in ks])
lab = [tok.encode(T[k]["text"], add_special_tokens=False)[:64] for k in ks]
n = min(map(len, lab)); lab = torch.tensor([x[:n] for x in lab], device=dev)

def per_position_loss(model, i, a):
    logits = model(input_ids=i, attention_mask=a, labels=lab).logits.float()
    return F.cross_entropy(logits.transpose(1, 2), lab, reduction="none").mean(0)    # (n,)

kw = dict(max_new_tokens=60, min_new_tokens=60, do_sample=False, no_repeat_ngram_size=4,
          pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, suppress_tokens=[tok.pad_token_id])
with torch.inference_mode():
    right = per_position_loss(m32, inp, att)
    wrong = per_position_loss(m32, inp.roll(1, 0), att.roll(1, 0))   # every continuation paired with another prompt
    print(f"C  loss on true continuation (fp32), right prompt vs mismatched prompt:")
    print(f"     first 8 tokens: {right[:8].mean():.3f} vs {wrong[:8].mean():.3f}   all {n} tokens: {right.mean():.3f} vs {wrong.mean():.3f}")
    g32 = tok.batch_decode(m32.generate(input_ids=inp[:6], attention_mask=att[:6], **kw), skip_special_tokens=True)
    print(f"B  distinct greedy outputs for 6 prompts, fp32: {len(set(g32))}/6")
    if dev == "cuda":
        m16 = BartForConditionalGeneration.from_pretrained(mdir).to(dev, torch.float16).eval()
        e16 = m16.get_encoder()(input_ids=inp[:6], attention_mask=att[:6]).last_hidden_state
        g16 = tok.batch_decode(m16.generate(input_ids=inp[:6], attention_mask=att[:6], **kw), skip_special_tokens=True)
        print(f"B  distinct greedy outputs for 6 prompts, fp16: {len(set(g16))}/6")
        print(f"A  fp16 encoder output finite: {torch.isfinite(e16).all().item()}  max |h|: {e16.float().abs().max().item():.1f}")
    else:
        print("A/B fp16 checks skipped (CPU runtime)")
for k, g in zip(ks[:3], g32[:3]):
    print(f"\n[{k}] prompt ends: ...{' '.join(P[k]['text'].split()[-15:])}\n     fp32 output: {g[:150]}")
