# Phase-2 BART checkpoint diagnostic (read-only). Paste into the BART notebook AFTER the "Tokenizer ... BIO tags" cell
# has run (uses TOK, TRAIN_IDS, VOCAB, p1_dir, DEVICE). Loads a fresh copy of the Phase-1 checkpoint; changes nothing.
from transformers import BartForConditionalGeneration

bart = BartForConditionalGeneration.from_pretrained(p1_dir("bart")).float().to(DEVICE).eval()
enc, dec = bart.model.encoder, bart.model.decoder
d = max(TRAIN_IDS, key=lambda k: len(TOK[k][0]))
x = torch.tensor([TOK[d][0][:128]], device=DEVICE); m = torch.ones_like(x)
x2 = x.clone(); x2[0, 40] = (x2[0, 40] + 1) % VOCAB

def ln_line(name, ln):
    w, b = ln.weight.detach().abs(), ln.bias.detach().abs()
    return f"  {name:30s} |gain| mean {w.mean().item():.2e}  max {w.max().item():.2e}   |bias| mean {b.mean().item():.2e}"

print("1) LayerNorm gains (initialised to 1.0)")
print(ln_line("encoder.layernorm_embedding", enc.layernorm_embedding))
for i, l in enumerate(enc.layers):
    print(ln_line(f"encoder.{i}.self_attn_layer_norm", l.self_attn_layer_norm))
    print(ln_line(f"encoder.{i}.final_layer_norm", l.final_layer_norm))
print(ln_line("decoder.layernorm_embedding", dec.layernorm_embedding))
for i, l in enumerate(dec.layers):
    print(ln_line(f"decoder.{i}.final_layer_norm", l.final_layer_norm))

pre_ln = {}                                   # largest |input| reaching each encoder LayerNorm (fp16 max is 65504)
hooks = [ln.register_forward_pre_hook(lambda mod, inp, n=n: pre_ln.__setitem__(n, inp[0].abs().max().item()))
         for n, ln in [(f"{i}.self_attn_ln", l.self_attn_layer_norm) for i, l in enumerate(enc.layers)]
                    + [(f"{i}.final_ln", l.final_layer_norm) for i, l in enumerate(enc.layers)]]
with torch.no_grad():
    h1 = enc(input_ids=x, attention_mask=m, output_hidden_states=True).hidden_states
    for h in hooks: h.remove()
    h2 = enc(input_ids=x2, attention_mask=m, output_hidden_states=True).hidden_states
    o1 = bart.model(input_ids=x, attention_mask=m, decoder_input_ids=x, decoder_attention_mask=m).last_hidden_state[0]
    o2 = bart.model(input_ids=x2, attention_mask=m, decoder_input_ids=x2, decoder_attention_mask=m).last_hidden_state[0]

print("\n2) Encoder: effect of changing token 40, after each layer (0 = embeddings + layernorm_embedding)")
for i, (a, b) in enumerate(zip(h1, h2)):
    a, b = a[0].float(), b[0].float()
    print(f"  after {i}: mean|h| {a.abs().mean().item():.2e}   change at pos 40 {(a[40] - b[40]).abs().max().item():.2e}"
          f"   at pos <40 {(a[:40] - b[:40]).abs().max().item():.2e}")
print("\n3) Largest |input| into each encoder LayerNorm:",
      {k: f"{v:.2e}" for k, v in pre_ln.items()}, " (fp16 overflows above 6.55e+04)")
print(f"\n4) BART_MODE='decoder' (document into encoder and decoder): mean|h| {o1.abs().mean().item():.2e}   change at pos 40 "
      f"{(o1[40] - o2[40]).abs().max().item():.2e}   at pos >40 {(o1[41:] - o2[41:]).abs().max().item():.2e}   "
      f"at pos <40 {(o1[:40] - o2[:40]).abs().max().item():.2e}")
del bart; gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
