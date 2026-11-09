"""Render phase1_arch_template.py into one self-contained notebook per architecture.

    python build_arch_notebooks.py           ->  P1_arch1_llama.ipynb, P1_arch2_bart.ipynb, ...
    python build_arch_notebooks.py bart      ->  only P1_arch2_bart.ipynb
"""
import os, sys, tempfile
from build_nb import build

HERE = os.path.dirname(os.path.abspath(__file__))

BART_NOTE = """
# **v2 retrain.** The v1 BART run (`cs781-p1-bart`) collapsed: the decoder ignored the encoder and wrote the same
# continuation for every prompt. v2 trains into a **new** repo (`cs781-p1-bart-v2`, so v1 stays as report evidence and is
# never resumed) and adds BART text-infilling as a second objective, a `<mask>` token, start-of-training wiring checks and
# encoder-use measurements at every evaluation. Later phases must load BART from `cs781-p1-bart-v2`."""

ARCHS = {   # ARCH: (tag, family, model, objective, repo suffix, note)
    "llama": ("Arch 1", "Decoder-only Transformer",
              "`LlamaForCausalLM` (RoPE, RMSNorm, SwiGLU), random init",
              "next-token prediction on every token", "", ""),
    "bart": ("Arch 2", "Encoder-decoder Transformer",
             "`BartForConditionalGeneration`, random init",
             "mixture per micro-batch: continuation (encoder reads the prompt, decoder writes the continuation, loss on "
             "continuation only) and BART text-infilling denoising (decoder rebuilds a span-masked window)",
             "-v2", BART_NOTE),
    "prefixlm": ("Arch 3", "Encoder/decoder hybrid (Prefix-LM)",
                 "`LlamaForCausalLM` + bidirectional prefix attention mask, random init",
                 "next-token prediction on continuation tokens only", "", ""),
    "mistral_swa": ("Arch 4", "Efficient Transformer (sliding-window attention + GQA)",
                    "`MistralForCausalLM`, random init",
                    "next-token prediction on every token", "", ""),
}

only = set(sys.argv[1:])
assert only <= set(ARCHS), f"unknown arch(s): {only - set(ARCHS)}"
template = open(os.path.join(HERE, "phase1_arch_template.py"), encoding="utf-8").read()
for i, (arch, (tag, family, model, objective, suffix, note)) in enumerate(ARCHS.items(), 1):
    if only and arch not in only:
        continue
    src = (template.replace("{{ARCH}}", arch).replace("{{TAG}}", tag).replace("{{FAMILY}}", family)
           .replace("{{MODEL}}", model).replace("{{OBJECTIVE}}", objective)
           .replace("{{REPO_SUFFIX}}", suffix).replace("{{ARCH_NOTE}}", note))
    assert "{{" not in src, "unfilled placeholder"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
    build(f.name, os.path.join(HERE, f"P1_arch{i}_{arch}.ipynb"))
    os.remove(f.name)
