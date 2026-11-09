"""Render phase2_lner_template.py into one self-contained L-NER notebook per Phase-1 architecture.

    python build_p2_notebooks.py        ->  P2_arch1_llama.ipynb, P2_arch2_bart.ipynb, ...
    python build_p2_notebooks.py bart   ->  only P2_arch2_bart.ipynb
"""
import os, sys, tempfile
from build_nb import build

HERE = os.path.dirname(os.path.abspath(__file__))

ARCHS = {   # ARCH: (tag, family, model)
    "llama": ("Arch 1", "Decoder-only Transformer",
              "Phase-1 `LlamaForCausalLM` body (`LlamaModel`), causal attention by default (`ATTN`)"),
    "bart": ("Arch 2", "Encoder-decoder Transformer",
             "Phase-1 `BartForConditionalGeneration`: encoder only, or encoder + decoder (`BART_MODE`)"),
    "prefixlm": ("Arch 3", "Encoder/decoder hybrid (Prefix-LM)",
                 "Phase-1 prefix-LM `LlamaModel`, whole window as the bidirectional prefix by default (`ATTN`)"),
    "mistral_swa": ("Arch 4", "Efficient Transformer (sliding-window attention + GQA)",
                    "Phase-1 `MistralForCausalLM` body (`MistralModel`), causal sliding-window attention by default (`ATTN`)"),
}

only = set(sys.argv[1:])
assert only <= set(ARCHS), f"unknown arch(s): {only - set(ARCHS)}"
template = open(os.path.join(HERE, "phase2_lner_template.py"), encoding="utf-8").read()
for i, (arch, (tag, family, model)) in enumerate(ARCHS.items(), 1):
    if only and arch not in only:
        continue
    src = (template.replace("{{ARCH}}", arch).replace("{{TAG}}", tag)
           .replace("{{FAMILY}}", family).replace("{{MODEL}}", model))
    assert "{{" not in src, "unfilled placeholder"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
    build(f.name, os.path.join(HERE, f"P2_arch{i}_{arch}.ipynb"))
    os.remove(f.name)
