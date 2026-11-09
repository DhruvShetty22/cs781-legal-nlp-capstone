"""Render phase6_summ_template.py into one self-contained summarization notebook per Phase-1 architecture.

    python build_p6_notebooks.py        ->  P6_arch1_llama.ipynb, P6_arch2_bart.ipynb, ...
    python build_p6_notebooks.py bart   ->  only P6_arch2_bart.ipynb
"""
import os, sys, tempfile
from build_nb import build

HERE = os.path.dirname(os.path.abspath(__file__))

DECODER_ONLY = ("`[document input | SEP_TEXT | summary]` in one sequence of at most 1024 positions; loss on the summary "
                "tokens; generation continues after `SEP_TEXT`")
ARCHS = {   # ARCH: (tag, family, model, input format)
    "llama": ("Arch 1", "Decoder-only Transformer", "Phase-1 `LlamaForCausalLM`, causal attention",
              DECODER_ONLY),
    "bart": ("Arch 2", "Encoder-decoder Transformer",
             "Phase-1 v2 `BartForConditionalGeneration` (`cs781-p1-bart-v2`)",
             "the encoder reads the document input (+ `<mask>` end marker when Phase-1 v2 trained with it); the decoder "
             "writes the summary starting from `<\\|endoftext\\|>` (as in Phase-1 continuation)"),
    "prefixlm": ("Arch 3", "Encoder/decoder hybrid (Prefix-LM)",
                 "Phase-1 prefix-LM `LlamaForCausalLM`: bidirectional attention over the prefix, causal after it",
                 DECODER_ONLY + "; the document input + `SEP_TEXT` is the bidirectional prefix (as in Phase-1 "
                 "pretraining), the summary is causal"),
    "mistral_swa": ("Arch 4", "Efficient Transformer (sliding-window attention + GQA)",
                    "Phase-1 `MistralForCausalLM`, causal sliding-window attention", DECODER_ONLY),
}

only = set(sys.argv[1:])
assert only <= set(ARCHS), f"unknown arch(s): {only - set(ARCHS)}"
template = open(os.path.join(HERE, "phase6_summ_template.py"), encoding="utf-8").read()
for i, (arch, (tag, family, model, inp)) in enumerate(ARCHS.items(), 1):
    if only and arch not in only:
        continue
    src = (template.replace("{{ARCH}}", arch).replace("{{TAG}}", tag).replace("{{FAMILY}}", family)
           .replace("{{MODEL}}", model).replace("{{INPUT}}", inp))
    assert "{{" not in src, "unfilled placeholder"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(src)
    build(f.name, os.path.join(HERE, f"P6_arch{i}_{arch}.ipynb"))
    os.remove(f.name)
