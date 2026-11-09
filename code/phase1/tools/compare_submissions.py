#!/usr/bin/env python3
"""Side-by-side reader for Phase-1 predictions: the end of each prompt, the reference continuation (if you have one),
then every submission's continuation for the same ID. Standard library only.

Official test set (no references exist locally):
    python3 compare_submissions.py arch1=submission/submission_arch1.zip arch2=submission/submission_arch2.zip \
        arch3=submission/submission.zip --prompts test.json
  test.json = Phase_1_Pretraining/Test/test.json from the Exploration-Lab/CS-781-Capstone dataset on Hugging Face.

Local held-out set (files from <hf-user>/cs781-p1-<arch>/submissions/v1/), with the true continuation shown:
    python3 compare_submissions.py llama=llama/heldout_pred_greedy_ngram4.json bart=bart/heldout_pred_greedy_ngram4.json \
        --prompts heldout_prompts.json --truth llama/heldout_truth.json

Writes comparison.html (open it in a browser), or prints to the terminal with --text.
"""
import argparse, html, json, random, sys, textwrap, zipfile


def load(path):
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            return json.loads(z.read("prediction.json"))
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def id_key(k):
    return (0, int(k), "") if k.isdigit() else (1, 0, k)


def tail(text, n):
    words = text.split()
    return ("… " if len(words) > n else "") + " ".join(words[-n:])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subs", nargs="+", metavar="LABEL=FILE", help="submission .zip or prediction .json, e.g. arch1=submission_arch1.zip")
    ap.add_argument("--prompts", help="prompt file ({id: {text}}), e.g. the official test.json or heldout_prompts.json")
    ap.add_argument("--truth", help="reference continuations ({id: {text}}), e.g. heldout_truth.json")
    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--ids", nargs="+", help="specific IDs to show")
    pick.add_argument("--n", type=int, default=12, help="number of random IDs to show (default 12)")
    pick.add_argument("--all", action="store_true", help="show every ID")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random ID sample")
    ap.add_argument("--prompt-words", type=int, default=60, help="how many words of the prompt ending to show (default 60)")
    ap.add_argument("--out", default="comparison.html")
    ap.add_argument("--text", action="store_true", help="print to the terminal instead of writing HTML")
    args = ap.parse_args()

    subs = []
    for s in args.subs:
        if "=" not in s:
            sys.exit(f"expected LABEL=FILE, got {s!r}")
        label, path = s.split("=", 1)
        subs.append((label, path, load(path)))
    prompts = load(args.prompts) if args.prompts else {}
    truth = load(args.truth) if args.truth else {}

    all_ids = sorted(set().union(*(set(p) for _, _, p in subs)), key=id_key)
    if args.ids:
        ids = args.ids
    elif args.all:
        ids = all_ids
    else:
        ids = sorted(random.Random(args.seed).sample(all_ids, min(args.n, len(all_ids))), key=id_key)
    text_of = lambda d, k: d[k]["text"] if k in d else None

    if args.text:
        wrap = lambda t: textwrap.fill(t, 110, initial_indent="    ", subsequent_indent="    ")
        for k in ids:
            print("=" * 114 + f"\nID {k}")
            if text_of(prompts, k):
                print("  PROMPT ENDING:\n" + wrap(tail(text_of(prompts, k), args.prompt_words)))
            if text_of(truth, k):
                print("  REFERENCE:\n" + wrap(text_of(truth, k)))
            for label, _, p in subs:
                t = text_of(p, k)
                print(f"  {label} ({len(t.split()) if t else 0} words):\n" + wrap(t if t else "(missing)"))
        return

    def block(cls, label, body):
        return f'<div class="{cls}"><span class="label">{label}</span>{html.escape(body)}</div>'

    parts = []
    for k in ids:
        p, r = text_of(prompts, k), text_of(truth, k)
        cards = []
        for label, _, sub in subs:
            t = text_of(sub, k)
            wc = f' <span class="wc">· {len(t.split())} words</span>' if t else ""
            cards.append(block("card", html.escape(label) + wc, t if t else "(missing)"))
        parts.append(f'<section><h2>ID {html.escape(k)}</h2>'
                     + (block("prompt", "Prompt ending", tail(p, args.prompt_words)) if p else "")
                     + (block("ref", "Reference continuation", r) if r else "")
                     + f'<div class="grid">{"".join(cards)}</div></section>')
    sources = " · ".join(f"<b>{html.escape(l)}</b> = {html.escape(path)}" for l, path, _ in subs)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predictions side by side</title>
<style>
:root {{ --bg:#fafaf8; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e2e1db; --prompt:#f0efe9; --ref:#ecf4ec; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161615; --fg:#e8e7e2; --muted:#9b9a94; --card:#1f1f1d; --line:#34342f; --prompt:#262622; --ref:#1d291e; }} }}
body {{ margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:24px 16px 64px; }}
h1 {{ font-size:20px; margin:0 0 4px; }}
.meta {{ color:var(--muted); font-size:13px; margin-bottom:20px; }}
section {{ border-top:1px solid var(--line); padding:20px 0; }}
h2 {{ font-size:15px; margin:0 0 10px; }}
.prompt, .ref, .card {{ padding:10px 12px; border-radius:6px; overflow-wrap:anywhere; }}
.prompt {{ background:var(--prompt); color:var(--muted); margin-bottom:10px; }}
.ref {{ background:var(--ref); margin-bottom:10px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(280px, 1fr)); gap:10px; }}
.card {{ background:var(--card); border:1px solid var(--line); }}
.label {{ display:block; margin-bottom:4px; font-size:12px; font-weight:600; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); }}
.wc {{ font-weight:400; text-transform:none; letter-spacing:0; }}
</style></head><body><main>
<h1>Predictions side by side</h1>
<div class="meta">{len(ids)} of {len(all_ids)} IDs · {sources}</div>
{"".join(parts)}
</main></body></html>"""
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"wrote {args.out} ({len(ids)} IDs) -- open it in a browser")


if __name__ == "__main__":
    main()
