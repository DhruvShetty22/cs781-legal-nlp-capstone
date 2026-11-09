"""Convert a percent-format .py file ('# %%' / '# %% [markdown]' cell markers) into a .ipynb."""
import json, sys, re

def build(src, dst):
    cells, cur, kind = [], [], None
    def flush():
        if kind is None: return
        text = "\n".join(cur).strip("\n")
        if kind == "markdown":
            text = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in text.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.splitlines(True)})
    for line in open(src, encoding="utf-8").read().splitlines():
        m = re.match(r"^# %%( \[markdown\])?\s*$", line)
        if m:
            flush(); cur = []; kind = "markdown" if m.group(1) else "code"
        elif kind is not None:
            cur.append(line)
    flush()
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    json.dump(nb, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"{dst}: {len(cells)} cells")

if __name__ == "__main__":
    build(sys.argv[1], sys.argv[2])
