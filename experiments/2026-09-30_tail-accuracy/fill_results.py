"""Refresh the generated blocks of RESULTS.md (<!-- NAME --> ... <!-- /NAME -->) from res/:
FRONTIER = frontier.py, PAIRS = summarize.py pairs (sorted by x), GRID = the survivors of summarize.py grid."""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = HERE / "RESULTS.md"
s = res.read_text()
run = lambda *a: subprocess.run([sys.executable, *a], capture_output=True, text=True, cwd=HERE).stdout.strip()
pairs = run("summarize.py", "pairs").splitlines()
head, body = pairs[:2], sorted(pairs[2:], key=lambda ln: float(ln.split("|")[3]) if ln.count("|") > 4 else 9.0)
blocks = {"FRONTIER": run("frontier.py"), "PAIRS": "\n".join(head + body)}
for k, v in blocks.items():
    fenced = f"<!-- {k} -->\n{v}\n<!-- /{k} -->"
    if f"<!-- {k} -->" in s:
        s = re.sub(rf"<!-- {k} -->.*?<!-- /{k} -->", lambda m: fenced, s, flags=re.S)
    else:
        s = s.replace(f"{k}_TABLE", fenced, 1)
res.write_text(s)
print("filled", [k for k in blocks if f"<!-- {k} -->" in s])
