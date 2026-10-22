"""Replace the TABLE_T1..T4 placeholders of RESULTS.md with the tables rendered by make_tables.py from the JSON
(or refresh them: the rendered blocks are fenced by <!-- TABLE_Tk --> ... <!-- /TABLE_Tk --> markers).

    ../../.venv-torch/bin/python fill_tables.py
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = HERE / "RESULTS.md"
s = res.read_text()
for k, sec in (("T1", "t1"), ("T2", None), ("T3", "t3"), ("T4", "t4")):
    if sec is None:   # T2 = the pricing table of every band arm (t1 + t1b + t1c), rendered by t2_price.py
        out = subprocess.run([sys.executable, str(HERE / "t2_price.py")], capture_output=True, text=True).stdout
        lines = [ln for ln in out.splitlines() if ln.strip()]
        def key(ln):
            try:
                return float(ln.split()[-4])
            except (ValueError, IndexError):
                return 9.0
        head, body = lines[0], sorted(lines[1:], key=key)
        md = ["| arm | raw × k3m | util (model) | util × | **projected ×** | max leg error | max cond(P) | kill ×0.97 |",
              "|---|---|---|---|---|---|---|---|"]
        for ln in body:
            parts = ln.split()
            if len(parts) < 8 or parts[-1] not in ("KILL", "survives", "anchor"):
                continue
            tag = " ".join(parts[:-7])
            rx, u, ux, px, le, cd, v = parts[-7:]
            md.append(f"| {tag} | {rx} | {u} | {ux} | **{px}** | {le} | {cd} | {v} |")
        block = "\n".join(md)
    else:
        block = subprocess.run([sys.executable, str(HERE / "make_tables.py"), sec], capture_output=True,
                               text=True).stdout.strip()
    fenced = f"<!-- TABLE_{k} -->\n{block}\n<!-- /TABLE_{k} -->"
    if f"<!-- TABLE_{k} -->" in s:
        s = re.sub(rf"<!-- TABLE_{k} -->.*?<!-- /TABLE_{k} -->", lambda m: fenced, s, flags=re.S)
    else:
        s = s.replace(f"TABLE_{k}", fenced, 1)
res.write_text(s)
print("filled", [k for k in ("T1", "T2", "T3", "T4") if f"<!-- TABLE_{k} -->" in s])
