"""Pareto front of results.jsonl: (utilisation, raw) -> projected adjusted score.

Projection, calibrated so that the k3l row reproduces its two MEASURED production numbers
(100-MLP flopscope run: final 2.1837e-8 at 26.27% of B -> adjusted 5.7387e-9, graded ~5.57e-9):
  raw_prod  = raw(torch, float64, n MLPs) * 0.94937      # torch n=4 -> flopscope float32 on 100 MLPs
  util_prod = util(model)                * 1.02665      # model -> measured k3j/k3l utilisation
  adjusted  = raw_prod * max(0.1, util_prod);  grader ~ 0.971 * adjusted
The raw factor folds two things: the n=4 -> n=10 subset shift (1.0058) and float32/flopscope vs the
torch chain (0.9439).  Both are measured on k3l only, and are ASSUMED config-independent.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
R_FAC = {4: 0.94937, 10: 0.94391}   # n=4: subset shift 1.0058 x float32/flopscope 0.9439;  n=10: 0.9439 only
U_FAC, GRADER = 1.02665, 0.971

rows, seen = [], {}
for fn in ("results.jsonl", "results10.jsonl"):
  for line in open(HERE / fn):
    r = json.loads(line)
    seen[(r["name"], r["n"])] = r                      # last run of a name wins
for r in seen.values():
    rp = r["raw"] * R_FAC[r["n"]]
    up = r["util"] * U_FAC
    rows.append(dict(name=r["name"], raw=r["raw"], raw_p=rp, util_p=up,
                     adj=rp * max(0.1, up), n=r["n"]))
base = next(x for x in rows if x["name"] == "k3l")
rows.sort(key=lambda x: x["adj"])
print(f"{'config':18s} {'n':>2s} {'raw(torch)':>11s} {'raw->prod':>10s} {'util':>7s} "
      f"{'adjusted':>10s} {'grader':>9s} {'vs k3l':>7s} {'pareto':>7s}")
front = []
for x in sorted(rows, key=lambda y: y["util_p"]):
    if not front or x["raw_p"] < min(f["raw_p"] for f in front):
        front.append(x)
fr = {f["name"] for f in front}
for x in rows:
    print(f"{x['name']:18s} {x['n']:2d} {x['raw']:11.4e} {x['raw_p']:10.4e} {100*x['util_p']:6.2f}% "
          f"{x['adj']:10.4e} {GRADER*x['adj']:9.4e} {x['adj']/base['adj']:7.3f} "
          f"{'  *' if x['name'] in fr else '':>7s}")
print(f"\n* = on the (util, raw) Pareto front.  k3l = the packaged submission (measured 5.7387e-9 local).")
ok = [x for x in rows if x["util_p"] <= 0.10 and x["raw_p"] <= 2.5e-8]
print(f"configs with util <= 10% AND raw <= 2.5e-8: {len(ok)}" + (f" -> {[x['name'] for x in ok]}" if ok else ""))
