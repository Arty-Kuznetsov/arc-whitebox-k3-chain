"""Tables of RESULTS.md straight from res/ (no hand transcription).
    python summarize.py [grid|bases|conf|full|pairs] -> markdown on stdout"""
import json
import sys

import numpy as np

import grid
import ta

which = sys.argv[1] if len(sys.argv) > 1 else "grid"
GATES = {"B9": None, "K3M": 2.3462e-08, "FULL": 2.2392e-08}


def row(tag, ref="B9", mlps=ta.SCREEN):
    s = ta.summary(tag, ref, mlps)
    if s is None:
        return None
    return s


if which == "bases":
    print("| base | raw (10 MLPs) | gate | per MLP x1e8 |\n|---|---|---|---|")
    for b, gv in GATES.items():
        per = ta.per_mlp(b)
        if len(per) < 10:
            print(f"| {b} | {len(per)}/10 done | {gv} | |")
            continue
        r = float(np.mean([per[m] for m in ta.SCREEN]))
        print(f"| {b} | {r:.4e} | {'—' if gv is None else f'{gv:.4e}'} | "
              + " ".join(f"{per[m] * 1e8:.4f}" for m in ta.SCREEN) + " |")
elif which == "grid":
    rows = []
    for tag in grid.ORDER:
        s = row(tag)
        if s is not None:
            rows.append((tag, s))
    rows.sort(key=lambda z: (z[1]["status"] != "done", z[1]["x"]))
    print("| arm | status | MLPs | raw x B9 | better | verdict |\n|---|---|---|---|---|---|")
    for tag, s in rows:
        v = ("SURVIVES" if s["status"] == "done" and s["x"] < 1 and s["better"] >= 7 else
             "kill" if s["status"] in ("done", "KILLED") else s["status"])
        print(f"| {tag} | {s['status']} | {s['n']} | {s['x']:.4f} | {s['better']}/{s['n']} | {v} |")
    print(f"\n{sum(1 for _, s in rows if s['status'] in ('done', 'KILLED'))} of {len(grid.ORDER)} arms finished")
elif which in ("conf", "full", "pairs"):
    ref = "FULL" if which == "full" else "B9"
    mlps = ta.CONF if which == "conf" else ta.SCREEN
    import os
    tags = sorted(p.name for p in ta.RES.iterdir() if p.is_dir() and (
        (which == "full" and p.name.startswith("FULL+")) or (which == "pairs" and "+" in p.name and not p.name.startswith("FULL+"))
        or which == "conf"))
    print(f"| arm | MLPs | raw x {ref} | better |\n|---|---|---|---|")
    for tag in tags:
        s = row(tag, ref, mlps)
        if s is not None and (which != "conf" or s["n"] >= 1):
            print(f"| {tag} | {s['n']} | {s['x']:.4f} | {s['better']}/{s['n']} |")
