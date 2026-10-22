"""The ablation table from the Mac runs (runs_mac/, Mac-vs-Mac ratios), in the cloud's format (tables.py)."""
import json
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
ORDER = (H / "arms/ORDER.txt").read_text().split()
B = 2 ** 41


def load(name):
    f = H / f"runs_mac/{name}_n32.json"
    if not f.exists():
        return None
    return {p["mlp_index"]: p for p in json.loads(f.read_text())["results"]["per_mlp"]}


def agg(d, idx):
    return (np.mean([d[i]["adjusted_final_layer_score"] for i in idx]), np.mean([d[i]["final_layer_mse"] for i in idx]),
            np.mean([max(0.1, d[i]["flops_used"] / B) for i in idx]))


base = load("base")
print("| arm | set | adjusted | raw | util | adj × | raw × | util × | MLPs worse (adj) |\n|---|---|---|---|---|---|---|---|---|")
for name in ORDER:
    d = load(name)
    if d is None:
        print(f"| {name} | — | not run on the Mac | | | | | | |")
        continue
    for tag, idx in (("0-31", range(32)), ("10-31", range(10, 32))):
        a, r, u = agg(d, idx)
        a0, r0, u0 = agg(base, idx)
        worse = sum(d[i]["adjusted_final_layer_score"] > base[i]["adjusted_final_layer_score"] for i in idx)
        fails = sum(bool(d[i].get("traceback")) for i in idx)
        print(f"| {name} | {tag} | {a:.4e} | {r:.4e} | {100*u:.2f}% | {a/a0:.4f} | {r/r0:.4f} | {u/u0:.4f} | {worse}/{len(idx)}"
              + (f" ({fails} failed)" if fails else "") + " |")
