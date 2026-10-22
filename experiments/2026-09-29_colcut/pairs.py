"""Prediction test of the consistency mechanism: matched (row, column) dead cuts. 10 MLPs.

sides.py found the dead-side column cut improves raw on 10/10 exactly at the dead-ROW cut's threshold (-2.5), and
explained it as consistency: a neuron gated to 0 on the row axis should not keep birthing kappa3 on the column axis.
If that is the mechanism it predicts that a MATCHED pair (deadcut = t, colcut dead side = t) beats the row cut alone
(deadcut = t, no column cut) at EVERY threshold t -- not only at -2.5 -- and that the joint optimum may move.
Base for each row of the table is the row-only cut at the same threshold.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-29_colcut/pairs.py
"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
BASE = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
            marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
            k4bparts=("sd",), pairx=("k22",), tet=3, k31mult=0.65)
INF = 1e9
gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
d = np.load(Path(__file__).resolve().parent.parent / "2026-09-27_residual" / "dump10.npz")
al = d["mu_pre"] / np.sqrt(d["var"]); L = al.shape[1]; w = np.array([L - 1 - s for s in range(L - 1)], float)
def dead_col_frac(t):
    return sum(w[s] * float((al[:, s, :] < t).mean()) for s in range(L - 1)) / w.sum()

def run(**kw):
    cfg = dict(BASE); cfg.update(kw)
    per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    return float(np.mean(per)), per

ref, refper = run(deadcut=-2.5)                     # k3m = the shipped configuration
print(f"k3m (row cut -2.5, no column cut): raw {ref:.4e}   (must be 2.3462e-08)\n")
out = {"k3m": {"raw": ref, "per_mlp": refper}}
print(f"{'t':>6} {'row only':>11} {'matched':>11} {'matched/row':>12} {'better':>7} {'matched/k3m':>12} {'dead cols':>10}")
for t in (-3.0, -2.75, -2.5, -2.25, -2.0):
    t0 = time.time()
    r_row, per_row = (ref, refper) if t == -2.5 else run(deadcut=t)
    r_mat, per_mat = run(deadcut=t, colcut=(t, INF))
    b = sum(1 for a, c in zip(per_mat, per_row) if a < c)
    out[str(t)] = {"row_only": r_row, "matched": r_mat, "matched_over_row": r_mat / r_row, "better_than_row": b,
                   "matched_over_k3m": r_mat / ref, "dead_col_frac": dead_col_frac(t),
                   "per_mlp_row": per_row, "per_mlp_matched": per_mat}
    print(f"{t:6.2f} {r_row:11.4e} {r_mat:11.4e} {r_mat/r_row:12.4f} {b:5d}/10 {r_mat/ref:12.4f} "
          f"{100*dead_col_frac(t):9.1f}%   [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "pairs_n10.json").write_text(json.dumps(out, indent=1))
