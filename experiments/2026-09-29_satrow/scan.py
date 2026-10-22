"""Saturated-ROW cut of the D pairing, alone and stacked on the matched dead cut. 10 baked MLPs.

Proposed independently by two lenses of the 2026-09-29 ideation workflow (convergence is evidence): D_ij =
kappa3[i,i,j] with i the doubled index. For a neuron saturated at the CURRENT layer (alpha_i > theta) its row of D
enters the nonlinear step only with weights that vanish there (d2_i ~ 0, d3_i ~ alpha phi(alpha)) and in K21, where
the pass-through part cancels against what the legs already carry. So the rows i of the expensive pairing
sum_s L1_s B_s^T + L2_s A_s^T could be skipped: legs untouched, zero construction, and it is the saturated-side
counterpart of the dead-row cut. Saving: fraction of saturated rows x the D-pairing share (10.6% of B).

NOTE the contrast with 2026-09-29_colcut: cutting saturated COLUMNS (parents) cost x1.12, because kappa3 passes
through a saturated neuron and its column carries transported content. This cuts saturated ROWS of D at the layer
where D is READ -- a different object. The measurement decides.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-29_satrow/scan.py
"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
K3M = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-2.5, k31mult=0.65)
INF = 1e9
gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
# fraction of saturated rows per layer (current-layer alpha), for the modelled saving
dd = np.load(OUT.parent / "2026-09-27_residual" / "dump10.npz")
al = dd["mu_pre"] / np.sqrt(dd["var"])
def sat_frac(th):   # layers that have a D pairing: 1..14 (layer 0 has no sources, the last layer reads only)
    return float(np.mean([(al[:, l, :] > th).mean() for l in range(1, 15)]))

ARMS = [("k3m base", {}), ("satrow 3.0", dict(satrow=3.0)), ("satrow 2.5", dict(satrow=2.5)),
        ("satrow 2.0", dict(satrow=2.0)), ("satrow 1.5", dict(satrow=1.5)),
        ("matched dead -2.25", dict(deadcut=-2.25, colcut=(-2.25, INF))),
        ("matched -2.25 + satrow 2.5", dict(deadcut=-2.25, colcut=(-2.25, INF), satrow=2.5)),
        ("matched -2.25 + satrow 2.0", dict(deadcut=-2.25, colcut=(-2.25, INF), satrow=2.0))]
out, base, bper = {}, None, None
print("10 MLPs, shippable k3m config. util modelled: satrow saves sat_frac x 10.6% of B (the D-pairing share).\n")
for tag, kw in ARMS:
    cfg = dict(K3M); cfg.update(kw)
    t0 = time.time()
    per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    sf = sat_frac(kw["satrow"]) if "satrow" in kw else 0.0
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": r / base, "better": b, "sat_frac": sf,
                "per_mlp": per}
    print(f"  {tag:28s} raw {r:.4e}  raw x{r/base:.4f}  better {b}/10  sat rows {100*sf:4.1f}% "
          f"(-{100*sf*0.106:.2f} pts of B)  [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "scan_n10.json").write_text(json.dumps(out, indent=1))
