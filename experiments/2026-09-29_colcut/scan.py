"""Birth-COLUMN cut on the SHIPPABLE base (k3m config), 10 baked MLPs.

The idea (the transpose of the dead-ROW cut we shipped): a source born at layer s has one column per parent neuron
p, weighted by d2_p = phi(alpha_p)/sigma_p. For |alpha_p| large -- dead OR saturated -- d2_p ~ 0 and the column
carries almost nothing, yet it is transported every layer for the rest of the source's life. NEGATIVE 48 looked at
this only at |alpha| < 3 (cost-weighted saving ~7%) and rejected it on 24 Sep when bigger levers existed. From the
per-layer alpha distributions (2026-09-27_residual/dump10.npz) the modelled saving grows much faster than the
removed birth weight:
    |alpha|>3.0: 8.2% of legs+D, 0.03% of weight  -> util ~23.0%
    |alpha|>2.5: 12.8%, 0.23%                     -> util ~22.0%
    |alpha|>2.0: 19.5%, 1.18%                     -> util ~20.5%
    |alpha|>1.5: 29.1%, 4.8%                      -> util ~18.3%
The weight share is only a proxy for accuracy; this measures the real thing.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-29_colcut/scan.py
Self-checks: colcut=None reproduces the k3m torch number; colcut=1e9 (every column kept) must equal it exactly.
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
UTIL = {None: 24.835, 1e9: 24.835, 3.0: 23.00, 2.5: 21.97, 2.25: 21.2, 2.0: 20.47, 1.75: 19.4, 1.5: 18.32}

gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
print("10 MLPs, shippable k3m config. proj = 5.3906e-9 (graded k3m) x raw ratio x util ratio (MODELLED util).\n")
for cc in (None, 1e9, 3.0, 2.5, 2.25, 2.0, 1.75, 1.5):
    t0 = time.time()
    per = [float(((run_chain(Ws, colcut=cc, **K3M)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    proj = 5.3906e-9 * (r / base) * (UTIL[cc] / 24.835)
    out[str(cc)] = {"raw": r, "raw_x": r / base, "better": b, "util_model": UTIL[cc], "proj_adjusted": proj,
                    "per_mlp": per}
    print(f"  colcut {str(cc):>5}: raw {r:.4e}  raw x{r/base:.4f}  better {b}/10  util~{UTIL[cc]:5.2f}%  "
          f"proj {proj:.3e} (x{proj/5.3906e-9:.3f})  [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "scan_n10.json").write_text(json.dumps(out, indent=1))
