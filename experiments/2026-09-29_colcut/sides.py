"""Which side of the birth-column cut is expensive? 10 MLPs, shippable k3m config.

scan_n10: the symmetric cut |alpha_p| > 3.0 removes 0.03% of the d2-weighted birth mass yet costs raw x1.122 --
a ~400x amplification of the proxy. NEGATIVE 48's premise was that for |alpha| large "all hub weights (d2, alpha,
dK3) are ~0". Physical reading: a DEAD parent (alpha << 0) outputs ~0 and passes nothing; a SATURATED parent
(alpha >> 0) has ReLU ~ identity, so kappa3 passes THROUGH it and its column may carry transported content rather
than new births. Split the two.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-29_colcut/sides.py
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
ARMS = [("base", None),
        ("DEAD side only, alpha < -3.0", (-3.0, INF)), ("DEAD side only, alpha < -2.5", (-2.5, INF)),
        ("DEAD side only, alpha < -2.0", (-2.0, INF)), ("DEAD side only, alpha < -1.5", (-1.5, INF)),
        ("SATURATED side only, alpha > 3.0", (-INF, 3.0)), ("SATURATED side only, alpha > 2.5", (-INF, 2.5))]
gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
# the dead-side column fraction, cost-weighted by source lifetime, from the dumped alpha distributions
d = np.load(Path(__file__).resolve().parent.parent / "2026-09-27_residual" / "dump10.npz")
al = d["mu_pre"] / np.sqrt(d["var"]); L = al.shape[1]; w = np.array([L - 1 - s for s in range(L - 1)], float)
def dropped(lo, hi):
    return sum(w[s] * float(((al[:, s, :] < lo) | (al[:, s, :] > hi)).mean()) for s in range(L - 1)) / w.sum()
out, base, bper = {}, None, None
for tag, cc in ARMS:
    t0 = time.time()
    per = [float(((run_chain(Ws, colcut=cc, **K3M)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    fr = 0.0 if cc is None else dropped(*cc)
    util = 24.835 * (1 - 0.9 * fr)
    proj = 5.3906e-9 * (r / base) * (util / 24.835)
    out[tag] = {"colcut": str(cc), "raw": r, "raw_x": r / base, "better": b, "dropped_frac": fr,
                "util_model": util, "proj_adjusted": proj, "per_mlp": per}
    print(f"  {tag:34s} raw x{r/base:.4f}  better {b}/10  cols dropped {100*fr:4.1f}%  util~{util:5.2f}%  "
          f"proj {proj:.3e} (x{proj/5.3906e-9:.3f})  [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "sides_n10.json").write_text(json.dumps(out, indent=1))
