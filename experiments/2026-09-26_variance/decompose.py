"""How much of the variance channel's x1.54 does a CHEAP correction buy?

The exact-variance substitution is x1.54 (a lower bound set by oracle noise). Here the measured relative
error e = (var_chain - var_true)/var_true of MLP 0 is decomposed and each piece is fed back through
chain.py's new `varfix` switch (var <- var * g, applied EARLY so legs, D, K4d and nonlin_slices all see it):

  exact      g = 1/(1+e)                      -> the ceiling, reproduces the var0 oracle arm
  level      g = 1/(1+mean_l e)               -> ONE scalar per layer, O(n) and free
  fit        g = 1/(1+X beta_l)               -> the per-layer least squares on cheap observables
  level+log  g = 1/(1+c_l + b log(v/<v>))     -> two numbers per layer, O(n) and free
  noiseonly  g = 1/(1+eps), eps = (A-B)/2     -> a CONTROL: pure oracle noise, must do nothing useful

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-26_variance/decompose.py [n_mlps]

Self-check: the base arm must print 2.4074e-08 on MLP 0 / 2.3135e-08 on 10 MLPs.
Every correction is built from MLP 0's oracle ONLY, so its value on MLPs 1.. is a held-out number.
"""
import sys
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

CHAIN = Path(__file__).resolve().parent.parent / "2026-09-24_k3-chain"
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
NM = int(sys.argv[1]) if len(sys.argv) > 1 else 1
STAGE = sys.argv[2] if len(sys.argv) > 2 else "late"   # "late" (nonlin_slices only) or "early" (everything)

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)

d = np.load(OUT / "diag_mlp0.npz")
e, X, noise, vt = d["e"], d["X"], d["noise"], d["var_true"]      # e, X from MLP 0
L, n = e.shape

lvl = e.mean(1)                                                  # per-layer level bias
fit = np.zeros_like(e)
lvlog = np.zeros_like(e)
for l in range(L):
    beta, *_ = np.linalg.lstsq(X[l], e[l], rcond=None)
    fit[l] = X[l] @ beta
    Z = np.stack([X[l][:, 2], np.ones(n)], 1)                    # log var and a constant only
    b2, *_ = np.linalg.lstsq(Z, e[l], rcond=None)
    lvlog[l] = Z @ b2

arms = {
    "base":                None,
    "exact (ceiling)":     1.0 / (1.0 + e),
    "level only":          1.0 / (1.0 + lvl[:, None] * np.ones((1, n))),
    "level + log var":     1.0 / (1.0 + lvlog),
    "full linear fit":     1.0 / (1.0 + fit),
    "CONTROL: noise only": 1.0 / (1.0 + noise),
}

gt_all = np.load(BAKED / "gt.npy")
res = {k: [] for k in arms}
for m in range(NM):
    W = np.load(BAKED / f"weights_{m}.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    g = torch.tensor(gt_all[m][-1])
    for name, vf in arms.items():
        kw = {("varfix" if STAGE == "early" else "varfix_late"): vf}
        pred, _ = run_chain(Ws, **kw, **K3L)
        res[name].append(float(((pred[-1] - g) ** 2).mean()))
    print(f"MLP {m}: " + "  ".join(f"{k}={res[k][-1]:.4e}" for k in arms), flush=True)

print(f"\nmean over {NM} MLP(s), stage={STAGE}   (corrections fitted on MLP 0's oracle only)")
b = float(np.mean(res["base"]))
for k in arms:
    v = float(np.mean(res[k]))
    print(f"  {k:22s} {v:.4e}   x{b / v:.3f} of base")
np.savez(OUT / f"decompose_{STAGE}_n{NM}.npz", **{k: np.array(v) for k, v in res.items()},
         level=lvl, fit=fit, lvlog=lvlog)
