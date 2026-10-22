"""Where is the per-neuron PRE-ACTIVATION VARIANCE of the chain wrong, and is the error predictable
from quantities the chain already has at that layer?

Substituting the oracle's exact var at every layer is x1.54 on the final MSE (the biggest single accuracy
channel; experiments/2026-09-26_accuracy/RESULTS.md). The LEVEL is right (slope 1.0006) -- the per-neuron
PATTERN is what costs, at 1e-4..1e-3 relative. If that pattern is a smooth function of alpha, var, the
kurtosis, ... then it is an O(n) correction and essentially free.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-26_variance/diagnose.py

Self-check: the chain arm must reproduce MLP 0's logged k3l value 2.4074e-08.
Two independent 1e7 oracles (default seed and 777) give both the pooled truth and the noise floor.
"""
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

CHAIN = Path(__file__).resolve().parent.parent / "2026-09-24_k3-chain"
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)

W = np.load(BAKED / "weights_0.npy")
gt = np.load(BAKED / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
L, n = len(Ws), W.shape[-1]

oa = np.load(CHAIN / "runs" / "mc_oracle_0.npz")
ob = np.load(CHAIN / "runs" / "mc_oracle_0_s777.npz")
print(f"oracles: N={float(oa['N']):.3g} and {float(ob['N']):.3g}")

pred, info = run_chain(Ws, record=True, **K3L)
mse = float(((pred[-1] - torch.tensor(gt[-1])) ** 2).mean())
print(f"self-check: final-layer MSE {mse:.4e}   (must be 2.4074e-08)")

va, vb = oa["var"].astype(np.float64), ob["var"].astype(np.float64)
vt = 0.5 * (va + vb)                      # pooled truth
noise = np.abs(va - vb) / (2.0 * vt)      # |A-B|/2 = the 1-sigma spread of the pooled value, relative

rows = []
print(f"\n{'l':>2} {'rms e':>9} {'mean e':>10} {'slope':>7} {'rms noise':>9} {'signal/noise':>12}")
feat_names = ["alpha", "alpha^2", "log var", "kurt", "skew", "rowsum C0^2", "1"]
for l in range(L):
    vc = info[l]["var"].numpy()
    e = (vc - vt[l]) / vt[l]
    sd = np.sqrt(vt[l])
    alpha = info[l]["mu_pre"].numpy() / sd
    k4d = info[l]["K4d"].numpy() / (vt[l] ** 2)
    skew = info[l]["t"].numpy() / (sd ** 3)
    D = info[l]["D"].numpy()
    rs = np.abs(D).sum(1) / (vt[l] * np.sqrt(n))       # a cheap proxy for off-diagonal weight
    slope = float(np.polyfit(vt[l], vc, 1)[0])
    nz = np.sqrt((noise[l] ** 2).mean())
    print(f"{l:2d} {np.sqrt((e**2).mean()):9.2e} {e.mean():10.2e} {slope:7.4f} {nz:9.2e} "
          f"{np.sqrt((e**2).mean())/nz:12.1f}")
    X = np.stack([alpha, alpha ** 2, np.log(vt[l] / vt[l].mean()), k4d, skew, rs, np.ones(n)], 1)
    rows.append((l, e, X, noise[l]))

print("\nper-layer least squares of e on the cheap observables (R^2, then the coefficients):")
print(f"{'l':>2} {'R2':>6}  " + "  ".join(f"{k:>11}" for k in feat_names))
for l, e, X, nz in rows:
    beta, *_ = np.linalg.lstsq(X, e, rcond=None)
    r2 = 1.0 - ((e - X @ beta) ** 2).sum() / ((e - e.mean()) ** 2).sum()
    print(f"{l:2d} {r2:6.3f}  " + "  ".join(f"{b:11.3e}" for b in beta))

print("\npooled fit over layers 1..15 (one coefficient set for the whole chain):")
Eall = np.concatenate([e for l, e, X, nz in rows if l >= 1])
Xall = np.concatenate([X for l, e, X, nz in rows if l >= 1])
beta, *_ = np.linalg.lstsq(Xall, Eall, rcond=None)
r2 = 1.0 - ((Eall - Xall @ beta) ** 2).sum() / ((Eall - Eall.mean()) ** 2).sum()
print(f"   R2 {r2:.3f}   " + "  ".join(f"{k}={b:.3e}" for k, b in zip(feat_names, beta)))
np.savez(OUT / "diag_mlp0.npz", beta=beta, e=np.stack([r[1] for r in rows]),
         X=np.stack([r[2] for r in rows]), noise=noise, var_true=vt,
         var_chain=np.stack([info[l]["var"].numpy() for l in range(L)]))
print(f"\nwritten {OUT / 'diag_mlp0.npz'}")
