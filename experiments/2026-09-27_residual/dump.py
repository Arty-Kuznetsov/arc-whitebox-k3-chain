"""Dump the chain's per-neuron state and its residual against the baked GT, for the 10 baked MLPs.

The point: our rms final-layer error is 1.47e-4 against a per-neuron signal std of 0.78, i.e. a relative
error of 1.9e-4, and the board's best is 1.24e-4 (only 18% better). Before betting on any accuracy work we
need to know the SHAPE of that residual: is it spread over all 1024 neurons or concentrated on a few, and is
it correlated with anything the estimator has at hand (alpha, var, kappa3, kappa4, ...)?

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-27_residual/dump.py

Self-check: the mean final-layer MSE over the 10 MLPs must be 2.3135e-08.
"""
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)

gt_all = np.load(BAKED / "gt.npy")
NM, L, n = gt_all.shape
keys = ("pred", "gt", "var", "mu_pre", "t", "K4d")
acc = {k: [] for k in keys}
mses = []
for m in range(NM):
    W = np.load(BAKED / f"weights_{m}.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    pred, info = run_chain(Ws, record=True, **K3L)
    p = pred.numpy()
    acc["pred"].append(p)
    acc["gt"].append(gt_all[m].astype(np.float64))
    for k in ("var", "mu_pre", "t", "K4d"):
        acc[k].append(np.stack([info[l][k].numpy() for l in range(L)]))
    mses.append(float(((p[-1] - gt_all[m][-1]) ** 2).mean()))
    print(f"MLP {m}: final MSE {mses[-1]:.4e}", flush=True)
print(f"\nmean {np.mean(mses):.4e}   (must be 2.3135e-08)")
np.savez_compressed(OUT / "dump10.npz", **{k: np.stack(v) for k, v in acc.items()})
print("written", OUT / "dump10.npz")
