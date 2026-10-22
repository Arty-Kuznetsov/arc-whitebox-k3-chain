"""Accuracy of our SIMPLE k=3 chain (chain.py) on baked MLPs, and ablations toward an affordable variant.
Usage: python ablate.py <config-name> [n_mlps]   (configs below). Prints final-layer MSE per MLP and mean."""
import sys, time
from pathlib import Path
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float32); torch.set_num_threads(3)
from chain import run_chain, run_chain_tucker

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
gt = np.load(BAKED / "gt.npy")
CONFIGS = {
    "full":        dict(),
    "J8":          dict(window=8),
    "J6":          dict(window=6),
    "J4":          dict(window=4),
    "J3":          dict(window=3),
    "J2":          dict(window=2),
    "corr_diag":   dict(corr="diag"),
    "corr_none":   dict(corr="none"),
    "no_k4":       dict(k4=False),
    "J4_corrdiag": dict(window=4, corr="diag"),
    "tk_y1_r64":   dict(tucker=True, young=1, rank=64),
    "tk_y1_r128":  dict(tucker=True, young=1, rank=128),
    "tk_y2_r64":   dict(tucker=True, young=2, rank=64),
    "tk_y2_r128":  dict(tucker=True, young=2, rank=128),
    "tk_y2_r256":  dict(tucker=True, young=2, rank=256),
    "tk_y3_r128":  dict(tucker=True, young=3, rank=128),
    "tk_y1_r32":   dict(tucker=True, young=1, rank=32),
    "k4memo":      dict(k4="memo22"),
    "edge2":       dict(edge2=True),
    "k4memo_edge2": dict(k4="memo22", edge2=True),
    "aug":         dict(aug=True),
    "k4memo_aug":  dict(k4="memo22", aug=True),
    "d21y2": dict(k4="memo22", d21_ages=2),
    "d21y4": dict(k4="memo22", d21_ages=4),
    "d21y6": dict(k4="memo22", d21_ages=6),
    "d21y8": dict(k4="memo22", d21_ages=8),
    "fullC_none": dict(k4="memo22", corr="fullC_none"),
    "fullC_diag": dict(k4="memo22", corr="fullC_diag"),
    "lr8": dict(k4="memo22", corr="lr8"),
    "lr16": dict(k4="memo22", corr="lr16"),
    "lr32": dict(k4="memo22", corr="lr32"),
    "lr64": dict(k4="memo22", corr="lr64"),
    "lr128": dict(k4="memo22", corr="lr128"),
}
name = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3
cfg = CONFIGS[name]
res = []
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float32) for l in range(W.shape[0])]
    t0 = time.time()
    if cfg.get('tucker'):
        c = {k: v for k, v in cfg.items() if k != 'tucker'}
        mu = run_chain_tucker(Ws, **c)
    else:
        mu, _ = run_chain(Ws, **cfg)
    mse = float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean())
    res.append(mse)
    print(f"  {name} mlp {i}: final_mse={mse:.4e}  ({time.time()-t0:.0f}s)", flush=True)
    del Ws, mu
print(f"RESULT {name}: mean final_mse over {N} MLPs = {np.mean(res):.4e}   (reference SIMPLE k=3 on MLPs 0-2: 4.2509e-08; k=2: 5.1494e-06)")
