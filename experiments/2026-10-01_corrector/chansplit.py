"""The k4split test for the OTHER two channels: is the pair-feature projection of the variance error or of the kappa3
error worth anything when removed (in sample, one channel at a time)? kappa4's was x0.916 / x0.918 (k4split.py) and
became a shipped corrector; if var's or kappa3's is too, the corrector can be extended.
"""
import sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
import k4split as K                                  # noqa: E402

CH = {"var": ("var", "var", "var_layers"), "k3": ("t", "k3", "k3_layers")}
for i in (0, 1):
    Ws = np.load(K.pb.BAKED / f"weights_{i}.npy")
    base, info = K.run(i, record="full")
    L, n = len(info), info[0]["mu_pre"].shape[0]
    ora = np.load(K.ORA / f"mc_oracle_{i}.npz")
    print(f"\n=== MLP {i}, base B9 {base:.6e}", flush=True)
    for c, (ck, ok, key) in CH.items():
        arrs = {k: {} for k in ("all", "Pown", "Pboth")}
        for l in range(1, L):
            d = info[l]
            x = d[ck].numpy().astype(np.float64)
            e = x - ora[ok][l]
            Fo = K.own(d, Ws, l, n); Fb = np.concatenate([Fo, K.pair(d)], 1)
            arrs["all"][l] = x - e
            arrs["Pown"][l] = x - K.proj(Fo, e)
            arrs["Pboth"][l] = x - K.proj(Fb, e)
        for k, lab in (("all", "remove ALL of e (channel ceiling)"), ("Pown", "remove P_own e"),
                       ("Pboth", "remove P_own+pair e")):
            orc = {ok: arrs[k], key: set(arrs[k])}
            r, _ = K.run(i, orc)
            print(f"  {c:4s} {lab:36s} x{r/base:.4f}", flush=True)
