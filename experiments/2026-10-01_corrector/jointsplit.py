"""The channel test that was never run: remove the feature-predictable part of the moment error in ALL THREE channels
AT ONCE (in sample, offline, through `oracle`). chansplit.py removed one channel at a time and found var and kappa3
HURT (x1.03-1.10) while kappa4 helps (x0.916/x0.918) -- but the exact correction is strongly synergistic (joint
x0.206/x0.302 against 0.69-1.09 per channel), so one-at-a-time may be the wrong test for a joint corrector.
Arms (per MLP): the joint exact gate (must reproduce exact_check's x0.206 / x0.302), joint P_own+pair, its pairs, and
the layer-8+ restriction. Upper bounds (in sample): a learned joint corrector would collect less out of sample.
"""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
import k4split as K                                  # noqa: E402

CH = {"var": ("var", "var", "var_layers"), "k3": ("t", "k3", "k3_layers"), "k4": ("K4d", "k4", "k4_layers")}
ARMS = [("exact", ("var", "k3", "k4"), "all", 1), ("Pboth", ("var", "k3", "k4"), "Pboth", 1),
        ("Pboth k3+k4", ("k3", "k4"), "Pboth", 1), ("Pboth var+k4", ("var", "k4"), "Pboth", 1),
        ("Pboth var+k3", ("var", "k3"), "Pboth", 1), ("Pboth, layers 8+", ("var", "k3", "k4"), "Pboth", 8),
        ("Pown", ("var", "k3", "k4"), "Pown", 1)]
for i in (0, 1):
    Ws = np.load(K.pb.BAKED / f"weights_{i}.npy")
    base, info = K.run(i, record="full")
    L, n = len(info), info[0]["mu_pre"].shape[0]
    ora = np.load(K.ORA / f"mc_oracle_{i}.npz")
    print(f"\n=== MLP {i}, base B9 {base:.6e}", flush=True)
    arrs = {c: {k: {} for k in ("all", "Pown", "Pboth")} for c in CH}
    for l in range(1, L):
        d = info[l]
        Fo = K.own(d, Ws, l, n); Fb = np.concatenate([Fo, K.pair(d)], 1)
        for c, (ck, ok, _) in CH.items():
            x = d[ck].numpy().astype(np.float64)
            e = x - ora[ok][l]
            arrs[c]["all"][l] = x - e
            arrs[c]["Pown"][l] = x - K.proj(Fo, e)
            arrs[c]["Pboth"][l] = x - K.proj(Fb, e)
    for lab, chans, kind, l0 in ARMS:
        orc = {}
        for c in chans:
            ok, key = CH[c][1], CH[c][2]
            sel = {l: v for l, v in arrs[c][kind].items() if l >= l0}
            orc[ok] = sel; orc[key] = set(sel)
        r, _ = K.run(i, orc)
        print(f"  {lab:20s} {'+'.join(chans):10s} x{r / base:.4f}", flush=True)
