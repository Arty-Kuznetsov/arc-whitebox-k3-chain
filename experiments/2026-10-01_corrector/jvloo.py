"""Leave-one-out over 10 MLPs for a JOINT (variance, kappa4) corrector -- the follow-up of jointsplit.py, which found
in sample that the variance projection, harmful alone (x1.03 / x1.10, chansplit.py), HELPS together with kappa4's:
var+k4 x0.885 / x0.900 against k4 alone x0.916 / x0.918. Does that synergy survive out of sample?

Same protocol as k4loo.py (pooled ridge on the 38 own+pair features + layer, relative error, fitted on 9 MLPs,
applied OFFLINE to the held-out one, raw through the chain vs the baked GT), now with a second ridge model for the
relative VARIANCE error, and two layer ranges (8-15 as shipped for kappa4, 2-15).
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
import k4split as K                                  # noqa: E402

RUNS = HERE.parent / "2026-09-24_k3-chain" / "runs"   # same oracles as k4loo.py (importing it would run its collect)
ORACLE = {0: K.ORA / "mc_oracle_0.npz", 1: K.ORA / "mc_oracle_1.npz"}
ORACLE.update({i: RUNS / f"mc_oracle_{i}.npz" for i in range(2, 10)})
LAM = 1e-3
CH = {"k4": ("K4d", "k4"), "var": ("var", "var")}


def collect(i, lo):
    Ws = np.load(K.pb.BAKED / f"weights_{i}.npy")
    base, info = K.run(i, record="full")
    L, n = len(info), info[0]["mu_pre"].shape[0]
    ora = np.load(ORACLE[i])
    rows = {c: {} for c in CH}
    for l in range(lo, L):
        d = info[l]
        F = np.concatenate([K.own(d, Ws, l, n), K.pair(d), np.full((n, 1), l / 16.0)], 1)
        for c, (ck, ok) in CH.items():
            x = d[ck].numpy().astype(np.float64)
            scale = np.abs(x).mean() + 1e-30
            rows[c][l] = (F, (x - ora[ok][l]) / scale, x, scale)
    return base, rows


def fit(train):
    F = np.concatenate([t[0] for t in train]); e = np.concatenate([t[1] for t in train])
    mu, sd = F.mean(0), F.std(0); sd[sd == 0] = 1.0
    Fz = (F - mu) / sd
    b = np.linalg.solve(Fz.T @ Fz + LAM * len(e) * np.eye(Fz.shape[1]), Fz.T @ (e - e.mean()))
    return b, e.mean(), mu, sd


def apply(model, rows, s):
    b, b0, mu, sd = model
    return {l: x - s * ((((F - mu) / sd) @ b + b0) * scale) for l, (F, e, x, scale) in rows.items()}


if __name__ == "__main__":
    avail = [i for i in range(10) if ORACLE[i].exists()]
    print(f"MLPs with an oracle: {avail}", flush=True)
    for lo in (8, 2):
        DATA = {i: collect(i, lo) for i in avail}
        # held-out R2 of each channel in feature space
        for c in CH:
            num = den = 0.0
            for h in avail:
                m = fit([DATA[i][1][c][l] for i in avail if i != h for l in DATA[i][1][c]])
                for l, (F, e, x, scale) in DATA[h][1][c].items():
                    pe = ((F - m[2]) / m[3]) @ m[0] + m[1]
                    num += ((e - pe) ** 2).sum(); den += ((e - e.mean()) ** 2).sum()
            print(f"layers {lo}-15  {c:3s} held-out R2 {1 - num / den:.3f}", flush=True)
        for chans, s in ((("k4",), 0.5), (("var",), 0.5), (("var", "k4"), 0.5), (("var", "k4"), 1.0)):
            ratios = []
            for h in avail:
                orc = {}
                for c in chans:
                    m = fit([DATA[i][1][c][l] for i in avail if i != h for l in DATA[i][1][c]])
                    arr = apply(m, DATA[h][1][c], s)
                    orc[CH[c][1]] = arr; orc[f"{CH[c][1]}_layers"] = set(arr)
                r, _ = K.run(h, orc)
                ratios.append(r / DATA[h][0])
            ratios = np.array(ratios)
            print(f"layers {lo}-15  {'+'.join(chans):7s} s={s}: held-out raw x mean {ratios.mean():.4f}  median "
                  f"{np.median(ratios):.4f}  better {int((ratios < 1).sum())}/{len(ratios)}   "
                  f"[{' '.join(f'{x:.3f}' for x in ratios)}]", flush=True)
