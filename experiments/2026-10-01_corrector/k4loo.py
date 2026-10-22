"""Leave-one-out over 10 MLPs: does a SMOOTH correction of the kappa4 diagonal generalise across networks?

In sample, removing the projection of the kappa4 error onto own+pair features is worth x0.916 / x0.918 (k4split.py);
fitted on one MLP and applied to the other it is worth nothing (k4xmlp.py) -- but two MLPs cannot tell a real
pattern from one network's idiosyncrasies. Here: oracles for MLPs 0-9 (0, 1 at 1e7; 2-9 at 2e6, one seed -- for a
TRAINING target the per-neuron noise averages over 1024 neurons), a pooled RIDGE model of the RELATIVE kappa4 error
on layers 8-15 fitted on 9 MLPs and applied OFFLINE to the held-out one, for every one of the 10.
Two feature sets: the 38 own+pair features, and a small 8-feature set of the dominant, least-collinear ones.
"""
import sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
import k4split as K                                  # noqa: E402   (guarded: runs nothing at import)

RUNS = HERE.parent / "2026-09-24_k3-chain" / "runs"
ORACLE = {0: K.ORA / "mc_oracle_0.npz", 1: K.ORA / "mc_oracle_1.npz"}
ORACLE.update({i: RUNS / f"mc_oracle_{i}.npz" for i in range(2, 10)})
SMALL = [22, 19, 8, 1, 17, 36, 25, 38]  # C0^2 row, t*a, sd, a^2, K4d, K22^2 row, C0^2 row/var, layer -- the
                                        # dominant features of the pooled fit, one per collinear family
LO = 8
LAMS = (1e-2, 1e-1, 1.0, 10.0)


def collect(i):
    Ws = np.load(K.pb.BAKED / f"weights_{i}.npy")
    base, info = K.run(i, record="full")
    L, n = len(info), info[0]["mu_pre"].shape[0]
    o4 = np.load(ORACLE[i])["k4"]
    rows = {}
    for l in range(LO, L):
        d = info[l]
        k4c = d["K4d"].numpy().astype(np.float64)
        F = np.concatenate([K.own(d, Ws, l, n), K.pair(d), np.full((n, 1), l / 16.0)], 1)
        scale = np.abs(k4c).mean() + 1e-30
        rows[l] = (F, (k4c - o4[l]) / scale, k4c, scale)
    return base, rows


def fit(train, cols, lam):
    F = np.concatenate([train[k][0][:, cols] for k in train])
    e = np.concatenate([train[k][1] for k in train])
    mu, sd = F.mean(0), F.std(0); sd[sd == 0] = 1.0
    Fz = (F - mu) / sd
    A = Fz.T @ Fz + lam * len(e) * np.eye(Fz.shape[1])
    b = np.linalg.solve(A, Fz.T @ (e - e.mean()))
    return b, e.mean(), mu, sd


def apply(model, rows, cols, s=1.0):
    b, b0, mu, sd = model
    arr = {}
    for l, (F, e, k4c, scale) in rows.items():
        pe = (((F[:, cols] - mu) / sd) @ b + b0) * scale
        arr[l] = k4c - s * pe
    return arr


avail = [i for i in range(10) if ORACLE[i].exists()]
print(f"MLPs with an oracle: {avail}", flush=True)
DATA = {i: collect(i) for i in avail}
ALL = list(range(DATA[avail[0]][1][LO][0].shape[1]))


def heldout_r2(cols, lam):
    """cheap: leave-one-MLP-out R^2 of the relative kappa4 error IN FEATURE SPACE (no chain runs)."""
    num = den = 0.0
    for h in avail:
        train = {(i, l): DATA[i][1][l] for i in avail if i != h for l in DATA[i][1]}
        b, b0, mu, sd = fit(train, cols, lam)
        for l, (F, e, k4c, scale) in DATA[h][1].items():
            pe = ((F[:, cols] - mu) / sd) @ b + b0
            num += ((e - pe) ** 2).sum(); den += ((e - e.mean()) ** 2).sum()
    return 1.0 - num / den


if __name__ == "__main__":
    best = {}
    for label, cols in (("38 own+pair (+layer)", ALL), ("8 small", SMALL)):
        scores = {lam: heldout_r2(cols, lam) for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0)}
        print(f"  {label:22s} held-out R2 by lambda: " + "  ".join(f"{l:g}:{r:.3f}" for l, r in scores.items()), flush=True)
        best[label] = (cols, max(scores, key=scores.get), max(scores.values()))

    print("\nthrough the chain, offline, every held-out MLP, at each set's best lambda:", flush=True)
    for label, (cols, lam, r2) in best.items():
        if r2 <= 0:
            print(f"  {label:22s} best held-out R2 {r2:.3f} <= 0: nothing to apply", flush=True)
            continue
        for s_ in (0.5, 1.0):
            ratios = []
            for h in avail:
                train = {(i, l): DATA[i][1][l] for i in avail if i != h for l in DATA[i][1]}
                model = fit(train, cols, lam)
                base_h, rows_h = DATA[h]
                arr = apply(model, rows_h, cols, s_)
                r, _ = K.run(h, {"k4": arr, "k4_layers": set(arr)})
                ratios.append(r / base_h)
            ratios = np.array(ratios)
            print(f"  {label:22s} lam {lam:g} s={s_}: held-out raw x  mean {ratios.mean():.4f}  median "
                  f"{np.median(ratios):.4f}  better {int((ratios < 1).sum())}/{len(ratios)}   "
                  f"[{' '.join(f'{x:.3f}' for x in ratios)}]", flush=True)
