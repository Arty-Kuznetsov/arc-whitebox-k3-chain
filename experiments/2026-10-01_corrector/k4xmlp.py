"""The decisive out-of-sample test of the kappa4 PATTERN axis.

In sample, removing the projection of the kappa4 error onto own+pair features is worth x0.916 / x0.918 (k4split.py).
Here: fit that projection on MLP A (per layer, standardised with A's statistics, OLS), apply it OFFLINE to MLP B
(B's own recorded trajectory, through `oracle`, kappa4 only), measure B's final raw. Both directions. Also a POOLED
variant (one model for layers 8-15 with the layer index as a feature) which has 8x more data per coefficient, and the
dominant features, as a hint toward a closed form.
"""
import sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.argv = [sys.argv[0]] + sys.argv[1:]
import k4split as K                                  # noqa: E402  (reuses run, own, pair; runs nothing at import? see guard)

FEATN = (["a", "a2", "a3", "a4", "phi", "Phi", "a*phi", "(a2-1)phi", "sd", "var", "skew", "exkurt", "u_next", "wcol",
          "rank", "mu", "t", "K4d", "Phi*var", "t*a", "K4d*a"]
         + ["|C0|r", "C0^2r", "C0^3r", "C0r", "C0^2r/var", "max|C0|", "C0^4r", "|C0|^1.5r",
            "|D|r", "D^2r", "Dr", "|D|c", "D^2c", "Dc", "|K22|r", "K22^2r", "K22r"])


def feats_and_err(i):
    Ws = np.load(K.pb.BAKED / f"weights_{i}.npy")
    base, info = K.run(i, record="full")
    L, n = len(info), info[0]["mu_pre"].shape[0]
    o4 = np.load(K.ORA / f"mc_oracle_{i}.npz")["k4"]
    out = {}
    for l in range(1, L):
        d = info[l]
        k4c = d["K4d"].numpy().astype(np.float64)
        F = np.concatenate([K.own(d, Ws, l, n), K.pair(d)], 1)
        out[l] = (F, k4c - o4[l], k4c)
    return base, out, L


def fit_perlayer(data):
    co = {}
    for l, (F, e, _) in data.items():
        mu, sd = F.mean(0), F.std(0); sd[sd == 0] = 1.0
        keep = (F.std(0) > 0) & np.isfinite(F).all(0)
        Fz = ((F - mu) / sd)[:, keep]
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(e)), Fz]), e, rcond=1e-8)
        co[l] = (b, mu, sd, keep)
    return co


def apply_perlayer(co, data, s=1.0):
    arr = {}
    for l, (F, e, k4c) in data.items():
        b, mu, sd, keep = co[l]
        Fz = ((F - mu) / sd)[:, keep]
        pe = np.column_stack([np.ones(len(k4c)), Fz]) @ b
        arr[l] = k4c - s * pe
    return arr


def fit_pooled(data, lo=8):
    Fs = [np.column_stack([data[l][0], np.full(len(data[l][1]), l / 16.0)]) for l in data if l >= lo]
    es = [data[l][1] / (np.abs(data[l][2]).mean() + 1e-30) for l in data if l >= lo]   # relative error per layer
    F = np.concatenate(Fs); e = np.concatenate(es)
    mu, sd = F.mean(0), F.std(0); sd[sd == 0] = 1.0
    keep = (F.std(0) > 0) & np.isfinite(F).all(0)
    Fz = ((F - mu) / sd)[:, keep]
    b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(e)), Fz]), e, rcond=1e-8)
    return (b, mu, sd, keep, lo)


def apply_pooled(co, data, s=1.0):
    b, mu, sd, keep, lo = co
    arr = {}
    for l, (F, e, k4c) in data.items():
        if l < lo:
            continue
        Fl = np.column_stack([F, np.full(len(k4c), l / 16.0)])
        pe = (np.column_stack([np.ones(len(k4c)), ((Fl - mu) / sd)[:, keep]]) @ b) * (np.abs(k4c).mean() + 1e-30)
        arr[l] = k4c - s * pe
    return arr


def raw_with(i, arr):
    r, _ = K.run(i, {"k4": arr, "k4_layers": set(arr)})
    return r


D = {}
for i in (0, 1):
    D[i] = feats_and_err(i)
for src, dst in ((0, 1), (1, 0)):
    base_d, data_d, _ = D[dst]
    _, data_s, _ = D[src]
    cpl, cpo = fit_perlayer(data_s), fit_pooled(data_s)
    print(f"\n--- fit on MLP {src} -> apply OFFLINE on MLP {dst}   (base {base_d:.6e})")
    for s in (0.5, 1.0):
        print(f"    per-layer OLS, s={s:3.1f}:  x{raw_with(dst, apply_perlayer(cpl, data_d, s))/base_d:.4f}")
    for s in (0.5, 1.0):
        print(f"    POOLED L8-15, s={s:3.1f}:   x{raw_with(dst, apply_pooled(cpo, data_d, s))/base_d:.4f}")

# which features carry the signal (pooled fit on both MLPs, layers 8-15, standardised |beta|)
allD = {}
for i in (0, 1):
    for l, v in D[i][1].items():
        allD[(i, l)] = v
Fs = [np.column_stack([v[0], np.full(len(v[1]), l / 16.0)]) for (i, l), v in allD.items() if l >= 8]
es = [v[1] / (np.abs(v[2]).mean() + 1e-30) for (i, l), v in allD.items() if l >= 8]
F = np.concatenate(Fs); e = np.concatenate(es)
sd = F.std(0); keep = sd > 0
Fz = (F[:, keep] - F[:, keep].mean(0)) / sd[keep]
b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(e)), Fz]), e - e.mean(), rcond=1e-8)
names = [n for n, k in zip(FEATN + ["layer"], keep) if k]
order = np.argsort(-np.abs(b[1:]))
print("\ndominant standardised features of the pooled kappa4-error model (both MLPs, layers 8-15):")
for j in order[:10]:
    print(f"   {names[j]:12s} beta {b[1 + j]:+.4f}")
