"""Separate the two failure modes of the learned corrector.

ONLINE (xmlp.py): the correction is a function of the LIVE state, so correcting layer 1 moves layer 2's state off the
distribution the fit saw, the linear model extrapolates, and it runs away (measured: NaN out of sample).
OFFLINE (here): predict the error from the RECORDED uncorrected trajectory, then re-run substituting those FIXED
arrays through chain.py's `oracle` path. No feedback. This answers "is the prediction itself useful?".

  python offline.py <A> <B>    fit on A, apply offline on B (and the reverse, and both in sample)
"""
import sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                            # noqa: E402
from chain import run_chain                          # noqa: E402
import xmlp                                          # noqa: E402  (reuse F_of, fit_on, B9, KEY)

torch.set_grad_enabled(False); torch.set_num_threads(3)
A, B = int(sys.argv[1]), int(sys.argv[2])
_GT = np.load(pb.BAKED / "gt.npy")
CH = ["var", "k3", "k4"]


def weights(i):
    return np.load(pb.BAKED / f"weights_{i}.npy")


def raw_of(i, oracle=None):
    pred, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in weights(i)],
                           record=True, oracle=oracle, **xmlp.B9)
    return float(((pred[-1] - torch.tensor(_GT[i], dtype=torch.float64)[-1]) ** 2).mean()), info


def predicted_arrays(coef, i, s, info, L):
    """chain value MINUS the predicted error, from the RECORDED trajectory -> absolute arrays for `oracle`."""
    Ws = weights(i)
    out = {c: {} for c in CH}
    for l in range(1, L):
        st = tuple(info[l][k].numpy().astype(np.float64) for k in ("mu_pre", "var", "t", "K4d"))
        F = xmlp.F_of(st, Ws, l, L)
        F = np.where(np.isfinite(F), F, 0.0)
        cur = {"var": st[1], "k3": st[2], "k4": st[3]}
        for c in CH:
            if l in coef[c]:
                beta, y0, mu_f, sd_f = coef[c][l]
                pe = ((F - mu_f) / sd_f) @ beta + y0
                out[c][l] = cur[c] - s * pe
    return out


for (src, dst) in ((A, B), (B, A), (A, A), (B, B)):
    coef, L = xmlp.fit_on(src)
    base, info = raw_of(dst)
    tag = "IN SAMPLE " if src == dst else "out of sample"
    print(f"\n--- fit on {src} -> offline on {dst}  [{tag}]   base {base:.6e}")
    for s in (0.25, 0.5, 1.0):
        arr = predicted_arrays(coef, dst, s, info, L)
        orc = {"var": {l: arr["var"][l] for l in arr["var"]}, "k3": {l: arr["k3"][l] for l in arr["k3"]},
               "k4": {l: arr["k4"][l] for l in arr["k4"]},
               "var_layers": set(arr["var"]), "k3_layers": set(arr["k3"]), "k4_layers": set(arr["k4"])}
        r, _ = raw_of(dst, orc)
        print(f"    s={s:4.2f}:  raw {r:.6e}   x{r/base:.4f}")
