"""Which PART of the moment error actually drives the final MSE: the smooth/feature-predictable part or the rest?

Decompose the exact per-neuron moment error at each layer into
    e = P e  (its projection onto the span of the chain's own smooth per-neuron features)  +  (I-P) e,
both computed from the RECORDED uncorrected trajectory and the MC oracle, and substitute each part alone (offline,
through the `oracle` path, no feedback).  Exact (both parts) is the x0.21-0.30 ceiling.
  * if removing (I-P)e alone recovers the ceiling, the damaging error is the ROUGH part and no smooth per-neuron
    corrector can ever collect it -- NEGATIVE 62's wall, measured in the corrector's own language;
  * if removing P e alone recovers it, the offline numbers are a fit-quality problem, not a structural one.
"""
import sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                            # noqa: E402
from chain import run_chain                          # noqa: E402
sys.argv = [sys.argv[0], "0", "1"]   # xmlp.py reads argv at import time
import xmlp                                          # noqa: E402

torch.set_grad_enabled(False); torch.set_num_threads(3)
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
_GT = np.load(pb.BAKED / "gt.npy")
CH = ["var", "k3", "k4"]
OK = {"var": "var", "k3": "k3", "k4": "k4"}


def raw_of(i, oracle=None):
    pred, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{i}.npy")],
                           record=True, oracle=oracle, **xmlp.B9)
    return float(((pred[-1] - torch.tensor(_GT[i], dtype=torch.float64)[-1]) ** 2).mean()), info


def arrays(i, info, L, part, s=1.0):
    """part = 'all' | 'proj' | 'orth': the absolute corrected moment arrays."""
    Ws = np.load(pb.BAKED / f"weights_{i}.npy")
    d = np.load(ORA / f"mc_oracle_{i}.npz")
    out = {c: {} for c in CH}
    for l in range(1, L):
        st = tuple(info[l][k].numpy().astype(np.float64) for k in ("mu_pre", "var", "t", "K4d"))
        F = xmlp.F_of(st, Ws, l, L)
        F = np.where(np.isfinite(F), F, 0.0)
        keep = (F.std(0) > 0) & np.isfinite(F).all(0)
        keep[0] = False
        Fz = ((F - F.mean(0)) / np.where(F.std(0) == 0, 1.0, F.std(0)))[:, keep]
        Q, _ = np.linalg.qr(np.column_stack([np.ones(Fz.shape[0]), Fz]))   # orthonormal basis incl. the intercept
        cur = {"var": st[1], "k3": st[2], "k4": st[3]}
        for c in CH:
            e = cur[c] - d[OK[c]][l]
            pe = Q @ (Q.T @ e)
            use = e if part == "all" else (pe if part == "proj" else e - pe)
            out[c][l] = cur[c] - s * use
    return out


def orc(a):
    return {"var": a["var"], "k3": a["k3"], "k4": a["k4"],
            "var_layers": set(a["var"]), "k3_layers": set(a["k3"]), "k4_layers": set(a["k4"])}


for i in (0, 1):
    base, info = raw_of(i)
    L = len(info)
    print(f"\n=== MLP {i}, base {base:.6e}   (features kept: the 20 smooth per-neuron ones + intercept)")
    for part in ("all", "proj", "orth"):
        line = f"  remove {part:5s}"
        for s in (1.0, 0.5):
            r, _ = raw_of(i, orc(arrays(i, info, L, part, s)))
            line += f"   s={s:3.1f} x{r/base:.4f}"
        print(line)
