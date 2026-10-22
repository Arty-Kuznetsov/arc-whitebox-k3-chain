"""The upper bound on ANY corrector with a given per-channel R^2.

An OLS predictor removes exactly the projection of the error onto the feature space, leaving a residual of magnitude
sqrt(1 - R^2) that is ORTHOGONAL to the features. A uniform exact-correction at strength s leaves (1-s)*error in the
SAME direction. Matching magnitudes gives s_eq = 1 - sqrt(1 - R^2). So:

  * the "s_eq" arm applies the EXACT error at the strength an R^2-accurate predictor would match in MAGNITUDE. It is
    the best case for a corrector of that accuracy IF only magnitude matters.
  * comparing it with the real learned corrector isolates how much of the damage is DIRECTION (the residual being
    orthogonal to the features) rather than magnitude.

Measured R^2 at the late layers (experiments/2026-10-01_corrector/RESULTS.md): var 0.10, k3 0.51, k4 0.56.
"""
import sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                            # noqa: E402
from chain import run_chain                          # noqa: E402

torch.set_grad_enabled(False); torch.set_num_threads(3)
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
K = {"var": "var", "k3": "k3", "k4": "k4"}
_GT = np.load(pb.BAKED / "gt.npy")


def run(i, mcorr=None):
    Ws = np.load(pb.BAKED / f"weights_{i}.npy")
    pred, _ = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], mcorr=mcorr, **B9)
    return float(((pred[-1] - torch.tensor(_GT[i], dtype=torch.float64)[-1]) ** 2).mean())


def mk(i, strengths, layers=None):
    d = np.load(ORA / f"mc_oracle_{i}.npz")
    def f(l, st):
        if l == 0 or (layers is not None and l not in layers):
            return None
        out = {}
        for c, s in strengths.items():
            cur = st["var"] if c == "var" else st["t"] if c == "k3" else st["K4d"]
            if cur is None or s == 0.0:
                continue
            out[c] = s * (cur - torch.tensor(d[K[c]][l], dtype=torch.float64))
        return out
    return f


R2 = {"var": 0.10, "k3": 0.51, "k4": 0.56}
SEQ = {c: 1.0 - np.sqrt(1.0 - r) for c, r in R2.items()}
print("R^2 ->  magnitude-equivalent exact strength s_eq = 1 - sqrt(1 - R^2):")
print("   " + "  ".join(f"{c}: R2 {R2[c]:.2f} -> s_eq {SEQ[c]:.3f}" for c in R2))
for i in (0, 1):
    base = run(i)
    print(f"\n=== MLP {i}, base {base:.6e}")
    print(f"  s_eq per channel (0.051/0.300/0.337)        x{run(i, mk(i, SEQ))/base:.4f}")
    print(f"  s_eq, layers 8-15 only                      x{run(i, mk(i, SEQ, set(range(8,16))))/base:.4f}")
    print(f"  s_eq but var UNTOUCHED (k3,k4 only)         x{run(i, mk(i, dict(SEQ, var=0.0)))/base:.4f}")
    print(f"  var at s_eq only                            x{run(i, mk(i, {'var': SEQ['var']}))/base:.4f}")
    # what if var were as predictable as k3/k4?
    print(f"  all three at s=0.30 (var as good as k3)     x{run(i, mk(i, {c: 0.30 for c in R2}))/base:.4f}")
