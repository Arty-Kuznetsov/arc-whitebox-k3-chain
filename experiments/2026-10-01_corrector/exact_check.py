"""Mandatory self-check of the xmlp.py harness, plus the per-channel and offline/online split.

ARM 1 (the gate): mcorr returns the EXACT error (chain value - oracle value) computed from the LIVE state. That is
identical to substituting the oracle value, i.e. cloud #9A's arm, which measured x0.117/0.128 joint. If this harness
does not reproduce that, nothing else it says means anything.
ARM 2: the same at partial strength s -- does a PARTIAL move toward the truth help at all?
ARM 3: per channel, exact.
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
KEY = {"var": "var", "k3": "k3", "k4": "k4"}
_GT = np.load(pb.BAKED / "gt.npy")


def run(i, mcorr=None):
    Ws = np.load(pb.BAKED / f"weights_{i}.npy")
    pred, _ = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], mcorr=mcorr, **B9)
    g = torch.tensor(_GT[i], dtype=torch.float64)
    return float(((pred[-1] - g[-1]) ** 2).mean())


def exact_mcorr(i, chans, s, layers=None, seed_file=None):
    d = np.load(ORA / (seed_file or f"mc_oracle_{i}.npz"))
    def f(l, st):
        if l == 0 or (layers is not None and l not in layers):
            return None
        out = {}
        for c in chans:
            cur = (st["var"] if c == "var" else st["t"] if c == "k3" else st["K4d"])
            if cur is None:
                continue
            out[c] = s * (cur - torch.tensor(d[KEY[c]][l], dtype=torch.float64))
        return out
    return f


for i in (0, 1):
    base = run(i)
    print(f"\n=== MLP {i}, base B9 raw {base:.6e}")
    for chans in (("var", "k3", "k4"), ("var",), ("k3",), ("k4",)):
        line = f"  exact {'+'.join(chans):14s}"
        for s in (1.0, 0.5, 0.25):
            r = run(i, exact_mcorr(i, chans, s))
            line += f"   s={s:4.2f} x{r/base:.4f}"
        print(line)
    # the second oracle, full strength, joint: the honest error bar of the ceiling
    r = run(i, exact_mcorr(i, ("var", "k3", "k4"), 1.0, seed_file=f"mc_oracle_{i}_s777.npz"))
    print(f"  [seed 777] exact joint s=1.00 x{r/base:.4f}")
    # late layers only
    for lay in (range(8, 16), range(12, 16)):
        r = run(i, exact_mcorr(i, ("var", "k3", "k4"), 1.0, layers=set(lay)))
        print(f"  exact joint, layers {min(lay)}-15 only: x{r/base:.4f}")
