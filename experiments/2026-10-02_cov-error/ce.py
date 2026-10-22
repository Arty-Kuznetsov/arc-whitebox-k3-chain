"""Harness of cloud #11 (the structure of the covariance error dC): the tail-accuracy harness (`ta.py`: per-(arm, MLP)
result files under res/<tag>/m<m>.json, resumable) with results in this folder, plus the oracle loaders.

Oracles per MLP 0 / 1:
  * covariance: cov_oracle.py, seeds SEEDS (1e7 each): C_post(l) at layers 0..14 and post means at every layer;
  * per-neuron variance: the Mac's two 1e7 seeds (../2026-09-24_k3-chain/oracles: default, 777) plus, from every
    covariance oracle, var_pre(l+1) = diag(W_{l+1}^T C(l) W_{l+1}) (the same samples, an exact identity).
Base for everything: B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep") (k3m + matched cut -2.5 + E2 2.5 sep)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-30_tail-accuracy"))
import ta  # noqa: E402  (puts chain.py and pb.py on the path, sets the thread count from PB_THREADS)

ta.RES = HERE / "res"
B9 = ta.B9
ORC = HERE / "oracles"
SEEDS = (2101, 2102)
MAC = {m: [HERE.parent / "2026-09-24_k3-chain" / "oracles" / f"mc_oracle_{m}{x}.npz" for x in ("", "_s777")]
       for m in (0, 1)}
L = 16


def cov(m, seed, mmap=True):
    """(L-1, n, n) float64 post-activation covariance, layers 0..14, of one 1e7 seed."""
    return np.load(ORC / f"cov_{m}_s{seed}_C.npy", mmap_mode="r" if mmap else None)


def cov_meta(m, seed):
    z = np.load(ORC / f"cov_{m}_s{seed}_meta.npz")
    return {k: z[k] for k in z.files}


def have_cov(m, seed):
    return (ORC / f"cov_{m}_s{seed}_C.npy").exists()


def var_from_cov(m, seed, Wn):
    """var_pre at layers 1..L-1 (row 0 left NaN) from the covariance oracle of one seed."""
    C = cov(m, seed)
    v = np.full((L, Wn[0].shape[1]), np.nan)
    for l in range(L - 1):
        Wl = Wn[l + 1]
        v[l + 1] = np.einsum("ji,jk,ki->i", Wl, C[l], Wl, optimize=True)
    return v


def var_seeds(m, Wn):
    """List of independent per-neuron pre-activation variance oracles (L, n): the Mac's two, then one per covariance
    seed available. Layer 0 is exact (Gaussian input): var = column norms of W_0 squared."""
    out = [np.load(f)["var"].copy() for f in MAC[m]]
    for s in SEEDS:
        if have_cov(m, s):
            out.append(var_from_cov(m, s, Wn))
    v0 = (Wn[0] ** 2).sum(0)
    for v in out:
        v[0] = v0
    return out


def wn(m):
    Ws, g = ta.mlp(m)
    return [w.numpy() for w in Ws], g


if __name__ == "__main__":
    # gate: B9 on the 10 screening MLPs (2.1967e-08 in cloud #9A/#10). Run as PB_THREADS=3 python ce.py
    ta.run_arm("B9", B9, ta.SCREEN, early=False)
    print("B9", ta.summary("B9", "B9", ta.SCREEN))
