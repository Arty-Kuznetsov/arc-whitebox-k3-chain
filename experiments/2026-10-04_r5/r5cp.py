"""R5 (2026-10-04): is the OLD kappa3 compressible by OPTIMISATION? Selection of hub columns (504aldo F76: capped atoms,
4n columns -> 39% D21 error) and subspaces (NEGATIVE 76: >= 320-384 dims) are dead; nobody measured the best rank-R
CP approximation (new rank-1 terms fitted by ALS) of the accumulated tensor. If R ~ n reproduced its readouts to ~2%,
the old tier would cost like one young source instead of ten.
Instrument: our float64 chain (research base B9 + marg2) on a random He MLP (n, L), stopped at layer l; the dense
pre-activation kappa3 of the OLD sources (age >= a) and of ALL sources (chain._k3_dense); non-symmetric CP-ALS of rank
R (3 factors, ridge 1e-10, SVD init) for R in a ladder; errors of the tensor, of its diagonal t and of its (2,1) slice
D -- at layer l, and after one exact transport+gate to layer l+1 (the readouts the next step uses).
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r5cp.py <n> <L> <layer> <age0> <seed> [Rs]
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("R5_THREADS", "4")))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-04_r1"))
from r1lib import BASE, weights                                  # noqa: E402
import chain                                                     # noqa: E402

n, L, lay, a0, seed = (int(x) for x in sys.argv[1:6])
Rs = [int(x) for x in sys.argv[6].split(",")] if len(sys.argv) > 6 else [n // 4, n // 2, n, 2 * n, 4 * n]
Ws = weights(n, L, seed)

# run the chain to layer `lay` and capture the sources there (pre-activation legs) via a hook on _k3_dense's inputs
cap = {}
orig = chain.nonlin_slices


def hook(*args, **kw):
    return orig(*args, **kw)


# simplest capture: re-run the chain truncated to lay+1 layers with record and grab `sources` through a k4young-free path:
# chain.run_chain does not return sources, so wrap _k3_dense through k4young impl="arc" at the target layer only
grab = {}
real_k3 = chain._k3_dense


def k3_spy(sources, n_, T3d=None, layer=None, couple=None):
    if layer == lay:
        grab["old"] = real_k3([s for s in sources if layer - s["born"] >= a0], n_, None)
        grab["all"] = real_k3(sources, n_, T3d)
        grab["d1"] = None
    return real_k3(sources, n_, T3d, layer=layer, couple=couple)


chain._k3_dense = k3_spy
_, info = chain.run_chain([W.clone() for W in Ws[:lay + 2]], record=True,
                          k4young=dict(impl="arc", conv="arc", ages=1, types=("32",), read=("K4d",), mult=0.0), **BASE)
chain._k3_dense = real_k3
assert "old" in grab, "no capture (check layer / ages)"
Wn = Ws[lay + 1]
mu_pre, var = info[lay]["mu_pre"], info[lay]["var"]
d1 = chain._Phi(mu_pre / var.sqrt())          # the gate of the transported kappa3 at layer lay


def slices(T):
    ii = torch.arange(n)
    t = T[ii, ii, ii].clone()
    D = T[ii, ii, :].clone()
    D[ii, ii] = 0.0
    return t, D


def nxt(T):   # one gate + transport: T'_{abc} = sum W_ia W_jb W_kc d1 d1 d1 T_ijk
    G = T * d1[:, None, None] * d1[None, :, None] * d1[None, None, :]
    return torch.einsum("ijk,ia,jb,kc->abc", G, Wn, Wn, Wn)


def rel(x, y):
    return float((x - y).norm() / y.norm().clamp_min(1e-300))


def cp_als(T, R, iters=60):
    # SVD init of the mode-1 unfolding
    U, S, _ = torch.linalg.svd(T.reshape(n, n * n), full_matrices=False)
    k = min(R, n)
    A = torch.cat([U[:, :k] * S[:k].sqrt()[None, :], torch.randn(n, R - k, dtype=T.dtype) * 1e-3], 1) if R > k else U[:, :R] * S[:R].sqrt()[None, :]
    B, C = A.clone(), A.clone()
    lam = 1e-10
    for it in range(iters):
        for which in range(3):
            if which == 0:
                M = torch.einsum("ijk,jr,kr->ir", T, B, C); G = (B.T @ B) * (C.T @ C)
                A = torch.linalg.solve(G + lam * torch.eye(R), M.T).T
            elif which == 1:
                M = torch.einsum("ijk,ir,kr->jr", T, A, C); G = (A.T @ A) * (C.T @ C)
                B = torch.linalg.solve(G + lam * torch.eye(R), M.T).T
            else:
                M = torch.einsum("ijk,ir,jr->kr", T, A, B); G = (A.T @ A) * (B.T @ B)
                C = torch.linalg.solve(G + lam * torch.eye(R), M.T).T
    Th = torch.einsum("ir,jr,kr->ijk", A, B, C)
    return (Th + Th.permute(0, 2, 1) + Th.permute(1, 0, 2) + Th.permute(1, 2, 0) + Th.permute(2, 0, 1) + Th.permute(2, 1, 0)) / 6.0


out = {"n": n, "L": L, "layer": lay, "age0": a0, "seed": seed, "rows": []}
for part in ("old", "all"):
    T = grab[part]
    T = (T + T.permute(0, 2, 1) + T.permute(1, 0, 2) + T.permute(1, 2, 0) + T.permute(2, 0, 1) + T.permute(2, 1, 0)) / 6.0
    t0, D0 = slices(T)
    Tn = nxt(T)
    t1, D1 = slices(Tn)
    srcs = sum(1 for _ in range(1))
    print(f"[{part}] |T| {float(T.norm()):.3e}  |t| {float(t0.norm()):.3e}  |D| {float(D0.norm()):.3e}", flush=True)
    for R in Rs:
        tt = time.time()
        Th = cp_als(T, R)
        th, Dh = slices(Th)
        Thn = nxt(Th)
        th1, Dh1 = slices(Thn)
        row = dict(part=part, R=R, tensor=rel(Th, T), t=rel(th, t0), D=rel(Dh, D0), t_next=rel(th1, t1), D_next=rel(Dh1, D1),
                   secs=time.time() - tt)
        out["rows"].append(row)
        print(f"  R={R:5d} ({R / n:.2f} n): tensor {row['tensor']:.3f}  t {row['t']:.3f}  D {row['D']:.3f}  | next layer t "
              f"{row['t_next']:.3f}  D {row['D_next']:.3f}  [{row['secs']:.0f} s]", flush=True)
(HERE / "res" / f"cp_n{n}_L{L}_l{lay}_a{a0}_s{seed}.json").write_text(json.dumps(out, indent=1))
