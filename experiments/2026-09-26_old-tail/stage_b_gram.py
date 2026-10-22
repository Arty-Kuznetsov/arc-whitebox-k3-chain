"""Stage B, done at production scale: the symmetric-CP rank of the old tail, without ever forming the tensor.

stage_b_cp.py builds T explicitly (n^3), which caps n at ~128 and left the fits optimizer-limited -- at
n=64, r=n the error still fell 1.31e-3 -> 6.6e-4 -> 4.2e-4 from 1.5k to 20k iterations, so those numbers are
UPPER BOUNDS and their n-scaling is not trustworthy.

Here the loss is expanded instead:  ||T - F||^2 = ||T||^2 - 2<T,F> + ||F||^2, and for
    S(a,b) = (a(x)b(x)b + b(x)a(x)b + b(x)b(x)a)/3
the inner products are closed-form in the four Gram matrices:
    <S(a,b), S(x,y)> = (1/3)(a.x)(b.y)^2 + (2/3)(a.y)(b.x)(b.y)
(check: a=b=x=y gives |a|^6).  So an iteration costs O(n R r) instead of O(n^3 r) -- a ~100x saving at
n=1024 and no n^3 memory, which is what makes a converged fit at the real width possible.

    ../../.venv-torch/bin/python stage_b_gram.py --n 512 --ranks 256,512,1024 --iters 3000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE))
from stage_b_cp import capture_tail, build_T, _reconstruct


def tail_columns(cols):
    """The tail as flat CP columns: hub leg a (weight folded in) and tree leg b, both n x R."""
    A = torch.cat([Am * g[None, :] for Am, _, g in cols], dim=1)
    Bm = torch.cat([Bt for _, Bt, _ in cols], dim=1)
    return A.contiguous(), Bm.contiguous()


def _ip(P, Q, U, V):
    """sum over column pairs of <S(a,b), S(x,y)> given the four Gram matrices."""
    return (P * V * V).sum() / 3.0 + 2.0 * (Q * U * V).sum() / 3.0


def norm2_T(A, Bm, block=512):
    """||T||^2 = sum_{i,j} <S(a_i,b_i), S(a_j,b_j)>, in blocks over i so the R x R Grams never materialise."""
    tot = 0.0
    R = A.shape[1]
    for s in range(0, R, block):
        e = min(s + block, R)
        Ai, Bi = A[:, s:e], Bm[:, s:e]
        tot += float(_ip(Ai.T @ A, Ai.T @ Bm, Bi.T @ A, Bi.T @ Bm))
    return tot


def fit(A, Bm, r, nT2, iters=3000, seed=0, lr=3e-2, init="heavy", log_every=0):
    """Minimise ||T - F||/||T|| over F = sum_q S(x_q, y_q).  Returns (rel_err, history)."""
    n, R = A.shape
    gen = torch.Generator().manual_seed(seed)
    if init == "heavy":
        w = A.norm(dim=0) * Bm.norm(dim=0) ** 2
        keep = torch.argsort(w, descending=True)[:min(r, R)]
        X0, Y0 = A[:, keep].clone(), Bm[:, keep].clone()
        if r > R:
            k = r - R
            X0 = torch.cat([X0, 1e-3 * torch.randn(n, k, generator=gen, dtype=A.dtype)], dim=1)
            Y0 = torch.cat([Y0, 1e-3 * torch.randn(n, k, generator=gen, dtype=A.dtype)], dim=1)
    else:
        sc = float(A.abs().mean())
        X0 = sc * torch.randn(n, r, generator=gen, dtype=A.dtype)
        Y0 = sc * torch.randn(n, r, generator=gen, dtype=A.dtype)
    nrm = nT2 ** 0.5

    def rel(Xa, Ya):
        P, Q = A.T @ Xa, A.T @ Ya
        U, V = Bm.T @ Xa, Bm.T @ Ya
        Gxx, Gyy, Gxy = Xa.T @ Xa, Ya.T @ Ya, Xa.T @ Ya
        f2 = (Gxx * Gyy * Gyy).sum() / 3.0 + 2.0 * (Gxy.T * Gxy * Gyy).sum() / 3.0
        return float(torch.sqrt((nT2 - 2.0 * _ip(P, Q, U, V) + f2).clamp_min(0.0)) / nrm)

    init_err = rel(X0, Y0)          # = the COLUMN-PRUNING merge: keep the r heaviest true CP columns
    if iters == 0:
        return init_err, [(0, init_err)]
    X = X0.clone().requires_grad_(True)
    Y = Y0.clone().requires_grad_(True)
    hist = [(-1, init_err)]
    with torch.enable_grad():
        opt = torch.optim.Adam([X, Y], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=lr / 300)
        best = float("inf")
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            P, Q = A.T @ X, A.T @ Y
            U, V = Bm.T @ X, Bm.T @ Y
            Gxx, Gyy, Gxy = X.T @ X, Y.T @ Y, X.T @ Y
            f2 = (Gxx * Gyy * Gyy).sum() / 3.0 + 2.0 * (Gxy.T * Gxy * Gyy).sum() / 3.0
            loss2 = nT2 - 2.0 * _ip(P, Q, U, V) + f2
            loss = torch.sqrt(loss2.clamp_min(1e-300)) / nrm
            loss.backward()
            opt.step(); sched.step()
            cur = float(loss.detach())
            best = min(best, cur)
            if log_every and it % log_every == 0:
                hist.append((it, cur))
                print(f"      it {it:6d} rel err {cur:.4e}", flush=True)
    return best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--a0", type=int, default=4)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--ranks", default="")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smax", type=int, default=0,
                    help="keep only the newest SMAX old sources; --smax 2 is the INCREMENTAL merge "
                         "step (an existing block's worth of columns + one graduating source)")
    ap.add_argument("--verify", action="store_true", help="cross-check the Gram loss against the explicit tensor")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default="logs/stage_b_gram.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    n = a.n
    cols = capture_tail(n, a.depth, a.a0, a.layer, thin=n)
    if a.smax:
        cols = cols[-a.smax:]
    A, Bm = tail_columns(cols)
    R = A.shape[1]
    nT2 = norm2_T(A, Bm)
    print(f"n={n} layer={a.layer} a0={a.a0}: S={len(cols)} old sources, R={R} CP columns, "
          f"||T||={nT2**0.5:.6e}", flush=True)
    if a.verify:   # the explicit tensor must agree with the Gram formula
        T, _ = build_T(cols, n)
        print(f"  VERIFY ||T||: explicit {float(torch.linalg.norm(T.reshape(-1))):.10e} vs gram "
              f"{nT2**0.5:.10e}  rel {abs(float(torch.linalg.norm(T.reshape(-1)))-nT2**0.5)/nT2**0.5:.2e}",
              flush=True)
        g = torch.Generator().manual_seed(1)
        X = torch.randn(n, 7, generator=g, dtype=A.dtype); Y = torch.randn(n, 7, generator=g, dtype=A.dtype)
        expl = float(torch.linalg.norm((T - _reconstruct(X, Y)).reshape(-1)) / torch.linalg.norm(T.reshape(-1)))
        P, Q, U, V = A.T @ X, A.T @ Y, Bm.T @ X, Bm.T @ Y
        Gxx, Gyy, Gxy = X.T @ X, Y.T @ Y, X.T @ Y
        f2 = (Gxx * Gyy * Gyy).sum() / 3.0 + 2.0 * (Gxy.T * Gxy * Gyy).sum() / 3.0
        gram = float(((nT2 - 2.0 * _ip(P, Q, U, V) + f2).clamp_min(0)) ** 0.5 / nT2 ** 0.5)
        print(f"  VERIFY loss at a random rank-7 F: explicit {expl:.10e} vs gram {gram:.10e}  "
              f"rel {abs(expl-gram)/expl:.2e}", flush=True)
    ladder = [int(x) for x in a.ranks.split(",")] if a.ranks else [n // 2, n, 2 * n]
    for r in ladder:
        t0 = time.time()
        err, hist = fit(A, Bm, r, nT2, iters=a.iters, lr=a.lr, log_every=a.log_every)
        rec = dict(n=n, layer=a.layer, a0=a.a0, S=len(cols), R=R, r=r, r_over_n=r / n, smax=a.smax,
                   rel_err=err, init_err=hist[0][1], iters=a.iters, lr=a.lr,
                   secs=round(time.time() - t0, 1))
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  r={r:6d} (r/n={r/n:5.2f})  rel err {err:.4e}   column-pruning init {rec['init_err']:.4e}"
              f"  [{a.iters} its, {rec['secs']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
