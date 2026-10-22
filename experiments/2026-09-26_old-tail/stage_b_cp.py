"""Stage B, the central question: is the old tail's 3-tensor compressible to rank ~n AFTER merging?

The tail's whole contribution to the chain is one symmetric 3-tensor
    T = sum_s sum_p g_s[p] * Sym( a_s[:,p] (x) b_s[:,p] (x) b_s[:,p] ),
whose diagonal is the tail's t and whose (2,1) slice is the tail's D.  Only the LEGS transport linearly
(L1 = 2 A o B o g and L2 = B o B o g are nonlinear in them and must be rebuilt every layer), so a merged
block MUST have the same symmetric-CP form with r columns -- and then it must match T as a TENSOR, not
just T's two slices, because later transports mix the rest of T into the readout.

One source contributes exactly n such rank-one terms, so S old sources give S*n.  The question is whether
r ~ n suffices.  This script measures that directly: capture the real tail's legs out of the chain (by
wrapping chain.read_slices -- chain.py itself is NOT edited for this), build T, and run ALS symmetric-CP
at a ladder of ranks.

SELF-CHECK built in: for S = 1 the tensor IS n rank-one terms, so ALS at r = n must reach ~0 error.
If it does, the fitter is right and the S >= 2 numbers mean what they say.

    ../../.venv-torch/bin/python stage_b_cp.py --n 128 --a0 4 --layer 12
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)


def capture_tail(n, depth, a0, layer, seed=3, thin=None):
    """Run the chain on a synthetic He-init MLP and return the (a, b, g) columns of the sources whose age
    is >= a0 at `layer`, by wrapping read_slices.  Nothing in chain.py is modified."""
    import chain
    rng = np.random.default_rng(seed)
    Ws = [torch.tensor(rng.standard_normal((n, n)) * np.sqrt(2.0 / n), dtype=torch.float64)
          for _ in range(depth)]
    grabbed = {}
    orig = chain.read_slices
    calls = {"i": 0}

    def wrapped(sources, nn, d21_ages=None, tailx=None):
        if calls["i"] == layer and layer not in grabbed:
            S = len(sources)
            old = [s for idx, s in enumerate(sources) if S - idx >= a0]
            grabbed[layer] = [(s["A"].clone(), s["B"].clone(), s["g"].clone()) for s in old]
        calls["i"] += 1
        return orig(sources, nn, d21_ages, tailx)

    chain.read_slices = wrapped
    try:
        ck = dict(K3L)
        if thin:
            ck["corr"] = f"split{thin}"
        chain.run_chain(Ws, **ck)
    finally:
        chain.read_slices = orig
    return grabbed.get(layer, [])


def build_T(cols, n):
    """T[i,j,k] = sum over all (source, column p) of g_p * Sym(a_p (x) b_p (x) b_p), the symmetrisation being
    the 3 distinct placements of the 'a' leg."""
    T = torch.zeros(n, n, n, dtype=torch.float64)
    R = 0
    for A, Bm, g in cols:
        R += A.shape[1]
        for p in range(A.shape[1]):
            a = A[:, p] * g[p]
            b = Bm[:, p]
            T += (torch.einsum('i,j,k->ijk', a, b, b)
                  + torch.einsum('i,j,k->ijk', b, a, b)
                  + torch.einsum('i,j,k->ijk', b, b, a)) / 3.0
    return T, R


def fit_sym_cp(T, r, cols=None, iters=4000, seed=0, verbose=False):
    """Fit T ~ sum_q Sym(x_q (x) y_q (x) y_q) by gradient descent with autograd.

    A hand-rolled symmetric ALS is biased here (x appears in all three modes, so a mode-1 least-squares
    update is not the exact conditional optimum), which would overstate the error and could fake a negative
    result.  Autograd on the Frobenius loss is correct by construction; the S = 1 control then has to reach
    ~0 at r = n, and that is what certifies the fitter.

    Init from the r heaviest TRUE columns when available -- the honest best case for a merge, and it also
    starts the fit AT the exact answer when r >= R."""
    n = T.shape[0]
    gen = torch.Generator().manual_seed(seed)
    if cols is not None:
        Xs = torch.cat([A * g[None, :] for A, _, g in cols], dim=1)     # hub leg, weight folded in
        Ys = torch.cat([Bm for _, Bm, _ in cols], dim=1)
        w = Xs.norm(dim=0) * Ys.norm(dim=0) ** 2
        if r <= Xs.shape[1]:
            keep = torch.argsort(w, descending=True)[:r]
            X0, Y0 = Xs[:, keep].clone(), Ys[:, keep].clone()
        else:
            k = r - Xs.shape[1]
            X0 = torch.cat([Xs, 1e-3 * torch.randn(n, k, generator=gen, dtype=T.dtype)], dim=1)
            Y0 = torch.cat([Ys, 1e-3 * torch.randn(n, k, generator=gen, dtype=T.dtype)], dim=1)
    else:
        sc = float(T.abs().mean()) ** (1.0 / 3.0)
        X0 = sc * torch.randn(n, r, generator=gen, dtype=T.dtype)
        Y0 = sc * torch.randn(n, r, generator=gen, dtype=T.dtype)
    X = X0.clone().requires_grad_(True)
    Y = Y0.clone().requires_grad_(True)
    nrm = torch.linalg.norm(T.reshape(-1))
    with torch.enable_grad():
        opt = torch.optim.Adam([X, Y], lr=3e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=1e-4)
        best = float("inf")
        for it in range(iters):
            opt.zero_grad(set_to_none=True)
            loss = torch.linalg.norm((_reconstruct(X, Y) - T).reshape(-1)) / nrm
            loss.backward()
            opt.step(); sched.step()
            best = min(best, float(loss))
            if verbose and it % 500 == 0:
                print(f"      it {it:5d} rel err {float(loss):.4e}", flush=True)
        lb = torch.optim.LBFGS([X, Y], max_iter=250, history_size=30, line_search_fn="strong_wolfe")

        def closure():
            lb.zero_grad(set_to_none=True)
            l = torch.linalg.norm((_reconstruct(X, Y) - T).reshape(-1)) / nrm
            l.backward()
            return l
        lb.step(closure)
        with torch.no_grad():
            final = float(torch.linalg.norm((_reconstruct(X, Y) - T).reshape(-1)) / nrm)
    return min(best, final), iters


def _reconstruct(X, Y):
    return (torch.einsum('ir,jr,kr->ijk', X, Y, Y)
            + torch.einsum('ir,jr,kr->ijk', Y, X, Y)
            + torch.einsum('ir,jr,kr->ijk', Y, Y, X)) / 3.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--a0", type=int, default=4)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smax", type=int, default=0, help="keep only the newest SMAX old sources (0 = all)")
    ap.add_argument("--ranks", default="", help="comma-separated absolute ranks; default = a ladder in n")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="logs/stage_b_cp.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    n = a.n
    cols = capture_tail(n, a.depth, a.a0, a.layer, thin=n)
    if a.smax:
        cols = cols[-a.smax:]
    print(f"n={n} layer={a.layer} a0={a.a0}: captured {len(cols)} old sources "
          f"-> {sum(c[0].shape[1] for c in cols)} CP columns", flush=True)
    T, R = build_T(cols, n)
    print(f"  tensor built, ||T|| = {float(torch.linalg.norm(T.reshape(-1))):.4e}, CP terms R = {R}", flush=True)
    ladder = ([int(x) for x in a.ranks.split(",")] if a.ranks
              else [x for x in (n // 4, n // 2, n, 3 * n // 2, 2 * n, 3 * n, 4 * n) if x <= max(R, 1) * 2])
    for r in ladder:
        t0 = time.time()
        err, its = fit_sym_cp(T, r, cols=cols, iters=a.iters)
        rec = dict(n=n, layer=a.layer, a0=a.a0, S=len(cols), R=R, r=r, r_over_n=r / n, smax=a.smax,
                   rel_err=err, iters=a.iters, tag=a.tag, secs=round(time.time() - t0, 1))
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"  r={r:5d} (r/n={r/n:5.2f})  rel err {err:.4e}  [{its} its, {rec['secs']:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
