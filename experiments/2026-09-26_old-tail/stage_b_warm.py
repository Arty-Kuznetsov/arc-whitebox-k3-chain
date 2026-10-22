"""How many sweeps does the INCREMENTAL merge need when WARM-STARTED from the previous block?

This is the production loop and the only thing that could still make the merge affordable.  A cold
(column-pruning) start needs O(10^2-10^3) sweeps at n=1024, while the whole saving allows about ONE sweep
per layer.  But production never starts cold: at layer l+1 it starts from the block it already had.

Measured here: fit rank r to the newest K old sources (cold, expensive, offline), then add ONE more source
and re-fit WARM from that solution, counting sweeps to reach a target error.  The comparison cold-vs-warm
at the same target is the number that decides the idea.

    ../../.venv-torch/bin/python stage_b_warm.py --n 256 --K 2 --target 0.02
"""
import argparse, json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE))
from stage_b_cp import capture_tail
from stage_b_gram import tail_columns, _ip, norm2_T


def sweeps_to(A, Bm, r, nT2, X0, Y0, target, max_it, lr):
    """Adam from (X0, Y0); return (sweeps to first reach `target`, best error, init error)."""
    nrm = nT2 ** 0.5
    X = X0.clone().requires_grad_(True)
    Y = Y0.clone().requires_grad_(True)

    def rel(Xa, Ya):
        P, Q, U, V = A.T @ Xa, A.T @ Ya, Bm.T @ Xa, Bm.T @ Ya
        Gxx, Gyy, Gxy = Xa.T @ Xa, Ya.T @ Ya, Xa.T @ Ya
        f2 = (Gxx * Gyy * Gyy).sum() / 3.0 + 2.0 * (Gxy.T * Gxy * Gyy).sum() / 3.0
        return torch.sqrt((nT2 - 2.0 * _ip(P, Q, U, V) + f2).clamp_min(1e-300)) / nrm

    init = float(rel(X0, Y0))
    hit, best = None, init
    with torch.enable_grad():
        opt = torch.optim.Adam([X, Y], lr=lr)
        for it in range(max_it):
            opt.zero_grad(set_to_none=True)
            loss = rel(X, Y)
            loss.backward(); opt.step()
            cur = float(loss.detach())
            best = min(best, cur)
            if hit is None and cur <= target:
                hit = it + 1
                break
    return hit, best, init, (X.detach(), Y.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--a0", type=int, default=4)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--K", type=int, default=2, help="sources in the block before the new one arrives")
    ap.add_argument("--target", type=float, default=0.02)
    ap.add_argument("--cold-it", type=int, default=20000)
    ap.add_argument("--max-it", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="logs/stage_b_warm.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    n, r = a.n, a.n
    cols = capture_tail(n, a.depth, a.a0, a.layer, thin=n)
    print(f"n={n}: {len(cols)} old sources available; block holds the newest K={a.K}, then +1 arrives",
          flush=True)

    def prep(K):
        c = cols[-K:]
        A, Bm = tail_columns(c)
        return A, Bm, norm2_T(A, Bm)

    # --- stage 1: the block over K sources, cold from column pruning
    A1, B1, nT1 = prep(a.K)
    w = A1.norm(dim=0) * B1.norm(dim=0) ** 2
    keep = torch.argsort(w, descending=True)[:r]
    hit1, best1, init1, (Xb, Yb) = sweeps_to(A1, B1, r, nT1, A1[:, keep].clone(), B1[:, keep].clone(),
                                             a.target, a.cold_it, a.lr)
    print(f"  block over K={a.K} sources, COLD from pruning: init {init1:.4e} -> best {best1:.4e}, "
          f"reached {a.target} after {hit1} sweeps", flush=True)

    # --- stage 2: one more source arrives. WARM from the block above vs COLD from pruning.
    A2, B2, nT2v = prep(a.K + 1)
    hit_w, best_w, init_w, _ = sweeps_to(A2, B2, r, nT2v, Xb, Yb, a.target, a.max_it, a.lr)
    w2 = A2.norm(dim=0) * B2.norm(dim=0) ** 2
    k2 = torch.argsort(w2, descending=True)[:r]
    hit_c, best_c, init_c, _ = sweeps_to(A2, B2, r, nT2v, A2[:, k2].clone(), B2[:, k2].clone(),
                                         a.target, a.max_it, a.lr)
    print(f"  +1 source, WARM from the block: init {init_w:.4e} -> best {best_w:.4e}, "
          f"reached {a.target} after {hit_w} sweeps", flush=True)
    print(f"  +1 source, COLD from pruning  : init {init_c:.4e} -> best {best_c:.4e}, "
          f"reached {a.target} after {hit_c} sweeps", flush=True)
    rec = dict(n=n, a0=a.a0, layer=a.layer, K=a.K, r=r, target=a.target, lr=a.lr,
               cold_block_sweeps=hit1, cold_block_best=best1,
               warm_sweeps=hit_w, warm_init=init_w, warm_best=best_w,
               cold_sweeps=hit_c, cold_init=init_c, cold_best=best_c)
    with open(HERE / a.out, "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
