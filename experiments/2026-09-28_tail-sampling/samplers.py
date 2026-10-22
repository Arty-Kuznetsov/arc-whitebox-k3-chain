"""STEP 2 (run as the VERIFICATION of step 1, not as a hope): actually draw the samplers and measure the
true relative tensor error with the Gram evaluator, at the real n=1024 tail.

Step 1 said kappa ~ 60, so the predicted error is ~kappa/sqrt(r) = 194% at r=n.  A formula that decides the
last open idea has to be checked against a real draw, and the variance-reduced variants have to be measured
rather than argued away -- otherwise "you did not try the hybrid" is a fair objection.

  (a) importance sampling with replacement, p ~ w
  (b) hybrid: top-k by weight kept deterministically, r-k sampled from the rest (k = r/2, r/4)
  (c) systematic / stratified on the cumulative weight (one uniform per stratum)
Also reports the weight concentration, which is what decides whether (b) can help at all.

    ../../.venv-torch/bin/python samplers.py --n 1024 --a0 1 --layer 15
"""
import argparse, json, sys, time
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE.parent / "2026-09-26_old-tail"))
from stage_b_cp import capture_tail
from stage_b_gram import tail_columns, norm2_T, _ip
from kappa import term_norms


def rel_err(A, Bm, nT2, X, Y):
    """||T - F|| / ||T|| with F = sum_q S(x_q, y_q), via the Gram form (verified to 3.6e-15 in 2026-09-26)."""
    P, Q, U, V = A.T @ X, A.T @ Y, Bm.T @ X, Bm.T @ Y
    Gxx, Gyy, Gxy = X.T @ X, Y.T @ Y, X.T @ Y
    f2 = (Gxx * Gyy * Gyy).sum() / 3.0 + 2.0 * (Gxy.T * Gxy * Gyy).sum() / 3.0
    return float(torch.sqrt((nT2 - 2.0 * _ip(P, Q, U, V) + f2).clamp_min(0.0)) / nT2 ** 0.5)


def scale_cols(A, Bm, idx, coef):
    """A term S(a,b) scales linearly in a, so the reweight g/(r p) is folded into the hub leg."""
    return A[:, idx] * coef[None, :], Bm[:, idx].clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--a0", type=int, default=1)
    ap.add_argument("--layer", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="logs/samplers.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    n = a.n
    cols = capture_tail(n, a.depth, a.a0, a.layer - 1)
    A, Bm = tail_columns(cols)
    R = A.shape[1]
    w = term_norms(A, Bm)
    W = float(w.sum())
    nT2 = norm2_T(A, Bm)
    kappa = W / nT2 ** 0.5
    order = torch.argsort(w, descending=True)
    cum = torch.cumsum(w[order], 0) / W
    print(f"n={n} a0={a.a0} true layer {a.layer}: S={len(cols)} sources, R={R} terms, "
          f"||T||={nT2**0.5:.4e}, W={W:.4e}, kappa={kappa:.2f}", flush=True)
    print(f"  weight concentration: top {n} terms hold {100*float(cum[n-1]):.1f}% of W, "
          f"top {2*n}: {100*float(cum[2*n-1]):.1f}%, top {4*n}: {100*float(cum[4*n-1]):.1f}%  "
          f"(uniform would be {100*n/R:.1f} / {100*2*n/R:.1f} / {100*4*n/R:.1f}%)", flush=True)
    p = (w / W).double()
    recs = []
    for r in (n // 2, n, 2 * n, 4 * n):
        pred = ((W * W - nT2) ** 0.5) / (r ** 0.5 * nT2 ** 0.5)
        for name, k in (("(a) IS w/ replacement", 0), ("(b) hybrid top r/4", r // 4),
                        ("(b) hybrid top r/2", r // 2), ("(c) systematic", -1)):
            errs = []
            for sd in range(a.seeds):
                g = torch.Generator().manual_seed(1000 * sd + r)
                if k == -1:                     # systematic on the cumulative weight
                    u = torch.rand(1, generator=g, dtype=torch.float64).item()
                    pts = (torch.arange(r, dtype=torch.float64) + u) / r
                    idx = torch.searchsorted(torch.cumsum(p, 0), pts.clamp(max=1 - 1e-12))
                    idx = idx.clamp(max=R - 1)
                    coef = 1.0 / (r * p[idx])
                elif k == 0:
                    idx = torch.multinomial(p, r, replacement=True, generator=g)
                    coef = 1.0 / (r * p[idx])
                else:                            # deterministic head + importance-sampled tail
                    head = order[:k]
                    rest = order[k:]
                    pr = p[rest] / p[rest].sum()
                    js = torch.multinomial(pr, r - k, replacement=True, generator=g)
                    idx = torch.cat([head, rest[js]])
                    coef = torch.cat([torch.ones(k, dtype=torch.float64),
                                      p[rest].sum() / ((r - k) * pr[js])])
                X, Y = scale_cols(A, Bm, idx, coef)
                errs.append(rel_err(A, Bm, nT2, X, Y))
            m = sum(errs) / len(errs)
            print(f"  r={r:5d} {name:24s} rel err {100*m:8.1f}%   (predicted {100*pred:6.1f}%)", flush=True)
            recs.append(dict(n=n, a0=a.a0, layer=a.layer, R=R, kappa=kappa, r=r, sampler=name, k=k,
                             rel_err=m, errs=errs, predicted=pred))
    with open(HERE / a.out, "a") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
