"""STEP 1: the one number that decides unbiased importance sampling of the old tail.

The tail is T = sum_c S(a_c, b_c), S(a,b) = (a(x)b(x)b + b(x)a(x)b + b(x)b(x)a)/3, one term per birth column
of each old source (the hub weight g is folded into a).  Unbiased importance sampling keeps r columns with
p_c ~ w_c = ||S(a_c,b_c)|| and reweights by 1/(r p_c).  It is unbiased, O(n^2), needs no optimisation pass and
is nested by construction -- so none of the three objections in NEGATIVE 58 applies to it.

Its error is set by ONE ratio.  With p_c = w_c / W, W = sum_c w_c:
    E||That - T||^2 = (1/r) ( sum_c w_c^2 / p_c - ||T||^2 ) = (W^2 - ||T||^2) / r
    rel err = sqrt(W^2 - ||T||^2) / (sqrt(r) ||T||)  ~=  kappa / sqrt(r),   kappa = W / ||T||.
kappa = 1 means the terms are perfectly aligned (coherent, the best case); kappa = sqrt(R) means they are
orthogonal (the worst case).  NEGATIVE 16 says the tail's terms are same-sign, which is why this is worth
measuring rather than assuming.

    ../../.venv-torch/bin/python kappa.py --n 1024 --a0 1 --layer 15
"""
import argparse, json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE.parent / "2026-09-26_old-tail"))
from stage_b_cp import capture_tail
from stage_b_gram import tail_columns, norm2_T


def term_norms(A, Bm):
    """w_c = ||S(a_c,b_c)|| = sqrt( (1/3)|a|^2 |b|^4 + (2/3)(a.b)^2 |b|^2 ), all O(n) per column."""
    a2 = (A * A).sum(0)
    b2 = (Bm * Bm).sum(0)
    ab = (A * Bm).sum(0)
    return torch.sqrt((a2 * b2 * b2) / 3.0 + (2.0 / 3.0) * ab * ab * b2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--a0", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--layer", type=int, nargs="+", default=[15],
                    help="TRUE layer index; read_slices is only called when sources exist, so the capture "
                         "hook fires at call index layer-1 (NEGATIVE 58 quoted 'layer 12' = this call index, "
                         "i.e. true layer 13)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="logs/kappa.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    n = a.n
    print(f"n={n}, production config (corr=split24rr).  kappa = sum_c w_c / ||T||;  "
          f"importance-sampling rel err ~ kappa/sqrt(r).\n")
    print(f"  {'layer':>5s} {'a0':>3s} {'S':>3s} {'R':>6s} {'||T||':>11s} {'W=sum w_c':>11s} "
          f"{'kappa':>9s} {'k/sqrt(R)':>9s} | {'r=n/2':>8s} {'r=n':>8s} {'r=2n':>8s} {'r=4n':>8s}")
    for layer in a.layer:
        for a0 in a.a0:
            cols = capture_tail(n, a.depth, a0, layer - 1)      # call index; thin=None -> production corr
            if not cols:
                print(f"  {layer:5d} {a0:3d}   (no sources of that age at this layer)")
                continue
            A, Bm = tail_columns(cols)
            R = A.shape[1]
            w = term_norms(A, Bm)
            W = float(w.sum())
            nT2 = norm2_T(A, Bm)
            nT = nT2 ** 0.5
            kappa = W / nT
            var_num = max(W * W - nT2, 0.0) ** 0.5            # exact numerator, not the kappa approximation
            errs = {rr: var_num / (rr ** 0.5 * nT) for rr in (n // 2, n, 2 * n, 4 * n)}
            print(f"  {layer:5d} {a0:3d} {len(cols):3d} {R:6d} {nT:11.4e} {W:11.4e} {kappa:9.2f} "
                  f"{kappa/R**0.5:9.3f} | " + " ".join(f"{100*errs[rr]:7.1f}%" for rr in
                                                       (n // 2, n, 2 * n, 4 * n)))
            with open(HERE / a.out, "a") as fh:
                fh.write(json.dumps(dict(n=n, layer=layer, a0=a0, S=len(cols), R=R, normT=nT, W=W,
                                         kappa=kappa, kappa_over_sqrtR=kappa / R ** 0.5,
                                         err={str(k): v for k, v in errs.items()})) + "\n")
    print("\n  STOPPING RULE (brief): if the error at r <= 2n cannot reach ~5%, the cheap sampler is dead.")


if __name__ == "__main__":
    main()
