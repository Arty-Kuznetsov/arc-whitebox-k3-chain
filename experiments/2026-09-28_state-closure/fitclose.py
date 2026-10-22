"""Can the all-distinct part be CLOSED from cheap O(n^2) observables?

closure.py measured that the part of the next kappa3 diagonal which (t, D) do not determine is 28.8% of the
readout, and tdchain.py measured that a closure must capture >=93% of it to stay within 3% of raw.  This
script asks whether any low-parameter closure can get there, by giving the fit an ORACLE target: least
squares of the true transported diagonal on a handful of cheap basis vectors, fitted per layer.  If even the
oracle-fitted version cannot reach 93%, the direction is closed; the fit is an upper bound on any closure.

Basis (all O(n^2) given what the chain already holds, apart from one n^3 product for diag(M C0 M^T), which
the chain computes anyway as the next layer's pre-activation covariance):
  b1 = sum_j M_ij^3 t_j                       (the t part -- exact, not a model)
  b2 = 3 sum_{j!=k} M_ij^2 M_ik D_jk          (the D part -- exact, not a model)
  b3 = (M t)_i * diag(M C0 M^T)_i             STAR ansatz: T_jkm ~ Sym(u_j C0_km) with u = t
  b4 = (M 1)_i * diag(M C0 M^T)_i             the same with a flat amplitude
  b5 = (M (t/var))_i * diag(M C0 M^T)_i       the same with the natural normalised amplitude

    ../../.venv-torch/bin/python fitclose.py --n 1024 --mlps 3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE))
from closure import CLEAN, two_slice_predict


def fit_capture(y, basis, fixed=2):
    """Least squares of y on `basis`; returns (residual norm, coefficients).
    The first `fixed` columns are the exact (t, D) parts -- they are refitted too, so the number reported is
    the best case for ANY closure built on this basis."""
    A = torch.stack(basis, dim=1)
    sol = torch.linalg.lstsq(A, y[:, None]).solution[:, 0]
    return float((y - A @ sol).norm()), sol


def run_one(mlp_idx, baked, depth=16):
    import chain
    W = np.load(baked / f"weights_{mlp_idx}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    n = Ws[0].shape[0]
    orig_read, orig_non = chain.read_slices, chain.nonlin_slices
    cap = {"call": 0, "src": {}, "d1": {}, "C0": {}, "var": {}}

    def wrapped_read(sources, nn, d21_ages=None, tailx=None):
        k = cap["call"]; cap["call"] += 1
        cap["src"][k] = [dict(s) for s in sources]
        return orig_read(sources, nn, d21_ages, tailx)

    def wrapped_non(mu, C, *a, **kw):
        o = orig_non(mu, C, *a, **kw)
        k = cap["call"] - 1
        cap["d1"][k] = o["d1"].clone()
        cap["C0"][k] = o["C0"].clone()
        cap["var"][k] = torch.diagonal(C).clone()
        return o

    chain.read_slices, chain.nonlin_slices = wrapped_read, wrapped_non
    try:
        chain.run_chain(Ws, **CLEAN)
    finally:
        chain.read_slices, chain.nonlin_slices = orig_read, orig_non

    rows = []
    for k in sorted(cap["src"]):
        layer = k + 1
        if k + 1 not in cap["src"] or layer + 1 >= len(Ws) or k not in cap["d1"]:
            continue
        src_l, src_n = cap["src"][k], cap["src"][k + 1]
        old = src_n[:len(src_l)]
        if len(old) != len(src_l) or not old:
            continue
        M = Ws[layer + 1].T * cap["d1"][k][None, :]
        chk = float((M @ src_l[0]["B"] - old[0]["B"]).abs().max() / old[0]["B"].abs().max())
        if chk > 1e-10:
            raise SystemExit(f"TRANSPORT SELF-CHECK FAILED layer {layer}: {chk:.2e}")
        t_l, D_l = orig_read(src_l, n)
        t_n, _ = orig_read(old, n)
        M2 = M * M
        b1 = (M2 * M) @ t_l
        b2 = 3.0 * (M2 * (M @ D_l.T)).sum(1)
        C0 = cap["C0"][k]
        q = ((M @ C0) * M).sum(1)                    # diag(M C0 M^T)
        var = cap["var"][k].clamp_min(1e-30)
        b3 = (M @ t_l) * q
        b4 = (M @ torch.ones(n, dtype=M.dtype)) * q
        b5 = (M @ (t_l / var)) * q
        dist = t_n - (b1 + b2)
        r2, _ = fit_capture(t_n, [b1, b2])
        r3, c3 = fit_capture(t_n, [b1, b2, b3])
        r5, c5 = fit_capture(t_n, [b1, b2, b3, b4, b5])
        nd = float(dist.norm())
        rows.append(dict(mlp=mlp_idx, layer=layer, norm_tn=float(t_n.norm()), norm_dist=nd,
                         cap_ab=1 - r2 / nd, cap_star=1 - r3 / nd, cap_all=1 - r5 / nd,
                         coef_ab=[float(x) for x in fit_capture(t_n, [b1, b2])[1]],
                         corr_dist_b2=float(torch.dot(dist, b2) / (dist.norm() * b2.norm()))))
        r = rows[-1]
        print(f"  MLP {mlp_idx} L{layer:2d}: |dist|/|t|={100*nd/r['norm_tn']:5.1f}%  "
              f"capture: rescale(a,b) {100*r['cap_ab']:6.1f}%   +star {100*r['cap_star']:6.1f}%   "
              f"+all5 {100*r['cap_all']:6.1f}%   (a,b)=({r['coef_ab'][0]:.3f},{r['coef_ab'][1]:.3f})  "
              f"corr(dist,b2)={r['corr_dist_b2']:+.3f}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--mlps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="logs/fitclose.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    baked = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    print("ORACLE-fitted capture of the all-distinct part (upper bound on any closure on this basis)\n"
          "target: >=93% to stay within 3% of raw (tdchain.py)\n")
    rows = []
    for i in range(a.mlps):
        rows += run_one(i, baked)
    with open(HERE / a.out, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    deep = [r for r in rows if r["layer"] >= 8]
    for key, name in (("cap_ab", "rescale (a,b) only"), ("cap_star", "+ star ansatz"),
                      ("cap_all", "+ all five")):
        v = [r[key] for r in deep]
        print(f"\n  layers >= 8, mean capture, {name:20s}: {100*sum(v)/len(v):6.2f}%  "
              f"(min {100*min(v):.1f}%, max {100*max(v):.1f}%)")
    print(f"\n  needed: 93%.")


if __name__ == "__main__":
    main()
