"""End-to-end: run the chain with a (t, D)-ONLY kappa3 state and measure raw MSE.

closure.py measured that one layer of transport pulls ~28% of the next kappa3 diagonal out of the part of
the tensor that (t, D) do not contain.  That is a per-layer number; what decides the idea is whether it
compounds over 15 layers.  Here the sources are still transported by the real chain, but their contribution
is IGNORED: the kappa3 slices fed to the nonlinear step are carried forward by the two-slice rule alone,
plus each layer's newborn computed exactly (a birth is O(n^2) from C0, D, t, kappa4, so a real (t,D)-state
chain could do it too).  So the raw MSE below is what an O(n^2)-state chain would actually score.

The (t, D)-determined parts of the transported slices, with u = M_i, w = M_k:
    t'_i   = sum_j u_j^3 t_j + 3 sum_{j!=k} u_j^2 u_k D_jk
    D'_ik  = sum_j u_j^2 w_j t_j + (u^2)^T D w + 2 (u o w)^T (D u)
           = [ (MoM)ot + (MoM)D + 2 Mo(M D^T) ] M^T  (row i, col k)
Both are verified against brute-force index masking at small n before any production number is quoted.

    ../../.venv-torch/bin/python tdchain.py --mode td --n 10
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
from closure import CLEAN, two_slice_predict


def two_slice_predict_D(M, t, D):
    M2 = M * M
    return (M2 * t[None, :] + M2 @ D + 2.0 * (M * (M @ D.T))) @ M.T


def selfcheck(n=40, seed=7):
    """Both closed forms against brute force over the index-coincidence masks."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, 6, generator=g, dtype=torch.float64)
    B = torch.randn(n, 6, generator=g, dtype=torch.float64)
    gg = torch.randn(6, generator=g, dtype=torch.float64)
    T = torch.zeros(n, n, n, dtype=torch.float64)
    for p in range(6):
        a, b = A[:, p] * gg[p], B[:, p]
        T += (torch.einsum('i,j,k->ijk', a, b, b) + torch.einsum('i,j,k->ijk', b, a, b)
              + torch.einsum('i,j,k->ijk', b, b, a))
    t = torch.diagonal(torch.diagonal(T, dim1=1, dim2=2), dim1=0, dim2=1).clone()
    D = torch.diagonal(T, dim1=0, dim2=1).T.clone(); D.fill_diagonal_(0.0)
    M = torch.randn(n, n, generator=g, dtype=torch.float64) / n ** 0.5
    idx = torch.arange(n)
    I, J, K = torch.meshgrid(idx, idx, idx, indexing="ij")
    keep = ((I == J) & (J == K)) | ((I == J) & (J != K)) | ((I == K) & (I != J)) | ((J == K) & (I != J))
    Tk = T * keep
    bf_t = torch.einsum('ai,aj,ak,ijk->a', M, M, M, Tk)
    bf_D = torch.einsum('ai,aj,bk,ijk->ab', M, M, M, Tk)
    e_t = float((two_slice_predict(M, t, D) - bf_t).abs().max() / bf_t.abs().max())
    e_D = float((two_slice_predict_D(M, t, D) - bf_D).abs().max() / bf_D.abs().max())
    print(f"  self-check: t-rule vs brute force {e_t:.2e}   D-rule vs brute force {e_D:.2e}", flush=True)
    assert e_t < 1e-12 and e_D < 1e-12, (e_t, e_D)


def run_one(mlp_idx, mode, baked, depth=16):
    import chain
    W = np.load(baked / f"weights_{mlp_idx}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    n = Ws[0].shape[0]
    orig_read, orig_non = chain.read_slices, chain.nonlin_slices
    st = {"k": 0, "t": None, "D": None, "d1": None}

    def wrapped_read(sources, nn, d21_ages=None, tailx=None):
        k = st["k"]; st["k"] += 1
        layer = k + 1
        if k == 0 or mode == "base":
            t, D = orig_read(sources, nn, d21_ages, tailx)
        else:
            M = Ws[layer].T * st["d1"][None, :]
            if mode == "exact":                       # plumbing self-check: must reproduce the base
                t_p, D_p = orig_read(sources[:-1], nn)
            else:                                      # "td" / "td<frac>": carry the two slices forward,
                t_p = two_slice_predict(M, st["t"], st["D"])      # optionally keeping a fraction of the
                D_p = two_slice_predict_D(M, st["t"], st["D"])    # all-distinct part an oracle closure gives
                frac = float(mode[2:]) if len(mode) > 2 else 0.0
                if frac:
                    t_e, D_e = orig_read(sources[:-1], nn)
                    t_p = t_p + frac * (t_e - t_p)
                    D_p = D_p + frac * (D_e - D_p)
            t_nb, D_nb = orig_read([sources[-1]], nn)
            t, D = t_p + t_nb, D_p + D_nb
            D = D.clone(); D.fill_diagonal_(0.0)
        st["t"], st["D"] = t, D
        return t, D

    def wrapped_non(*a, **kw):
        o = orig_non(*a, **kw)
        st["d1"] = o["d1"].clone()
        return o

    chain.read_slices, chain.nonlin_slices = wrapped_read, wrapped_non
    try:
        mu, _ = chain.run_chain(Ws, **CLEAN)
    finally:
        chain.read_slices, chain.nonlin_slices = orig_read, orig_non
    return mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", default=["base", "exact", "td"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default="logs/tdchain.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    selfcheck()
    baked = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    gt = np.load(baked / "gt.npy")
    base = None
    for mode in a.mode:
        mses, t0 = [], time.time()
        for i in range(a.n):
            mu = run_one(i, mode, baked)
            mses.append(float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean()))
        raw = float(np.mean(mses))
        if mode == "base":
            base = raw
        rel = f"  x{raw/base:8.3f}" if base else ""
        print(f"  mode {mode:6s}: raw {raw:.4e}{rel}   ({time.time()-t0:.0f}s)", flush=True)
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(dict(mode=mode, n=a.n, raw=raw, per_mlp=mses)) + "\n")


if __name__ == "__main__":
    main()
