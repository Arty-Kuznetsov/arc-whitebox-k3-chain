"""Is (t, D, C, kappa4) a sufficient state, i.e. do the two slices the nonlinear step reads carry the chain?

Five attempts (NEGATIVE 54-58) all tried to represent the kappa3 TENSOR more cheaply.  None asked whether it
has to be represented at all: `nonlin_slices` reads only t and D.  If (t, D) were closed under one layer, the
whole per-source machinery (legs 12.8% + pairings 10.6% of budget) would collapse to O(n^2) state.

EXACTLY it is not closed.  All legs transport linearly, so with M = W^T diag(d1) the whole kappa3 tensor
transports as T -> (M,M,M).T, and splitting the contraction by index coincidence:

    t^(l+1)_i = sum_j M_ij^3 t_j            <- in t
              + 3 sum_{j!=k} M_ij^2 M_ik D_jk   <- in D
              + sum_{j,k,m all distinct} M_ij M_ik M_im T_jkm   <- NOT in (t, D)

so the question is quantitative: how big is the third term?  It is computable as a RESIDUAL, without ever
forming T: the full contraction is just the chain's own transported readout.

Self-checks: (1) M reproduces the chain's own leg transport to machine precision; (2) at small n the explicit
tensor reproduces the split exactly.

    ../../.venv-torch/bin/python closure.py --n 1024 --mlps 3
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))

# k3l_t minus the path births: with e3age=1 the path leg is DROPPED at age 2, which is not a transport and
# would pollute the residual.  paths=None makes the kappa3 state transport exactly, which is what the
# structural question needs.  The production arm is measured separately and the caveat reported.
CLEAN = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
             marg=True, k31pat=True, paths=None, e3age=None, pathD="full", k4b=1, k4bparts=("sd",),
             pairx=("k22",), tet=3, deadcut=None, k31mult=0.65)


def two_slice_predict(M, t, D):
    """The part of the next layer's kappa3 diagonal that (t, D) alone determine."""
    M2 = M * M
    term_t = (M2 * M) @ t                       # sum_j M_ij^3 t_j
    term_D = 3.0 * (M2 * (M @ D.T)).sum(1)      # 3 sum_{j!=k} M_ij^2 M_ik D_jk  (D has zero diagonal)
    return term_t + term_D


def run(n, depth, mlp_idx, cfg, baked, verbose=True):
    import chain
    W = np.load(baked / f"weights_{mlp_idx}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    cap = {"d1": {}, "src": {}, "call": 0}
    orig_read, orig_non = chain.read_slices, chain.nonlin_slices

    def wrapped_read(sources, nn, d21_ages=None, tailx=None):
        k = cap["call"]
        cap["src"][k] = [dict(s) for s in sources]          # shallow: leg tensors are replaced, not mutated
        cap["call"] += 1
        return orig_read(sources, nn, d21_ages, tailx)

    def wrapped_non(*a, **kw):
        o = orig_non(*a, **kw)
        cap["d1"][cap["call"] - 1] = o["d1"].clone()        # the d1 that gates the legs at the end of this layer
        return o

    chain.read_slices, chain.nonlin_slices = wrapped_read, wrapped_non
    try:
        chain.run_chain(Ws, **cfg)
    finally:
        chain.read_slices, chain.nonlin_slices = orig_read, orig_non

    rows = []
    for k in sorted(cap["src"]):
        if k + 1 not in cap["src"] or k not in cap["d1"]:
            continue
        src_l, src_n = cap["src"][k], cap["src"][k + 1]
        old = src_n[:len(src_l)]                            # the same sources, transported and gated
        if not old or len(old) != len(src_l):
            continue
        layer = k + 1                                       # read_slices only runs once sources exist
        if layer + 1 >= len(Ws):
            continue
        # between the readout at layer l and the readout at l+1 the legs are gated by d1^(l) and then
        # transported by W_{l+1}: M = W_{l+1}^T diag(d1^(l)).  The self-check below is what pins this down.
        M = Ws[layer + 1].T * cap["d1"][k][None, :]
        chk = float((M @ src_l[0]["B"] - old[0]["B"]).abs().max() /
                    old[0]["B"].abs().max())                # SELF-CHECK 1: M is the chain's own transport
        if chk > 1e-10:
            raise SystemExit(f"TRANSPORT SELF-CHECK FAILED at layer {layer}: {chk:.3e} -- M is wrong, "
                             f"every number below it would be meaningless")
        t_l, D_l = orig_read(src_l, n)
        t_n, _ = orig_read(old, n)
        pred = two_slice_predict(M, t_l, D_l)
        resid = t_n - pred
        rows.append(dict(mlp=mlp_idx, layer=layer, S=len(src_l), transport_check=chk,
                         rel_resid=float(resid.norm() / t_n.norm()),
                         frac_t=float(((M * M * M) @ t_l).norm() / t_n.norm()),
                         frac_D=float((3.0 * ((M * M) * (M @ D_l.T)).sum(1)).norm() / t_n.norm()),
                         norm_tn=float(t_n.norm())))
        if verbose:
            r = rows[-1]
            print(f"  MLP {mlp_idx} layer {layer:2d} (S={r['S']:2d}): "
                  f"|distinct|/|t| = {100*r['rel_resid']:7.2f}%   "
                  f"(from t {100*r['frac_t']:6.2f}%, from D {100*r['frac_D']:6.2f}%)   "
                  f"transport self-check {r['transport_check']:.1e}", flush=True)
    return rows


def selfcheck_small(n=48, depth=6, seed=5):
    """SELF-CHECK 2: build the explicit kappa3 tensor of the sources at small n and verify that
    diag-part + two-equal-part + all-distinct-part reproduces the full contraction exactly."""
    import chain
    g = torch.Generator().manual_seed(seed)
    Ws = [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]
    cap = {}
    orig = chain.read_slices

    def wrapped(sources, nn, d21_ages=None, tailx=None):
        cap.setdefault("s", [dict(s) for s in sources] if len(sources) >= 2 else None)
        if cap.get("s") is None and len(sources) >= 2:
            cap["s"] = [dict(s) for s in sources]
        return orig(sources, nn, d21_ages, tailx)
    chain.read_slices = wrapped
    try:
        chain.run_chain(Ws, **CLEAN)
    finally:
        chain.read_slices = orig
    src = cap["s"]
    t, D = orig(src, n)
    # explicit tensor of the TREE part only (the piece whose algebra we quote); E/feed add more terms, so the
    # check is run on a source set restricted to its tree contribution
    T = torch.zeros(n, n, n, dtype=torch.float64)
    for s in src:
        A, B, gg = s["A"], s["B"], s["g"]
        for p in range(A.shape[1]):
            a, b = A[:, p] * gg[p], B[:, p]
            T += (torch.einsum('i,j,k->ijk', a, b, b) + torch.einsum('i,j,k->ijk', b, a, b)
                  + torch.einsum('i,j,k->ijk', b, b, a))
    tt = torch.diagonal(torch.diagonal(T, dim1=1, dim2=2), dim1=0, dim2=1).clone()
    DD = torch.diagonal(T, dim1=0, dim2=1).T.clone()          # DD[j,k] = T[j,j,k]
    DD.fill_diagonal_(0.0)
    v = torch.randn(4, n, generator=g, dtype=torch.float64)
    full = torch.einsum('ai,aj,ak,ijk->a', v, v, v, T)
    # my closed form for the part (t, D) determine
    part = (v ** 3) @ tt + 3.0 * ((v * v) * (v @ DD.T)).sum(1)
    # the same thing by BRUTE FORCE over index-coincidence masks -- this is what makes the check real
    idx = torch.arange(n)
    I, J, K = torch.meshgrid(idx, idx, idx, indexing="ij")
    m_all = (I == J) & (J == K)
    m_two = (((I == J) & (J != K)) | ((I == K) & (I != J)) | ((J == K) & (I != J)))
    m_dist = ~(m_all | m_two)
    Tall, Ttwo, Tdist = T * m_all, T * m_two, T * m_dist
    bf_all = torch.einsum('ai,aj,ak,ijk->a', v, v, v, Tall)
    bf_two = torch.einsum('ai,aj,ak,ijk->a', v, v, v, Ttwo)
    bf_dist = torch.einsum('ai,aj,ak,ijk->a', v, v, v, Tdist)
    e_form = float((part - (bf_all + bf_two)).abs().max() / full.abs().max())
    e_sum = float((full - (bf_all + bf_two + bf_dist)).abs().max() / full.abs().max())
    print(f"  self-check 2a  closed form == brute-force (diag + two-equal):  rel {e_form:.2e}", flush=True)
    print(f"  self-check 2b  the three masks partition the contraction:      rel {e_sum:.2e}", flush=True)
    print(f"  self-check 2c  (n={n}, random probes) all-distinct / full = "
          f"{100*float(bf_dist.norm()/full.norm()):.2f}%", flush=True)
    assert e_form < 1e-12 and e_sum < 1e-12, (e_form, e_sum)
    return float(bf_dist.norm() / full.norm())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--mlps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--small-only", action="store_true")
    ap.add_argument("--out", default="logs/closure.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    print("SELF-CHECKS")
    selfcheck_small()
    if a.small_only:
        return
    baked = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    print(f"\nHow much of the next layer's kappa3 diagonal is NOT determined by (t, D)?  n={a.n}", flush=True)
    allrows = []
    for i in range(a.mlps):
        t0 = time.time()
        allrows += run(a.n, a.depth, i, CLEAN, baked)
        print(f"  [MLP {i} done in {time.time()-t0:.0f}s]", flush=True)
    with open(HERE / a.out, "a") as fh:
        for r in allrows:
            fh.write(json.dumps(r) + "\n")
    deep = [r["rel_resid"] for r in allrows if r["layer"] >= 8]
    print(f"\n  mean |distinct|/|t| over layers >= 8: {100*sum(deep)/len(deep):.2f}%")
    print(f"  max transport self-check: {max(r['transport_check'] for r in allrows):.2e}")


if __name__ == "__main__":
    main()
