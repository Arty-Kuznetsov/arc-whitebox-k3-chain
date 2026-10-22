"""Proposition 3: the share of the next layer's kappa3 diagonal that (t, D) do not carry, derived.
The next diagonal is t'_i = sum_{abc} w_a w_b w_c G_abc with w = W_{l+1}[:, i] i.i.d. N(0, s^2), s^2 = 2/n, independent
of G = T o (d1 x d1 x d1) (the chain's kappa3 tensor with the gates of layer l on its legs). By Isserlis,
    E[t'_i^2] = s^6 (6 ||G||_F^2 + 9 ||v||^2),   v_a = sum_b G_abb,
and the all-distinct part of the contraction is uncorrelated with the rest, E[AD_i^2] = 6 s^6 ||G_AD||_F^2 (entries with
three different indices). Hence, for every neuron and so for the norm over neurons,
    share = ||AD|| / ||t'||  ~=  sqrt( 6 ||G_AD||^2 / (6 ||G||^2 + 9 ||v||^2) ).
This script measures the share with the network's own W_{l+1} (as ../2026-09-28_state-closure/closure.py does) and
compares it with the formula, at widths where the dense tensor fits; random He-init networks, the chain config of
closure.py (CLEAN).
usage: python prop3.py <n> <depth> <n nets>
    ARC=$(cd ../../external/mlp_cumulant_propagation && pwd); env PYTHONPATH="$ARC/src:../2026-09-24_k3-chain" \\
        nice -n 10 uv run --project "$ARC" python prop3.py 128 16 2
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain")); sys.path.insert(0, str(HERE.parent / "2026-09-28_state-closure"))
import chain                                                     # noqa: E402
from closure import CLEAN, two_slice_predict                    # noqa: E402


def energies(G):
    n = G.shape[0]
    tot = float((G * G).sum())
    idx = torch.arange(n)
    diag = G[idx, idx, idx]
    Gjjk = G[idx, idx, :]                                        # (j, j, k), k free
    two_eq = float((Gjjk * Gjjk).sum() - (diag * diag).sum())   # sum over j != k of G_jjk^2
    rep = float((diag * diag).sum()) + 3.0 * two_eq
    v = Gjjk.sum(0)                                              # v_k = sum_j G_jjk (G symmetric)
    return tot, rep, tot - rep, float(v @ v)


def run(n, depth, net):
    g = torch.Generator().manual_seed(9_000_000 + 1000 * n + net)
    Ws = [torch.randn(n, n, generator=g) * (2.0 / n) ** 0.5 for _ in range(depth)]
    cap = {"d1": {}, "src": {}, "call": 0}
    orig_read, orig_non = chain.read_slices, chain.nonlin_slices
    def wrapped_read(sources, nn, d21_ages=None, tailx=None):
        k = cap["call"]; cap["src"][k] = [dict(s) for s in sources]; cap["call"] += 1
        return orig_read(sources, nn, d21_ages, tailx)
    def wrapped_non(*a, **kw):
        o = orig_non(*a, **kw); cap["d1"][cap["call"] - 1] = o["d1"].clone(); return o
    chain.read_slices, chain.nonlin_slices = wrapped_read, wrapped_non
    try:
        rows_m, _ = chain.run_chain(Ws, **CLEAN)
    finally:
        chain.read_slices, chain.nonlin_slices = orig_read, orig_non
    rows = []
    for k in sorted(cap["src"]):
        if k + 1 not in cap["src"] or k not in cap["d1"]: continue
        src_l, src_n = cap["src"][k], cap["src"][k + 1]
        old = src_n[:len(src_l)]
        if not old or len(old) != len(src_l): continue
        layer = k + 1
        if layer + 1 >= len(Ws): continue
        d = cap["d1"][k]
        M = Ws[layer + 1].T * d[None, :]
        chk = float((M @ src_l[0]["B"] - old[0]["B"]).abs().max() / old[0]["B"].abs().max())
        if chk > 1e-10: raise SystemExit(f"transport self-check failed at layer {layer}: {chk:.2e}")
        t_l, D_l = orig_read(src_l, n); t_n, _ = orig_read(old, n)
        resid = t_n - two_slice_predict(M, t_l, D_l)
        measured = float(resid.norm() / t_n.norm())
        T = chain._k3_dense(src_l, n)
        tchk = float((torch.diagonal(torch.diagonal(T, 0, 0, 1), 0, 0, 1) - t_l).abs().max() / t_l.abs().max())
        G = T * d[:, None, None] * d[None, :, None] * d[None, None, :]
        tot, rep, ad, vv = energies(G)
        idx_ = torch.arange(n); v_vec = G[idx_, idx_, :].sum(0); m_post = rows_m[layer]
        cos_vm = float(v_vec @ m_post / (v_vec.norm() * m_post.norm()))
        pred = (6 * ad / (6 * tot + 9 * vv)) ** 0.5
        # the realized contraction from the dense tensor, as a check of the residual (all-distinct part)
        Wn = Ws[layer + 1]
        full = torch.einsum("abc,ai,bi,ci->i", G, Wn, Wn, Wn)
        rows.append(dict(n=n, net=net, layer=layer, S=len(src_l), measured=measured, predicted=pred,
                         ratio=measured / pred, cos_v_m=cos_vm, f_ad=ad / tot, q_v=9 * vv / (6 * tot), t_check=tchk,
                         full_check=float((full - t_n).norm() / t_n.norm())))
        r = rows[-1]
        print(f"n={n} net {net} layer {layer:2d} S={r['S']:2d}: measured {100*measured:6.2f}%  predicted {100*pred:6.2f}%  "
              f"ratio {r['ratio']:.3f} | cos(v, m) {r['cos_v_m']:+.3f} | AD energy {100*r['f_ad']:5.1f}% of ||G||^2, 9||v||^2/6||G||^2 = {r['q_v']:.3f} | "
              f"checks t {tchk:.1e}, contraction {r['full_check']:.1e}", flush=True)
    return rows


if __name__ == "__main__":
    n, depth, nets = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
    allrows = []
    for net in range(nets):
        t0 = time.time(); allrows += run(n, depth, net); print(f"  net {net} done in {time.time()-t0:.0f}s", flush=True)
    json.dump(allrows, open(HERE / "res" / f"prop3_n{n}_L{depth}{sys.argv[4] if len(sys.argv) > 4 else ''}.json", "w"), indent=1)
