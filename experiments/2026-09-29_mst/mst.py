"""Slice-Markov tail (MST): young sources exact, old ones carried as (tau, Delta) by the repeated-index rule.

NEGATIVE 62 closed the case where ALL sources are replaced by the two-slice rule (x25.6).  It left one door
open: a LARGER state.  MST is that door — sources younger than a0 keep their legs and stay exact; at age a0 a
source is folded into a tail state (tau, Delta) with its exact slices, and from then on the tail moves forward
by the exact repeated-index rule only (the t-term and the D-term of closure.py section 1).  The only thing
lost is the OLD tail's all-distinct part.  62 measured 28.8% for ALL sources together; the old tail's own
share has never been measured, and 62's own layer profile (35-61% when everything is young, 28.8% deep)
suggests fresh content is the slice-poor part.

Two anchoring self-checks, both exact:
  a0 = 1   -> every source is in the tail, only births exact == tdchain.py's `td` mode (x25.6 on CLEAN)
  a0 = 99  -> nothing ever graduates == the base
plus the transport self-check of closure.py (M reproduces the chain's own leg transport to ~1e-16).

    ../../.venv-torch/bin/python mst.py diag --a0 2 3 4 5 --mlps 3
    ../../.venv-torch/bin/python mst.py e2e  --a0 99 5 4 3 2 --mlps 10
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE.parent / "2026-09-28_state-closure"))
from closure import CLEAN, two_slice_predict
from tdchain import two_slice_predict_D, selfcheck

# MST base = the production k3l_t term set with deadcut off, so the gate d1 is exactly nonlin_slices' d1
# (with deadcut the chain zeroes d1 on dead rows inside run_chain and the reconstruction of M would need the
# mask; deadcut is worth x1.002 on accuracy, so nothing structural rides on it).
PROD = dict(CLEAN); PROD.update(paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge")
CFGS = {"clean": CLEAN, "prod": PROD}


def _hooks(chain, Ws, on_read):
    st = {"k": 0, "d1": None, "prevB": None}
    orig_read, orig_non = chain.read_slices, chain.nonlin_slices

    def wrapped_read(sources, nn, d21=None, tailx=None):
        out = on_read(st, sources, nn, orig_read, Ws)
        st["k"] += 1
        st["prevB"] = sources[0]["B"].clone() if sources else None
        return out

    def wrapped_non(*a, **kw):
        o = orig_non(*a, **kw)
        st["d1"] = o["d1"].clone()
        return o
    return st, orig_read, orig_non, wrapped_read, wrapped_non


def _M(Ws, st, layer, sources, orig_read):
    """M = W_layer^T diag(d1_{layer-1}), with the self-check that it IS the chain's own leg transport."""
    M = Ws[layer].T * st["d1"][None, :]
    if st["prevB"] is not None and sources:
        e = float((M @ st["prevB"] - sources[0]["B"]).abs().max() / sources[0]["B"].abs().max())
        if e > 1e-10:
            raise SystemExit(f"TRANSPORT SELF-CHECK FAILED at layer {layer}: {e:.2e}")
    return M


def run_diag(mlp_idx, a0s, cfg, baked, depth=16):
    """Measurement 1: the all-distinct share of the OLD TAIL's own transported contribution."""
    import chain
    W = np.load(baked / f"weights_{mlp_idx}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    n = Ws[0].shape[0]
    rows, keep = [], {}

    def on_read(st, sources, nn, orig_read, Ws):
        k, layer = st["k"], st["k"] + 1
        keep[k] = [dict(s) for s in sources]
        if k > 0 and layer < len(Ws):
            M = _M(Ws, st, layer, sources, orig_read)
            prev = keep[k - 1]
            Sp = len(prev)
            for a0 in a0s:
                n_old = max(0, Sp - a0 + 1)          # sources of age >= a0 at the previous layer
                if n_old == 0:
                    continue
                old_p, old_n = prev[:n_old], sources[:n_old]
                t_p, D_p = orig_read(old_p, nn)
                t_n, _ = orig_read(old_n, nn)
                pred = two_slice_predict(M, t_p, D_p)
                rows.append(dict(mlp=mlp_idx, layer=layer, a0=a0, n_old=n_old,
                                 f=float((t_n - pred).norm() / t_n.norm()),
                                 share=float(t_n.norm())))
        return orig_read(sources, nn)

    st, orig_read, orig_non, wr, wn = _hooks(chain, Ws, on_read)
    chain.read_slices, chain.nonlin_slices = wr, wn
    try:
        chain.run_chain(Ws, **cfg)
    finally:
        chain.read_slices, chain.nonlin_slices = orig_read, orig_non
    return rows


def run_e2e(mlp_idx, a0, cfg, baked, depth=16):
    """Measurement 2: the MST chain itself.  a0=None -> base."""
    import chain
    W = np.load(baked / f"weights_{mlp_idx}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    n = Ws[0].shape[0]
    tail = {"tau": None, "Del": None}

    def on_read(st, sources, nn, orig_read, Ws):
        if a0 is None:
            return orig_read(sources, nn)
        k, layer = st["k"], st["k"] + 1
        S = len(sources)
        n_old = max(0, S - a0 + 1)                    # age >= a0 : indices 0 .. S-a0
        if tail["tau"] is None:
            tau = torch.zeros(nn, dtype=torch.float64); Dl = torch.zeros(nn, nn, dtype=torch.float64)
        else:                                          # move the tail by the repeated-index rule only
            M = _M(Ws, st, layer, sources, orig_read)
            tau = two_slice_predict(M, tail["tau"], tail["Del"])
            Dl = two_slice_predict_D(M, tail["tau"], tail["Del"])
        if S >= a0:                                    # the source that reaches age a0 here joins, exactly
            tg, Dg = orig_read([sources[S - a0]], nn)
            tau, Dl = tau + tg, Dl + Dg
        # the tail's Delta must carry a ZERO diagonal: the diagonal of the transported (2,1) slice is the
        # transported diagonal, which tau already holds, and the repeated-index rule assumes D_jj = 0 (else
        # the j=k term is counted twice).  Keeping it non-zero is what made a0=1 read x28.06 against
        # tdchain's x27.16 -- caught by the anchoring self-check.
        Dl = Dl.clone(); Dl.fill_diagonal_(0.0)
        tail["tau"], tail["Del"] = tau, Dl
        young = sources[n_old:]
        ty, Dy = orig_read(young, nn) if young else (torch.zeros(nn, dtype=torch.float64),
                                                     torch.zeros(nn, nn, dtype=torch.float64))
        t, D = ty + tau, Dy + Dl
        D = D.clone(); D.fill_diagonal_(0.0)
        return t, D

    st, orig_read, orig_non, wr, wn = _hooks(chain, Ws, on_read)
    chain.read_slices, chain.nonlin_slices = wr, wn
    try:
        mu, _ = chain.run_chain(Ws, **cfg)
    finally:
        chain.read_slices, chain.nonlin_slices = orig_read, orig_non
    return mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["diag", "e2e"])
    ap.add_argument("--a0", type=int, nargs="+", default=[2, 3, 4, 5])
    ap.add_argument("--mlps", type=int, default=10)
    ap.add_argument("--cfg", default="prod", choices=list(CFGS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    cfg, baked = CFGS[a.cfg], HERE.parent / "2026-09-16_kprop-arc" / "baked"
    out = HERE / (a.out or f"logs/mst_{a.what}_{a.cfg}.jsonl")
    if a.what == "diag":
        selfcheck()
        rows = []
        for i in range(a.mlps):
            rows += run_diag(i, a.a0, cfg, baked)
        with open(out, "a") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"\nall-distinct share of the OLD TAIL's own contribution ({a.cfg}, {a.mlps} MLPs)")
        print("  (62 measured 28.8% for ALL sources together)")
        for a0 in a.a0:
            for lo, hi, nm in ((1, 7, "layers 1-7"), (8, 99, "layers >=8")):
                v = [r["f"] for r in rows if r["a0"] == a0 and lo <= r["layer"] <= hi]
                if v:
                    print(f"  a0={a0}  {nm:12s}  f = {100*sum(v)/len(v):6.2f}%  "
                          f"(min {100*min(v):5.2f}%, max {100*max(v):5.2f}%, n={len(v)})")
    else:
        gt = np.load(baked / "gt.npy")
        base = None
        for a0 in ([None] + a.a0 if None not in a.a0 else a.a0):
            mses, t0 = [], time.time()
            for i in range(a.mlps):
                mu = run_e2e(i, a0, cfg, baked)
                mses.append(float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean()))
            raw = float(np.mean(mses))
            if a0 is None or base is None:
                base = base or raw
            tag = "base" if a0 is None else f"a0={a0}"
            print(f"  {tag:8s} raw {raw:.4e}   x{raw/base:8.4f}   ({time.time()-t0:.0f}s)", flush=True)
            with open(out, "a") as fh:
                fh.write(json.dumps(dict(cfg=a.cfg, a0=a0, n=a.mlps, raw=raw, x=raw / base,
                                         per_mlp=mses)) + "\n")


if __name__ == "__main__":
    main()
