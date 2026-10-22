"""Self-checks of src/estimator_k3m.py (k3l + deadcut -2.5 + optional low-rank path leg).

fs part (whest env):   python k3m_check.py fs [small]
  (1) k3m(deadcut=-3.0, pathrank=None) must be BITWISE equal to k3l on 1024x16, with the same FLOP count
      -- the standard of every port in this project (k3i_check.py, k3j_check.py).
  (2) k3m default (deadcut=-2.5): FLOPs / util / residual, against the Mac's measured 24.894%.
  (3) machinery of the low-rank path leg: pathrank = n reconstructs Y exactly, so it must reproduce
      pathrank=None to float32 rounding through a completely different product path.
  (4) the decisive cost measurement: util of pathrank in {None, 384, 256, 192, 128, 64} x {svd, rangefinder}.
      'small' writes the n=128 outputs the torch part reads.

torch part (torch env): python k3m_check.py torch
  the n=128, depth 8, thin=128 (corr='split128') cross-check against chain.py, for pathrank None and 64.
"""
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
# the k3l/k3m term set in chain.py's language (corr is set per call: split24rr in production, split128 for the
# small cross-check, where the correction leg is exact and cannot hide a mismatch behind the range finder)
K3L_T = dict(k4="memo22", c0d=True, k4pat=True, k22pat=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2",
             paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",),
             tet=3, k31mult=0.65)
B = 2 ** 41

if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    import estimator_k3l as Lm, estimator_k3m as M

    def mlp(n, depth, seed):
        rng = np.random.default_rng(seed)
        Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
        class Mm:
            width = n
        Mm.depth, Mm.seed, Mm.weights = depth, seed, [fnp.asarray(W) for W in Ws]
        return Mm, np.stack(Ws)

    def run(est, m):
        with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            t0 = time.perf_counter()
            out = np.asarray(est.predict(m, B), dtype=np.float64)
            return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0

    if len(sys.argv) < 3 or sys.argv[2] != "small":
        A, _ = mlp(1024, 16, 11)
        o0, f0, r0, w0 = run(Lm.Estimator(), A)
        o1, f1, r1, w1 = run(M.Estimator(deadcut=-3.0), A)
        print(f"(1) k3m(deadcut=-3.0, pathrank=None) vs k3l, 1024x16: BITWISE {np.array_equal(o0, o1)}, "
              f"flops equal {f0 == f1} ({100*f0/B:.3f}% of B), residual {r1:.3f}s (k3l {r0:.3f}s)")
        o2, f2, r2, w2 = run(M.Estimator(), A)
        d = np.abs(o2 - o0)
        print(f"(2) k3m default (deadcut=-2.5): {100*f2/B:.3f}% of B (k3l {100*f0/B:.3f}%), residual {r2:.3f}s, "
              f"wall {w2:.1f}s (k3l {w0:.1f}s); final-layer max|k3m-k3l| {d[-1].max():.2e} "
              f"rel {d[-1].max()/np.abs(o0[-1]).max():.1e}")
        o3, f3, r3, w3 = run(M.Estimator(deadcut=-3.0, pathrank=1024, pathsvd=True), A)
        d3 = np.abs(o3 - o0)
        print(f"(3) pathrank = n (exact reconstruction, factored transport) vs k3l: final-layer rel "
              f"{d3[-1].max()/np.abs(o0[-1]).max():.1e}, all-layer rel {(d3/np.abs(o0).max()).max():.1e} "
              f"-- the machinery is exact; {100*f3/B:.3f}% of B")
        print("(4) cost of the low-rank path leg on the real meter (deadcut=-2.5, 1024x16):")
        rows = [("None", None, True)]
        for r in (384, 256, 192, 128, 64):
            rows.append((f"{r} svd", r, True))
            rows.append((f"{r} rangefinder", r, False))
        base = None
        recs = []
        for tag, r, sv in rows:
            o, f, rw, wl = run(M.Estimator(pathrank=r, pathsvd=sv), A)
            if base is None:
                base = f
            dd = np.abs(o - o2)
            print(f"    pathrank {tag:16s} {100*f/B:7.3f}% of B  ({f/base:6.4f} x pathrank=None)  "
                  f"residual {rw:.3f}s  wall {wl:6.1f}s  final rel diff {dd[-1].max()/np.abs(o2[-1]).max():.2e}")
            recs.append(dict(tag=tag, pathrank=r, pathsvd=sv, flops=int(f), util=f / B, ratio=f / base,
                             residual=rw, wall=wl))
        (HERE / "logs" / "cost_pathrank.json").write_text(json.dumps(recs, indent=1))
    S, W = mlp(128, 8, 3)
    np.save(HERE / "k3m_small_W.npy", W)
    for name, kw in (("none", dict()), ("p64", dict(pathrank=64, pathsvd=True))):
        out, _, _, _ = run(M.Estimator(thin=128, **kw), S)
        np.save(HERE / f"k3m_small_{name}.npy", out)
    print("(5) small-MLP outputs written (n=128, depth 8, thin=128, deadcut=-2.5)")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
    from chain import run_chain
    W = np.load(HERE / "k3m_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    for name, pr in (("none", None), ("p64", 64)):
        R, _ = run_chain(Ws, corr="split128", deadcut=-2.5, pathrank=pr, **K3L_T)
        out = np.load(HERE / f"k3m_small_{name}.npy")
        e = [np.abs(out[l] - R[l].numpy()).max() / np.abs(R[l].numpy()).max() for l in range(W.shape[0])]
        print(f"(6) pathrank={str(pr):>4s}: max rel diff flopscope vs torch per layer "
              + " ".join(f"{x:.1e}" for x in e))
    Rn, _ = run_chain(Ws, corr="split128", deadcut=-2.5, pathrank=None, **K3L_T)
    R64, _ = run_chain(Ws, corr="split128", deadcut=-2.5, pathrank=64, **K3L_T)
    eff = [np.abs(R64[l].numpy() - Rn[l].numpy()).max() / np.abs(R64[l].numpy()).max() for l in range(W.shape[0])]
    print("    size of the rank-64 truncation itself (torch p64 vs none): " + " ".join(f"{x:.1e}" for x in eff))
