"""Self-checks of src/estimator_k3o.py (k3m + the matched dead-side column cut, transposed wide leg layout).

fs part (whest env):     python k3o_check.py fs      -> (1) k3o(colcut=None) vs k3m on 1024x16: agreement to float32
                                                        rounding + FLOPs; (2) small-MLP outputs (n=128, depth 8,
                                                        thin=128 = exact correction legs) for the torch comparison
torch part (ARC env):    python k3o_check.py torch   -> (2) vs chain.py with the k3m config, corr='split128',
                                                        deadcut=t, colcut=(t, inf) at t in {-2.5, -1.5, -1.0}
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
K3M_T = dict(k4="memo22", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True,
             paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3,
             k31mult=0.65)
TS = (-2.5, -1.5, -1.0)
if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    flops.configure(symmetry_warnings=False)
    import estimator_k3m as M, estimator_k3o as O

    def mlp(n, depth, seed):
        rng = np.random.default_rng(seed)
        Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
        class Mo:
            width = n
        Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
        return Mo, np.stack(Ws)

    def run(est, m):
        with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            t0 = time.perf_counter()
            out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
            return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0

    if "small" not in sys.argv:
        A, _ = mlp(1024, 16, 11)
        o0, f0, r0, w0 = run(M.Estimator(), A)
        o1, f1, r1, w1 = run(O.Estimator(colcut=None), A)
        d = np.abs(o1 - o0).max(axis=1) / np.abs(o0).max(axis=1)
        print(f"(1) k3o(colcut=None) vs k3m, 1024x16: max rel diff per layer {d.max():.1e} (layer 15 {d[-1]:.1e}); "
              f"flops ratio {f1 / f0:.5f} ({100 * f1 / 2 ** 41:.3f}% vs {100 * f0 / 2 ** 41:.3f}% of B); "
              f"residual {r1:.3f} s (k3m {r0:.3f}), wall {w1:.1f} s (k3m {w0:.1f})")
    S, W = mlp(128, 8, 3)
    np.save(HERE / "k3o_small_W.npy", W)
    for t in TS:
        out, _, _, _ = run(O.Estimator(thin=128, deadcut=t, colcut=t), S)
        np.save(HERE / f"k3o_small_{t}.npy", out)
        out, _, _, _ = run(O.Estimator(thin=128, deadcut=t, colcut=None), S)
        np.save(HERE / f"k3o_small_{t}_rowonly.npy", out)
    print("(2) small-MLP outputs written")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
    from chain import run_chain
    W = np.load(HERE / "k3o_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    for t in TS:
        ref, info = run_chain(Ws, corr="split128", deadcut=t, colcut=(t, 1e9), record=True, **K3M_T)
        row, _ = run_chain(Ws, corr="split128", deadcut=t, **K3M_T)
        cut = [int(((r["mu_pre"] / r["var"].sqrt()) < t).sum()) for r in info]
        out = np.load(HERE / f"k3o_small_{t}.npy")
        outr = np.load(HERE / f"k3o_small_{t}_rowonly.npy")
        e = [np.abs(out[l] - ref[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        er = [np.abs(outr[l] - row[l].numpy()).max() / np.abs(row[l].numpy()).max() for l in range(W.shape[0])]
        eff = np.abs(ref[-1].numpy() - row[-1].numpy()).max() / np.abs(ref[-1].numpy()).max()
        print(f"t={t}: dead/cut neurons per layer {cut}")
        print(f"   matched cut, flopscope vs torch, max rel diff per layer: " + " ".join(f"{x:.1e}" for x in e))
        print(f"   row cut only, flopscope vs torch:                      " + " ".join(f"{x:.1e}" for x in er))
        print(f"   size of the column cut's effect (torch, final layer):   {eff:.1e}")
