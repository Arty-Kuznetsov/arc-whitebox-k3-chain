"""Self-checks of src/estimator_k3q.py (k3p + E2: saturated-row cut of D with the kappa4 level restored, "sep").

fs part (whest env):  python k3q_check.py fs [small]  -> (1) k3q(satrow=None) == k3p bitwise (1024x16 unless small);
                                                       (2) small-MLP outputs (n=128, depth 8, thin=128, no band)
torch part (ARC env): python k3q_check.py torch CHAIN_DIR -> (2) vs the cloud's chain.py (satrow=theta, satrow_k4="sep")
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
K3M_T = dict(k4="memo22", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True,
             paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3,
             k31mult=0.65)
CASES = [(-2.5, 1.5), (-2.5, 1.0), (-1.5, 1.0)]   # (matched dead cut, saturation threshold)
if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    flops.configure(symmetry_warnings=False)
    import estimator_k3p as P, estimator_k3q as Q

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
        o0, f0, r0, w0 = run(P.Estimator(), A)
        o1, f1, r1, w1 = run(Q.Estimator(satrow=None), A)
        print(f"(1) k3q(satrow=None) vs k3p, 1024x16: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
              f"({100 * f1 / 2 ** 41:.3f}% of B); residual {r1:.3f} s (k3p {r0:.3f})")
    S, W = mlp(128, 8, 3)
    np.save(HERE / "k3q_small_W.npy", W)
    o0, f0, _, _ = run(P.Estimator(thin=128, band=None), S)
    o1, f1, _, _ = run(Q.Estimator(thin=128, band=None, satrow=None), S)
    print(f"(1s) k3q(satrow=None) vs k3p, 128x8: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1}")
    for t, th in CASES:
        out, f, _, _ = run(Q.Estimator(thin=128, band=None, deadcut=t, colcut=t, satrow=th), S)
        np.save(HERE / f"k3q_small_{t}_{th}.npy", out)
        outr, fr, _, _ = run(Q.Estimator(thin=128, band=None, deadcut=t, colcut=t, satrow=None), S)
        print(f"  t={t} satrow={th}: flops x{f / fr:.4f} of no-satrow, finite {np.isfinite(out).all()}")
    print("(2) small-MLP outputs written")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    sys.path.insert(0, sys.argv[2])
    from chain import run_chain
    W = np.load(HERE / "k3q_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    for t, th in CASES:
        ref, info = run_chain(Ws, corr="split128", deadcut=t, colcut=(t, 1e9), satrow=th, satrow_k4="sep", record=True,
                              **K3M_T)
        base, _ = run_chain(Ws, corr="split128", deadcut=t, colcut=(t, 1e9), **K3M_T)
        naive, _ = run_chain(Ws, corr="split128", deadcut=t, colcut=(t, 1e9), satrow=th, **K3M_T)
        nsat = [int(((r["mu_pre"] / r["var"].sqrt()) > th).sum()) for r in info]
        out = np.load(HERE / f"k3q_small_{t}_{th}.npy")
        e = [np.abs(out[l] - ref[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        eff = np.abs(ref[-1].numpy() - base[-1].numpy()).max() / np.abs(base[-1].numpy()).max()
        effk = np.abs(ref[-1].numpy() - naive[-1].numpy()).max() / np.abs(naive[-1].numpy()).max()
        print(f"t={t} satrow={th}: saturated per layer {nsat}")
        print(f"   flopscope vs torch (sep), max rel diff per layer " + " ".join(f"{x:.1e}" for x in e))
        print(f"   effect of the cut + restore (torch, final layer) {eff:.1e};  of the kappa4 restoration alone {effk:.1e}")
