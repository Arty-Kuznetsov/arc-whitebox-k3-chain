"""Self-checks of src/estimator_k3k.py (k3j + shared subspace band).
fs part (whest env):    python k3k_check.py fs [small] -> (1) k3k(band_age=None) bitwise == k3j on 1024x16; k3k default:
    FLOPs, residual, final diff; (2) small-MLP outputs (n=128, depth 10, thin=128, band 4:64, over 16).
torch part (ARC env):   python k3k_check.py torch -> vs chain.py ssp=(4, 64, 1, 16) on the k3j configuration."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
K3J_T = dict(k4="memo22", c0d=True, k4pat=True, k22pat=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2",
             paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3,
             deadcut=-3.0)
SMALL = dict(thin=128, band_age=4, band_rank=64, band_over=16)
if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    import estimator_k3j as J, estimator_k3k as K
    def mlp(n, depth, seed):
        rng = np.random.default_rng(seed)
        Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
        class M:
            width = n
        M.depth, M.seed, M.weights = depth, seed, [fnp.asarray(W) for W in Ws]
        return M, np.stack(Ws)
    def run(est, m):
        with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            t0 = time.perf_counter(); out = np.asarray(est.predict(m, 2**41), dtype=np.float64)
            return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0
    if len(sys.argv) < 3 or sys.argv[2] != "small":
        A, _ = mlp(1024, 16, 11)
        o0, f0, r0, w0 = run(J.Estimator(), A)
        o1, f1, r1, w1 = run(K.Estimator(band_age=None), A)
        print(f"(1) k3k(band_age=None) vs k3j, 1024x16: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
              f"({100*f0/2**41:.3f}%), residual {r1:.3f}s (k3j {r0:.3f}s)")
        o2, f2, r2, w2 = run(K.Estimator(), A)
        d = np.abs(o2 - o0)
        print(f"    k3k default: {100*f2/2**41:.3f}% of B (k3j {100*f0/2**41:.3f}%), residual {r2:.3f}s, wall {w2:.1f}s "
              f"(k3j {w0:.1f}s); final-layer max|k3k-k3j| {d[-1].max():.2e}, rel {d[-1].max()/np.abs(o0[-1]).max():.1e}")
    S, W = mlp(128, 10, 5)
    np.save(HERE / "k3k_small_W.npy", W)
    for name, kw in (("band", SMALL), ("off", dict(thin=128, band_age=None))):
        out, _, _, _ = run(K.Estimator(**kw), S)
        np.save(HERE / f"k3k_small_{name}.npy", out)
    print("(2) small-MLP outputs written")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
    from chain import run_chain
    W = np.load(HERE / "k3k_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    base, _ = run_chain(Ws, corr="split128", **K3J_T)
    ref, _ = run_chain(Ws, corr="split128", ssp=(4, 64, 1, 16), **K3J_T)
    for name, R in (("off", base), ("band", ref)):
        out = np.load(HERE / f"k3k_small_{name}.npy")
        e = [np.abs(out[l] - R[l].numpy()).max() / np.abs(R[l].numpy()).max() for l in range(W.shape[0])]
        print(f"(2) {name:4s}: max rel diff flopscope vs torch per layer " + " ".join(f"{x:.1e}" for x in e))
    eff = [np.abs(ref[l].numpy() - base[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
    print("    size of the band approximation itself (torch): " + " ".join(f"{x:.1e}" for x in eff))
