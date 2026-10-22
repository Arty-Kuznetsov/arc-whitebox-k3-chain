"""Self-checks of src/estimator_k3e.py.
flopscope part (whest env):  python k3e_check.py fs    -> (1) k3e(closure=False, c0d=False) bitwise == k3d on a
    1024x16 MLP; (2) writes k3e outputs on a small MLP (n=128, depth 6, thin=128 = full rank) for three switch settings.
torch part (ARC env):        python k3e_check.py torch -> compares those outputs with chain.py (corr='freefull' = exact
    split, the full-rank equivalent of the thin leg) with the same switches."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SETTINGS = {"c0d": dict(closure=False, c0d=True), "closure": dict(closure=True, c0d=False), "both": dict(closure=True, c0d=True)}
TORCH = {"c0d": dict(c0d=True), "closure": dict(k4pat=True, k22pat=True, k4lvl=True),
         "both": dict(c0d=True, k4pat=True, k22pat=True, k4lvl=True)}
if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    import estimator_k3d as K3D, estimator_k3e as K3E
    def mlp(n, depth, seed):
        rng = np.random.default_rng(seed)
        Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
        class M:
            width = n
        M.depth, M.seed, M.weights = depth, seed, [fnp.asarray(W) for W in Ws]
        return M, np.stack(Ws)
    def run(est, m):
        with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            out = np.asarray(est.predict(m, 2**41), dtype=np.float64)
            return out, ctx.flops_used, ctx.residual_wall_time_s
    A, _ = mlp(1024, 16, 11)
    o0, f0, r0 = run(K3D.Estimator(), A)
    o1, f1, r1 = run(K3E.Estimator(closure=False, c0d=False), A)
    print(f"(1) k3e(closure=False, c0d=False) vs k3d, 1024x16: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1}")
    o2, f2, r2 = run(K3E.Estimator(), A)
    d = np.abs(o2 - o0)
    print(f"    k3e default: {100*f2/2**41:.3f}% of B (k3d {100*f0/2**41:.3f}%), residual {r2:.3f}s (k3d {r0:.3f}s); "
          f"final-layer max|k3e-k3d| {d[-1].max():.2e}, mean {o2[-1].mean():.4f}")
    S, W = mlp(128, 6, 3)
    np.save(HERE / "runs" / "k3e_small_W.npy", W)
    for name, kw in SETTINGS.items():
        out, _, _ = run(K3E.Estimator(thin=128, **kw), S)
        np.save(HERE / "runs" / f"k3e_small_{name}.npy", out)
    print("(2) small-MLP outputs written")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    from chain import run_chain
    W = np.load(HERE / "runs" / "k3e_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    base, _ = run_chain(Ws, k4="memo22", corr="freefull")
    for name in SETTINGS:
        out = np.load(HERE / "runs" / f"k3e_small_{name}.npy")
        ref, _ = run_chain(Ws, k4="memo22", corr="freefull", **TORCH[name])
        e = [np.abs(out[l] - ref[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        eff = [np.abs(ref[l].numpy() - base[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        print(f"(2) {name:8s}: max rel diff flopscope vs torch per layer " + " ".join(f"{x:.1e}" for x in e)
              + "   | size of the change itself " + " ".join(f"{x:.1e}" for x in eff))
