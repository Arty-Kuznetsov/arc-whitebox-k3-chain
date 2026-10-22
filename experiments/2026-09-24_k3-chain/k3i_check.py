"""Self-checks of src/estimator_k3i.py.
flopscope part (whest env):  python k3i_check.py fs    -> (1) k3i(paths=False, star=False, c0k22=False) bitwise == k3h
    on a 1024x16 MLP; k3i default: FLOPs, residual; (2) small-MLP outputs (n=128, depth 6, thin=128 = full rank).
torch part (ARC env):        python k3i_check.py torch -> those outputs vs chain.py (corr='split128' = exact at n=128)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
OFF = dict(paths=False, star=False, c0k22=False)
SETTINGS = {"paths": dict(OFF, paths=True), "star": dict(OFF, star=True), "c0k22": dict(OFF, c0k22=True), "all": dict()}
K3F_T = dict(k4="memo22", c0d=True, k4pat=True, k22pat=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2")
P_T = dict(paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge")
S_T = dict(k4b=1, k4bparts=("sd",))
C_T = dict(pairx=("k22",), tet=3)
TORCH = {"paths": dict(K3F_T, **P_T), "star": dict(K3F_T, **S_T), "c0k22": dict(K3F_T, **C_T), "all": dict(K3F_T, **P_T, **S_T, **C_T)}
if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    import estimator_k3h as H, estimator_k3i as I
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
        o0, f0, r0, w0 = run(H.Estimator(), A)
        o1, f1, r1, w1 = run(I.Estimator(**OFF), A)
        print(f"(1) k3i(all off) vs k3h, 1024x16: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
              f"({100*f0/2**41:.3f}%), residual {r1:.3f}s (k3h {r0:.3f}s)")
        o2, f2, r2, w2 = run(I.Estimator(), A)
        d = np.abs(o2 - o0)
        print(f"    k3i default: {100*f2/2**41:.3f}% of B (k3h {100*f0/2**41:.3f}%), residual {r2:.3f}s, wall {w2:.1f}s "
              f"(k3h {w0:.1f}s); final-layer max|k3i-k3h| {d[-1].max():.2e}")
    S, W = mlp(128, 6, 3)
    np.save(HERE / "runs" / "k3i_small_W.npy", W)
    for name, kw in SETTINGS.items():
        out, _, _, _ = run(I.Estimator(thin=128, **kw), S)
        np.save(HERE / "runs" / f"k3i_small_{name}.npy", out)
    out, _, _, _ = run(I.Estimator(thin=128, **OFF), S)
    np.save(HERE / "runs" / "k3i_small_off.npy", out)
    print("(2) small-MLP outputs written")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    from chain import run_chain
    W = np.load(HERE / "runs" / "k3i_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    base, _ = run_chain(Ws, corr="split128", **K3F_T)
    off = np.load(HERE / "runs" / "k3i_small_off.npy")
    e0 = [np.abs(off[l] - base[l].numpy()).max() / np.abs(base[l].numpy()).max() for l in range(W.shape[0])]
    print("(2) off     : max rel diff flopscope vs torch per layer " + " ".join(f"{x:.1e}" for x in e0))
    for name in SETTINGS:
        out = np.load(HERE / "runs" / f"k3i_small_{name}.npy")
        ref, _ = run_chain(Ws, corr="split128", **TORCH[name])
        e = [np.abs(out[l] - ref[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        eff = [np.abs(ref[l].numpy() - base[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        print(f"(2) {name:8s}: max rel diff flopscope vs torch per layer " + " ".join(f"{x:.1e}" for x in e)
              + "   | size of the change itself " + " ".join(f"{x:.1e}" for x in eff))
