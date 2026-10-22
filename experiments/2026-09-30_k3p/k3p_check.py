"""Self-checks of src/estimator_k3p.py (k3o + the transported band basis in coefficient form).

fs part (whest env):   python k3p_check.py fs     -> (1) k3p(band=None) == k3o bitwise on 1024x16 (+ FLOPs); (2) small-MLP
                                                     outputs (n=128, depth 8, thin=128) with a band, for the torch check
torch part (ARC env):  python k3p_check.py torch CHAIN_DIR -> (2) vs the cloud's chain.py `bandz` (branch
                                                     cloud/port-battery, fbc1bd5..edafaa4), same config, float32 band
The torch comparison uses init="cols" and colcut=None (the cloud's torch A is not masked by the column cut and its "rr"
probes come from another generator, so only this configuration defines the same first basis in both)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
K3M_T = dict(k4="memo22", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True,
             paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3,
             k31mult=0.65)
BANDS = [(3, 64), (4, 48), (3, 96)]
if sys.argv[1] == "fs":
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import flopscope as flops
    import flopscope.numpy as fnp
    flops.configure(symmetry_warnings=False)
    import estimator_k3o as O, estimator_k3p as P

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
        o0, f0, r0, w0 = run(O.Estimator(), A)
        o1, f1, r1, w1 = run(P.Estimator(band=None), A)
        print(f"(1) k3p(band=None) vs k3o, 1024x16: bitwise {np.array_equal(o0, o1)}, flops equal {f0 == f1} "
              f"({100 * f1 / 2 ** 41:.3f}% of B); residual {r1:.3f} s (k3o {r0:.3f})")
    S, W = mlp(128, 8, 3)
    np.save(HERE / "k3p_small_W.npy", W)
    out, _, _, _ = run(P.Estimator(thin=128, colcut=None, band=None), S)
    np.save(HERE / "k3p_small_none.npy", out)
    for a0, r in BANDS:
        out, _, _, _ = run(P.Estimator(thin=128, colcut=None, band=(a0, r, "cols")), S)
        np.save(HERE / f"k3p_small_{a0}_{r}.npy", out)
    print("(2) small-MLP outputs written")
else:
    import torch
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
    sys.path.insert(0, sys.argv[2])
    from chain import run_chain
    W = np.load(HERE / "k3p_small_W.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
    base, _ = run_chain(Ws, corr="split128", deadcut=-2.5, **K3M_T)
    out = np.load(HERE / "k3p_small_none.npy")
    e = [np.abs(out[l] - base[l].numpy()).max() / np.abs(base[l].numpy()).max() for l in range(W.shape[0])]
    print("no band: flopscope vs torch, max rel diff per layer " + " ".join(f"{x:.1e}" for x in e))
    for a0, r in BANDS:
        cfg = dict(a0=a0, r=r, k=None, dtype=torch.float32, proj="qr", entry="pre", init="cols")
        ref, _ = run_chain(Ws, corr="split128", deadcut=-2.5, bandz=cfg, **K3M_T)
        out = np.load(HERE / f"k3p_small_{a0}_{r}.npy")
        e = [np.abs(out[l] - ref[l].numpy()).max() / np.abs(ref[l].numpy()).max() for l in range(W.shape[0])]
        eff = np.abs(ref[-1].numpy() - base[-1].numpy()).max() / np.abs(base[-1].numpy()).max()
        print(f"band a0={a0} r={r}: flopscope vs torch per layer " + " ".join(f"{x:.1e}" for x in e)
              + f"   | the band's own effect (torch, final layer) {eff:.1e}")
