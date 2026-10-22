"""Self-checks of src/estimator_k3d.py against the frozen src/estimator_k3c.py on random width-1024 depth-16 MLPs.
(1) k3d(batch_base=False) must be BITWISE equal to k3c, fresh and with the workspace reused across MLPs A, B, A.
(2) k3d (batched base products) must differ from k3c only by float32 rounding (compare with the L0-vs-L3 spread)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
import estimator_k3c as K3C
import estimator_k3d as K3D

n, depth = 1024, 16
def make(seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]
    class M:
        width = n
    M.depth, M.seed, M.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return M

def run(est, mlp):
    with flops.BudgetContext(flop_budget=2**50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        dt = time.perf_counter() - t0
        return out, ctx.flops_used, dt, ctx.residual_wall_time_s

A, B = make(11), make(12)
ref = {}
for nm, mlp in (("A", A), ("B", B)):
    ref[nm] = run(K3C.Estimator(strassen_levels=3), mlp)
    print(f"k3c L3 {nm}: {100*ref[nm][1]/2**41:.3f}% of B, wall {ref[nm][2]:.1f}s, residual {ref[nm][3]:.3f}s")
refL0 = run(K3C.Estimator(strassen_levels=0), A)

print("(1) k3d(batch_base=False), one instance, MLPs A, B, A:")
est = K3D.Estimator(strassen_levels=3, batch_base=False)
for nm, mlp in (("A", A), ("B", B), ("A", A)):
    out, fl, dt, res = run(est, mlp)
    same = np.array_equal(out, ref[nm][0]) and fl == ref[nm][1]
    print(f"   {nm}: bitwise equal to k3c: {same}  (max|diff| {np.abs(out-ref[nm][0]).max():.1e}, flops equal {fl == ref[nm][1]}), residual {res:.3f}s")

print("(2) k3d (batched base products), one instance, MLPs A, B, A:")
est = K3D.Estimator(strassen_levels=3)
outs = []
for nm, mlp in (("A", A), ("B", B), ("A", A)):
    out, fl, dt, res = run(est, mlp)
    outs.append(out)
    d = np.abs(out - ref[nm][0])
    print(f"   {nm}: {100*fl/2**41:.3f}% of B (k3c {100*ref[nm][1]/2**41:.3f}%), wall {dt:.1f}s, residual {res:.3f}s; "
          f"vs k3c L3: final max|diff| {d[-1].max():.1e} rms {np.sqrt((d[-1]**2).mean()):.1e}, all-layer max {d.max():.1e}")
print(f"   A run twice on one instance bitwise equal: {np.array_equal(outs[0], outs[2])}")
d0 = np.abs(refL0[0] - ref["A"][0])
print(f"   reference spread k3c L0 vs L3 on A: final max|diff| {d0[-1].max():.1e} rms {np.sqrt((d0[-1]**2).mean()):.1e}, all-layer max {d0.max():.1e}")
dz = np.abs(outs[0] - refL0[0])
print(f"   k3d vs k3c L0 on A: final max|diff| {dz[-1].max():.1e} rms {np.sqrt((dz[-1]**2).mean()):.1e}")
