"""Meter (flopscope, synthetic 1024x16 as experiments/2026-10-02_costmap): the FLOPs the young-kappa4 3+2 reader adds to
k3z through the restructured D pairing -- one D = Dx V product per age at every layer with a D pairing, and the old
sources' pairing at the last layer (where k3z computes none) -- inserted on zero buffers of the exact shapes."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3z as Z
import estimator_k4meter as M
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mo:
    width = 1024
Mo.depth, Mo.seed, Mo.weights = 16, 11, [fnp.asarray(W) for W in Ws]
def run(est):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        return np.asarray(est.predict(Mo, 2 ** 41), dtype=np.float64), ctx.flops_used
o0, f0 = run(Z.Estimator())
print(f"k3z: {f0:.6e} FLOPs", flush=True)
mode = sys.argv[1] if len(sys.argv) > 1 else "full"
cases = {"last": ((1, True, False), (2, True, False)), "nolast": ((1, False, True), (2, False, True)),
         "full": ((0, False, False), (1, False, False), (2, False, False))}[mode]
for a, lo, nl in cases:
    e = M.Estimator(meter_ages=a); e.meter_last_only = lo; e.meter_no_last = nl
    o1, f1 = run(e)
    print(f"meter_ages={a} last_only={lo} no_last={nl}: {f1:.6e}  x{f1 / f0:.5f}  (+{100 * (f1 / f0 - 1):.2f}%)  outputs equal {np.array_equal(o0, o1)}", flush=True)
