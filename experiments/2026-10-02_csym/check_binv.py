"""binv prototype (explicit R^-1 at the band entry) vs k3y: bitwise when off, FLOPs and max rel move when on."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3y as Y, estimator_binv as B
from check import mlp, run
m = mlp(1024, 16, 11)
o0, f0, w0 = run(Y.Estimator(), m)
o1, f1, w1 = run(B.Estimator(), m)
print(f"binv off == k3y: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1}", flush=True)
o2, f2, w2 = run(B.Estimator(binv=True), m)
print(f"binv on: FLOPs x{f2 / f0:.5f}  util {f2 / 2**41:.4%}  max rel move {np.abs(o2 - o0).max() / np.abs(o0).max():.2e}  "
      f"finite {np.isfinite(o2).all()}  wall {w0:.1f} -> {w2:.1f} s", flush=True)
