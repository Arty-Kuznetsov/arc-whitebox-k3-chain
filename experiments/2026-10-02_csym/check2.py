import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3x as X, estimator_csym as Y
from check import mlp, run
m = mlp(1024, 16, 11)
o0, f0, w0 = run(X.Estimator(), m)
for c in (1, 2):
    o2, f2, w2 = run(Y.Estimator(csym=c), m)
    rel = np.abs(o2 - o0).max() / np.abs(o0).max()
    print(f"csym={c}: FLOPs x{f2 / f0:.5f}  max rel move {rel:.2e}  wall {w0:.1f} -> {w2:.1f} s", flush=True)
