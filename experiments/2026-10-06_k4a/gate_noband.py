"""Diagnostic: G1 with band=None (no members) -- if the restructured D equals k3z's to float rounding here, the 8e-4
move of the full configuration is the band's projection on the transition members (dense snapshot vs P Z)."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3z as Zm, estimator_k4a as A
from gate import mlp, run
m = mlp(1024, 16, 11)
o0, f0, w0 = run(Zm.Estimator(band=None), m)
o1, f1, w1 = run(A.Estimator(band=None, young=None), m)
rel = np.abs(o1 - o0).max() / np.abs(o0).max()
print(f"band=None: restructure only vs k3z: FLOPs x{f1 / f0:.5f}  max rel move {rel:.2e}  per-layer max rel: " +
      " ".join(f"{np.abs(o1[i]-o0[i]).max()/np.abs(o0[i]).max():.1e}" for i in range(o0.shape[0])), flush=True)
