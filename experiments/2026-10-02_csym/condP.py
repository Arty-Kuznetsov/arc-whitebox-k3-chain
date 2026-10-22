"""How ill-conditioned is the band basis P at its entries? (decides whether the R-only QR + semi-normal equations,
which square the condition number, could ever be safe in float32). Logs singular values of every (n x r) QR input."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3y as Y
from check import mlp, run
log = []
_qr = fnp.linalg.qr
def qr_log(a, *args, **kw):
    A = np.asarray(a, dtype=np.float64)
    if A.shape[1] >= 256:
        s = np.linalg.svd(A, compute_uv=False)
        log.append((A.shape, s[0] / s[-1]))
    return _qr(a, *args, **kw)
fnp.linalg.qr = qr_log
for seed in (11, 12):
    log.clear()
    run(Y.Estimator(), mlp(1024, 16, seed))
    print(f"seed {seed}: " + "  ".join(f"{c:.2e}" for _, c in log), flush=True)
