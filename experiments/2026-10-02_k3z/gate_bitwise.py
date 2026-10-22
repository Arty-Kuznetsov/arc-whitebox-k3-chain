"""Bitwise gate with both estimators WARM (the first predict of an instance also bills the workspace construction's
reshapes, ~0.4%): winograd=False, csym=1 must equal k3y in outputs and FLOPs."""
import numpy as np
from gate import mlp, run, Y, Z
for (n, d, kw) in ((1024, 16, {}), (128, 8, dict(thin=128, band=None, transport_l5=False))):
    m = mlp(n, d, 11)
    o0, f0, _ = run(Y.Estimator(**kw), m, 2)
    o1, f1, _ = run(Z.Estimator(winograd=False, csym=1, **kw), m, 2)
    print(f"{n}x{d} warm: bitwise {np.array_equal(o0, o1)}  flops equal {f0 == f1}  ({f0:.6e} / {f1:.6e})")
