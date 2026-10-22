"""Exception hunt at full width over depths and budgets (the smoke test uses a shape other than 16 x 1024).
usage: python depth_grid.py <estimator path> [width] [depths comma list]"""
import sys, time, traceback
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP
from whestbench.loader import load_estimator_from_path
est_path = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
depths = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else "2,3,4,5,6,8,12".split(","))]
est, _ = load_estimator_from_path(Path(est_path))
est.setup(SetupContext(width=w, depth=16, flop_budget=2**41, api_version="2.0", seed=0))
for d in depths:
    for tag, B in (("full", 2**41), ("scaled", int(2**41 * (w / 1024) ** 2 * d / 16))):
        rng = np.random.default_rng(w * 1000 + d)
        weights = [fnp.asarray((rng.standard_normal((w, w)) * np.sqrt(2.0 / w)).astype(np.float32)) for _ in range(d)]
        mlp = MLP(width=w, depth=d, weights=weights, seed=0); mlp.validate()
        t = time.time()
        try:
            with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                out = est.predict(mlp, B)
                fin = bool(fnp.all(fnp.isfinite(out)))
                used = ctx.flops_used
            print(f"OK   w={w} d={d:<2d} {tag:6s} {time.time()-t:5.1f}s util={used/B:7.3%} finite={fin}", flush=True)
        except Exception as e:
            print(f"FAIL w={w} d={d:<2d} {tag:6s} {time.time()-t:5.1f}s {type(e).__name__}: {str(e)[:300]}", flush=True)
            tb = traceback.format_exc().strip().splitlines()
            print("     " + "\n     ".join(tb[-8:]), flush=True)
