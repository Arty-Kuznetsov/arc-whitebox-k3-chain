"""Exception hunt over network shapes and budgets, in-process flopscope (whest venv).
usage: python shape_grid.py <estimator path> [maxwidth]"""
import sys, time, traceback
from pathlib import Path
import numpy as np
ARC = str(__import__("pathlib").Path(__file__).resolve().parents[2])   # the repository root
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP
from whestbench.loader import load_estimator_from_path
est_path = sys.argv[1]; maxw = int(sys.argv[2]) if len(sys.argv) > 2 else 768
est, _ = load_estimator_from_path(Path(est_path))
est.setup(SetupContext(width=1024, depth=16, flop_budget=2**41, api_version="2.0", seed=0))
widths = [w for w in (16, 32, 64, 96, 128, 192, 256, 384, 512, 768) if w <= maxw]
depths = (1, 2, 3, 4, 6, 8, 12, 16)
for d in depths:
    for w in widths:
        for tag, B in (("full", 2**41), ("scaled", max(10**7, int(2**41 * (w / 1024) ** 2 * d / 16)))):
            rng = np.random.default_rng(w * 1000 + d)
            weights = [fnp.asarray((rng.standard_normal((w, w)) * np.sqrt(2.0 / w)).astype(np.float32)) for _ in range(d)]
            mlp = MLP(width=w, depth=d, weights=weights, seed=0); mlp.validate()
            t = time.time()
            try:
                with flops.BudgetContext(flop_budget=B, quiet=True) as ctx:
                    out = est.predict(mlp, B)
                    fin = bool(fnp.all(fnp.isfinite(out)))
                    used = ctx.flops_used
                print(f"OK   w={w:<4d} d={d:<2d} {tag:6s} {time.time()-t:5.1f}s util={used/B:7.3%} finite={fin} shape={tuple(out.shape)}", flush=True)
            except Exception as e:
                print(f"FAIL w={w:<4d} d={d:<2d} {tag:6s} {time.time()-t:5.1f}s {type(e).__name__}: {str(e)[:200]}", flush=True)
                tb = traceback.format_exc().strip().splitlines()
                print("     " + "\n     ".join(tb[-6:]), flush=True)
