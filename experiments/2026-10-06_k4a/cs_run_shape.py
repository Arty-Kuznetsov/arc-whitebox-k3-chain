"""Estimator through the flopscope CLIENT against the local SERVER at an arbitrary shape (random weights made on the server).
usage: FLOPSCOPE_SERVER_URL=... python cs_run4.py <estimator module> <width> <depth>"""
import os, sys, time, traceback, importlib, math
ARC = str(__import__("pathlib").Path(__file__).resolve().parents[2])   # the repository root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cs_stub")); sys.path.insert(0, os.path.join(ARC, "src"))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP
name, w, d = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mod = importlib.import_module(name); est = mod.Estimator()
t = time.time(); stage = "setup"
try:
    with flops.BudgetContext(flop_budget=2**41) as ctx:
        est.setup(SetupContext(width=w, depth=d, flop_budget=2**41, api_version="2.0", seed=0))
        stage = "weights"
        rng = fnp.random.default_rng(w * 1000 + d)
        weights = [fnp.asarray(rng.standard_normal((w, w)) * math.sqrt(2.0 / w), dtype=fnp.float32) for _ in range(d)]
        mlp = MLP(width=w, depth=d, weights=weights, seed=0); mlp.validate()
        stage = "predict"; t = time.time()
        out = est.predict(mlp, 2**41)
        fin = bool(fnp.all(fnp.isfinite(out))); used = ctx.flops_used
    print(f"{name} {w}x{d}: predict OK in {time.time()-t:.1f} s; util {used/2**41:.3%}; finite {fin}; shape {tuple(out.shape)}", flush=True)
except Exception:
    print(f"{name} {w}x{d}: {stage} FAILED after {time.time()-t:.1f} s", flush=True); traceback.print_exc()
