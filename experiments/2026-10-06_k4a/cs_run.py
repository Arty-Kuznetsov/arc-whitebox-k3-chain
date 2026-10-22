"""Estimator through the flopscope CLIENT against a local flopscope SERVER; everything inside one budget session.
usage: FLOPSCOPE_SERVER_URL=ipc:///tmp/fs.sock python cs_run2.py <estimator module in src/> [mlp index] [layers]"""
import os, sys, time, traceback, importlib
ARC = str(__import__("pathlib").Path(__file__).resolve().parents[2])   # the repository root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "cs_stub")); sys.path.insert(0, os.path.join(ARC, "src"))
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP
name = sys.argv[1]; idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
depth = int(sys.argv[3]) if len(sys.argv) > 3 else 16
print("flopscope module:", flops.__file__, "version", getattr(flops, "__version__", "?"), flush=True)
mod = importlib.import_module(name)
est = mod.Estimator()
t = time.time(); stage = "setup"
try:
    with flops.BudgetContext(flop_budget=2**41) as ctx:
        est.setup(SetupContext(width=1024, depth=depth, flop_budget=2**41, api_version="2.0", seed=0))
        stage = "load"
        W = fnp.load(f"{ARC}/experiments/2026-09-16_kprop-arc/baked/weights_{idx}.npy")
        weights = [fnp.array(fnp.asarray(W[i], dtype=fnp.float32)) for i in range(depth)]   # contiguous copies, as the grader's worker builds them
        mlp = MLP(width=1024, depth=depth, weights=weights, seed=0); mlp.validate()
        print(f"setup+load {time.time()-t:.1f} s; weight type {type(weights[0]).__name__}; flops so far {ctx.flops_used}", flush=True)
        stage = "predict"; t = time.time()
        out = est.predict(mlp, 2**41)
        used = ctx.flops_used
        print(f"predict OK in {time.time()-t:.1f} s; flops_used {used} ({used/2**41:.4%}); out shape {getattr(out,'shape',None)}", flush=True)
        try:
            print("out[-1][:4] =", [float(x) for x in out[depth-1][:4]], flush=True)
        except Exception as e:
            print("out read failed:", repr(e), flush=True)
except Exception:
    print(f"{stage} FAILED after {time.time()-t:.1f} s", flush=True)
    traceback.print_exc()
