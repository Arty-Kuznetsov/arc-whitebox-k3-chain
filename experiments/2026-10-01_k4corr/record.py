"""Record the kappa4-corrector features on the PRODUCTION trajectory (src/estimator_k3v.py with the corrector off and
recording on) for the first 10 MLPs of the mini split -- the same weights AND seeds the runner uses -- and save them with
the Monte-Carlo oracle kappa4 of each layer, for training. Also checks that the dataset MLPs are the baked ones."""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import whestbench as wb
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3v as V

ORA0 = ROOT / "experiments/2026-09-24_k3-chain/oracles"
ORAR = ROOT / "experiments/2026-09-24_k3-chain/runs"
BAKED = ROOT / "experiments/2026-09-16_kprop-arc/baked"
ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
if not hasattr(ds, "column_names"):
    ds = ds["mini"]
LO_I = int(sys.argv[1]) if len(sys.argv) > 1 else 0
HI_I = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUTF = HERE / (sys.argv[3] if len(sys.argv) > 3 else "recorded.npy")
out = {}
for i, mlp in enumerate(wb.iter_mlps(ds)):
    if i >= HI_I:
        break
    if i < LO_I:
        continue
    W = np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights])
    same = np.array_equal(W, np.load(BAKED / f"weights_{i}.npy")) if (BAKED / f"weights_{i}.npy").exists() else None
    ora = np.load((ORA0 if i < 2 else ORAR) / f"mc_oracle_{i}.npz")["k4"]
    est = V.Estimator(k4corr=False)   # record on the UNcorrected trajectory (the model is applied online)
    est._k4rec = []
    t0 = time.perf_counter()
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True):
        est.predict(mlp, 2 ** 41)
    rec = [(l, np.asarray(F, dtype=np.float64), np.asarray(k, dtype=np.float64)) for (l, F, k) in est._k4rec]
    out[i] = [(l, F, k4, ora[l]) for (l, F, k4) in rec]
    print(f"  MLP {i} ({getattr(mlp, 'name', '?')}, seed {mlp.seed}): baked-identical {same}, recorded layers "
          f"{[r[0] for r in rec]}  {time.perf_counter()-t0:.1f}s", flush=True)
np.save(OUTF, np.array(out, dtype=object), allow_pickle=True)
print(f"saved {OUTF.name}")
