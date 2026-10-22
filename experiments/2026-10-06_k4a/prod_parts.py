"""Record the production reader's parts per (layer, generation) on baked MLP m, and compare with torch_parts_m{m}.npz."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k4a as A
class pb:
    BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
m = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Ws = np.load(pb.BAKED / f"weights_{m}.npy")
class Mo:
    width = Ws.shape[2]
Mo.depth, Mo.seed, Mo.weights = Ws.shape[0], m, [fnp.asarray(W.astype(np.float32)) for W in Ws]
e = A.Estimator(); e._drec = []
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    out = np.asarray(e.predict(Mo, 2 ** 41), dtype=np.float64)
gt = np.load(pb.BAKED / "gt.npy")
print("prod final MSE", float(((out[-1] - gt[m][-1]) ** 2).mean()))
T = np.load(HERE / f"torch_parts_m{m}.npz")
def cmp(a, b):
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    msk = a != 0   # production zeros the cut rows of the 3+2; compare on the common support for the total
    c = np.corrcoef(a[msk], b[msk])[0, 1] if msk.sum() > 2 else float("nan")
    return f"corr {c:+.3f} |p|/|t| {np.linalg.norm(a[msk]) / max(np.linalg.norm(b[msk]), 1e-30):.3f}"
for rec in e._drec:
    if rec[0] != "k4y": continue
    _, l, born, pd, st, tot = rec
    tpd, tst, t32 = (T.get(f"l{l}_m{born}_pd"), T.get(f"l{l}_m{born}_st"), T.get(f"l{l}_m{born}_32"))
    line = f"l={l:2d} born={born:2d}: "
    if pd is not None and tpd is not None: line += " pathdiag " + cmp(pd, tpd)
    if st is not None and tst is not None: line += " | star " + cmp(st, tst)
    if tot is not None:
        ttot = sum(x for x in (tpd, tst, t32) if x is not None)
        line += " | total " + cmp(tot, ttot)
        if t32 is not None:
            p32 = np.asarray(tot, dtype=np.float64) - (0 if pd is None else np.asarray(pd, dtype=np.float64)) - (0 if st is None else np.asarray(st, dtype=np.float64))
            line += " | 3+2 " + cmp(p32, t32)
        else:
            line += " | (torch has no 3+2 here)"
    print(line)
