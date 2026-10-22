"""Where do flopscope's billed reshapes (numel(input) each, view or not) come from? Per call site of the estimator."""
import sys, traceback, collections
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src")); sys.path.insert(0, str(HERE))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
from flopscope._ndarray import FlopscopeArray
import estimator_k3y as Y
from check import mlp, run
acc = collections.Counter(); cnt = collections.Counter()
_orig = FlopscopeArray.reshape
def logged(self, *a, **k):
    st = [f for f in traceback.extract_stack(limit=6) if 'estimator_k3y' in f.filename]
    key = st[-1].lineno if st else -1
    acc[key] += int(np.prod(self.shape)); cnt[key] += 1
    return _orig(self, *a, **k)
FlopscopeArray.reshape = logged
o, f, w = run(Y.Estimator(), mlp(1024, 16, 11))
tot = sum(acc.values())
print(f"total FLOPs {f:.4e}; reshape numel logged {tot:.4e} = {tot / f:.3%} of FLOPs, {sum(cnt.values())} calls")
src = (HERE.parents[1] / "src" / "estimator_k3y.py").read_text().split("\n")
for ln, v in acc.most_common(15):
    print(f"  line {ln:5d}: {v / f:.3%}  calls {cnt[ln]:4d}   {src[ln - 1].strip()[:110] if ln > 0 else '?'}")
