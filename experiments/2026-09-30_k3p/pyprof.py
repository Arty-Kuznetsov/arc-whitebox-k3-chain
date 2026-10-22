"""Where does the residual (non-flopscope Python) time go? cProfile of one k3p predict on a 1024x16 MLP; the
flopscope calls themselves are included in the cumulative times, so read tottime of OUR functions and of views."""
import cProfile, pstats, sys, io
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3p as P
rng = np.random.default_rng(11)
Ws = [(rng.standard_normal((1024, 1024)) * np.sqrt(2.0 / 1024)).astype(np.float32) for _ in range(16)]
class Mo:
    width = 1024
Mo.depth, Mo.seed, Mo.weights = 16, 11, [fnp.asarray(W) for W in Ws]
est = P.Estimator(band=(5, 448, "rr"))
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:  # warm the workspace
    est.predict(Mo, 2 ** 41)
    print("warm residual", round(ctx.residual_wall_time_s, 3))
pr = cProfile.Profile()
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    pr.enable(); est.predict(Mo, 2 ** 41); pr.disable()
    print("profiled residual", round(ctx.residual_wall_time_s, 3), "ops", len(ctx.op_log))
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(30)
print(s.getvalue()[:6000])
s = io.StringIO()
st = pstats.Stats(pr, stream=s).sort_stats("tottime")
st.print_stats("estimator_k3p|_ndarray|getitem|reshape|swapaxes|_array_ops", 25)
print(s.getvalue()[-7000:])
