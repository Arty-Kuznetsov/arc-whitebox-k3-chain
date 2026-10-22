import time, gc
import flopscope as flops
import flopscope.numpy as fnp
rng = fnp.random.default_rng(0)
A = rng.standard_normal((12, 1024, 1024), dtype=fnp.float32)
def run(label, f, reps=10):
    with flops.BudgetContext(flop_budget=2**55, wall_time_limit_s=600.0, quiet=True) as ctx:
        for _ in range(reps):
            f()
    print(f"{label:50s} resid/rep {ctx.residual_wall_time_s/reps*1e3:7.2f} ms   backend/rep {ctx.flopscope_backend_time_s/reps*1e3:7.2f} ms")
run("x = A + A; del x   (48 MB alloc+free)", lambda: (A + A))
def keep():
    x = A + A
    y = x * 2.0
    del x
    return y
run("two temporaries", keep)
big = [None]
def big_tmp():
    x = fnp.concatenate([A] * 8, axis=0)  # 384 MB
    return None
run("384 MB temporary", big_tmp, reps=4)
gc.disable()
run("384 MB temporary, gc disabled", big_tmp, reps=4)
gc.enable()
