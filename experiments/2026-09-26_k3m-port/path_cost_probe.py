"""Per-layer flopscope bill of the path leg: the full Strassen transport it costs today, against every product
a rank-r factorisation needs.  Unit = one plain n x n x n product (2 n^3 FLOPs, 1/1024 of B at n=1024).

    ../../.venv-whest/bin/python path_cost_probe.py [chunk]

chunk only sets how many matrices ride one Strassen call; the per-product billing is chunk-independent, so a
small chunk is used here to keep the workspace off the box while the 100-MLP run holds 4 GB.
"""
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
from estimator_k3m import _Strassen, _F32

n, chunk, lv = 1024, int(sys.argv[1]) if len(sys.argv) > 1 else 2, 4
UNIT = 2 * n ** 3
rng = np.random.default_rng(0)

with flops.BudgetContext(flop_budget=2 ** 55, wall_time_limit_s=3600.0, quiet=True):
    sw = _Strassen(n, lv, chunk)
    Wt = fnp.asarray(rng.standard_normal((n, n)).astype(np.float32))
    Yp = fnp.asarray(rng.standard_normal((n, n)).astype(np.float32))
    X2 = fnp.asarray(rng.standard_normal((2, n, n)).astype(np.float32))
    O2 = fnp.empty((2, n, n), dtype=_F32)
    On = fnp.empty((1, n, n), dtype=_F32)

res = {}
def meas(name, f):
    with flops.BudgetContext(flop_budget=2 ** 55, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter(); f(); w = time.perf_counter() - t0
        u = ctx.flops_used / UNIT
    res[name] = u
    print(f"  {name:52s} {u:7.4f} u   wall {w*1e3:8.1f} ms")
    return u

print(f"n={n}, Strassen L{lv}, chunk={chunk}.  Unit = one plain n^3 product = {1/1024:.4%} of B\n")
print("what the path leg costs TODAY (k3l/k3m default): it rides the 'full' shared stack next to C")
s2 = meas("shared(Wt, stack of 2 [C, Y])            ", lambda: sw.shared(Wt, X2, O2))
s1 = meas("shared(Wt, stack of 1 [C only])          ", lambda: sw.shared(Wt, X2[:1], O2[:1]))
full = s2 - s1
print(f"  -> the path leg's own transport: {full:.4f} u per layer "
      f"({full * 15 / 1024:.3%} of B over 15 layers)\n")

print("what a rank-r path leg needs instead (all three products through the same workspace Strassen)")
print(f"  {'r':>5s} {'svd(k=r)':>9s} {'materialise':>12s} {'thin transp':>12s} {'rebuild Y':>10s} "
      f"{'TOTAL svd':>10s} {'TOTAL rf':>9s} {'vs full':>8s}")
for r in (384, 256, 192, 128, 96, 64, 32):
    pw = min(n, r + 16)
    with flops.BudgetContext(flop_budget=2 ** 55, wall_time_limit_s=3600.0, quiet=True):
        PF = fnp.empty((1, n, r), dtype=_F32); PV = fnp.empty((1, r, n), dtype=_F32)
        PT = fnp.empty((1, n, r), dtype=_F32)
        PFw = fnp.empty((1, n, pw), dtype=_F32); PVw = fnp.empty((1, pw, n), dtype=_F32)
        PTw = fnp.empty((1, n, pw), dtype=_F32); POm = fnp.asarray(rng.standard_normal((1, n, pw)).astype(np.float32))
    q = {}
    def m(name, f):
        with flops.BudgetContext(flop_budget=2 ** 55, wall_time_limit_s=3600.0, quiet=True) as ctx:
            t0 = time.perf_counter(); f(); q[name + "_w"] = time.perf_counter() - t0
            q[name] = ctx.flops_used / UNIT
    m("svd", lambda: fnp.linalg.svd(Yp, full_matrices=False, k=r))
    m("mat", lambda: sw.shared(PF[0], PV, On))              # (n x r)(r x n) -> n x n : materialise Wt Y
    m("tra", lambda: sw.shared(Wt, PF, PT))                 # (n x n)(n x r) -> n x r : transport the factor
    # range finder: sketch Y @ Om, QR, V = Q^T Y  (the probe draw itself is O(n^2) and billed separately)
    m("ske", lambda: sw.shared(Yp, POm, PFw))
    m("qr", lambda: fnp.linalg.qr(PFw[0]))
    m("qty", lambda: sw.shared(fnp.multiply(fnp.swapaxes(PFw[0], 0, 1), 1.0), Yp[None], PVw))
    m("matw", lambda: sw.shared(PFw[0], PVw, On))
    m("traw", lambda: sw.shared(Wt, PFw, PTw))
    tot_svd = q["svd"] + q["mat"] + q["tra"] + q["mat"]          # svd + rebuild Y (for dK21) + transport + materialise
    tot_rf = q["ske"] + q["qr"] + q["qty"] + q["matw"] + q["traw"] + q["matw"]
    print(f"  {r:5d} {q['svd']:9.4f} {q['mat']:12.4f} {q['tra']:12.4f} {q['mat']:10.4f} "
          f"{tot_svd:10.4f} {tot_rf:9.4f} {'':>2s}{tot_svd/full:5.2f}x / {tot_rf/full:.2f}x")
    res[f"r{r}"] = (tot_svd, tot_rf)
print(f"\n  TOTAL svd = svd(k=r) + rebuild Y for the (2,1) birth slice + thin transport + materialise Wt Y")
print(f"  TOTAL rf  = sketch + qr + Q^T Y + rebuild + thin transport + materialise (probes r+16)")
print(f"  'vs full' > 1 means the rank-r path leg bills MORE than the full transport it replaces ({full:.4f} u).")
print(f"\n  break-even rank: svd  {int(n * full / (res['r256'][0] / (256/n)) ):d}   "
      f"rf  {int(n * full / (res['r256'][1] / (256/n))):d}   (linear in r)")
