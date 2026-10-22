"""Price E3's CONSTRUCTION before building it (project lesson #9: a model that prices only the saved product says yes
every time). Measures, on the real shipping estimator at 1024x16:
  * the receiver fraction at theta_out = -3.0 per layer (how much of the transport's output axis E3 removes),
  * R (rows of the leg stack), npp (alive previous neurons) and wn (the newest source's rows, which must stay FULL
    width because arm "zero" costs x1.017 raw against arm "b"'s x1.002),
  * the transport's own share of the FLOPs, so the saving can be compared with the construction:
      3 chunked Strassen calls instead of 1 + a (R x n) scatter per layer + an (n_recv x n x n) fill-in.
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3r as R

TH = -3.0
rows = []


def patched_predict(self, mlp, budget):
    """Re-run the shipped predict but record the shapes we need, by monkeypatching the two helpers it calls."""
    return R.Estimator.predict(self, mlp, budget)


rng = np.random.default_rng(11)
n, depth = 1024, 16
Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]


class Mo:
    width = n


Mo.depth, Mo.seed, Mo.weights = depth, 11, [fnp.asarray(W) for W in Ws]

# --- receiver fraction: replay the chain's own alpha per layer with a plain float64 numpy mirror of the mean/cov step
mu = np.zeros(n)
C = np.eye(n) / n
fr = []
for l in range(depth):
    W = Ws[l].astype(np.float64)
    mu_pre = W.T @ mu
    C_pre = W.T @ C @ W
    var = np.maximum(np.diag(C_pre), 1e-12)
    a = mu_pre / np.sqrt(var)
    fr.append((float((a < TH).mean()), float((a < -2.5).mean()), float((a > 2.5).mean())))
    # Gaussian ReLU step (good enough for the alpha histogram)
    sd = np.sqrt(var)
    from math import erf
    Ph = 0.5 * (1.0 + np.array([erf(z / np.sqrt(2)) for z in a]))
    ph = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    mu = mu_pre * Ph + sd * ph
    d1 = Ph
    C = (d1[:, None] * C_pre) * d1[None, :]
    np.fill_diagonal(C, (mu_pre ** 2 + var) * Ph + mu_pre * sd * ph - mu ** 2)
print(f"{'layer':>5s} {'recv frac (a<-3.0)':>19s} {'dead (a<-2.5)':>14s} {'sat (a>2.5)':>12s}")
for l, (r3, d25, s25) in enumerate(fr):
    print(f"{l:5d} {r3*100:18.2f}% {d25*100:13.2f}% {s25*100:11.2f}%")
m = np.mean([x[0] for x in fr[1:]])
print(f"\nmean receiver fraction over layers 1-15: {m*100:.2f}%  (E3 removes this share of the transport's output axis)")

# --- the FLOPs breakdown of one shipped predict, to size the transport
with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
    t0 = time.perf_counter()
    _ = R.Estimator().predict(Mo, 2 ** 41)
    wall = time.perf_counter() - t0
print(f"\nshipped k3r on this MLP: flops {ctx.flops_used:.4e}  util {ctx.flops_used/2.2e12*100:.3f}%  "
      f"residual {ctx.residual_wall_time_s:.3f}s  wall {wall:.1f}s")
