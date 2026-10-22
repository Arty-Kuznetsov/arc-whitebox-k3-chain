"""Is Strassen level 5 shippable? (util / residual / wall / peak RSS on the production path)

    uv run --project external/whest-starterkit python experiments/2026-09-26_strassen5/probe.py [chunk ...]

`_Strassen.__init__` in src/ stops the recursion at `m < 128`, which caps n=1024 at four levels (leaf 64).
This probe monkeypatches ONLY that guard (module global `_Strassen`, so src/ is untouched) to `m < 64`, giving
five levels and a 32x32 leaf, and measures what the Mac alone can decide:
  * utilisation  -- the modelled prize is billing 0.636 -> ~0.560, i.e. 24.84% -> ~21.9% (x1.14, no accuracy cost);
  * residual time -- cap 0.400 s; grader/local-runner = 0.753 measured on 332253/332403;
  * wall time    -- cap 120 s; the grader runs ~3.1x this runner and 332403's WORST MLP was already 75.3 s;
  * peak RSS     -- cap 8 GB; the L5 level region alone is 2 x 7^5 x chunk x 32 x 32 floats.
The dead-row padding quantum is `2 ** len(sw.bufs)`, so it follows the level count automatically (16 -> 32).
"""
import resource, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k3m as K


class S5(K._Strassen):
    """k3m's _Strassen with the recursion floor at 32 instead of 128 (five levels at n=1024)."""

    def __init__(self, n, levels, chunk):
        self.levels, self.chunk = levels, chunk
        self.bufs, self.regions = [], []
        m, G = n, 1
        for _ in range(levels):
            if m % 2 or m < 64:          # <-- the only change: was `m < 128`
                break
            m //= 2
            G *= 7
            half = G * chunk * m * m
            R = fnp.empty((2 * half,), dtype=K._F32)
            self.regions.append(R)
            self.bufs.append((R[:half].reshape((G, chunk, m, m)), R[half:].reshape((G, chunk, m, m)),
                              fnp.empty((G, 1, m, m), dtype=K._F32)))
        self.leaf_tmp = None
        L = len(self.bufs)
        if L >= 2:
            Gl, ml = 7 ** L, n // 2 ** L
            self.leaf_tmp = self.regions[L - 2][:Gl * chunk * ml * ml].reshape((Gl, chunk, ml, ml))
        elif L == 1:
            self.leaf_tmp = fnp.empty((7, chunk, n // 2, n // 2), dtype=K._F32)


def mlp(n, depth, seed):
    rng = np.random.default_rng(seed)
    Ws = [(rng.standard_normal((n, n)) * np.sqrt(2.0 / n)).astype(np.float32) for _ in range(depth)]

    class Mo:
        width = n
    Mo.depth, Mo.seed, Mo.weights = depth, seed, [fnp.asarray(W) for W in Ws]
    return Mo


def run(est, m):
    with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
        t0 = time.perf_counter()
        out = np.asarray(est.predict(m, 2 ** 41), dtype=np.float64)
        return out, ctx.flops_used, ctx.residual_wall_time_s, time.perf_counter() - t0


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 3   # darwin: bytes


A = mlp(1024, 16, 11)
print(f"levels built at chunk 16, guard 128 (production): {len(K._Strassen(1024, 5, 16).bufs)}")
base = K.Estimator()
o_ref, f_ref, r_ref, w_ref = run(base, A)
print(f"L4  chunk 16 : {100 * f_ref / 2 ** 41:7.3f}% of B   residual {r_ref:.3f}s  wall {w_ref:5.1f}s  "
      f"peak RSS {rss_gb():.2f} GB")
del base

K._Strassen = S5
for chunk in [int(x) for x in (sys.argv[1:] or ["16", "8"])]:
    est = K.Estimator(strassen_levels=5, chunk=chunk)
    o, f, r, w = run(est, A)
    d = np.abs(o - o_ref)[-1].max() / np.abs(o_ref[-1]).max()
    print(f"L5  chunk {chunk:<2}: {100 * f / 2 ** 41:7.3f}% of B   residual {r:.3f}s  wall {w:5.1f}s  "
          f"peak RSS {rss_gb():.2f} GB   x{f / f_ref:.4f} of L4   final rel|L5-L4| {d:.2e}")
    del est
