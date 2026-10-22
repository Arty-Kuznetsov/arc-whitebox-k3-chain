"""Local probe: does the tetrachoric/Hermite ReLU-covariance correction help, and how
does its effect scale with width (toward the 1024 competition regime)?

No baked ground truth (dataset declined), so we build a high-sample Monte-Carlo
reference and compare the estimator at several `n_corr_terms` against it. K=0 is the
gain-method baseline. We report:
  - final_layer_mse per K (vs MC ref),
  - a low-variance PAIRED improvement  dMSE(K) = mean_m[ mse(est_K) - mse(est_0) ]  (<0 = helps),
  - a Monte-Carlo noise floor (half-split), so we know which deltas are real.

Width 256 weakens CLT; 1024 is the competition shape. The TREND across width tells us
whether A's payoff is regime-dependent. Absolute magnitudes still need the baked 1e9 GT.

Run:  uv run --project ../../external/whest-starterkit python probe.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp

ROOT = Path(__file__).resolve().parents[2]
# src first so OUR estimator wins over the kit's same-named estimator.py; kit still on path for local_engine.
sys.path.insert(0, str(ROOT / "external" / "whest-starterkit"))
sys.path.insert(0, str(ROOT / "src"))
from estimator import Estimator  # noqa: E402
from local_engine import build_mlp, monte_carlo_layer_means  # noqa: E402

WIDTHS = [256, 512, 768]
DEPTH = 16
N_CHUNK = 120_000
N_CHUNKS = 4  # even; split into halves for the noise floor
MLP_SEEDS = [0, 1, 2]
KS = [0, 1, 2, 3]
BUDGET = 2**41


def mc_reference(mlp, base_seed):
    acc = None
    halves = [None, None]
    counts = [0, 0]
    for c in range(N_CHUNKS):
        m = monte_carlo_layer_means(mlp, N_CHUNK, seed=base_seed * 10_000 + c)
        acc = m if acc is None else acc + m
        h = c % 2
        halves[h] = m if halves[h] is None else halves[h] + m
        counts[h] += 1
    return acc / N_CHUNKS, halves[0] / counts[0], halves[1] / counts[1]


def mse_final(pred, ref):
    d = pred[-1] - ref[-1]
    return float(fnp.mean(d * d))


def main():
    print(f"DEPTH={DEPTH}  ref N={N_CHUNK * N_CHUNKS:,}  MLP_SEEDS={MLP_SEEDS}  KS={KS}\n")
    for width in WIDTHS:
        t0 = time.time()
        per_k_final = {k: [] for k in KS}
        per_k_paired = {k: [] for k in KS}  # per-mlp (mse_k - mse_0)
        per_k_flops = {k: 0 for k in KS}
        noise = []
        for s in MLP_SEEDS:
            mlp = build_mlp(width=width, depth=DEPTH, seed=s)
            ref, ha, hb = mc_reference(mlp, s)
            noise.append(mse_final(ha, hb))
            m0 = None
            for k in KS:
                est = Estimator(n_corr_terms=k)
                with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
                    p = est.predict(mlp, BUDGET)
                per_k_flops[k] = int(ctx.flops_used)
                mk = mse_final(p, ref)
                per_k_final[k].append(mk)
                if k == 0:
                    m0 = mk
                per_k_paired[k].append(mk - m0)

        mean = lambda xs: sum(xs) / len(xs)
        nf = mean(noise)
        print(f"width={width}  (noise floor ~{nf:.2e}, {time.time()-t0:.0f}s)")
        print("   K | final_layer_mse | paired dMSE vs K0 |    %budget | verdict")
        print("   " + "-" * 74)
        for k in KS:
            fin = mean(per_k_final[k])
            dm = mean(per_k_paired[k])
            pct = 100.0 * per_k_flops[k] / BUDGET
            if k == 0:
                verdict = "baseline"
            elif abs(dm) < nf:
                verdict = "within noise"
            elif dm < 0:
                verdict = f"HELPS ({m0_ratio(fin, per_k_final[0], mean):.2f}x)"
            else:
                verdict = "hurts"
            print(f"   {k} |     {fin:.4e} |       {dm:+.3e} | {pct:8.4f}% | {verdict}")
        print()


def m0_ratio(fin_k, base_list, mean):
    base = mean(base_list)
    return base / fin_k if fin_k > 0 else float("nan")


if __name__ == "__main__":
    main()
