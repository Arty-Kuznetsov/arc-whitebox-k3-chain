"""Baked-GT evaluation at the real 1024x16: K=0 vs K=1, with a per-layer error curve.

Loads the mini split (cache hit after the first `whest run` download) and scores our
estimator in-process against the baked 1e9 ground-truth means, for several K. Reports the
authoritative final_layer_mse / adjusted_final_layer_score, and the per-layer MSE curve so
we can see where error accumulates (→ which lever is next: more A, or vector B).

Run:  uv run --project ../../external/whest-starterkit python eval_baked.py [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
import whestbench as wb

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "external" / "whest-starterkit"))
sys.path.insert(0, str(ROOT / "src"))
from estimator import Estimator  # noqa: E402

REPO = "aicrowd/arc-whestbench-public-2026"
REVISION = "v2-phase2"  # pin the Phase 2 round (1024x16, 2**41 budget)
BUDGET = 2**41
KS = [0, 1]
GT_COLS = ("all_layer_means", "all_means", "layer_means", "means")


def load_mini(limit=0):
    ds = wb.load_dataset(REPO, revision=REVISION, split="mini")
    if not hasattr(ds, "column_names"):  # DatasetDict -> pick split
        ds = ds["mini"]
    n = len(ds) if not limit else min(len(ds), limit)
    cols = list(ds.column_names)
    gt_col = next((c for c in GT_COLS if c in cols), None)
    if gt_col is None:
        raise KeyError(f"no ground-truth column among {GT_COLS}; dataset columns={cols}")
    print(f"loaded mini: {len(ds)} MLPs (using {n}); columns={cols}; GT column='{gt_col}'")
    gt_all = np.asarray(ds.with_format("numpy")[gt_col][:n], dtype=np.float32)  # (n, depth, width)
    if gt_all.ndim == 2:  # only final means were stored -> promote to (n,1,width)
        gt_all = gt_all[:, None, :]
    mlps = []
    for i, mlp in enumerate(wb.iter_mlps(ds)):
        if i >= n:
            break
        mlps.append(mlp)
    return mlps, gt_all, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    mlps, gt_all, n = load_mini(args.limit)
    width, depth = mlps[0].width, mlps[0].depth
    final_only = gt_all.shape[1] == 1 and depth > 1

    agg = {k: {"perlayer_sse": np.zeros(depth), "flops": 0} for k in KS}
    for i in range(n):
        mlp = mlps[i]
        gt = gt_all[i]  # (depth,width) or (1,width)
        for k in KS:
            est = Estimator(n_corr_terms=k)
            with flops.BudgetContext(flop_budget=BUDGET, quiet=True) as ctx:
                pred = np.asarray(est.predict(mlp, BUDGET), dtype=np.float32)
            agg[k]["flops"] = int(ctx.flops_used)
            if final_only:
                d = pred[-1] - gt[0]
                agg[k]["perlayer_sse"][-1] += float(np.sum(d * d))
            else:
                d = pred - gt
                agg[k]["perlayer_sse"] += np.sum(d * d, axis=1)

    print(f"\nwidth={width} depth={depth}  N_mlps={n}  (final-only GT: {final_only})\n")
    per_k_curve = {}
    for k in KS:
        pl = agg[k]["perlayer_sse"] / (n * width)  # per-layer MSE
        per_k_curve[k] = pl
        final = float(pl[-1])
        allm = float(pl.mean()) if not final_only else float("nan")
        fl = agg[k]["flops"]
        mult = max(0.1, fl / BUDGET)
        print(f"K={k}: final_layer_mse={final:.4e}  all_layers_mse={allm:.4e}  "
              f"adjusted={final * mult:.4e}  flops={fl:,} ({100 * fl / BUDGET:.3f}%)")

    if 0 in KS and 1 in KS:
        f0, f1 = per_k_curve[0][-1], per_k_curve[1][-1]
        print(f"\nA gain (K1 vs K0) on final layer: {100 * (f1 - f0) / f0:+.1f}%  "
              f"({f0 / f1:.3f}x)" if f1 > 0 else "")

    if not final_only:
        print("\nper-layer MSE (L: K0 -> K1):")
        for l in range(depth):
            print(f"  L{l:2d}: {per_k_curve[0][l]:.3e} -> {per_k_curve[1][l]:.3e}")


if __name__ == "__main__":
    main()
