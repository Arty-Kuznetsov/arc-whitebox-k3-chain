# Multi-source Edgeworth estimator — authoritative score (2026-09-23)

> **⚠️ Correction (2026-09-23 evening):** the k=2 + tetrachoric baseline quoted below as **4.058e-6** (adjusted
> 4.058e-7) is wrong. The stored report `2026-09-16_baked-gt/k1_mini.json` and a fresh re-run
> (`2026-09-23_writeup-ablations/report_k2.json`, bit-identical) both give **3.8502e-6** (adjusted 3.8502e-7,
> all-layers 2.1184e-6). 4.058e-6 is MLP 0's value. Ratios against the baseline below are therefore
> overstated by 5.4%; all other numbers in this file stand.

Full public `mini` split, 100 MLPs, baked N=1e9 ground truth, width 1024, depth 16.
Change vs the previous submission: the third-cumulant correction now sums over the **3 newest
sources** instead of regenerating from the previous layer alone (`n_sources=3`).

| metric | banked k=2 | submitted J=1 (id 331885) | **J=3 (new)** | vs J=1 |
|---|---|---|---|---|
| **adjusted_final_layer_score** | 4.058e-7 | 2.2767e-7 | **1.16775e-7** | **1.95×** |
| final_layer_mse | 4.058e-6 | 2.2767e-6 | 1.16775e-6 | 1.95× |
| all_layers_mse | 2.150e-6 | 1.1591e-6 | 6.047e-7 | 1.92× |
| mean_compute_utilization | 2.351% | 3.823% | **9.103%** | still under the 10% floor |
| mean_score_multiplier | 0.1 | 0.1 | 0.1 | — |
| best / worst per-MLP adjusted | 2.15e-7 / 5.50e-7 | 1.231e-7 / 3.348e-7 | 7.477e-8 / 1.608e-7 | — |
| failed MLPs | 0/100 | 0/100 | **0/100** | — |

Cumulative improvement over the September baseline: **3.47×**, entirely analytic, with no sampling
and no constant fitted to the public MLPs.

## Why it works
A third-cumulant source is born at every ReLU layer and transported forward, so the pre-activation
at depth l carries l of them. The previous version modelled only the newest, which is why its
regenerated κ₃ correlated 0.998 with truth at layer 1 but only 0.64 at depth. Carrying a window of
the J newest sources costs 2 matmuls per source per layer (birth is free: the hub leg starts as the
identity). Measured raw gain over plain k=2: 1.61× / 2.37× / 3.16× / 3.94× for J = 1/2/3/4, saturating
near 4.5× and degrading at J=15 — the leading Wick diagram is a poor model for old sources.
J = 3 is the largest window that still fits under the 10% multiplier floor, where extra compute is free.

## Safety margins on the grader contract
Per-MLP residual wall time 0.0439 s against the 0.4 s cap (9× margin); wall time 0.73 s against 120 s;
`failure_breakdown` all zeros. The J=1 version ran at 0.0166 s residual, so the cost scales with the
number of carried sources as expected.

## Reproduce
```bash
uv run --project external/whest-starterkit whest run --estimator src/estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
    --runner local --max-threads 4 --format json
```
Derivation and the source sweep: `../2026-09-16_kprop-arc/RESULTS-edgeworth.md`, `multisource.py`.
The map of what each ingredient is worth: `../2026-09-16_kprop-arc/logs/state_split3.txt`.

## Headroom left
Utilization is 9.10%, so 0.9 percentage points (≈2e10 FLOPs, about 9 matmuls) remain free below the
floor. A fourth source needs ~32 matmuls and would cross it; measured J=4 is only ~2% better adjusted.
