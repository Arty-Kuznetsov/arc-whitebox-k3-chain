# Baked-GT authoritative score (1024×16, mini = 100 MLPs)

> **⚠️ Correction (2026-09-23 evening):** the k=2 + tetrachoric baseline quoted below as **4.058e-6** (adjusted
> 4.058e-7) is wrong. The stored report `2026-09-16_baked-gt/k1_mini.json` and a fresh re-run
> (`2026-09-23_writeup-ablations/report_k2.json`, bit-identical) both give **3.8502e-6** (adjusted 3.8502e-7,
> all-layers 2.1184e-6). 4.058e-6 is MLP 0's value. Ratios against the baseline below are therefore
> overstated by 5.4%; all other numbers in this file stand.

Run 2026-09-16 · `whest run --estimator src/estimator.py --dataset ...@v2-phase2 --split mini --runner local`.
Our estimator with `n_corr_terms=1` (tetrachoric). Dataset download completed (7.03 GB, HF cache).

| metric | value |
|---|---|
| **adjusted_final_layer_score** | **4.058e-7** |
| final_layer_mse | 4.058e-6 |
| all_layers_mse | 2.150e-6 |
| mean_effective_compute | 5.179e10 FLOPs (2.35% of 2**41) |
| mean_score_multiplier | 0.1 (floored) |
| per-MLP best / worst adjusted | 2.15e-7 / 5.50e-7 |

## Reading
- **Essentially identical to the K=0 covariance baseline** (starter-kit docs: final 4.050e-6). The tetrachoric
  correction is **~neutral at width 1024** — consistent with the width-sweep trend (benefit shrinks with width,
  →~0 by 1024). **Pure-structural vector A is confirmed a dead end at competition scale.**
- **Leaderboard placement:** adjusted 4.06e-7 is worse than rank #100 (1.05e-7) → **~#120–130 / 164**.
  Top = 2.5e-9 (162× better), shared cluster = 5.4e-9 (75× better). Confirms the method-class gap → pivot.
- **This is now the honest baseline** the hybrid must beat by ~2 orders of magnitude. all_layers_mse (2.15e-6)
  < final_layer_mse (4.06e-6): error concentrates at the final/deepest layer (bias accumulates with depth) —
  which is exactly what the self-consistent-closure / 1-over-n angles target.
