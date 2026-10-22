# Authoritative score of the Edgeworth-corrected estimator (2026-09-23)

> **⚠️ Correction (2026-09-23 evening):** the k=2 + tetrachoric baseline quoted below as **4.058e-6** (adjusted
> 4.058e-7) is wrong. The stored report `2026-09-16_baked-gt/k1_mini.json` and a fresh re-run
> (`2026-09-23_writeup-ablations/report_k2.json`, bit-identical) both give **3.8502e-6** (adjusted 3.8502e-7,
> all-layers 2.1184e-6). 4.058e-6 is MLP 0's value. Ratios against the baseline below are therefore
> overstated by 5.4%; all other numbers in this file stand.

`whest run --estimator src/estimator.py --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2
--split mini --runner local --max-threads 4 --format json` → `report.json`.
Full public split, **100 MLPs**, baked N=1e9 ground truth, width 1024, depth 16.

| metric | banked k=2 (2026-09-16) | **Edgeworth (new)** | gain |
|---|---|---|---|
| **adjusted_final_layer_score** | 4.058e-7 | **2.2767e-7** | **1.78×** |
| final_layer_mse | 4.058e-6 | 2.2767e-6 | 1.78× |
| all_layers_mse | 2.150e-6 | 1.1591e-6 | 1.85× |
| mean_compute_utilization | 2.351% | 3.823% | — |
| mean_effective_compute | 5.179e10 | 8.406e10 | — |
| mean_score_multiplier | 0.1 (floored) | 0.1 (floored) | — |
| best / worst per-MLP adjusted | 2.15e-7 / 5.50e-7 | 1.231e-7 / 3.348e-7 | — |
| failed MLPs | 0 / 100 | **0 / 100** | — |

`failure_breakdown` is all zeros: no budget exhaustion, no timeouts, no residual-time breaches.

## Reading
- The multiplier stays **floored at 0.1**, so the extra 1.47 percentage points of budget spent on the
  correction cost nothing on the metric. The whole 1.78× is accuracy.
- The predicted FLOP cost was exact: hand calculation gave 5.179e10 + 15 × (2n³ − n²) = 8.40e10 against
  a measured 8.406e10.
- Per-MLP spread tightened as well: the worst MLP improved 1.64× and the best 1.75×, so the gain is not
  carried by a few outliers.
- For context, the best published analytic-plus-sampling blend on this split (forum #18200) reports
  4.050e-7 → 3.044e-7, a 1.33× gain at 9.345% utilization. This estimator reaches **2.277e-7 at 3.82%**,
  is purely analytic, and has no fitted constants.

## Confirmed on the real grader (2026-09-22)

Submitted as AIcrowd **submission id 331885** (account `artiem_kuznietsov`), description
"covariance propagation + Wick-regenerated Edgeworth mean correction". Graded score:
**2.2219498214326448e-07**, against our local 2.2767e-7 on the public `mini` split — the grader is
**2.5% better**, i.e. local and remote agree closely. No failure reported. Note the two numbers are
measured on different MLP sets from the same distribution: the grader's live score uses the 50 public
MLPs of the Phase-2 suite, while `mini` is the 100 openly released verification examples. For
reference, forum write-up #18200 reported a 1.433% local-to-live gap and #18218 a consistent ~6.5% one.
Tracking: https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/331885

## Still outside the leaderboard
2.277e-7 remains far from the public top 100 (rank 100 was 5.08e-8 on 2026-09-22). This is a principled
improvement to a generalizing baseline, not a competitive placement. Its value is (a) a materially better
designated submission that carries into the private re-evaluation unchanged, and (b) the algorithmic
contribution behind it.

## Reproduce
```bash
uv run --project external/whest-starterkit whest run --estimator src/estimator.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini \
    --runner local --max-threads 4 --format json
```
Derivation and the probes behind the method: `../2026-09-16_kprop-arc/RESULTS-edgeworth.md`.

## Ablation: the tetrachoric covariance term (`estimator_notetra.py` → `report_notetra.json`)

| configuration | adjusted | final_layer_mse | all_layers_mse | utilization | failures |
|---|---|---|---|---|---|
| Edgeworth + tetrachoric (**shipped**) | **2.2767e-7** | 2.2767e-6 | 1.1591e-6 | 3.823% | 0/100 |
| Edgeworth, tetrachoric off | 2.4527e-7 | 2.4527e-6 | 1.2601e-6 | 3.819% | 0/100 |

Switching the tetrachoric term off costs **7.7%**, so it is kept on. This is an interaction worth
stating: measured **alone** in September the tetrachoric correction was neutral (4.058e-6 against the
kit's 4.050e-6), and `RESULTS-layer-blame.md` showed that improving the covariance *alone* actively
hurts (up to +16%). Once the mean is corrected, the same covariance improvement pays. The three
observations are one mechanism: a more accurate covariance is only useful when paired with a mean of
matching quality, which is the constructive side of the "accuracy ⊥ stability" effect reported at
depth 32 in Phase-1 write-up #326024.
