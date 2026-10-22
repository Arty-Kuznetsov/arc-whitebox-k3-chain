# Tetrachoric correction — width-sweep probe

**Run:** 2026-09-13 · `probe.py` · depth=16, MC ref N=480,000 (4×120k chunks), MLP seeds [0,1,2].
No baked ground truth — reduced-width MC reference. Tests DIRECTION, not the 1024 magnitude.

## Result (paired ΔMSE vs K=0; negative = the correction helps)

| width | baseline `final_layer_mse` (K=0) | K=1 ΔMSE | K=2 ΔMSE | K=3 ΔMSE | noise floor | verdict |
|---|---|---|---|---|---|---|
| 256 | 5.83e-5 | +1.27e-6 | +1.30e-6 | +1.34e-6 | 9.0e-7 | **hurts** |
| 512 | 2.22e-5 | −1.76e-6 | −1.77e-6 | −1.77e-6 | 9.5e-7 | **helps ~8%** |
| 768 | 7.58e-6 | −0.42e-6 | −0.42e-6 | −0.43e-6 | 5.5e-7 | helps (≈ noise floor) |

## Reading it
- **Sign flip with width.** The correction *hurts* at width 256 and *helps* from 512 up. Mechanism:
  the tetrachoric series makes the covariance exact **under the Gaussian pre-activation assumption**.
  At small width that assumption is wrong (weak CLT), so a "better Gaussian covariance" is worse —
  there the bottleneck is non-Gaussianity (vector B). Once the CLT holds (width ≥512), the covariance
  approximation is a real error source and the correction removes it.
- **K=1 is the whole story.** K=2, K=3 add <1e-8 at every width → ship `n_corr_terms=1`; more terms
  are pure cost. (Consistent with small pre-activation correlations ρ at large width: ρⁿ decays fast.)
- **Benefit shrinks with width** (8% → 5.6% relative), so extrapolating to 1024 suggests a *modest*
  gain, not a knockout. Sign is very unlikely to flip back (hurting only occurred in the anti-CLT regime).
- Baseline `final_layer_mse` tracks toward the documented 1024 value (5.8e-5 → 2.2e-5 → 7.6e-6 → 4.05e-6).

## Limits / why the next step needs the dataset
Local MC can't resolve the ~1.5e-7 effect expected at width 1024 without N≳1e7 (≈hours + big RAM).
The baked 1e9 ground truth (`arc-whestbench-public-2026@v2-phase2`, mini split) resolves it instantly
and is the only way to (a) confirm A's true 1024 gain and (b) read per-layer error to pick the next lever.

## Reproduce
```
uv run --project ../../external/whest-starterkit python probe.py
```
