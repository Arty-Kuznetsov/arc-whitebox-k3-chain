# Structure of the final-layer residual — why no post-hoc or learned corrector can work

**Date:** 2026-09-27, Mac. **Scripts:** `dump.py` (runs the torch chain on the 10 baked MLPs and saves every layer's
prediction, GT, pre-activation mean/variance, κ₃ diagonal t and κ₄ diagonal; `dump10.npz` is gitignored — rerun
`dump.py`, self-check mean final MSE **2.3135e-08**, k3l config with `deadcut=-3.0`) and `analyze.py` (the analysis).
Note: the baked ground truth covers **10 MLPs** (`gt.npy` is (10, 16, 1024)); the 100-MLP set lives only behind
`whest run`.

**Why it was run.** The user asked for broader thinking about first place; the obvious candidate was a learned or
post-hoc corrector of our final-layer error (shipping data files is allowed). Before building one, measure whether the
residual has any structure a corrector could learn.

## 1. Size of the signal vs our error
Per-neuron std of the ground truth: layer 0 **0.0125**, layer 4 0.641, layer 8 0.741, layer 12 0.793, **layer 15
0.781** (mean 0.564). Our rms error at the graded raw 2.165e-8 is **1.47e-4**, i.e. R² ≈ 0.99999994 against a
constant-per-MLP baseline. The leaders' rms error (raw 1.54e-8) is 1.24e-4 — only ~18% better than ours in rms.

## 2. The residual is incoherent, mildly heavy-tailed, and lives on the "alive" neurons
- Worst 1% of neurons carry 12.9% of the MSE; 5% → 39.9%; 10% → 59.0%; 25% → 86.4%; 50% → 98.7%. Kurtosis 5.0.
- **Coherent (per-MLP mean) share of the MSE: 0.46%** (rms of the per-MLP mean 1.04e-5 against rms 1.52e-4). Any
  global or level correction is worth at most 0.5%.
- By α = μ/σ at the last layer: α < −3 (192 neurons/MLP) carries 0.0%; α ∈ [1, 2) 19.8%; **α ≥ 2 (284 neurons/MLP)
  51.5%**. The error grows with how alive a neuron is — for α ≫ 0 the ReLU is the identity, so the residual there is
  simply the error of the propagated mean.

## 3. It is uncorrelated with every local feature, and even with itself one layer earlier
| feature at the last layer | corr with the residual |
|---|---|
| α | −0.0156 |
| σ | −0.0238 |
| var | −0.0235 |
| skewness t/σ³ | −0.0288 |
| excess kurtosis K4d/var² | +0.0109 |
| the prediction itself | −0.0193 |
| the last step's change of the prediction | −0.0131 |

**R² ≤ 0.001 from any of them.** And across depth the residual is scrambled: corr(residual at layer l, residual at
layer 15) = +0.0064 (l = 8), +0.0109 (10), +0.0058 (12), +0.0042 (13), **+0.0040 (14)**, while its rms grows
8.5e-5 → 1.43e-4 → 1.52e-4 (≈30% of the final MSE is already present at layer 8).

## 4. Mechanism and verdict
Each layer multiplies the error vector by a fresh random matrix, so the final-layer residual is a random projection of
the accumulated upstream error — not a function of anything local. **A post-hoc corrector, learned or analytic, has
nothing to learn** (NEGATIVE §64). Independently confirmed by 504aldo (forum topic 18218, their row 10: "final-layer
features correlate < 0.09 with the error"). The only way to reduce this error is to make fewer errors upstream, at every
layer — i.e. better propagation, not a better readout.
