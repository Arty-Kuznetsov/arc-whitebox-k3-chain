# The variance channel — why the ×1.54 is not reachable with a cheap correction

**Date:** 2026-09-26, Mac. **Scripts:** `diagnose.py`, `decompose.py`, `scan.py`.
**Base:** the k3l/k3m torch config, MLP 0, self-check **2.4074e-08** printed by every arm's `c = 0` row.
**Oracles:** the two independent 1e7 MC runs (`mc_oracle_0.npz`, `mc_oracle_0_s777.npz`).
New switches in `chain.py`: `varfix` (early, before the source loop) and `varfix_late` (late, only
`nonlin_slices` sees it) — both default `None`, and `ablate.py k4memo 3` still prints **3.7533e-08**.

## 1. Where the variance is wrong (`diagnose.py`)

`e_i(l) = (var_chain − var_true)/var_true`, pooled over the two oracles; the noise floor is |A−B|/2.

| l | rms e | mean e | slope | rms oracle noise | signal/noise |
|---|---|---|---|---|---|
| 0 | 3.17e-4 | −3.2e-6 | 0.9999 | 3.14e-4 | **1.0** |
| 4 | 3.77e-4 | −2.0e-5 | 1.0001 | 3.07e-4 | 1.2 |
| 8 | 7.45e-4 | +5.7e-5 | 1.0010 | 3.19e-4 | 2.3 |
| 12 | 1.31e-3 | +3.1e-4 | 1.0027 | 2.96e-4 | 4.4 |
| 15 | 1.85e-3 | +4.5e-4 | 1.0035 | 3.27e-4 | **5.7** |

- **Layers 0–6 are at the oracle noise floor**: at 1e7 samples we cannot even see the chain's variance error
  there. The measurable error is a *deep-layer* phenomenon that accumulates: rms 3.2e-4 → 1.85e-3, with a
  small systematic positive bias (the chain over-estimates var by +4.5e-4 at layer 15).
- **The pattern is barely predictable from cheap observables.** Least squares of e on
  (α, α², log var, K4d/var², t/σ³, Σ|D|/…, 1) gives **R² = 0.098 pooled** over layers 1–15 (per layer:
  0.05 at l=7, 0.13 at l=14). The strongest single feature is log var (coefficient +1.0e-3 pooled,
  +2.0e-3 at l=15).

## 2. The chain sits in an extremely sharp minimum in the variance (`scan.py`, MLP 0, late stage)

`var ← var(1+c)` at every layer — one scalar, a function of the current state only, so no staleness:

| c | final-layer MSE | vs base |
|---|---|---|
| −1e-3 | 5.17e-6 | ×0.005 (215× worse) |
| −3e-4 | 5.10e-7 | ×0.047 |
| −1e-4 | 8.51e-8 | ×0.283 |
| −3e-5 | 3.18e-8 | ×0.758 |
| **0 (base)** | **2.4074e-8** | **1.000** |
| +3e-5 | 2.55e-8 | ×0.945 |
| +1e-4 | 6.42e-8 | ×0.375 |
| +3e-4 | 4.48e-7 | ×0.054 |
| +1e-3 | 4.99e-6 | ×0.005 |

**MSE ≈ 2.4e-8 + 5·c²**, a parabola whose minimum is at **c ≈ 0 to within 3e-5**. A relative variance level
error of **1e-4 costs ×2.7**; 1e-3 costs ×200.

- **So the variance LEVEL is already calibrated to ≈1e-5 relative and there is nothing to win there.**
  This closes level corrections on the current base for the third time (cf. §37, §43).
- **And it explains the whole channel.** A *coherent* variance shift c moves every neuron the same way and
  its effect on the readout adds up; an *incoherent* per-neuron pattern is suppressed by √n. Our rms pattern
  error at depth is 1.8e-3, yet it costs only the ×1.54 gap — equivalent to a coherent error of
  c_eff = √(0.85e-8/5) = **4.1e-5**, i.e. **1.8e-3/4.1e-5 ≈ 44 ≈ √n × 1.4**. The suppression is exactly the
  incoherent-sum factor this project keeps meeting.

## 3. Therefore every cheap correction is harmful (`decompose.py`, MLP 0)

Corrections built from MLP 0's own oracle, fed back through `varfix_late` (`early` gives the same to 3 digits):

| arm | final MSE | vs base |
|---|---|---|
| base | 2.4074e-8 | 1.000 |
| "exact", stale multiplicative factors | 3.66e-8 | ×0.658 |
| level only (per-layer mean of e) | 4.06e-8 | ×0.593 |
| level + log var (2 numbers per layer) | 4.42e-8 | ×0.544 |
| full linear fit (7 features per layer) | 4.50e-8 | ×0.535 |
| CONTROL: |A−B|/2 (oracle noise) | 3.70e-7 | ×0.065 |

- **⚠️ The "exact" row is NOT the ×1.54 ceiling and must not be read as a contradiction of it.** A
  multiplicative factor measured on the base trajectory is **stale from layer 1 on**: once you touch the
  variance, the chain's own variance moves, so `g = var_true/var_chain_old` no longer lands on `var_true`.
  With a ×2.7-per-1e-4 stiffness, that drift alone is fatal. The oracle arm that gives ×1.54 substitutes the
  **absolute** value, which is trajectory-independent. Any future test of a correction must therefore be a
  **function of the current state**, which is also the only thing production could compute.
- **The smooth fits are worse than doing nothing**, and §2 says why: a function of (α, log var, …) is
  *smooth across neurons*, i.e. largely **coherent**, so it injects exactly the kind of error the readout
  amplifies, while the part of e it captures (R² ≈ 0.1) is incoherent and worth √n less.

## 4. Verdict

**The variance channel is closed to cheap corrections.** The ×1.54 lives in the incoherent per-neuron
residual, which (a) is invisible below layer 7 at 1e7 samples, (b) has R² ≈ 0.1 against every cheap
observable we have, and (c) cannot be approached with a smooth fit, because the coherent component of any
such fit costs more than the incoherent component it captures. To halve the pattern error — which would be
≈×1.36 on raw — needs a **structurally better covariance propagation**, not a fit: the error is the
transported error of the post-activation covariance (FRONTIER: driven by D, corr 0.993). Measuring *that*
needs a pair-slice oracle with N ≫ 1e7 (§50), which is the real blocker on this axis.

## 5. What is measured vs modelled

Everything above is **measured** (torch float64, baked MLPs, self-check 2.4074e-08 / 3.7533e-08 on every run).
Single-MLP caveat: §1–§3 are **MLP 0 only**; the `scan.py` parabola should be re-run on 10 MLPs before it is
quoted as a general constant, though the ×0.758 at c = −3e-5 leaves no room for a different conclusion.
