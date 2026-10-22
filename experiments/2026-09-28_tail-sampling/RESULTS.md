# Unbiased sampling of the old tail: dead by arithmetic, κ ≈ 62 at n = 1024

**Date:** 2026-09-28, cloud session, branch `cloud/tail-sampling`.
**What was tested:** the crack honestly left open in NEGATIVE §58. There the only cheap construction was the
**biased** selection of the r heaviest columns (62.8% error at r=n), and it loses mass **coherently**, which is exactly
the mechanism of §16. **Unbiased importance sampling with reweighting** (p_c ∝ w_c, weight 1/(r·p_c)) is a different
estimator: it is O(n²), needs not a single optimisation pass and is **nested by construction**, so none of the three
objections of §58 applies to it. The objection was accepted; the idea was tested.

## Short answer

**The idea is dead, and dead on the very first number.** The stopping rule from the brief is "if κ/√n is above ~20%,
the idea is dead":

> **κ = 62.08, κ/√n = 194%** (n=1024, a0=1, layer 15). The threshold is exceeded **by a factor of 10.**

This is not an artefact of the choice of a0 or of the layer: κ = 55.9–62.1 over the whole grid
a0 ∈ {1,2,3,4} × layer ∈ {11,13,15}. The measurement confirms the formula to within 1–2% (§2), and the variance-reduced
variants (hybrid, systematic sampling) give at best 86.2% at r=2n, **17 times above the 3–5% budget**.

**The mechanism, and this is what is worth recording.** The coherence of §16 ("the tail is a sum of same-sign terms")
holds for the **diagonal readout t**, but the sampling variance is governed by the **Frobenius norm of the tensor**,
and in that norm the tail's terms are nearly orthogonal: κ/√R = 0.50, with the limits 1/√R ≈ 0.008 (perfect
coherence) and 1.0 (orthogonality). So the premise that made unbiased sampling look promising **does not hold in the
norm that actually sets the error.** Same-sign t and orthogonal T do not contradict each other: t is one diagonal of the
tensor, while sampling corrupts the whole tensor.

On top of that, **Stage A at a0=1 (§3, not measured before) makes the situation worse, not better:** the prize of
getting under the floor exists only at a0=1, and the tolerance there is ≈1% instead of 3–5% at a0=4. The gap between
what is needed and what is achievable is ×100.

The cost axis is closed for good: §54, §55, §56, §58 and now this.

---

## 1. STEP 1: one number (`kappa.py`, `logs/kappa.jsonl`)

For T = Σ_c S(a_c,b_c), S(a,b) = Sym(a⊗b⊗b), unbiased sampling of r columns with p_c = w_c/W gives
```
E‖T̂ − T‖² = (W² − ‖T‖²)/r,   W = Σ_c w_c,   w_c = √( (1/3)|a|²|b|⁴ + (2/3)(a·b)²|b|² )
rel. error = √(W² − ‖T‖²) / (√r ‖T‖)  ≈ κ/√r,  κ = W/‖T‖.
```
`w_c` is computed in O(n) per column (three per-column contractions), i.e. **the whole step is O(n²)**, as promised.

Production configuration (`corr=split24rr`), n = 1024:

| layer | a0 | S | R | ‖T‖ | W | **κ** | κ/√R | r=n/2 | **r=n** | r=2n | r=4n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | **1** | 15 | 15360 | 5.572e+0 | 3.459e+2 | **62.08** | 0.501 | 274.3% | **194.0%** | 137.2% | 97.0% |
| 15 | 2 | 14 | 14336 | 4.916e+0 | 3.004e+2 | 61.10 | 0.510 | 270.0% | 190.9% | 135.0% | 95.5% |
| 13 | 1 | 13 | 13312 | 6.352e+0 | 3.906e+2 | 61.49 | 0.533 | 271.7% | 192.1% | 135.9% | 96.1% |
| 11 | 4 | 8 | 8192 | 4.362e+0 | 2.438e+2 | 55.89 | 0.618 | 247.0% | 174.6% | 123.5% | 87.3% |

(the full grid of 12 rows is in `logs/kappa.jsonl`; κ changes by only 10% over the whole grid)

**Reaching 5% needs r = (κ/0.05)² ≈ 1.5·10⁶ columns**, 100 times more than the R = 15360 columns the tail contains in
total. In other words, the sampling converges to "carry the whole tail", i.e. to no compression. This is not a matter of
tuning; it is the arithmetic of the estimator.

## 2. STEP 2: a measurement instead of the formula (`samplers.py`, `logs/samplers.jsonl`)

A formula that closes the last idea cannot be left unchecked. Real draws, with the error measured by the Gram
estimator (the same one that was cross-checked against the explicit tensor to 3.6e-15), n=1024, a0=1, layer 15,
3 seeds:

| r | (a) IS with replacement | (b) hybrid top r/4 | (b) hybrid top r/2 | (c) systematic | prediction |
|---|---|---|---|---|---|
| n/2 | 271.7% | 275.4% | 302.9% | 261.7% | 274.3% |
| **n** | **191.7%** | 175.2% | 176.0% | 173.0% | **194.0%** |
| 2n | 138.8% | 101.4% | **86.2%** | 105.7% | 137.2% |
| 4n | 95.9% | 53.1% | **39.0%** | 60.0% | 97.0% |

* **The formula is confirmed:** (a) reproduces the prediction to within 1–2% at all four ranks.
* **The variance-reduced variants work but do not rescue the idea.** The hybrid helps precisely because the weights
  are concentrated: **the top 1024 terms hold 34.6% of W, the top 4096 hold 77.4%** (uniform weights would give 6.7%
  and 26.7%). But the best result within the floor (r ≤ 2n) is **86.2%** against a budget of 3–5%: **a miss by a
  factor of 17.**
* Extrapolating the hybrid (×2.2 per doubling of r) to 5% gives r ≈ 28n ≈ 2·R, again "carry everything and as much
  again".

**The stopping rule fired:** the cheap sampler does not reach ~5% at r ≤ 2n. Step 3 (compounding) was not run: there
is no point in measuring how an error accumulates when it is already 86–194% on a single merge.

## 3. STEP 4: Stage A at a0 = 1, 2, 3 (not covered in §58)

At a0=1 the "tail" is ALL the sources, so the merge error there is the error of the whole κ₃ state.
10 baked MLPs, base 2.3135e-08 (`logs/stage_a_low*.jsonl`).

| arm | **a0=1** | a0=2 | a0=4 (§58) |
|---|---|---|---|
| `exact`: self-check of the separate accumulation | **×1.000** | ×1.000 | ×1.000 (and a0=3 ×1.000) |
| coherent error 1e-3 (`scaleTD`) | ×1.003 | — | ×1.000 |
| coherent 1e-2 | **×1.051** | ×1.021 | ×1.001 |
| coherent 1e-1 | **×4.028** | — | ×1.471 |
| incoherent 1e-3 (`noiseTD`) | ×1.000 | — | — |
| incoherent 1e-2, 2 seeds | **×1.018 / ×1.022** | ×1.014 | ×1.006–1.011 |

**The budget at a0=1 is ~5 times TIGHTER than at a0=4:** for ≤3% raw it is ≈0.7% coherent or ≈1.5% incoherent
error (against 3–5% at a0=4). This makes things worse for sampling, not better: **the prize (getting under the
floor) exists only at a0=1, and that is exactly where the tolerance is narrowest.** The resulting gap between what is
needed (≈1%) and what the cheapest unbiased estimator gives at r ≤ 2n (86%) is **about ×100.**

`exact` at a0 = 1, 2, 3 gives exactly ×1.000, i.e. the separate accumulation of t_old/D_old is correct on the full set
of sources as well; the `tailx` switch is verified in this mode too.

## 4. The prize all this was for (`cost_merge.py`, a0 = 1 and 2, which the brief asked to add)

| a0 | r | m=0 | m=1 | m=2 | m=4 |
|---|---|---|---|---|---|
| **1** | n | **5.85%** | **7.31%** | **8.78%** | 11.71% |
| 2 | n | 7.82% | 9.19% | 10.55% | 13.29% |
| 3 | n | 10.38% | 11.65% | 12.92% | 15.46% |
| 4 | n | 12.69% | 13.87% | 15.04% | 17.38% |

At a0=1 and m ≤ 2 the merge goes **under the 10% floor**, i.e. adjusted = raw × 0.10 = **1.92e-9** at the raw ceiling
of 2026-09-28_ceiling (rank 2–3). The prize was real and much larger than the 3.0e-9 that the old brief had at
a0=4; **this is exactly why the check was worth it, and exactly why the negative answer here is valuable.**

## 5. What is measured, what is derived

**Measured (torch float64, baked MLPs, production configuration):** κ and the whole grid of §1; all errors of §2,
which are real draws evaluated in Gram form; the weight concentration; §3.
**Derived (and checked by measurement):** the variance formula for unbiased sampling; it matched (a) to 1–2%.
**Model:** util in §4 (`cost_merge.py` on top of the calibrated `cost.py`).
**Not touched:** `chain.py` did not change in this session (empty diff against `main`), so both gates are the same as
on main; `CLAUDE.md`, `notes/*.md`, `src/` were not edited; nothing was submitted to AIcrowd.

## 6. Files

| file | what |
|---|---|
| `kappa.py` | STEP 1: κ and the predicted sampling error, O(n²) |
| `samplers.py` | STEP 2: real draws (a)/(b)/(c) + weight concentration, error via Gram |
| `stage_a_low.py` | STEP 4: Stage A at a0 = 1, 2, 3 |
| `logs/` | `kappa.jsonl`, `samplers.jsonl`, `stage_a_low*.jsonl`, gate |
