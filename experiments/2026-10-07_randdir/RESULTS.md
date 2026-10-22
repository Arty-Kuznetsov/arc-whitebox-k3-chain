# Random directions: the κ₃ analogue of the random-projection theorem (2026-10-07, Mac; cloud #12 "task C", never run before)

**Question (write-up §6.4, Propositions 1–2).** Layer l's pre-activation moments are projections of layer l−1's
post-activation joint cumulants through the column w_i of W_l, which is independent of them. The chain computes them the
same way from its own state, so its per-neuron errors are the same projections of its joint errors: for the variance a
coherent level (2/n)·tr ΔC plus i.i.d. noise; for κ₃ no level at all (odd moments vanish). The parity test
(`../2026-10-02_parity`) checked the means. This test checks the **spread**: read the same post-activations along
random directions V_l (i.i.d. N(0, 2/n), independent of the network) both in the chain (W_l replaced by V_l) and in a
Monte-Carlo pass (same samples as the real columns, 10⁷ × 2 seeds), and compare the real columns' errors with the
random directions' errors — the prediction is i.i.d. with that variance, and the same mean (the coherent part).

**Instrument.** `randdir_mc.py` (MC, float32 forward / float64 moments, MLP 0, layers 4, 8, 11, 14, seeds 1001/1002),
`randdir.py` (chain B9 = `pb.matched(-2.5, satrow=2.5, satrow_k4="sep")`, one run per target layer with W_l := V_l;
noise = half the seed difference, subtracted from the energies). Output `res/randdir_0.json`.

## Result [M] (`res/randdir_0.json`, `res/chain_run.log`; MLP 0, chain B9, MC 10⁷ × 2 seeds; n = 1024 real columns vs 1024 random directions)
Errors e = chain − MC (seed average); noise ν = half the seed difference; spreads are noise-subtracted standard deviations about
the mean; z = mean / √(mean(e²)/n); Δz = (mean_real − mean_rand) / combined SE; "signal" = 1 − noise energy / raw energy (real).
| l | order | mean real (z) | mean rand (z) | Δz | sd real | sd rand | **ratio** | signal | q01/q99 real | q01/q99 rand |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | var | 2.5e-5 (3.2) | 2.9e-5 (3.9) | −0.32 | 1.48e-4 | 1.24e-4 | 1.20 | 0.36 | −2.39 / 2.35 | −2.29 / 2.26 |
| 4 | **κ₃** | −4.5e-6 (−0.31) | 1.0e-5 (0.70) | −0.71 | 3.59e-4 | 3.69e-4 | **0.97** | 0.60 | −2.41 / 2.27 | −2.33 / 2.26 |
| 4 | κ₄* | −1.4e-4 (−3.1) | −1.5e-4 (−3.3) | +0.19 | 1.28e-3 | 1.36e-3 | 0.95 | 0.87 | −2.62 / 2.25 | −2.62 / 2.14 |
| 8 | var | 2.7e-5 (3.8) | 1.3e-5 (1.9) | +1.37 | 2.01e-4 | 1.99e-4 | **1.01** | 0.81 | −2.35 / 2.62 | −2.28 / 2.43 |
| 8 | **κ₃** | 5.9e-6 (0.44) | 5.5e-6 (0.46) | +0.02 | 4.17e-4 | 3.73e-4 | **1.12** | 0.95 | −2.68 / 2.38 | −2.45 / 2.60 |
| 8 | κ₄* | −1.5e-4 (−6.8) | −1.1e-4 (−4.8) | −1.40 | 6.95e-4 | 7.07e-4 | 0.98 | 0.97 | −2.61 / 2.51 | −2.56 / 2.41 |
| 11 | var | 7.4e-5 (8.3) | 6.3e-5 (7.3) | +0.91 | 2.67e-4 | 2.58e-4 | **1.03** | 0.94 | −2.37 / 2.79 | −2.43 / 2.55 |
| 11 | **κ₃** | 3.3e-5 (2.00) | 2.5e-5 (1.54) | +0.33 | 5.23e-4 | 5.18e-4 | **1.01** | 0.98 | −2.72 / 2.81 | −2.41 / 3.20 |
| 11 | κ₄* | −1.5e-4 (−7.0) | −9.2e-5 (−4.6) | −1.96 | 6.63e-4 | 6.30e-4 | 1.05 | 0.99 | −3.30 / 2.65 | −3.18 / 2.44 |
| 14 | var | 9.4e-5 (8.3) | 8.6e-5 (7.8) | +0.53 | 3.44e-4 | 3.35e-4 | **1.03** | 0.97 | −2.32 / 3.00 | −2.23 / 2.89 |
| 14 | **κ₃** | 2.4e-6 (0.11) | −2.6e-5 (−1.19) | +0.89 | 7.34e-4 | 6.89e-4 | **1.06** | 1.00 | −2.67 / 2.61 | −2.78 / 2.73 |
| 14 | κ₄* | −1.6e-4 (−8.4) | −1.7e-4 (−8.2) | +0.32 | 5.75e-4 | 6.22e-4 | 0.92 | 0.99 | −3.43 / 2.14 | −3.62 / 2.06 |
\* κ₄ is a closure in the chain (3·var²-type), not a projection; shown for contrast only.

**MLP 1, same instrument** (`res/randdir_1.json`, `res/run_mlp1.log`):
| l | order | mean real (z) | mean rand (z) | Δz | sd real | sd rand | **ratio** | signal |
|---|---|---|---|---|---|---|---|---|
| 4 | var | 2.5e-6 (0.33) | 6.5e-7 (0.08) | +0.17 | 1.31e-4 | 1.33e-4 | 0.99 | 0.28 |
| 4 | **κ₃** | −4.2e-6 (−0.27) | 1.1e-7 (0.01) | −0.20 | 3.81e-4 | 3.70e-4 | **1.03** | 0.61 |
| 8 | var | 5.3e-5 (6.9) | 3.6e-5 (4.8) | +1.62 | 2.17e-4 | 2.12e-4 | **1.02** | 0.83 |
| 8 | **κ₃** | −1.5e-5 (−0.95) | 8.1e-6 (0.55) | −1.07 | 4.79e-4 | 4.59e-4 | **1.05** | 0.95 |
| 11 | var | 6.6e-5 (7.3) | 7.5e-5 (8.1) | −0.71 | 2.72e-4 | 2.77e-4 | **0.98** | 0.93 |
| 11 | **κ₃** | 3.7e-6 (0.21) | −1.6e-5 (−0.94) | +0.80 | 5.47e-4 | 5.25e-4 | **1.04** | 0.98 |
| 14 | var | 8.0e-5 (6.5) | 1.0e-4 (8.0) | −1.16 | 3.78e-4 | 3.83e-4 | **0.99** | 0.97 |
| 14 | **κ₃** | −4.0e-5 (−1.58) | 2.0e-5 (0.76) | −1.65 | 8.03e-4 | 8.38e-4 | **0.96** | 1.00 |
(κ₄, contrast: the same negative level on both sides, Δz ≤ 1.3 except layer 4 (3.3, noise), spreads 0.95–1.10.)

**Pooled over the two networks (8 layer–network pairs):** κ₃ spread ratios **0.96–1.12**, both means zero (|z| ≤ 2.0 real,
≤ 1.5 random; the two means within 1.7 SE of each other); variance at layers 8–14: spread ratios **0.98–1.03**, the same
coherent level on both sides (z 1.9–8.3; means within 1.6 SE); layer 4's variance error below the noise floor on both
networks (signal share 0.36 / 0.28).

**Reading.**
1. **κ₃ (Proposition 2, now with its spread).** Over real columns and over random directions the mean error is zero
   (|z| ≤ 2.0 real, ≤ 1.5 random; the two means within 1.7 SE of each other) and the noise-subtracted spreads agree:
   ratios 0.97, 1.12, 1.01, 1.06 (MLP 0) and 1.03, 1.05, 1.04, 0.96 (MLP 1) at layers 4, 8, 11, 14, with the standardised
   1 %/99 % quantiles within 0.1–0.4 of each other. The per-neuron κ₃ errors are i.i.d. draws from the law of the chain's error along a random direction, with
   that law's variance — the full statement of the proposition, not only "no coherent level".
2. **Variance (Proposition 1).** Both sides carry the same coherent level (z 1.9–8.3; the means within 1.6 SE) and the
   spreads agree to 0.98–1.03 at layers 8–14 of both networks. At layer 4 the variance error is below the oracle's noise
   floor (signal share 0.36 / 0.28), so its ratios (1.20 / 0.99) are not measurements.
3. **κ₄ (contrast).** The closure's error has the same negative level along real columns and random directions (Δz ≤ 2),
   as a bias of the joint state must; its spread also matches (0.92–1.05). The projection law is a property of the state
   plus independent weights, not of the chain's arithmetic.
4. **Instrument.** The MC along random directions costs two extra products per target layer in the same pass (10⁷ samples,
   ≈ 10 min per seed on 3 threads); the chain side is one float64 run per target layer with W_l := V_l (≈ 1.5 min each).
   Tails are heavy (kurtosis 4–11, 30–48 at layer 14 for κ₃ on both sides), so spread ratios carry a few per cent of
   sampling scatter of their own; the agreement is within that.
