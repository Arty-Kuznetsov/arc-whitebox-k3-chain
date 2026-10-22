# A constant-free analytic mean correction: derivation, three probes, end-to-end result (2026-09-23)

Follows `RESULTS-layer-blame.md`, which established that the whole k=2 final-layer error is carried
by the propagated MEAN (fixing the covariance never helps and is up to 16% worse). This is the
attempt to fix the mean. Logs: `logs/diag_k3_probe.txt`, `logs/wick_regen_probe.txt`,
`logs/edgeworth_chain.txt`. Scripts: `diag_k3_probe.py`, `wick_regen_probe.py`, `edgeworth_chain.py`.

## The correction

For `z ~ N(mu, sigma^2)` the k=2 mean update `mu_post = mu*Phi(a) + sigma*phi(a)`, `a = mu/sigma`,
is exact. Its leading error in the pre-activation's own third cumulant is (derived here):

    mu_post ≈ mu*Phi(a) + sigma*phi(a) − kappa3_diag * a * phi(a) / (6*sigma^2)

`kappa3_diag[i] = kappa3_pre[i,i,i]` is a **vector**. Verified against 40M-sample Monte Carlo:
removes 61–87% of the Gaussian-closure mean error at `a != 0`, and vanishes identically at `a = 0`
(where the residual is a 4th-cumulant effect).

## Getting the diagonal: two routes, one works

### Route 1 — transport it (FAILS)
Exact: `kappa3_pre[i,i,i] = sum_abc W[a,i]W[b,i]W[c,i] kappa3_post[a,b,c]`. The memoryless
(diagonal-dominant) approximation keeps only `a=b=c`, one `O(n^2)` matvec. Fed the TRUE post
diagonal from the k=3 chain, it predicts the true pre diagonal one layer on with:

| layer | 1 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| correlation | 0.808 | 0.593 | 0.435 | 0.500 | 0.450 |
| **magnitude ratio** | 0.658 | 0.136 | 0.053 | 0.024 | **0.017** |

**At depth the diagonal term carries only 1.7% of the true magnitude.** The third cumulant is made
by off-diagonal cross terms, not by its own diagonal. This independently reproduces, and quantifies
more sharply, the "memoryless tensor on (2,1)/(3,) slices" dead end of forum write-up #18218.

### Route 2 — regenerate it from the covariance (WORKS, partially)
The cross terms are themselves a functional of the pre-activation mean and covariance, which the
k=2 chain already carries. Wick/cluster expansion with derivative coefficients
`d_k = E[d^k ReLU(Z)]` (`d1 = Phi(a)`, `d2 = phi(a)/sigma`, both checked against the reference
`wick.py` to 0.00e+00) gives, for the leading connected diagram,

    kappa3_post[a,b,c] ≈ d2_a d1_b d1_c C_ab C_ac + d1_a d2_b d1_c C_ab C_bc + d1_a d1_b d2_c C_ac C_bc

and the contraction with the next layer's weights collapses to **one matmul**:

    V = (Wm * d1) @ C                         # Wm[i,a] = W[a,i]
    kappa3_pre[i,i,i] ≈ 3 * sum_c Wm[i,c] * d2_c * V[i,c]^2

| layer | 1 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| correlation | **0.998** | 0.798 | 0.706 | 0.707 | 0.636 |
| magnitude ratio | **0.918** | 0.424 | 0.268 | 0.191 | 0.148 |
| relative RMS error | 0.104 | 0.709 | 0.833 | 0.876 | 0.913 |

Near-exact at layer 1 (the pre-activations there are exactly Gaussian, so the leading diagram is the
whole story), decaying with depth as accumulated off-diagonal history takes over. Adding the
transported diagonal on top changes the correlation not at all (0.763 either way).
**Favourable alignment:** the regeneration is most accurate exactly where `RESULTS-layer-blame.md`
found the per-layer blame is largest (layers 1–3, ≈7e-7 each), and degrades where blame is smallest.

## End-to-end (3 MLPs, baked N=1e9 GT, width 1024, depth 16)

Same chain code for all three arms; only the mean update differs.
**Self-check: the uncorrected arm reproduces the reference k=2 exactly (4.5018e-6 on MLP 0).**

| arm | final_layer_mse | all_layers_mse | vs k=2 | % of B | adjusted |
|---|---|---|---|---|---|
| no correction | 5.1494e-6 | 2.6955e-6 | 1.00× | 2.36% | 5.149e-7 |
| **Wick-regenerated (deployable)** | **3.0860e-6** | **1.5373e-6** | **1.67×** | **3.82%** | **3.086e-7** |
| oracle diagonal (upper bound) | 1.7257e-6 | 7.8680e-7 | 2.98× | — | — |

Per MLP the gain is consistent: 1.66× / 1.57× / 1.77×.

## What this establishes
1. **A deployable 1.67× on final-layer MSE, free on the metric.** The extra cost is 15 matmuls of
   `2n^3 − n^2` FLOPs, taking the estimator from 2.36% to 3.82% of budget — still under the 10%
   multiplier floor, where only accuracy counts. **Zero fitted constants; every quantity is derived.**
   For comparison, the published covariance + whitened-antithetic-MC blend (#18200) buys 1.33× and
   leans on sampling, which ARC states it values less.
2. **The regeneration captures 56% of what a perfect diagonal would buy** (1.67× of 2.98×).
3. **A hard ceiling on the whole direction: ~3×.** Even a perfect diagonal third cumulant with the
   exact first-order Edgeworth term gives only 2.98×, against the 110× of full k=3. So the k=3 gain
   does **not** live in the diagonal readout — it lives in what the transported full tensor does to
   the state through depth. Combined with the earlier finding that the covariance is irrelevant,
   this narrows where the remaining 40× must come from.

## Next
Port the Wick arm to `flopscope.numpy` and score it on the full public `mini` split (100 MLPs) for
the authoritative number and a real submission ID.
