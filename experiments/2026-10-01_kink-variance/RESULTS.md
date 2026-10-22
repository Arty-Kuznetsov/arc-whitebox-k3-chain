# Cloud #10 — the kink-core variance hunt (`notes/CLOUD-BRIEF.md`, re-pointed 2026-10-01) — RESULTS

Branch `cloud/kink-variance` from `origin/main` fa056a1 (contains #9A, #9B and the Mac's `mcorr` hook). Torch chain,
float64; FLOPs and MSE only (CLOUD.md §2). Harness: the tail-accuracy one (`kv.py` wraps `ta.py`, results in `res/`).
Oracles: per MLP 0 / 1 FOUR independent 1e7 MC runs (tail-accuracy seeds 1001 / 1002 + the Mac's default / 777 from
`../2026-09-24_k3-chain/oracles/`), pooled = 4e7.

## 0. Gates (every new switch off; `logs/q_*`, `res/<base>/`)
| gate | required | measured on this branch |
|---|---|---|
| `ablate.py k4memo 3` | 3.7533e-08 | **3.7533e-08** (again after the task-2 patch: 3.7533e-08) |
| `ablate.py k3l_t 10` | 2.3135e-08 | **2.3135e-08** |
| k3m BASE, 10 MLPs | 2.3462e-08 | **2.34616e-08** |
| B9 = k3m + matched -2.5 + E2 2.5 sep | 2.1967e-08 | **2.19667e-08** |
| FULL = #8's t4_final25a | 2.23915e-08 | **2.23915e-08** |

## 1. Level vs pattern of the kink-core variance error (task 1; `kv1.py`, `res1/decomp.json`, `res1/arm_*.json`)
Core = |alpha_chain| < 1 on B9, layers 1-15; e_i(l) = var_chain - var_exact (pooled 4e7). (a) level c(l) = <e, v> / <v, v>;
(b) a linear model of the rest on 8 standardized cheap features (var, mu, alpha, t, K4d, incoming and outgoing weight
norms, alpha-rank) + intercept; (c) the residual. The oracle's own noise (variance across the 4 seeds / 4) is
subtracted from ||e||^2 to get the signal energy.

| MLP | ||e||^2 | oracle noise | level share (of signal) | pattern share (of signal) | level + pattern | relative error L1 -> L15 |
|---|---|---|---|---|---|---|
| 0 | 3.086e-04 | 1.713e-04 (55.5%) | 4.6% | 7.3% | **11.9%** | 2.2e-4 -> 1.5e-3 |
| 1 | 3.249e-04 | 1.674e-04 (51.5%) | 4.4% | 5.8% | **10.2%** | 2.2e-4 -> 1.5e-3 |

Per layer (where the signal dominates, L >= 9): level 4-13%, pattern 2-12%, residual 83-93%. The level c(l) is
positive at depth (the chain's core variance is too large by ~3e-4 relative). **Below the brief's 25% kill line.**
Because energy share is not effect (NEGATIVE 57: a coherent level error is amplified by the readout, an incoherent one
discounted by ~sqrt(n)), the chain arms were run anyway — absolute values substituted for the core neurons at layers
1-15 (`oracle` var with `regime`): full = var_exact; level = var_chain_base - c(l) var_exact; lp = var_chain_base -
(level + pattern):

| arm (core variance) | MLP 0 raw x B9 | MLP 1 raw x B9 |
|---|---|---|
| **exact (full oracle, 4e7)** | **0.5708** | **0.7690** |
| exact on the whole kink abs alpha <= 2.5 | 0.6098 | 0.7928 |
| **level only (a per-layer scalar)** | **1.0635 (worse)** | **1.0086 (worse)** |
| level + pattern | 1.0749 (worse) | 1.0104 (worse) |

**Closed on that number: a per-layer scalar recovers nothing of the variance channel — it is harmful (MLP 0 x1.063,
MLP 1 x1.009) — and the smooth pattern makes it worse still.** The whole gain of the exact core variance (x0.57 /
x0.77, larger than #9A's 2-seed x0.64 / x0.82 because the 4e7 oracle is less noisy) lives in the rough residual,
orthogonal to everything per-neuron and cheap: the same wall as NEGATIVE 70 found for kappa3 / kappa4, now for the
variance. Two more facts: the exact variance of the CORE alone beats the exact variance of the whole kink (x0.571
vs x0.610, x0.769 vs x0.793): giving the shoulder neurons an exact but inconsistent variance costs — the consistency
mechanism of #9A (saturated exact variance x1.02-1.09) again; and a partial correction in the right direction is not
monotone here (the level is in the right direction and still hurts), unlike §70's s = 0.25 scaling of the WHOLE
error: what matters is the error's rough structure, not its size.

## 2. What term is the pair expansion missing? (task 2; `chain.py` `pairnew`, `marg3`; `kv_grid.T2`)
Next-order pair diagrams from the bivariate cumulant exponential E[f g] = E_marg[exp(sum k_ab d_x^a d_y^b / (a! b!)) f g]
(k_11 = C0, k_21 = D, k_31 = K31, k_22 = K22): C0.K31 (k11 k31 / 6, a4 b2), D^2 (k21^2 / 8, a4 b2), D.D^T (k21 k12 / 4,
a3 b3), K22^2 (/32, a4 b4), D.K22 (/8, a4 b3), C0^2.D (/4, a4 b3), C0^2.K22 (/8, a4 b4), and marg3 (the k = 3 coefficient's
own marginal correction; `relu_wick` extended to k = 7, new keys only). Off = bitwise (`selfcheck_t2.py`). All are
Hadamard products of matrices the step already holds: O(n^2), zero products.

| arm | B9 (10) | FULL (10) | B9, 32 fresh | FULL, 32 fresh |
|---|---|---|---|---|
| **C0.K31** | 0.9893 (7/10) | 0.9889 (7/10) | **0.9922 (25/32)** | **0.9922 (26/32)** |
| **D^2** | 0.9936 (7/10) | 0.9930 (7/10) | 0.9947 (21/32) | 0.9948 (20/32) |
| C0.K31 + D^2 | 0.9896 | 0.9886 | — | 0.9938 (19/32) |
| D.D^T / K22^2 / D.K22 / C0^2.D / C0^2.K22 / marg3 | killed (1.0017 / 1.0001 / 0.9998 / 1.0017 / 1.0008 / 1.0016) | | | |
| all seven together | 0.9924 | — | | |

**The three-way comparison on the shipping base** (32 fresh MLPs 10-41, x FULL): **FULL + marg2 0.9842 (28/32)** vs
FULL + C0.K31 0.9922 (26/32) vs FULL + D^2 0.9948 (20/32) vs FULL + C0.K31 + D^2 0.9938 (19/32); and on top of marg2
the pair is x1.003 (screen) / **x1.008 (fresh, 16/32)**, the singles x0.9942 / x1.0023 (screen, 10). **marg2 is the
best substitute; no shipping change.**

## 3. The consistency audit (task 3; `chain.py` `k4cons`, `k2pair`; `kv_grid.T3`)
Off = bitwise (`selfcheck_t3.py`). Every arm killed on B9 AND on FULL, so none reaches the FULL + marg2 check:

| arm (one object entering twice, forced to agree) | B9 | FULL |
|---|---|---|
| K4d's kappa4 level := K22's level (`k4d_from_k22`) | 1.0000 (4/8, killed) | 1.0000 (4/8) |
| K22's level := K4d's (`k22_from_k4d`) | 1.0003 (killed) | 1.0002 |
| the scale-mixture v of K31 and the feed := K22's (`v_from_k22`) | 1.0000 (killed) | 1.0000 |
| next variance = the pair expansion's own diagonal (`k2pair`, Hermite series to k = 3) | 3.9e4 | 3.8e4 |
| k = 1 marginal correction off in Q22 only | 1.0026 (killed) | 1.0027 |
| k = 1 marginal correction off everywhere (consistently) | **1.664** | 1.649 |

The kappa4 level is already consistent between its two paths (forcing either onto the other moves nothing), the
truncated Hermite variance is unusable (ReLU's series converges far too slowly at the kink), and the k = 1
correction is essential in every channel — the opposite of k = 2.

## 4. The order in 1/n (task 4) — DROPPED for budget
Baking ground truth at three new widths does not fit the remaining credit, and after tasks 1-2 the answer would be a
write-up statement rather than a lever (`notes/CLOUD-CONTINUE.md`).

## 5. Verdict
(a) **Best shipping stack: FULL + marg2 = k3r + E3, projected 3.909e-9 (3.58e-9 with L5, which is not shippable on wall
time, NEGATIVE 69)**, honest recipe: fresh-MLP ratio x0.9842 (28/32) on FULL's fresh-scaled raw, meter-anchored util
18.95% (`../2026-09-30_tail-accuracy/frontier.py`). Nothing from tasks 1-3 improves it.

(b) **The mechanism of task 2.** Three independent corrections — the Gaussian k = 2 Wick coefficient (marg2), the C0.K31
diagram and the D^2 diagram — each improve the chain by 0.5-1.6% alone and are mutually exclusive: every one of them
acts through the k = 2 leg of the pair expansion (coef(2,p), E[f''] = the density at the kink), and once one is in,
the others double-count (x1.008). So the expansion has ONE defect in its k = 2 channel — the first-order marginal
Edgeworth correction of the second-order coefficient over-corrects, consistently in Q11 and Q21 — and the "missing
diagrams" are partial projections of the same correction; the cleanest and strongest form is to drop the correction
(marg2).

(c) **Nothing free is left in the pair expansion.** Across this session and #9A/#9B: 149 tail restrictions, 53 catalog
switches, 7 next-order diagrams + marg3, 6 consistency arms — the only free gain is the k = 2 channel (marg2, shipped).
What remains (exact kink-core variance x0.57 / x0.77, joint moments x0.12-0.14) lives in a rough, high-rank residual
that no per-layer scalar (x1.06, harmful), no smooth per-neuron model (NEGATIVE 70) and no O(n^2) diagram of the
expansion reaches: it needs pair-level information per neuron, i.e. the n^3 wall of NEGATIVE 54-59.
