# Where k3l's remaining error lives — and why the old attribution method no longer resolves it
2026-09-25 evening, MLP 0, torch chain, base = `k3l_t` (self-check: 2.4074e-08, the logged k3l value).

Redone because the attribution in `2026-09-24_k3-chain/oracle_subst3.py` was measured on **k3f**, and
k3i + k3j + k3l have since taken raw from 2.80e-8 to 2.31e-8 (10 MLPs). A channel worth ×2 on k3f can
be worth ×1.05 now.

## 1. Oracle substitutions on the k3l base (`oracle_k3l.py`)

| exact quantity | final MSE | × vs k3l |
|---|---|---|
| — (k3l) | 2.4074e-8 | 1.00 |
| variance | 1.6880e-8 | 1.43 |
| κ₄ (diag) | 1.8457e-8 | 1.30 |
| κ₃ (diag) | 2.2051e-8 | 1.09 |
| κ₃ + κ₄ | 1.4261e-8 | 1.69 |
| **variance + κ₃ + κ₄** | **4.7941e-9** | **5.02** |
| D (κ₃ (2,1) slice) | 2.8404e-8 | 0.85 |
| C0 (off-diagonal) | 2.3332e-8 | 1.03 |
| C (full covariance) | 2.5047e-8 | 0.96 |
| C + D | 2.3846e-8 | 1.01 |
| C + D + κ₃ + κ₄ | 1.4685e-8 | 1.64 |

Read naively this says "exact second moments make the chain worse, so the errors must be cancelling".
**That reading is wrong**, and the next section is why. What survives is the last-but-one line of §3.

## 2. The sanity arm that broke it (`oracle_stage.py`)

`var_layers` substitutes the diagonal of `C_pre` late (only `nonlin_slices` sees it); `C_layers`
substitutes the whole `C_pre` early (the legs, D and K4d are built from it too). To tell an error in
the variance apart from an inconsistency between the two stages, `chain.py` gained `var0_layers`
(exact diagonal, early; default off). Sanity arm: `var0 + C0` substitutes exactly what `C` does, so
the two must agree.

| arm | final MSE | × |
|---|---|---|
| variance exact, LATE | 1.6880e-8 | 1.43 |
| variance exact, EARLY | 1.6869e-8 | 1.43 |
| C0 exact, EARLY | 2.3332e-8 | 1.03 |
| **var0 + C0 EARLY (must equal `C`)** | **1.6839e-8** | **1.43** |
| **C exact, EARLY** | **2.5047e-8** | **0.96** |

They disagree by 48%. The stage does not matter (1.43 late = 1.43 early). The whole difference is
**which array the diagonal comes from**:

* `mc_oracle_0.npz` (**N = 1e7**) → expected variance noise √(2/N) = **4.5e-4**
* `mc_oracle_all_0.npz` (**N = 4e6**) → **7.1e-4**
* measured difference of the two diagonals: **6.6e-4 relative** — the quadrature sum of the two
  noises (8.4e-4) to within the spread over layers. So they differ by **MC noise only**, with no
  systematic part, and that noise alone moves the final MSE by 48%.

## 3. The response curve (`var_sensitivity.py`)

Walking the line `d(t) = var_{1e7} + t · (diag C_{4e6} − var_{1e7})`, i.e. one unit of t = 6.6e-4
relative, purely in the noise direction:

| t | −1.0 | −0.5 | **0.0** (1e7) | **0.25** | 0.5 | 0.75 | **1.0** (4e6) | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|---|
| final MSE (e-8) | 4.454 | 2.618 | **1.687** | **1.561** | 1.660 | 1.986 | **2.538** | 4.321 | 7.009 |
| × vs k3l | 0.54 | 0.92 | 1.43 | **1.54** | 1.45 | 1.21 | 0.95 | 0.56 | 0.34 |

A sharp bowl, minimum at **t ≈ 0.25**, floor **1.56e-8 (×1.54)**.

**The minimum lands where statistics says the truth is.** For two unbiased estimates with noises
σ₁ = 4.47e-4 and σ₂ = 7.07e-4, the maximum-likelihood combination sits at
t* = σ₁²/(σ₁²+σ₂²) = **0.29**. Measured minimum: **0.25**. So the final MSE is not merely sensitive
to the variance — along this line it is a *detector* of the true variance, and both oracles are
already off the bottom.

(The baked ground truth itself is not the limit here: the dataset was baked at 1e9 samples, so its
own noise contributes ~1.6e-10 to the MSE, two orders below the 2.4e-8 we are reading.)

## 4. What this changes

1. **Arms fed from `mc_oracle_all_0.npz` (C, C0, D) measure oracle noise, not a channel.** That
   file is N=4e6, its variance noise is 7.1e-4, and §3 shows what 6.6e-4 does. **"Exact C is worse
   than doing nothing" is an artefact, not evidence of error cancellation.** The arms fed from the
   1e7 file (variance, κ₃, κ₄) are reproducible — see §5, which measured that instead of assuming it.
2. **The variance is worth ×1.5, best estimate ×1.54** (at the ML-combined oracle, t = 0.25), and it
   is the largest single channel. It is **not** on CLAUDE.md's candidate list; the notes call the
   variance "right" because its *level* is (slope 1.0006) — its per-neuron *pattern* is what costs.
3. **κ₃ is nearly exhausted: ×1.09.** Candidate #1 in CLAUDE.md (the κ₃-driven part of G_off) feeds
   K31, i.e. the κ₄ channel, whose whole ceiling is ×1.30, and K31 is only a slice of it. The
   listed ordering "accuracy via κ₃ first" is stale.
4. **The three diagonals together are worth ≥8.2× while the product of the singles is ×2.31.** The
   channels are strongly super-additive — which is the mechanism behind this project's history of
   ~×1.05–1.15 per lever, and an argument for changes that move variance, κ₃ and κ₄ jointly.
5. **The scale that matters for the variance is 1e-4–1e-3 relative.** Any cheap correction of the
   variance pattern at that accuracy is worth more than a new κ₃ diagram. Conversely, a change that
   perturbs the variance slightly can swing the score by tens of percent in either direction — which
   is a warning for cost work that truncates transports.

## 5. Error bars: the same arms against two independent 1e7 oracles (`oracle_errorbar.py`)

A second oracle was run with a different seed (`mc_oracle.py 0 1e7 777`, 16 min; the script now takes
an optional seed and writes its own file). Both are N=1e7 estimates of the same truth, so the spread
between them is the honest uncertainty of every arm, and their mean ("pooled") has √2 less noise.

| arm | oracle A | oracle B | **pooled** | spread A/B |
|---|---|---|---|---|
| variance | 1.43× | 1.31× | **1.54×** | 1.09 |
| κ₄ (diag) | 1.30× | 1.29× | **1.31×** | 1.01 |
| κ₃ (diag) | 1.09× | 1.16× | **1.14×** | 1.07 |
| κ₃ + κ₄ | 1.69× | 1.70× | **1.76×** | 1.01 |
| **variance + κ₃ + κ₄** | 5.02× | 4.46× | **8.21×** | 1.13 |

**Every arm is reproducible (spread ≤ 1.13), so §1's single-channel numbers stand** — the noise
problem of §2 was specific to the N=4e6 file, exactly where it was diagnosed. The two oracles differ
on κ₃ by 48% and on κ₄ by 39% *relative* (max 939× and 680×), but that is division by near-zero on
the many neurons whose cumulants are tiny; in absolute terms that noise averages out of the readout,
which is why those arms are the most reproducible of all (1.01–1.07).

**The pooled column is the finding.** Halving the oracle noise does not nudge the arms, it moves
them: the joint arm goes 5.02 / 4.46 → **8.21**. The chain's final MSE is a detector of the true
cumulants (§3 said the same about the variance), so **every ceiling measured this way is a LOWER
BOUND set by the oracle, not by the method.**

* Exact diagonals are therefore worth **≥8.2×**, not the ×5.5 standing in FRONTIER: raw ≤2.93e-9.
  At today's 26.27% that would be an adjusted **7.7e-10** — 2.6× better than the current #1.
* Pooled ordering: **variance 1.54 > κ₄ 1.31 > κ₃ 1.14**, product 2.31 against 8.21 jointly. The
  super-additivity is extreme and it is the central fact about this chain.
* The variance's pooled 1.54 equals the bowl minimum of §3 (1.54 at t=0.25) exactly — two independent
  routes to the same number.

## 6. Caveats and the next measurement

* **One MLP (0).** Per-MLP spread in this project is ±15%, and the project's own rule is not to
  judge a 5–10% effect on few MLPs. The ×1.5 and ×5.0 are large enough to survive that; the ordering
  of κ₃ (1.09) against κ₄ (1.30) is not safe on one MLP.
* ✅ **Done:** two independent 1e7 runs, §5. What they show is that the *ceilings* are still
  oracle-limited (pooling two oracles moved the joint arm from ~4.7 to 8.2), so the honest statement
  is "≥8.2×", and a bigger oracle would raise it further. N ≈ 2e8 (~6 h of `mc_oracle.py`, or one
  cloud session at 1.6× the Mac's time) would settle it — worth doing only if the ceiling itself
  becomes the decision, since ≥8.2× already says the accuracy axis is nowhere near exhausted.

Scripts: `oracle_k3l.py`, `oracle_stage.py`, `var_sensitivity.py` (+ logs next to them). `chain.py`
gained one switch, `var0_layers`, default off — every existing configuration is bitwise unchanged.
