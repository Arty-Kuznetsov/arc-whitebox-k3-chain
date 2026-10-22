# Saturated-ROW cut of the D pairing — naive version fails; the κ₄-level restoration is untested

**Date:** 2026-09-29, Mac. **Script:** `scan.py`. **Switch:** `chain.py satrow=θ` (default None; `k4memo 3` still
3.7533e-08): before the nonlinear step, rows i of D (= κ₃[i,i,j]) with α_i > θ at the CURRENT layer are zeroed — in
production those rows of the D pairing Σ_s L1_s B_sᵀ + L2_s A_sᵀ would simply not be computed (legs untouched).
10 baked MLPs, shippable k3m config (`deadcut -2.5`), base **2.3462e-08**.

**Why.** Proposed independently by two lenses of the ideation workflow (convergence counts as evidence): for a
saturated row, D enters the step only with weights that vanish there (d2_i ≈ 0, d3_i ≈ α φ(α)) and in K21 where the
pass-through part cancels against what the legs carry. Saving = saturated fraction × the D-pairing share (10.6% of B).
Not the same object as the saturated *column* cut of `2026-09-29_colcut/` (which cost ×1.12): this cuts rows of D at
the layer where D is read.

| arm | raw × base | better | saturated rows | modelled saving |
|---|---|---|---|---|
| satrow 3.0 | **1.0094** | 6/10 | 7.9% | −0.84 pts of B |
| satrow 2.5 | 1.3436 | 0/10 | 11.3% | −1.20 |
| satrow 2.0 | 3.0704 | 0/10 | 15.8% | −1.68 |
| satrow 1.5 | 10.977 | 0/10 | 21.7% | −2.29 |
| matched dead −2.25 (row+column, reference) | 1.0294 | 1/10 | — | — |
| matched −2.25 + satrow 2.5 | 1.3650 | 0/10 | 11.3% | — |
| matched −2.25 + satrow 2.0 | 3.0926 | 0/10 | 15.8% | — |

**Verdict (naive form): not a lever.** At 3.0 it is a wash (×1.0094 for −0.84 points ≈ ×0.975 projected, but 6/10 and
a per-MLP spread of ×0.84–×1.35 per the synthesis), and it falls off a cliff from 2.5. The ideation synthesis's
diagnosis: the κ₄ *level* is computed from sums over D, so zeroing rows shifts that stiff level for everyone — which is
why the proposal came with "restore the κ₄ level exactly in O(n²)". **That restored version has not been run**; it is
item 3/E2 of the cloud's port battery (`notes/CLOUD-BRIEF.md`). Kill rule there: at θ = 3.0 raw worse than +1.0% or
fewer than 9/10 better.
