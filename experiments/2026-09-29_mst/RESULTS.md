# Slice-Markov tail (MST): closed, the old tail is exactly as "off-slice" as everything else

**Date:** 2026-09-29, cloud session, branch `cloud/mst`.
**What was tested:** the door that §62 left open in so many words: "this does NOT close a closure with a
larger state". MST is that door: sources **younger than a0** are carried exactly (with their legs), at age
a0 a source is added through its exact slices into the tail state (τ, Δ), and from then on the tail moves
**only by the exact repeated-index rule** (the t term and the D term from §1 of §62). Only the all-distinct part
**of the old tail** is lost, not that of all sources as in §4 of §62.

## Short answer

**Closed by the brief's stopping rule: worse by ×2 at any a0.** And the cause has been measured:

> **f, the share of the all-distinct part in the old tail ITSELF, is 27.5–28.2% on layers ≥8 (10 MLPs, pooled 27.76%,
> n=320) and almost independent of a0.**
> For all sources together §62 measured 28.8%. The brief's hypothesis ("fresh content is less slice-like, so
> the old tail's own share may be much smaller") is **wrong**: the old tail is exactly the same.

The end-to-end curve is monotone and **strictly dominated by k3m at every point**:

| a0 | raw ×base | util (model) | adjusted at graded raw 2.1652e-8 |
|---|---|---|---|
| 1 (anchor §62) | ×27.16 | — | — |
| 2 | ×18.75 | **9.57%** (under the floor) | 4.060e-8 |
| 3 | ×12.38 | 12.63% | 3.385e-8 |
| 4 | ×8.14 | 15.44% | 2.721e-8 |
| 5 | ×5.33 | 18.00% | 2.077e-8 |
| 6 | ×3.69 | 20.32% | 1.623e-8 |
| 8 | ×1.89 | 24.21% | 9.888e-9 |
| 10 | ×1.20 | 27.10% | 7.035e-9 |
| 12 | ×1.02 | 29.00% | **6.380e-9** |
| ∞ (= base of this configuration) | ×1.00 | 29.81% | 6.455e-9 |
| — | — | — | **k3m today: 5.3906e-9** |

The multiplier floor is reached only at a0=2 (util 9.57%), where raw is already ×18.75 → adjusted 4.06e-8, i.e.
7.5 times worse than k3m. The curve has a **very shallow optimum at a0≈12: 6.380e-9 against 6.455e-9 for its own
base, ×0.988**. That is 1.2%, well below the resolution of the instrument (§60: nothing worth less than 2–3% can be
judged by a submission), and still 1.2 times worse than k3m, which has a different (better) configuration:
`deadcut=−2.5`, util 24.835%.

⚠️ I wrote the row "limit = k3m" in the first version of this report incorrectly: the limit at a0→∞ is the base of
**this** configuration (`deadcut=None`, util 29.81% → 6.455e-9), while k3m is a separate configuration with deadcut.
The conclusion does not change (MST is dominated at every point), but the numbers in the table have now been
recomputed.

## 1. Self-checks: three anchors, all exact

| check | result |
|---|---|
| transport: M = Wᵀdiag(d₁) reproduces the chain's own leg transport | **≤1e-10** (hard `raise` in the code) |
| **a0 = 99** (no source graduates) == base | **×1.0000** |
| **a0 = 1** (everything in the tail, only the births exact) == mode `td` of §62 | **6.8521e-07 = ×27.1640**, matched §62 to the last digit |
| cost model at a0=None == `cost.util` | 29.813% vs 29.813% |

⚠️ **The a0=1 anchor first gave ×28.06 instead of ×27.16 and caught a bug.** I was storing Δ with a nonzero
diagonal in the tail state. The diagonal of the carried (2,1)-slice is the carried diagonal, which τ already carries,
and the repeated-index rule is derived with D_jj = 0; otherwise the j=k term is counted twice. Fixed; after that the
anchor is exact. Without this anchor the bug would have gone unnoticed as "MST is a bit worse than expected".

## 2. Measurement 1: f(a0, layer), the old tail's own share (`mst.py diag`, prod, **10 MLPs**)

| a0 | layers 1–7 | **layers ≥8** |
|---|---|---|
| 2 | 38.4% | **28.23%** (25.9–33.6) |
| 3 | 38.1% | **27.78%** (25.2–31.6) |
| 4 | 39.6% | **27.53%** (24.7–32.2) |
| 5 | 42.1% | **27.49%** (24.2–32.6) |

Pooled over all a0 on layers ≥8: **27.76%** (n=320). The per-MLP means are 26.7–29.3%: **the spread between MLPs is
smaller than the distance to any useful threshold**, so this is not the noise of three networks.

Flat in a0 and equal to the overall number of §62 (28.8%). **So the 35–61% in layers 1–3 from §62 is an effect of
the shallow layers (few sources), not a "young/old" difference.** The hypothesis the brief rested on is refuted
directly.

<small>The first run was on 3 MLPs (`logs/diag_prod.txt`): 27.99 / 27.58 / 27.29 / 27.22%, the same numbers to within
0.3 pp, but the brief required 10 MLPs, and the verdict rests on exactly this number, so it was re-measured on the
full 10 (`logs/diag_prod10.txt`, `logs/mst_diag_prod10.jsonl`).</small>

## 3. Why the curve is still monotone although f is constant

Restricting the rule to the old sources does not reduce the error **per transport** (it is the same 28%); it reduces
**the number of approximate transports** that the content of each source goes through: for the first a0−1 layers
the source is exact. Hence the monotonicity in a0, and hence also the absence of a gain: for the error to become
tolerable, almost all sources have to stay exact (a0≈10), and then the cost is almost the same as before. No
compromise exists, because **cost and accuracy are driven by one and the same parameter, in the same direction**.

## 4. Verdict

**The "larger state" door that §62 left open is closed.** MST is its most natural form: the state
(t, D) + exact young sources, i.e. exactly "(t,D) plus any number of additional O(n²)/O(n³) objects".
It does not beat k3m at any point, and the mechanism is the same as in §62: what is missing is orthogonal to the
slices, and its share does not depend on whose it is, a young source's or an old one's.

Together with §54–§59 and §62 this means that the cost axis is closed not by five accidents and not by one
principle, but by **a principle tested at its own weakest point.**

## 5. What is measured, what is a model

**Measured (torch float64, baked MLPs, n=1024, 10 MLPs both for end-to-end and for f):** all `raw ×` and all `f`.
The prod base = 2.3057e-08 (this is k3l_t with `deadcut=None`; the k3l_t gate = 2.3135e-08 at deadcut=−3.0, ratio
1.0034, consistent with "deadcut off costs ×1.002–1.005").
**Model:** the `util` column (`cost_mst.py` on top of the calibrated `cost.py`; the tail is priced at 3 dense
products per layer, as in the brief). The model's self-check is the exact reproduction of `cost.util` at a0=None.
**Not touched:** `chain.py` did not change (everything through a monkeypatch), `CLAUDE.md`, `notes/*.md`, `src/`
were not edited; nothing was submitted to AIcrowd.

## 6. Files

| file | what |
|---|---|
| `mst.py` | `diag`: measurement 1; `e2e`: the whole MST chain; three anchor self-checks |
| `cost_mst.py` | the cost of MST in `cost.py` units, self-check against `cost.util` |
| `logs/` | `e2e_prod.txt`, `e2e_hi.txt`, `diag_prod10.txt` (10 MLPs, main), `diag_prod.txt` (first screen on 3), `anchor.jsonl`, `mst_*.jsonl` |
