# Cost-unconstrained accuracy — the ceiling moves from 1.92e-9 to 1.724e-9

**Date:** 2026-09-28, Mac. **Scripts:** `sweep.py`, `combine.py`, `deep.py`, `k4b_ship.py`.
**Base:** the ×1.11 stack of `2026-09-28_ceiling/` (knobs + `allages` + `thin192` + `deadcut` off) =
**2.0844e-08** on 10 baked MLPs, reproduced as the self-check of all three runs.
**Conversion used throughout:** grader raw = torch × 0.920; at the 10% multiplier floor the score is that × 0.10.
Board 2026-09-28: **#1 J2W 1.70e-9** (raw 1.62e-8, util 10.5%) · #2 1.90e-9 · #5 2.20e-9.

## Why this sweep

Many terms here were rejected with the words *"does not pay"*, which always meant a **cost/benefit at 25% of
budget** — never an accuracy verdict. At the floor that judgement is void. So every one of them was re-opened.

## 1. What is alive and what is dead on ACCURACY (`sweep.py`, on top of the stack)

| arm | raw | × | better |
|---|---|---|---|
| **`pathD="full"`** (we ship `"merge"`) | 2.0332e-8 | **×1.0252** | 7/10 |
| **`k4k3=1.0`** (κ₄→κ₃ coupling) | 2.0528e-8 | ×1.0154 | **9/10** |
| **`k4b=2`** (κ₄ birth depth) | 2.0564e-8 | ×1.0136 | 7/10 |
| `thin384` | 2.0680e-8 | ×1.0079 | **9/10** |
| `corr="split1024rr"` (exact) | 2.0709e-8 | ×1.0066 | 8/10 |
| full κ₄ birth parts (NEG §47) | 2.0831e-8 | ×1.0006 | 5/10 |
| `q22x` = 1 / 2 | 2.0845 / 2.0837e-8 | ×1.000 | 4/10, 7/10 |
| `tet=4` | 2.0857e-8 | ×0.9994 | 2/10 |
| `edge2` (2nd-order Edgeworth) | 2.1242e-8 | ×0.981 | 2/10 |
| **`e3` births (NEG §45)** | 2.2081e-8 | **×0.944** | 0/10 |
| **`aug` (AUGMENT dense diagrams)** | 4.4998e-8 | **×0.463** | 0/10 |

- **`aug` and `e3` are now closed on ACCURACY, not on cost.** `aug` more than doubles the error: its dense
  diagrams double-count what we already derived independently (`c0d`, `pairx`, `k31pat`, the path births).
  This updates §45 — the E3 verdict was "a real mechanism that does not pay"; on the modern base it is simply
  wrong by 5.6%.
- **`corr="split1024rr"` is WORSE than `split384rr`.** The randomized correction-leg rank is *non-monotone*
  past 192: the truncation is acting as regularisation, not only as an approximation. Anyone raising `thin`
  should scan it rather than assume more is better.

## 2. The optima and the combination (`combine.py`, `deep.py`)

`k4k3` has an interior optimum near **0.5** (0.5 → ×1.0396, 1.0 → ×1.0314, 1.5 → ×1.0017, 2.0 → ×0.954).
`k31mult` moves from 0.65 to **0.55** (0.45 → ×1.0718, 0.55 → ×1.0776, 0.75 → ×1.0395).
`k4b` saturates slowly: 3 → ×1.0721, 4 → ×1.0845, 5 → ×1.0878 — **all on 10/10 MLPs**.

| configuration | raw | ×stack | better | **floor** |
|---|---|---|---|---|
| stack (starting point) | 2.0844e-8 | 1.0000 | — | 1.918e-9 |
| **STRUCTURAL only: `pathD=full` + `thin384` + `k4b=5`** | **1.9161e-8** | **×1.0878** | **10/10** | **1.763e-9** |
| + the fitted pair (`k4k3=0.5`, `k31mult=0.55`), `k4b=4` | **1.8741e-8** | **×1.1122** | **10/10** | **1.724e-9** |

**Every arm in the table improves all ten MLPs.** Against the board that is rank 1–2 — *if* we reach the floor.

⚠️ **Structural vs fitted.** `pathD`, `thin384`, `k4b` are structural (a term added or a rank raised, no fitted
constant) and should transfer. `k4k3`, `k31mult`, `feedmult`, `k22c` are **scalars fitted on these same 10
MLPs**, and `2026-09-28_k3n/` measured that exact shrinkage in production: torch ×1.0089 on 10 → flopscope
×1.0030 on 100, a factor of 3. Discount the fitted half accordingly: the honest projection is **≈1.75e-9**,
and the structural-only **1.763e-9** is the number to quote.

## 3. ⚠️ The κ₄ birth depth is an INTERACTION, not a lever (`k4b_ship.py`)

`k4b` with our `k4bparts=("sd",)` is **O(n²) per source** — the `"pd"/"p22"` branch is skipped by
`if "pd" in parts or "p22" in parts`, and so is `"s22"`, leaving only `addd += 4·sdmult·(Y∘B²).sum(1)`.
So raising it to 5 costs ≈0.03% of budget, and it looked like a **free** ×1.05 available immediately.
**It is not.** On the shippable base (k3n config: `e3age=1`, `pathD="merge"`, `thin24`, `deadcut −2.5`):

| k4b | 1 | 2 | 3 | 5 | 8 | 15 |
|---|---|---|---|---|---|---|
| raw | 2.3249e-8 | 2.3363e-8 | 2.3503e-8 | 2.3803e-8 | 2.4086e-8 | 2.4149e-8 |
| × | 1.000 | 0.995 | 0.989 | **0.977** | 0.965 | 0.963 |

**Monotonically worse.** The κ₄ birth depth only pays *together with* all-age path births — with `e3age=1`
the deeper κ₄ births are inconsistent with a κ₃ that has no path content at those ages, and adding them
makes the mismatch worse. So nothing in this document is available before the cost breakthrough.

## 4. What this changes strategically

The value of reaching the floor went **up**: it was worth 1.92e-9 (rank 2–3) this morning and is now worth
**1.72–1.76e-9 (rank 1–2)**. Nothing here is shippable at 24.8% utilisation — every live lever costs budget,
and the one that looked free turned out to be conditional on the ones that do not. The cost breakthrough is
now the only thing standing between us and first place, and it is the cloud's current task.
