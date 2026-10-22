# The old tail: can the sources of age ≥ a0 be carried as one object?

**Date:** 2026-09-27, cloud session (4 vCPU / 16 GB, x86_64), branch `cloud/old-tail`.
**Base:** k3l/k3m torch configuration, 10 baked MLPs, `ablate.py k3l_t 10` = **2.3135e-08**.

## Short answer

**Accuracy is not the obstacle. The obstacle is the cost of THE MERGE ITSELF.**

* **Stage A (measured).** The tail cannot be dropped along either of the two axes: without its contribution to D,
  ×26.5; without its contribution to t, ×41.8 (a0=4). But it is not demanding either: **a coherent 1% error in t AND
  in D costs 0.1%, an incoherent 1% costs 0.6–1.1%**, and 10% coherent costs ×1.47. So **the answer to the main
  question of Stage A is negative in the good sense: the tail does NOT need 1e-3 accuracy, the percent level is
  enough.** On top of that, its D has an effective rank of ≈256 out of 1024 (truncating to 256 costs ×1.008).
* **Stage B (measured).** The merged block has to reproduce the tail's **3-tensor**, not only its two slices: only the
  legs are carried linearly, while L1 = 2A∘B∘g and L2 = B∘B∘g are nonlinear in them and are rebuilt every layer.
  The tail tensor is a sum of R = S·n ≈ 10n symmetrised terms of rank 1. A good approximation of rank r ≈ n
  **exists** (n=64: 4.29e-4 at r=n, 5.02e-5 at r=2n), the opposite of every projection compression (§18, §54).
  But:
  1. **the only cheap (O(n²)) CP compression, keeping the r heaviest columns, gives 62.8% error at the real n=1024
     with r=n** (76.1% at r=n/2, 43.5% at 2n, 20.8% at 4n);
  2. **finding a good block costs incomparably more than it saves.** One fitting pass is
     8·n·R·r flops = **8 units of n³ per layer** at n=1024 with incremental merging, while the entire saving gives
     the merge **8.8 units per layer**, i.e. exactly ONE pass. Measured at n=1024: 400 passes give 4.3% and are
     still falling, a gap of ≈400×;
  3. **and a warm start does not help: CP has no nestedness.** Adding one source to a converged block throws the
     error from 2.9% up to **48.7%**, worse than cold column selection (42.2%), and refining it takes the same ~10⁴
     passes. The merge has to be solved from scratch every layer.

**Verdict: the tail can be merged in principle, but not within the budget, and the reason is structural, not an
engineering one.** The compression exists and the accuracy is sufficient, but **the representation that compresses
cannot be updated cheaply**: the CP optimum is not nested, so every new source requires a full re-solve. This is the
third case of one mechanism: in `pathrank` (§55) the basis is not free, in `k3k` (§54) entry into the band is not
free, here the merge itself is not free. The cost model counts the saved products and does not count the construction.

**And the prize is smaller than the task assumed.** Even a free, perfect merge at a0=4, r=n gives
util 13.87% → adjusted **3.0e-9** (×1.8 vs k3m), not top-5: ≤2.30e-9 requires util ≤10.6%, i.e. a0=3.

Nothing was submitted to AIcrowd. `CLAUDE.md`, `notes/*.md`, `src/` were not edited.

---

## 0. Warm-up and gates

| check | result |
|---|---|
| `ablate.py k4memo 3` on the patched `chain.py` | **3.7533e-08** ✅ |
| `ablate.py k3l_t 10` on the patched `chain.py` | **2.3135e-08** ✅ |
| `base` arm of Stage A (2 streams instead of 3) | 2.3135e-08, ×1.000 ✅ |

`chain.py` got one switch, `tailx` (default `None`): the contributions of sources **of age ≥ a0** are accumulated
into separate `t_old` / `D_old` and transformed before they are added. Both gates above were run **after** the patch;
with `tailx=None` the path is the same operation for operation (`tA, DA = t, D`, the same objects and order).

## 1. Self-checks of the switch (MLP 0, `logs/sanity_n1.jsonl`)

| check | result |
|---|---|
| `base` | 2.4074420792934517e-08 |
| `tailx mode="exact"` (separate accumulation, then addition) | 2.4074420792907588e-08 → **1.1e-12** relative |
| `tailx mode="dropD"` vs the existing `d21_ages=3` | **2.0e-13** relative (10 MLPs: matched to 5 digits) |
| `tailx mode="diagD"` vs `dropD` | **exactly 0.0** |

The last row means that **arm A4 of the task is degenerate**: `read_slices` zeroes the diagonal of D at the end, so
"keep only diag(D_old)" is identical to "drop D_old". That end of the ladder is carried by A3 (`rankD`).

## 2. Stage A: how accurate the tail has to be

All numbers: torch float64, 10 baked MLPs, base 2.3135e-08. `logs/stage_a_w*.jsonl`.

### 2.1 Which half carries what part of the cost (A1, A2)

| arm | a0=4 | a0=5 | a0=6 |
|---|---|---|---|
| **A1** drop the tail's D, t exact | **×26.5** | ×15.2 | ×8.5 |
| **A2** drop the tail's t, D exact | **×41.8** | ×28.4 | ×18.8 |

**Neither half can be dropped**, and t matters slightly more than D. (A1 is the existing `d21_ages`; §19 measured it
on the old base 4.25e-8 and got ×9.1 at a0=4. The factor grew not because the switch changed but because the base
became 1.8 times more accurate.) So the merge has to preserve **both** contractions. This is the worst possible case,
and it also rules out "keep only t".

### 2.2 How compressible the tail's D is (A3, a0=4)

| truncation rank of D_old | 64 | 128 | **256** | 1024 |
|---|---|---|---|---|
| ×raw | ×2.035 | ×1.245 | **×1.008** | ×1 (exact) |

**The tail's D has an effective rank of ≈256 out of 1024.** A block of r columns gives a D of rank ≤2r, so along the
D axis r ≈ 128 would be enough. This is an upper bound (SVD-optimal truncation; the block's factors are coupled).

### 2.3 Error budget (A5, a0=4): the main result of Stage A

| tail perturbation | ε=1e-3 | ε=1e-2 | ε=1e-1 |
|---|---|---|---|
| coherent, t only (`scaleT`) | ×0.998 | ×0.987 | ×1.220 |
| coherent, t AND D (`scaleTD`) | ×1.000 | ×1.001 | ×1.471 |
| incoherent, t only (`noiseT`) | ×1.000 (2 seeds) | **×1.002** (7 seeds, 1.000–1.005) | **×1.253** (2 seeds, 1.252/1.254) |
| incoherent, t AND D (`noiseTD`) | — | **×1.008** (3 seeds, 1.006–1.011) | — |

The spread over seeds is tiny (at ε=1e-2 the whole range is 1.000–1.005, at ε=1e-1 it is 1.252 vs 1.254), so these
are stable numbers, not a single realisation of the noise.

**The task asked whether the tail needs 1e-3 accuracy. It does not: the percent level is enough.** A relative error of
1% in both contributions costs 0.1–1.1% raw; the budget for ≤3% raw is roughly **3–5% relative error in the tail**.
Note that `scaleT` at 1e-2 gives ×0.987, i.e. slightly better than the base: the chain slightly under-weights the tail.

The contrast with the variance channel (§57) is sharp: there a coherent 1e-4 cost ×2.7. **The tail is a soft
channel.** One more measured difference from §57: at ε=1e-1 an incoherent perturbation of t (×1.253) costs **as much
as a coherent one** (×1.220), so **there is no √n suppression here**. The reason is that what gets perturbed is the
already summed contribution of the tail (one vector of n numbers), not 1024 independent inputs as it was for the
variance. In practice this means the merge error can be measured with an ordinary relative norm, without splitting it
into coherent and incoherent parts, and that is what Stage B does.

## 3. What exactly the merged block has to reproduce

The tail's contribution to both contractions is generated **only** by the legs:
```
t_i += 3 Σ_p A[i,p] g[p] B[i,p]²  (+ E term, + P2)          D += L1 Bᵀ + L2 Aᵀ,
L1 = 2 A∘B∘g,   L2 = B∘B∘g
```
L1 and L2 are **nonlinear** in the legs and are rebuilt every layer, so only the legs can be carried, and the block
has to have the same symmetric-CP form: r columns (x_q, y_q). It then reproduces the tail on all later layers if and
only if it matches the **tensor**
```
T = Σ_s Σ_p g_s[p] · Sym( a_s[:,p] ⊗ b_s[:,p] ⊗ b_s[:,p] ),
```
whose diagonal and (2,1)-slice are exactly the tail's t and D. Matching the two slices on one layer is not enough: the
transport mixes the rest of T into the readout. One source is exactly n such terms, S sources are R = S·n.
Where this breaks explicitly: the d1 gating is diagonal and the block survives a layer, but **the birth of a new source
cannot be folded into the block without re-solving the rank problem**, and that is exactly what is evaluated below.

## 4. Stage B: measurements

The tensor is never built explicitly: `stage_b_gram.py` expands ‖T−F‖² through four Gram matrices,
```
⟨S(a,b), S(x,y)⟩ = (1/3)(a·x)(b·y)² + (2/3)(a·y)(b·x)(b·y)
```
(the check a=b=x=y gives |a|⁶), so a pass costs O(n·R·r), not O(n³·r). **Cross-check against the explicit tensor:
‖T‖ to 2.6e-15, the loss value on a random F of rank 7 to 3.6e-15.** ✅

### 4.1 Certifying the fitter

A hand-written symmetric ALS is biased here (x enters all three modes), which would inflate the error and could
"manufacture" a negative result, so the fit is autograd on the Frobenius norm. Control: **S=1, where an exact
solution of rank n exists by construction** → `rel err 4.2e-16` ✅.

### 4.2 Existence of a block of rank ~n (n=64, 40000 passes)

| r/n | 0.5 | 1.0 | 2.0 |
|---|---|---|---|
| rel. tensor error | 2.89e-3 | **4.29e-4** | 5.02e-5 |

**A 10× compression (R=640 → r=64) with error 4.3e-4 exists.** This is the direct opposite of what every projection
compression gave (§18 Tucker, §54 band): the tail is **not** incompressible.

⚠️ These numbers are **upper bounds**, and they depend on the number of passes: at n=64, r=n
1500 → 1.31e-3, 6000 → 6.62e-4, 20000 → 4.22e-4, 40000 → 4.29e-4. The first runs of `stage_b_cp.py`
(1200–1500 passes) had **not converged**, and the "dependence on n" read off them was an optimiser artefact; it is
recomputed in §4.3 from converged runs.

### 4.3 Dependence on n at fixed r/n

Full tail (S=10, R=10n), r/n = 1, converged runs:

| n | 64 | 128 |
|---|---|---|
| rel. error at r=n | 4.29e-4 | 3.03e-3 |
| rel. error at r=2n | 5.02e-5 | 5.57e-4 |

At fixed r/n the error **grows with n** (×7 per doubling at r=n). Runs at n=256/512/1024 for the full tail are
limited by the number of passes, not by the mathematics, so I will not extrapolate this column: **the verdict below
rests only on measurements at the real n=1024** (§4.4, §4.5, §4.6), where no extrapolation is needed.

### 4.4 Cheap merge at the real n=1024: measured

The only CP compression that costs O(n²) (i.e. is practically free) is to **keep the r heaviest columns** of the
tail. It is also the initialisation of the fit, so its error is measured by a single loss evaluation:

| r | n/2 = 512 | n = 1024 | 2n = 2048 | 4n = 4096 |
|---|---|---|---|---|
| rel. error (n=1024, S=10, R=10240) | 76.1% | **62.8%** | 43.5% | 20.8% |

**62.8% at r=n against a budget of 3–5%.** The cheap merge misses by a factor of 15–20.

### 4.5 The cost of finding the block: this is where everything dies

Counted in the same units as everything else: 1 unit = one dense n³ product = 2n³ flops, the whole budget of one
MLP = 1024 units.

**What the merge can afford at all.** The saving being fought for (§5): util 24.14% → 13.87% = 10.27 points =
**105 units for the whole MLP**. The tail exists on 12 layers (a0=4, L=16), i.e. **≈8.8 units per layer**, and that
is ALL the merge may cost and still pay off.

**What one fitting pass costs.** In Gram form a pass (a gradient or an equivalent ALS sweep) is four Gram matrices,
8·n·R·r flops. For an incremental merge (a block of r=n columns + one graduating source of n columns, i.e. R=2n) at
n=1024: 8·1024·2048·1024 = 1.72e10 flops = **8.0 units**. So **the merge budget is exactly ONE pass per layer.**

**How many passes are needed (measured at n=1024, incremental step, r=n):**

| passes | 0 (column selection) | 50 | 200 | 300 | 400 |
|---|---|---|---|---|---|
| rel. error | 16.8% | — | 4.54% | 4.38% | **4.32%** (still falling) |

400 passes alone are 3200 units per layer = **38 400 units per MLP, i.e. 37 full budgets**, for a saving of 105
units. The gap is **≈400×** on this step alone, and the 3% target has not been reached yet.

### 4.6 Why incremental merging does not help: the CP optimum is not nested (main result)

One hope remained: production never starts "from scratch"; on layer l+1 it starts from the block that already
existed. Measurement (`stage_b_warm.py`, n=256, r=n, a converged block over K=2 sources, then a third one arrives):

| start | error at start | best over 12000 passes |
|---|---|---|
| block over K=2, cold start from column selection | 24.2% | 2.86% |
| **+1 source, WARM start from the existing block** | **48.7%** | 2.87% |
| +1 source, cold start from column selection | 42.2% | 3.02% |

**The warm start is not warm: adding ONE source to a converged block throws the error from 2.9% to 48.7%, worse
than cold column selection (42.2%), and refining it takes the same ~10⁴ passes.**

Mechanism: the CP decomposition **has no nestedness property**. For an SVD/subspace, adding a vector moves the basis
only slightly, so the update is cheap (this is how `split24rr` works). For symmetric CP the optimal block of rank r
for K sources is neither a part nor a small perturbation of the optimal block for K+1: a new source rearranges the
whole factorisation. So the merge has to be **solved from scratch every layer**, and its cost is not an "update" but a
full nonlinear problem: ~10⁴ passes × 8 units = 8·10⁴ units per layer against a budget of 1024 units for the whole
MLP.

**This is what closes the idea.** Not because the tail is incompressible (it is compressible, §4.2), and not because
high accuracy is needed (it is not, §2.3), but because **the representation that compresses cannot be updated
cheaply.**

## 5. What this would cost at all (`cost_merge.py`)

The model imports `cost.py` and **reproduces `cost.util` exactly** at `a0=None` (0.0000 points).
`m` is the merge charge, in units of one dense n³ product per layer.

| a0 | r | m=0 | m=1 | m=2 | m=4 |
|---|---|---|---|---|---|
| 4 | n | 12.69% | **13.87%** | 15.04% | 17.38% |
| 4 | 2n | — | ≈16% | — | — |
| 3 | n | 10.38% | 11.65% | 12.92% | 15.46% |
| 6 | n | 16.62% | 17.60% | 18.57% | 20.53% |

At k3m's graded raw (2.1652e-8): a0=4, r=n, m=1 → **3.00e-9**; a0=3, m=1 → **2.52e-9**;
a0=3, m=2 → 2.80e-9. Top-5 (≤2.30e-9) requires util ≤10.6%, i.e. **a0=3 and m≈0**: a perfect free merge of
everything older than age 2. **The prize of the whole idea is ×1.8, not top-5.**

## 6. Verdict and what would change it

The idea is closed by §4.6, not by accuracy. What would have to be found to reopen it: **a closed-form incremental
symmetric-CP merge** (a block of rank r + one graduating source → a block of rank r) costing ≤1–2 units of n³ per
layer. Measured alternatives: column selection 62.8% (§4.4), cold fitting ≳400 passes (§4.5), warm start **does not
work at all** (§4.6). There is no need to try more compression heuristics: the problem is not finding a better
heuristic, it is that CP factorisation has no nestedness, so a cheap update does not exist in principle. The only
thing that would remain to try seriously is to rewrite the tail in a representation that is NESTED by construction
(like the subspace in `split24rr`); but all such representations are projections, and those are closed by
measurements (§18, §20–§22, §54).

Practical conclusion for the remaining three weeks: the cost axis is closed completely (all the switches of §54–§56
plus this one), the prize of the last idea is ×1.8 and it is outside the budget → **it makes sense to move the time to
the write-up** (24 Oct).

## 7. What is measured, what is a model

**Measured (torch float64, baked MLPs, gates 3.7533e-08 / 2.3135e-08 on every run):** all of §2 (Stage A),
§4.1–§4.4, including n=1024 in §4.4.
**Measured and cross-checked:** Gram form vs the explicit tensor (2.6e-15 / 3.6e-15), certification of the fitter
(4.2e-16).
**Model:** every `util` in §5 (`cost_merge.py` on top of the calibrated `cost.py`), and the charge `m` is a parameter,
not a measurement.
**Extrapolation, marked as such:** the rescaling of §4.3 to n=1024.
**Not measured and not transferable:** residual and wall (CLOUD.md §2); here they are not an argument at all.

## 8. Files

| file | what |
|---|---|
| `stage_a.py` | Stage A arms through the new `tailx`; `logs/stage_a_w*.jsonl`, `logs/w*.txt` |
| `stage_b_cp.py` | explicit tensor + autograd fit (certification S=1); usable up to n≈128 |
| `stage_b_gram.py` | Gram form of the loss: fitting without the n³ tensor, up to n=1024; `--verify` cross-checks it against the explicit one |
| `cost_merge.py` | cost of the merged tail in `cost.py` units; the self-check reproduces `cost.util` |
| `logs/sanity_n1.jsonl` | self-checks of the switch (§1) |
| `logs/stage_b_gram.jsonl`, `logs/stage_b_prune.jsonl`, `logs/stage_b_incr.jsonl` | raw results of Stage B |
