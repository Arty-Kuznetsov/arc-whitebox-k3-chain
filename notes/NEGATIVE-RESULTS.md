# NEGATIVE RESULTS — everything we tried that did not work, with numbers and mechanisms

**Read this before proposing anything.** Every entry was measured against the baked N=1e9 ground
truth (or Monte Carlo where stated), with a self-check that the unmodified arm reproduces a known
reference. Scripts and raw logs are in `../experiments/2026-09-16_kprop-arc/` and its `logs/`.
Chronology is in `WORKLOG.md`; the method that survived is in `METHOD.md`.

> **Structure.** Sections A–E (§1–§15) are from the Part-I method (submission 331944, Sept 13–23). **Section F
> (§16–§54) is from the k=3 chain phase (2026-09-23 night → 09-25)**, the current method in `METHOD.md` Part II.
> **Section G (§55–§66) is the cost-boundary phase (2026-09-25 → 09-29):** five cheap-representation ideas closed
> by one mechanism (§54–§59), the principle underneath them (§62), the grader's resolution (§60), what re-opening
> "does not pay" terms at the floor showed (§61, §65), the jump of the old tier (§63), the post-hoc corrector
> (§64) and the Slice-Markov tail — §62's own "larger state" door (§66). Numbering is chronological, not by section order in the file; entries §58–§66 sit near the end.
> **Two things in section G are NOT negatives and live elsewhere:** the dead-side birth-column cut (§48 revised,
> `experiments/2026-09-29_colcut/`) and the transported band basis (`experiments/2026-09-29_band/`) — both alive.
> Its raw data is in `../experiments/2026-09-24_k3-chain/RESULTS.md`. Some Part-I verdicts do not carry over:
> §7 ("κ₄ path: budget says no") is **superseded**. In the k=3 chain, κ₄ is worth 10.7× and costs 2 products per
> layer (§32).

Two of these entries are **corrections of our own earlier conclusions**. Both came from conflating
two effects in one test. That failure mode is the single most useful lesson here: **split a compound
change before judging it** (§5, §11).

---

## A. Cheap-cost levers on ARC's reference k=3 (Sept 17–19)

Baseline facts: at 1024×16 the reference k=3 reaches `final_layer_mse = 4.1e-8` but costs **148% of
budget**; 95% of that is the third-cumulant linear step; cost scales `O(L^2 n^3)` (measured ×4.06–4.27
per depth doubling); rank grows `R = 2n(l+1)`.

### 1. Low-rank compression of the κ₃ CP tensor — DEAD
Per-layer SVD of the factor matrix: **effective rank stays ≈ n = 1024 at every layer** (σ > 1e-6·σ_max:
1024 → 885). Capping `R` by keeping top columns by norm:

| R_max | 128 | 256 | 512 | 1024 | 2048 | 4096 | full |
|---|---|---|---|---|---|---|---|
| final MSE | 3.96e-6 | 3.55e-6 | 3.24e-6 | 2.94e-6 | 1.86e-6 | 8.49e-7 | **3.71e-8** |
| vs full | ×107 | ×96 | ×88 | ×79 | ×50 | ×23 | 1 |

**Mechanism:** column truncation is not valid CP compression; κ₃ has essentially full multilinear rank
and its column importance is flat. Forum write-up #18218 reports the same ("cap 4n: error 0.39; 8n:
0.03–0.14; column importance flat"). Even a *perfect* orthogonal re-compression to `R = n` saves only
`~L ≈ 8×`, i.e. 148% → ~18% — still short of the frontier.
`rank_diag.py`, `rcap_test.py`.

### 2. Activation-regime (on/off/kink) pruning — DEAD
Mean fraction of neurons with `|alpha| < 2` is **0.70** (1.00 at layer 0, still 0.42 at layer 15);
firmly on or dead totals only 38% even at the deepest layer. He-init keeps `mu ~ 0, sigma ~ 1`, so most
neurons sit at the ReLU elbow at every depth and cannot be handled linearly. `kink_diag.py`.

### 3. Layer-adaptive cumulant order — DEAD
Self-verified at both endpoints (`m=0` reproduces k=2, `m=16` reproduces k=3). Forward (k=3 for the
first m layers): `m=4` gives ×1.2 at 10.6% of budget. Reverse (k=3 for the last j): `j=8` gives ×1.4 at
37.6%. The 110× gain appears only when κ₃ is tracked through **all 16** layers.
**Mechanism:** blame is spread across depth with no hot spot — the per-layer marginal contribution is
≈7e-7 at layers 1–3 decaying to ≈2e-7 at 9–15, and **layer 0 contributes exactly zero** (its input is
exactly Gaussian, which is also a free validation of the measurement).
`layer_adaptive.py`, `reverse_adaptive.py`, `layer_blame.py`.

## B. Shortcuts that would have changed the cost class

### 4. Richardson extrapolation across cumulant ORDER — DEAD
The ladder (3.33e-4, 4.50e-6, 4.10e-8) has ratios 74× and 110×, matching `(L/n)^k` with `n/L = 64`, and
`k=1` costs 0.07% of `k=2` — so the correction would have been free.

| rung | corr(step, error) | fitted slope | theory | after theory |
|---|---|---|---|---|
| `mu^2 − mu^1` | 0.169 | **+0.0273** | −0.0159 | 0.92× (worse) |
| `mu^3 − mu^2` | 0.179 | **+0.0154** | −0.0159 | 0.92× (worse) |

**Mechanism:** the `(L/n)^k` law governs the *aggregate* MSE; per-neuron error *directions* decorrelate
between orders. The fitted slope even has the opposite sign. Even an oracle-fitted slope buys 1.03–1.08×.
This closes "resummation of the depth recursion", which our own literature review had flagged as
organiser-endorsed and unexplored. `ladder_extrap.py`.

### 5. A low-dimensional covariance subspace — DEAD (and an instructive trap)
The pre-activation covariance really does concentrate with depth: 90% of the energy in **145 of 1024**
directions by layer 15, participation ratio 511 → 65, numerically rank-deficient from layer 13. Since
the k=3 cost is concentrated in exactly those deep layers, this looked like the one lever that changes
the cost *class*. Projecting the chain onto the top-m **exact** eigenspace:

| m | 1024 | 512 | 300 | 200 | 150 | 64 |
|---|---|---|---|---|---|---|
| vs untruncated | 1.00× | **3424×** | 10475× | 16662× | 20984× | 31721× |

**Mechanism:** the mean reads `sigma_i = sqrt(C_ii)` and the diagonal aggregates *every* eigendirection.
The discarded directions carry little total variance but contribute **coherently** to all n diagonal
entries, so truncation biases sigma downward for every neuron at once — a systematic bias, not a small
perturbation. **Variance concentration in a spectrum does not imply low-dimensional dynamics when the
readout is a diagonal functional.** `subspace_probe.py`, `subspace_truncate.py`.

### 6. Low-rank compression of the source PROPAGATORS — real effect, does not pay
Unlike §1 this is a **matrix**, where SVD truncation is optimal (Eckart–Young), and `A_m` is a product
of `age` random matrices whose singular values spread exponentially (Lyapunov). Both premises check out:

| age | top-128 | top-256 | top-512 |
|---|---|---|---|
| 1 | 0.376 | 0.623 | 0.894 |
| 4 | 0.840 | 0.975 | 1.000 |

and truncation is nearly free in accuracy (rank 512 from age 2 → 0.98×; rank 256 from age 3 → 0.97×).
`fnp.linalg.qr` and `svd` exist, and a randomized range finder was implemented and validated.
**It still does not pay, for two compounding reasons:**
1. Sources live only 4 layers, so truncating at age 3 leaves exactly **one** layer that uses the
   factored form, while the sketch itself costs ≈0.9 of a matmul.
2. The contractions need `A` and `B` **materialised** anyway — `kappa3[i] = 3 sum_a d2_a A[i,a] B[i,a]^2`
   is elementwise in `(i,a)`; in factored form it costs `O(n r^3)`, worse. Materialising costs `2 n^2 r`
   each, cancelling what the factored `D21` product saves.

Measured: utilization 14.98% → 14.54%, score 7.93e-8 → 7.82e-8 on the same 6 MLPs. **1.4%.** Reverted.
`lowrank_propagator.py`.

## C. Extra terms in the expansion

### 7. Fourth cumulant (path diagram) — works, but the budget says no
The diagrams were validated at width 64 against 2e7 MC samples: the **star** term is identically zero
wherever `alpha = 0` (its hub carries `d3 = -alpha*phi/sigma^2`), and the **path** diagram alone
correlates **0.985** with the true κ₄. Control: the κ₃ prefactor fits 3.30 against the derived 3.0 with
correlation 0.997. The path template reuses matrices the κ₃ machinery already carries, so it costs one
extra matmul per source per layer. **End-to-end it buys only 1.04–1.06× while costing 4 points of
budget**, which pushes past the multiplier floor: adjusted 1.512e-7 → 1.912e-7. A third κ₃ source beats
a κ₄ term. `k4_diagrams.py`, `k4_chain.py`.

### 8. Higher-order birth diagrams — not affordable by anyone
The next terms are the **covariance triangle** and **κ₃ × edge products**. Our own contraction analysis
puts the triangle at `O(n^4)`. Independently, ARC's reference states in `factor_k3.py` that the
augmented term set "includes diagrams with no O(n)-rank factorization (e.g. the covariance triangle and
kappa_3 x edge products), which this implementation **DROPS**". The reference still reaches 121× without
them, so **they are not the gap**.

### 9. Rescaling the birth magnitude — the derived value is already optimal
Our regenerated source carries only 0.918 of the true magnitude at layer 1, and `0.918^16 = 0.25`, so a
compounding deficit looked plausible. Sweeping a multiplicative factor on the born source:

| scale | 0.90 | **1.00** | 1.05 | 1.089 | 1.15 | 1.25 | 1.40 |
|---|---|---|---|---|---|---|---|
| vs scale 1 | 0.92× | **1.00×** | 0.98× | 0.93× | 0.82× | 0.63× | 0.39× |

**The optimum is exactly the derived value**, and moving toward the "missing" magnitude makes it worse.
Worth stating in the write-up: the method has no fitted constant, and fitting one does not help.
`birth_scale.py`.

## D. Cost engineering

### 10. Deeper Strassen-Winograd — bounded by residual time, not by the meter
flopscope genuinely bills the op tree (0.88 / 0.78 / 0.70 / 0.65 at 1–4 levels) and the organisers
permit it. But each level multiplies Python-level `fnp` calls by ~7, and the grader caps non-flopscope
time at 400 ms per MLP. **Level 2 measured 0.325 s on 20 of 20 MLPs — 81% of the cap** on a machine
faster than the grader's single physical core, and a breach zeroes every prediction. Level 1 ships.
A mixed-depth scheme (level 2 on ~27% of the products) would claim ~0.6 points of budget; untested.

### 11. Applying the D21 covariance term to a subset of sources — 3%
At `J = 4`, restricting the correction to the newest k: best is k=3 at 8.298e-8 against k=4's 8.558e-8.
Not worth a change. `d21_subset.py`.

### 12. Diagonal variance correction — HARMFUL, and it masked a real win
The same cumulant rule gives `Delta var_i = d1_i d2_i kappa3[i,i,i]`, free (no matmul). Applied at J=4:

| applied | vs none |
|---|---|
| diagonal only | **0.22× (harmful)** |
| off-diagonal only | **1.82× (a real win)** |
| both together | 0.56× |

**Our first test applied both at once and concluded "the covariance correction hurts" — which was
wrong and cost hours.** The diagonal must keep the exact rectified-Gaussian variance. Similarly, of the
two leg assignments in the symmetrisation, keeping only the **first** beats keeping both (2.15× vs
1.82×) at half the cost. `covcorr.py`, `vardiag.py`, `offdiag_sweep.py`.

### 13. Sources beyond J = 4–5 — saturates, then degrades
Without the covariance correction: 1.61× / 2.37× / 3.16× / 3.94× / 4.43× / 4.54× / 4.25× at
J = 1/2/3/4/6/8/15. With it: 1.66× / 2.15× / 2.15× / 2.00× / 1.71× at J = 3/4/5/6/8.
**Mechanism:** the leading Wick diagram is a poor model for old sources, so adding them eventually adds
more error than signal. `multisource.py`, `offdiag_sweep.py`.

## E. Earlier dead ends (Sept 13–16)

### 14. Tetrachoric / Hermite covariance correction — NOT a negative result (corrected 2026-09-23)
**This entry used to say "neutral alone (4.058e-6 vs the kit's 4.050e-6)". The 4.058e-6 was MLP 0's value copied
as the split mean.** Re-measured on 100 MLPs, same pipeline (`experiments/2026-09-23_writeup-ablations/`):
gain-only k=2 **4.0504e-6** (= kit docs; per MLP it matches ARC's float64 reference k=2 to 3 s.f.), with the
first tetrachoric term **3.8502e-6** — a **5.2% gain alone**. It is also worth 7.7% at J=1 (2.4527e-7 vs
2.2767e-7) and 3.1% final / 14.6% all-layers in the shipped configuration. It stays on. The width sweep still
stands: at width 256 it hurts, from 512 up it helps. The published covariance-only results (#18200 −1.2% for a
full second-order Hermite correction; #18214 +7.4% for the exact bivariate covariance) are in the same small range.

### 15. Ideas closed by the published corpus (never re-tested here)
From Phase-1 write-ups #326024 / #317660 and Phase-2 #18218, all reported closed with mechanisms:
characteristic-function propagation · randomized-truncation unbiased analytic · particle/cubature ·
off-diagonal 4th cumulants · learned autoregressive or per-layer operators (they *diverge*: a learned
operator 40% more accurate one-step diverged by +12,000% when iterated) · MLMC · nonlinear controls ·
mid-depth re-anchoring · per-MLP adaptive sample counts · input-space QMC/stratification ("does not
survive fusion") · final-layer-only correctors · spherical quadrature · median-of-means · antithetic
RQMC · learned output corrections (<0.09 correlation with the error) · Monte-Carlo control variates for
a k=3-class chain (need 33–300× variance reduction, best realizable 1.5×) · the finite-width literature
(annealed over weights, so odd cumulants vanish; quenched literature stops at covariance).

---

## F. The k=3 chain phase (2026-09-23 night → 2026-09-24)

These numbers are for 3 baked MLPs (0–2), final-layer MSE. The faithful chain (radial κ₄) gives 4.2509e-8, the
same as ARC's reference; the memo22 base gives 3.7533e-8. Configs are named as in
`experiments/2026-09-24_k3-chain/ablate.py`. **The common thread:** the κ₃ diagonal readout
`t_i = Σ_p g_p A_ip B_ip² + …` and the D slice are **sums of many positive-weight contributions**. A truncation
that throws away a "small" tail therefore throws it away **coherently**, so its error does not average out. This is
the same failure mode as §5 and §6, now seen in five more forms (§16–§22).

### Cost levers that failed

#### 16. Window: keep only the newest J sources — DEAD
J8 gives 1.50e-7, J6 3.51e-7, J4 8.68e-7, J3 1.26e-6 and J2 1.83e-6 (full: 4.25e-8). In Part I, J = 4 was optimal
only because Part I's *regenerated* sources were a poor model of old sources. In the exact chain every source is
correct and old sources are needed: at layer 15, sources older than 4 carry **69% of ‖t‖**, and those older than 6
carry 53% (`old_vs_young.py`).

#### 17. Regenerate old-source content from the young sources — DEAD
The old part correlates with the young part at only 0.6–0.8. A per-layer linear fit leaves 40–60% of the old part
unexplained. The old sources carry information of their own; they are not an echo of the young ones.

#### 18. Tucker / shared-basis compression of old sources — DEAD (lossless only when it saves nothing)
Sources older than y are folded into one basis P (n×r, transported like a leg) plus a symmetric core G (r³), with a
HOSVD refresh. It is exact at r = n. Results:
- y=1: r=64 1.43e-6; r=128 6.91e-7
- y=2: r=128 2.75e-7; r=256 8.14e-8
- y=3, r=128: 1.38e-7
- y=5 9.66e-8; y=6 6.37e-8; y=8 4.76e-8; y=10 4.32e-8

It is lossless only at y ≥ 10, where the cost saving is 0–11%. **Mechanism:** the fold keeps the top-r subspace,
and the discarded remainder enters t through positive weights. The same failure has been reported for "shared
subspace" on the forum (#18218: a 1.3× lever, with a cliff below).

#### 19. (2,1) slice D from the newest y sources only — DEAD
d21y8 gives 6.17e-8, y6 1.37e-7, y4 3.85e-7 and y2 1.02e-6 (the diagonal t still came from all sources). D feeds
every pair cumulant, so an error in D corrupts the covariance and propagates.

#### 20. Low-rank tree leg U (hence B) — DEAD
Rank 256 gives 7.55e-8 and rank 128 9.87e-8. Applying it only to deep births is better but still lossy:
from layer 8 on, 4.85e-8; from layer 11 on, 4.10e-8.
U = diag(d1)·C0 inherits the spectrum of C0, the zero-diagonal covariance, which is spread out: the off-diagonal of
a random-network covariance is "noise-like". **B must stay full rank.**

#### 21. Low-rank D — DEAD
Rank 256 gives 7.75e-8, rank 128 1.36e-7 and rank 64 2.41e-7. D is a sum over sources of products of full-rank legs.

#### 22. Per-source low rank (project a source's legs on its top-r subspace from age a) — DEAD, and why
sr4_256 gives 6.27e-8, sr4_128 1.75e-7, sr3_256 1.17e-7, sr6_128 6.64e-8 and sr2_512 5.41e-8.

The split test is the informative part: in production form, rank 256 from age 4, projecting **A only gives
5.53e-8** and **B only gives 4.06e-8**. **The hub leg A is the sensitive one.** A enters t as Σ_p A_ip·(g_p B_ip²),
where g·B² ≥ 0, so the discarded tail of A adds up coherently over p. B enters squared, so its tail is
second-order.

#### 23. Low-rank ΔK21 in the correction leg (C0-tree form) — DEAD as posed, FIXED by an exact split
Rank 16 gives 8.07e-7, rank 64 1.86e-7 and rank 128 2.03e-7. That the error does not improve with rank is the
tell: ΔK21 contains a **full-rank structured piece** diag(α)·C0·diag(d1) (the Gaussian a1b1·C0 term of K21). No
low rank can capture it. Splitting it off analytically and carrying it as 3B·diag(α) left a remainder R that *is*
low-rank. The result was thin E (`METHOD.md` §10), a lossless 6→4. **Lesson: before approximating a matrix by
low rank, remove its analytically known full-rank part.**

#### 24. Tree legs built from the full C instead of C0, without/with a diagonal correction — DEAD
fullC_none gives 3.29e-7 and fullC_diag 3.16e-7. With the exact correction (fcfull) it is 3.7533e-8, as it
should be, and a rank-k correction of the resulting ΔK21′ is only partly lossless (fclr64: 4.03e-8). Superseded by
§23's split.

#### 25. Correction leg simplified — DEAD
Diagonal only (ΔK3, no ΔK21) gives 1.74e-6 (41× worse); no correction leg gives 2.80e-6 (66×). The repeated-index
slices (3,) and (2,1) must be exact at birth.

### Accuracy levers that failed

#### 26. More terms in the SIMPLE expansion — no gain or worse
- Single-index κ₃² Edgeworth term (`edge2`): 4.37e-8, worse.
- Dense AUGMENT pair diagrams (C0³, t·C0, C0·D, K4d·C0, C0·K22): 6.59e-8, or 6.23e-8 on the memo22 base; worse.
  **Mechanism:** these add some second-order diagrams without their companions (inconsistent truncation), which
  biases the covariance.
- Higher Gaussian terms ρ³…ρ⁶ in every pair cumulant (tet3/4/6): 3.775e-8, no gain.
- The one-step readout is not where the error is: fed MC-exact moments it gives 1e-10 (§31).

#### 27. κ₄ births (tree paths 4·X·C0·Xᵀ, star diagrams) from the newest 1/2/all sources — no gain
k4b1 gives 3.93e-8, k4b1_nostar 3.71e-8, k4b2 4.13e-8 and k4b15 4.62e-8 (base 3.7533e-8). Against MC the paths
raise the κ₄-diagonal correlation only slightly (0.23 → 0.34). The stars *lower* it (→ −0.23), so the star formula
is wrong or harmful. Neither fixes the level deficit, which is the real problem (§30).

#### 28. κ₃² pair diagrams (D·Dᵀ, D², t·D) in K22, and ρ⁴…ρ⁶ in K22 — do not move the (2,2) deficit
Tested inside the chain (`q22x`) and in the one-step decomposition (`k22_step.py`). The post-ReLU (2,2) deficit
(0.94 → 0.69 of MC over layers 6–14) is unchanged.

#### 29. "κ₄[iijj] ∝ C_ij" closure (504aldo's fitted form) — NOT supported by MC
In the MC oracle the (2,2) slice is not proportional to C, which is sign-random. About 30% of its pair variation is
explained by C0², and its mean is 1/3 of the κ₄ diagonal mean at every layer (radial-like). A C-proportional closure
would get the structure wrong.

#### 30. Rescaling the chain's κ₄ level by oracle ratios every layer — UNSTABLE (not a valid test)
κ₄ sits in a feedback loop: a larger κ₄ changes the nonlinear step, which changes the next κ₄. Per-layer rescaling
diverged. The valid tests are substitution of MC moments (`oracle_subst.py`) and the one-step decomposition
(`k22_step.py`); see `FRONTIER.md`.

#### 31. "Improve the readout" — closed by the MC oracle
With exact per-neuron (μ, σ², κ₃, κ₄), the one-step Edgeworth readout gives a layer-15 MSE of 9.7e-11. With κ₃²
added it gives 8.1e-11. **All the error is in the upstream moments** (mostly κ₄, then κ₃), not in the readout.

#### 32. No κ₄ at all — 10.7× worse (4.54e-7)
This is the reverse of Part I's §7: in the full chain κ₄ is essential. The radial scalar already buys 10.7×, and
the memoryless (2,2) transport adds a further 1.13×.

### Engineering failures (the grader zeroes an MLP whose residual time exceeds 0.4 s)

#### 33. Strassen L4 / unbuffered ports — residual-time failures
- **First port `estimator_k3.py`**: 6.6 s of residual time per MLP. The causes were Python per-source loops,
  concatenation of non-contiguous views, and large frees. Fixed by the buffered layout (`METHOD.md` §11).
- **k3c L4 (chunk 8) on 100 MLPs**: MLPs 4–7 failed at 0.408, 0.411, 0.422 and 0.453 s residual (wall 30–33 s,
  normal 18 s). The MC oracles were running in parallel. Residual time grows with the Python-call count, which rises
  with the Strassen level, and **with CPU contention**. On 3 MLPs on a quiet machine the residual was 0.24 s, so
  the margin was only 1.7×. **Rule:** ship only configurations whose local residual is ≤0.2 s. The grader's residual
  ≈ local, but its host load is unknown.
- **k3b L3/L4 grader wall**: 2.9× local, so L4 (25 s local) would take ~73 s against the 120 s cap. Not shipped.

### κ₄ phase 2 (2026-09-24 after midnight; numbers in `experiments/2026-09-24_k3-chain/RESULTS.md` §7)
What worked is in RESULTS §7 and METHOD §15: the edge×κ₃ diagram C0·D plus the scale-mixture closure of the κ₄ pattern,
×1.13 on 10 MLPs. What did not:

#### 34. Pure Gaussian scale-mixture higher orders (FRONTIER A1 as stated) — wrong SIGN
The pre-activation (2,2) level is radial (v from q equals v from K22/(varᵢvarⱼ) to 3 digits), and the common scale is
strongly skewed (skewness 1.9 → 4.8). But the exact scale-mixture average over s, relative to its first order in v,
LOWERS the post (2,2) (0.996/0.986/0.977 at layers 6/10/14), while the measured one-step deficit needs
×1.025/1.039/1.046. Radial κ₆ and Var(s)² are not the missing piece.

#### 35. Marginal Edgeworth correction of the pair Wick coefficients (a_k with t, K4d) — wrong direction
In the one-step with MC inputs it lowers the post (2,2) further (0.956 → 0.951 at layer 14) and tilts K21.

#### 36. The κ₄ (3,1) slice κ₄[iiij] — harmful in the chain
With the MC (3,1) slice the one-step moves only +0.3%. The scale-mixture closure K31 = 3v·varᵢ·C0ᵢⱼ in the chain:
4.06e-8 alone and 4.91e-8 on top of the closure (base 3.52e-8, 10 MLPs). It is first order in v and ∝ C, so it
biases the covariance, whose errors compound.

#### 37. Fixing the κ₄ LEVEL alone (C0·D without the pattern closure) — harmful
C0·D in every Q raises the chain's pre (2,2) level from 0.70 to 0.91 of MC at depth, but the final MSE goes
3.75e-8 → 4.64e-8. By channel: Q22 only 0 (it moves the post (2,2) by 1%), Q11 only −2.8% (better), Q21 only
×1.33 worse. **Mechanism:** the chain's per-neuron κ₄ pattern was wrong (corr 0.23–0.53 with MC), so the too-low
level had acted as shrinkage; raising it added error (the κ₄-induced readout error at layer 15 went 1.9e-9 → 4.2e-9).
**Lesson: fix the pattern before the level.** With the σ⁴ pattern closure, the same C0·D gives ×1.13.

#### 38. Other second-order terms of the one-step — no effect on the post (2,2)
ρ³…ρ⁶ (0.963/0.956), K22² (0.963/0.956), C0·K22 + C0²·K22 (0.963/0.956), t·C0 (0.960/0.953), κ₃² D-blocks
(0.960/0.951) at layers 10/14, against 0.963/0.956 without them.

#### 39. Replacing old sources by the amplitude-mode closure — the right shape, not enough accuracy
Positive homogeneity predicts an amplitude-mode κ₃ ∝ Sym(μ ⊗ C). The old-source κ₃ diagonal does follow μᵢvarᵢ
(R² 0.92 at depth), and the old (2,1) slice follows (2μᵢC0ᵢⱼ + μⱼvarᵢ) (R² 0.87). Still, window 3 + absorbed tail
gives 6.0e-7, and even a per-layer oracle fit of the form gives 5.2e-7 (full chain 3.1e-8). The old tail is ~70% of
|t|, so 8% unexplained variance is a ~20% error in t, while the chain needs ~5%. **The cost of old sources cannot be
removed with this form;** it would need R² ≳ 0.998.

#### 40. κ₄→κ₃ feed, part P1 (v × tree births) — slightly harmful
The scale-mixture (2,1,1) slice gives P1 = v × (tree birth). Implemented as g → (1+v) d2, it is slightly worse:
3.11e-8 vs 3.09e-8 on 3 MLPs, and P1P2_mk 2.85e-8 vs P2_mk 2.80e-8 on 10. A likely reason: the v·tree (2,1) part
goes into the thin residual R. Only P2 is shipped.

#### 41. The κ₄→κ₃ feed is NOT what limits κ₃
The MC-measured feed (`mc_feed.py`) is 6–9% of |t| and matches the closure (corr 0.97). The chain's own κ₃ error
(5–6%) is uncorrelated with it (corr 0.00–0.04). Alone, P2 is worth ×1.03; with marg + K31, ×1.125. The κ₃ error
has another source (open; candidates: second-order all-distinct pass-through terms, upstream covariance errors).


#### 42. Marginally corrected leg gating (d1, d2 → Edgeworth-corrected) — slightly harmful
Gating the legs, building the tree and forming the correction all with d1c = d1 + d4·t/6 + d5·K4d/24 (and d2c, w12c)
is the natural extension of "marg" to the κ₃ pass-through. 3 MLPs: 2.714e-8 vs 2.698e-8. It also moves the κ₃ slopes
the wrong way (D 0.964 → 0.962, t 0.985 → 0.983 at layer 15).

#### 43. Rescaling the chain's κ₃ LEVEL to MC — no gain
On the k3f configuration, the κ₃ slopes (MC regressed on the chain) are 0.985 (t) and 0.964 (D) at depth. Part of
that is attenuation from the chain's own pattern noise (corr 0.997 / 0.993). Scaling t and D per layer by these
slopes: t 2.77e-8, D 2.65e-8, both 2.75e-8, vs 2.66e-8 (`k3_level_test.py`). As with κ₄ (§37): **the κ₃ error that
matters is the per-neuron pattern, not the level.**

#### 44. Per-neuron κ₄ pattern beyond varᵢ² — no readout gain
Scale-mixture features with a per-neuron variance response do not help the readout (`k4_pattern2.py`), even when
they raise the correlation (0.86 → 0.89 at depth). The candidates were ρᵢ from the MC (2,2) row sums, varᵢ·ρᵢ and
(varᵢ + c Σⱼ C0ᵢⱼ²/varⱼ)². The best-scalar readout MSE stays best with varᵢ² (1.1e-9 at depth, against 1e-10 with
exact κ₄).

#### 45. The E3 diagram (edge × κ₃(2,1) tree births) — a real mechanism that does not pay **— UPDATED 2026-09-28: on the modern base it is simply WORSE on accuracy (×0.944 on top of the cost-free stack, 0/10 MLPs), not merely unaffordable; see §65**
> ⚠️ **2026-09-24 evening: this entry was measured with a BUG.** `chain.py`'s E3 source carried hub weight ½·d3, i.e.
> half of the reference normalisation (ARC `factor_k3.py` augment legs; checked to 1e-16 by `paths_algebra_check.py`),
> while the correction leg subtracted the FULL-strength (2,1) birth slice. So the numbers below are for a half-strength
> E3 with an inexact (2,1) slice at birth. Re-measured with the `paths` switch — see §51.
With exact inputs, E3 explains a third of the κ₃ one-step residual at depth (coef 0.87–0.92, theory 1). In the chain it
gives ×1.057, but its extra leg costs +2 products per source-layer (+50%). Age-limited versions keep little of the gain
(age ≤1 ×1.02, ≤4 ×1.04). E3 content persists after birth, like the tree's.

#### 46. A "memoryless" κ₃ chain with the amplitude closure for old content — closed
The next-layer diagonal is dominated by repeated-index slices, which SIMPLE computes exactly. The old content
enters only through the all-distinct pass-through (~23% of |t′|). That part matches the amplitude form with R² only
0.09–0.39, so the closure would leave an ~18% error in t (the chain has 7%).

#### 47. Young κ₄ births: only the free star-diagonal part is worth keeping (×1.017)
Pair (2,2) births are useless on top of the closure. Path-diagonal births cost 1 product per layer for +0.6%.

#### 48. Carrying only the "kink" hub columns of the legs — ~~not worth it~~ **REVISED 2026-09-29: the DEAD side is a lever (×0.9928 raw on 10/10, projected ×0.936), the SATURATED side is not (×1.12)** — `experiments/2026-09-29_colcut/RESULTS.md`. The premise below holds only for dead parents; a saturated neuron is ≈ identity, so κ₃ passes *through* it and its column carries transported content that is not ∝ d2. Original text:
For neurons in the linear regime (|μ/σ| large), all hub weights (d2, α, ΔK3) are ≈0, so their leg columns are
negligible. But the sources born early live longest and dominate the cost, and at layers 0–2 every neuron is at the
kink. The cost-weighted kept fraction for |a| < 3 is 0.93, i.e. a saving of about 7%. (The P2 feed also needs
full columns.)

#### 49. Strassen engineering dead ends (k3g, L4 at chunk 8, einsum)
- Rewriting the combos so that no op takes two quadrant views (`estimator_k3g.py`, bitwise identical) did not
  lower the residual, although a microbenchmark had suggested it would. The residual is set by the op count.
- Plain L4 at chunk 8: residual 0.25–0.30 s and wall 20 s locally. Superseded by the k3h layout (§17 of METHOD).
- flopscope `einsum('gkij,gkjl->gil')`: correct billing, 70× slower. Use matmul + sum.

#### 51. ~~The factorable path births are harmful alone~~ — WRONG: a 3-MLP artefact; on 10 MLPs they are ×1.10 (→ k3i)
`chain.py` switch `paths`: all O(n)-rank (1,1,1) births of ARC's `factor_k3.py` with `augment=True` whose other edge is
C0 share the hub leg A and the tree leg B, so together they are Sym(T·Y ⊗ 3A ⊗ B), ONE extra leg, hub weights inside
Y = d1 Dᵀ d3 (E3) + d2 D d2 + ½ d2 K22 d3 + d2 C0² d3 (verified vs ARC's code to 5e-16, `paths_algebra_check.py`).
- 3 MLPs (0–2) said "worse" (2.7449e-8 vs 2.6983e-8). **10 MLPs: 2.5499e-8 vs 2.7989e-8 (×1.098), 8/10 better** —
  MLPs 0–2 happen to be the unlucky ones. **Lesson: 3 MLPs cannot resolve ~5–10% effects** (per-MLP spread ±15%);
  use 10 for any accept/reject decision.
- Against MC (MLP 0, `aug_vs_chain.py`): with paths the κ₃-diagonal readout injection drops to the MC noise floor at
  every layer, like ARC's AUGMENT reference.
- Only the first readout matters: age ≤1 2.5748e-8, ≤2 2.5846e-8, ≤4 2.5679e-8, all 2.5499e-8; low-rank Y (64)
  2.6766e-8 (bad). At age ≤1, t-only 2.5817e-8, t + merged D pairings 2.5780e-8, + the (A∘B)Xᵀ pairing 2.5748e-8.
- Shipped as k3i (METHOD §18). The E3-alone test is still worse on 3 MLPs (2.7973e-8); not re-run on 10.

#### 54. Shared subspace for old sources in production (k3k) — exact, but the entry update eats the saving
torch: 6:320 +1.3% MSE (ideal), transported basis with one subspace iteration +1.2%. The flopscope port
(`src/estimator_k3k.py`) matches the torch mirror to 3–6e-7 but costs **27.78% of B vs k3j 26.62%**: every layer a source
enters the band, and the update (two Gram applications on r+32 vectors, QR, eigh, rotating all band coefficients, new
coefficients) is ≈4–7 n³-products in plain billing, while the saving is ≈1.2 per band source-layer. Best case after
optimisation ≈×1.11. Parked (experiments/2026-09-25_cost/RESULTS.md §7).

#### 53. Folding the path leg into the tree leg at age 2 — catastrophic
Instead of dropping the path leg after its first readout, B ← B + X·diag(1/(2g)) makes Sym(B̃⊗3A⊗B̃ g) = tree + path +
a quadratic term, at no cost. 10 MLPs: 1.1530e-7 (k3i2 2.4441e-8). B is also used by the correction leg (3B diag α) and
the P2 feed, and X/(2d2) is not small at depth (d3/d2 = −α/σ), so the side effects dominate.

#### 52. Rescaling the κ₄ level at every layer — a compounding gain, not a level test
`k4mult` (K4d, K22 × f at every layer, k3f): f = 0.95 → 3.49e-8, 1.05 → 7.61e-8, 1.10 → 2.56e-7 (base 2.70e-8).
The factor compounds through the memo22 level transport (1.05¹⁵ ≈ 2). The current level sits at the optimum.

#### 50. Per-layer oracle substitution of the MC covariance / (2,1) slice at 4e6 samples — not a valid oracle
The MC pair noise (≈0.4% per pair for C) adds up coherently in the next variance to ≈1e-3 relative, the same size
as the chain's own error. Result: "C exact" 2.72e-8 vs 2.66e-8 (no gain), "D exact" 3.55e-8 (worse). The diagonals
(var, κ₃, κ₄ from 1e7 samples) are valid oracles; the pair slices need ≫1e7 samples.

## G. The cost-boundary phase (2026-09-25 → 2026-09-29): cheap representations, the principle, the floor

Everything here was tried after k3j/k3l/k3m were banked, to get util from ~25% towards the ~10.5% the leaders sit at.
Five cheap-representation ideas close by one mechanism (§54 k3k, §55, §56, §58, §59: *a cost model that prices the
saved product and not the construction says yes every time*), then the principle underneath (§62: (t, D) is not a
sufficient κ₃ state; the missing part is orthogonal to everything O(n²)), then its weakest point (§66 MST, §63 jump).
§57/§60/§61/§64/§65 are the accuracy side. **Two live levers came out of this phase and are NOT here:** the matched
dead-side row + column cut (§48 revised; `experiments/2026-09-29_colcut/`) and the transported band basis
(`experiments/2026-09-29_band/`).

#### 55. `pathrank` (low-rank path leg) — the cost model's saving does not exist in production (arithmetic, 2026-09-26)
The cost-frontier sweep (2026-09-26_cost-frontier §2.2, §4) put `pathrank 256` first on the port list: +0.66% MSE on
10 MLPs for −0.7 points of B, ×0.979 adjusted. Its cost model is `q_path = ρ/n` — a **pure volume reduction of the
transport**, which assumes the rank-ρ factors are already in hand. They are not, and producing them costs more than
the transport it replaces.

The path leg is read exactly once, at the first readout (`e3age=1`):
`t_i = 3 Σ_j A_ij B_ij (Wᵗ Y)_ij` with `A = Wᵗ`, so the whole use of Y is one **trilinear** contraction
`Σ_k Σ_j Wᵗ_ik Y_kj M_ij`, `M = Wᵗ ∘ (Wᵗ B)` (free: both factors already exist). Count in units of one dense
n³ product, q = ρ/n = ¼ at ρ = 256, the Strassen billing 0.636 applying to everything that rides a stack:

| what | now | with `Y ≈ Q Z`, Q n×ρ |
|---|---|---|
| basis of Y (`Y Ω`, then `Z = Qᵗ Y`) | — | 2q |
| transport / contraction (`Wᵗ Q`, then `(Wᵗ Q) Z` or `M Zᵗ`) | 1 | 2q |
| **total** | **1** | **4q = 1.0** |

So ρ = 256 is **exactly break-even before the QR** (`fnp.linalg.qr` on n×ρ adds ≈0.125 units), i.e. strictly worse,
and ρ < n/4 only helps if the basis were free. A power iteration (the range finder the chain uses everywhere else,
`corr="split24rr"`) makes it 5q–6q, far worse. The break-even rank for the transport alone (2q < 1) is ρ < 512, which
is where the accuracy of the truncation is no longer interesting.

**A cheap deterministic basis does not exist either.** With the closure's K22 (`K22_ij = v(var_i var_j + 2C0_ij²)`),
`Y = d1 Dᵗ d3 + d2 D d2 + (1+v) d2 C0² d3 + (rank-1 in var)`. The rank-1 part *is* free (it contracts to
`(Wᵗa)_i (M b)_i`, O(n²)), and it is worth nothing on its own; everything else needs a low-rank basis of D, Dᵗ or
C0², all closed by measurement (§18 Tucker, §20 low-rank D, §22 per-source low rank).

#### 56. Strassen level 5 — the meter pays (×0.9165) but the wall time doubles (2026-09-26, measured)
`experiments/2026-09-26_strassen5/RESULTS.md`. Lifting the recursion floor from `m < 128` to `m < 64` gives five
levels at n = 1024: util **25.169% → 23.067% (×0.9165**, less than the modelled ×0.88, still 8.4%), final-layer
difference 1.5e-6 relative, residual fine. **Wall 19.7 s → 38.4 s (×1.95)** and peak RSS 7.81 GB of an 8 GB cap.
332403 measured the grader at 60.9 s mean / **75.3 s max** for 19.4 s local (3.1× mean, **3.9× worst MLP**), so
×1.95 means ≈119 s mean and ≈147 s worst against a **120 s cap** — a breach zeroes that MLP. `chunk 8` halves the
region but not the wall (38.2 s) and costs residual (0.364 s), as §49 already found for L4. Mechanism: level 5
replaces 38 416 matmuls of 64³ with 268 912 of 32³ and, worse, runs 20 element-wise combine/recombine passes over a
275 M-float (1.1 GB) buffer — the level is **memory-bandwidth bound**, which no layout change fixes (§49, k3g).
**With this, the cost axis is exhausted on every known switch** (deadcut shipped, pathrank §55, band §54, window §16,
srank neutral, L5 here).

#### 57. Cheap corrections of the variance PATTERN — all harmful, and the reason is coherence (2026-09-26, measured)
`experiments/2026-09-26_variance/RESULTS.md`. New `chain.py` switches `varfix` / `varfix_late` (default off,
`k4memo 3` still 3.7533e-8).
- **The chain sits in a parabola of half-width ~1e-5.** `var ← var(1+c)` at every layer, MLP 0:
  MSE ≈ 2.4e-8 + 5c², minimum at **c ≈ 0 to within 3e-5**; c = 1e-4 costs **×2.7**, c = 1e-3 costs ×200.
  So the variance *level* is already calibrated to ≈1e-5 — level fixes are closed for the third time (§37, §43).
- **Every low-parameter pattern correction makes it worse:** per-layer level −0.59×, level + log var −0.54×,
  a 7-feature per-layer least squares −0.535× (all "worse than base" factors). R² of the fit is only **0.098**.
- **Mechanism.** A smooth function of (α, log var, κ₄/var², …) is *coherent across neurons*, and the readout sums
  coherently; an incoherent per-neuron error is suppressed by √n. Measured: rms pattern error 1.8e-3 at depth is
  worth the same as a coherent 4.1e-5 shift — a factor **44 ≈ √n**. So a fit pays the coherent price of its own
  smoothness and collects only the √n-discounted part of what it captures.
- ⚠️ **Method warning that cost us an hour:** a *multiplicative* correction measured on the base trajectory is
  **stale from layer 1 on** (touching var moves the chain's own var), and at ×2.7-per-1e-4 stiffness the drift is
  fatal — the "exact" arm built that way reads ×0.658, which is NOT a contradiction of the ×1.54 oracle arm
  (that one substitutes the trajectory-independent absolute value). Test only corrections that are a function of
  the current state — which is also all production could compute.
- Layers 0–6 are **at the 1e7 oracle noise floor** (rms e 3.2e-4 vs noise 3.1e-4): the variance error is a
  deep-layer accumulation, and seeing it at all needs N ≫ 1e7.

**Measured on the real meter by the cloud (2026-09-26/27, `experiments/2026-09-26_k3m-port/`), which sharpens the
arithmetic above:** the dense path leg rides the Strassen stack and costs **0.602 units/layer**, while the truncation
machinery (factorisation + rebuilding Y for the birth's (2,1) slice + thin transport + materialisation) gets no Strassen
discount and costs ≈3.8·r/n. **pathrank 256 raises util 25.169% → 25.715% (×1.0217)** and raw +0.77%; break-even is
rank ≈158. The accuracy penalty saturates at ≈1.7% by rank 128, so every rank that saves FLOPs costs ≈1.7% raw; the
best measured point, **rank 64: util ×0.980 × raw ×1.017 = ×0.9966**, is inside the grader's resolution. Closed.
**Consequence:** the port list of 2026-09-26_cost-frontier §4 item 1 is withdrawn; `deadcut −2.5` (item 2, shipped as
k3m) stands, and the remaining modelled cost lever with no accuracy cost is **Strassen L5** (billing 0.636 → ≈0.560,
×1.14), which is a residual-time/RSS decision on the Mac. The general lesson repeats §54 (shared subspace): a cost
model that prices only the *saved* product and not the *basis it needs* overstates every projection compression.

#### 58. Merging the old tail into one block — compresses, but the construction does not fit in the budget (2026-09-27, cloud, measured)
`experiments/2026-09-26_old-tail/RESULTS.md`. `chain.py` gained `tailx` (default None; both gates still exact).
**The accuracy side came out well and the cost side killed it.**
- **The tail cannot be dropped on either axis:** without its contribution to D ×26.5, without its contribution to
  t ×41.8 (a0=4, 10 MLPs). Both halves are load-bearing, so a merge has to preserve the whole 3-tensor, not
  two slices: only the legs transport linearly, while `L1 = 2A∘B∘g` and `L2 = B∘B∘g` are rebuilt every layer.
- **But the tail is a SOFT channel:** 1% relative error on t and D costs 0.1–1.1% raw, 10% costs ×1.25–1.47,
  so the budget for ≤3% raw is **3–5% relative error**. The contrast with §57 is the point: there a *coherent*
  1e-4 cost ×2.7. And there is **no √n discount here** — at ε=0.1 the incoherent perturbation (×1.253) costs the
  same as the coherent one (×1.220), because what is perturbed is the already-summed tail vector, not 1024
  independent inputs. So a merge error can be judged by a plain relative norm.
- **The tail's D is effectively rank ≈256** (truncating it there costs ×1.008). This does *not* contradict §21:
  that truncated the FULL D including the young sources, and the young part is the sensitive one.
- **A rank-n block EXISTS** — n=64, converged: 4.29e-4 at r=n, 5.02e-5 at r=2n, i.e. a 10× compression of
  R = 10n rank-one terms. **The tail is not incompressible**, unlike every projection compression (§18, §54).
- **Three measured reasons it still cannot be built:**
  1. the only O(n²) construction, keeping the r heaviest columns, gives **62.8% at r=n on the real n=1024**
     (76.1% at n/2, 43.5% at 2n, 20.8% at 4n) against a 3–5% budget — a 15–20× shortfall;
  2. **the whole merge budget is one fitting pass.** The saving being fought for is 105 units per MLP ≈ 8.8
     units per layer; one Gram-form pass on the incremental problem (block n + graduate n) is 8·n·R·r flops =
     **8.0 units**. Measured at n=1024: 200 passes → 4.54%, 400 → **4.32% and still falling**. A ≈400× gap;
  3. **CP has no nestedness, so warm starting does not help.** Adding one source to a converged block throws the
     error from 2.9% to **48.7%** — worse than cold column selection (42.2%) — and it takes ~10⁴ passes to come
     back. Every layer is a fresh global solve.
- **Verdict: the representation that compresses does not update cheaply.** This is the *fourth* instance of one
  mechanism in this project: §54 (the band's entry update), §55 (pathrank's basis), §56 (L5's memory traffic),
  and here. **A cost model that prices the saved product and not the construction it needs will say yes every
  time.** Write that into any future cost estimate before trusting it.
- ⚠️ **Not yet closed:** the only cheap construction tested is *biased* (keep the heaviest columns), which loses
  the dropped mass coherently — exactly the failure mode of §16. **Unbiased importance sampling with
  reweighting** (p ∝ term weight, weight g/(r·p)) is O(n²), needs no optimisation and is nested by
  construction, and its error is ≈ (Σ‖term‖)/(√r·‖T‖). Whether that ratio is 3% or 300% at n=1024 is one cheap
  number and has not been measured. **→ measured and closed, §59.**

#### 59. Unbiased importance sampling of the tail's CP terms — closed on the first number, κ = 62 (2026-09-28, cloud)
`experiments/2026-09-28_tail-sampling/RESULTS.md`. The crack left open in §58, tested and shut.
For T = Σ_c Sym(a_c⊗b_c⊗b_c), unbiased sampling of r columns with p_c ∝ w_c gives relative error
≈ κ/√r with **κ = (Σ_c w_c)/‖T‖**. Measured at the real n = 1024: **κ = 62.08, κ/√n = 194%** against a 3–5%
budget — the stopping rule's threshold exceeded **10×**. Stable across the grid (κ = 55.9–62.1 for
a0 ∈ {1,2,3,4} × layer ∈ {11,13,15}); the formula was verified against real draws to 1–2%; variance-reduced
variants (hybrid top-k + sampled remainder, systematic sampling) reach at best 86.2% at r = 2n.
**To reach 5% would need r ≈ 1.5e6 columns — 100× more than the R = 15360 the tail contains**, i.e. sampling
converges to "carry the whole tail", which is no compression at all.
- **The mechanism is the part worth keeping.** §16's coherence ("the tail is a sum of same-sign terms") is true
  of the **diagonal readout t**, but the sampling variance is governed by the **Frobenius norm of the tensor**,
  and in *that* norm the tail's terms are nearly orthogonal: κ/√R = 0.50, against 0.008 for perfect coherence
  and 1.0 for orthogonality. So the premise that made unbiased sampling promising **does not hold in the norm
  that sets the error**. t being same-sign and T being near-orthogonal are not in conflict — t is one diagonal
  of the tensor, while sampling damages all of it.
- Stage A at a0 = 1 (never measured before) makes it worse, not better: the only configuration whose prize is
  the multiplier floor is a0 = 1, and its tolerance is ≈1% rather than the 3–5% at a0 = 4.
- **With this the cost axis is closed five times over** — §54, §55, §56, §58, §59 — and each time by the same
  mechanism: the construction costs what it saves.

#### 62. (t, D) is not a sufficient κ₃ state, and what is missing is orthogonal to everything cheap — the cost axis closes on a PRINCIPLE (2026-09-28, cloud)
`experiments/2026-09-28_state-closure/RESULTS.md`. The question under §54–§59: all five tried to *represent* the κ₃
tensor more cheaply; none asked whether it must be represented at all, since `nonlin_slices` reads only **t** and **D**.
- **Exact decomposition.** With the leg transport M = Wᵗ·diag(d₁) (verified to reproduce the chain's own transport
  to ≤5.1e-16), the next diagonal splits by index coincidence into a t-term (Σⱼ Mᵢⱼ³tⱼ), a D-term
  (3Σ_{j≠k} Mᵢⱼ²Mᵢₖ Dⱼₖ) and an **all-distinct** term (Σ over pairwise-distinct j,k,m of Mᵢⱼ Mᵢₖ Mᵢₘ Tⱼₖₘ), the last
  taken as a remainder without ever building n³. Closed forms checked against brute force to 6e-17 / 4.3e-16.
- **The all-distinct part is 28.8% of the next diagonal per layer** (layers ≥ 8, range 27.2–31.2%; 35–61% in layers 1–3).
  Of the predictable part, D carries ≈95% and t only ≈0.9%. That is far *less* than the naive n× dominance a generic
  tensor would give — the tensor really is concentrated on the two slices — but it compounds: **a (t, D)-state chain
  is ×25.6 on raw end-to-end** (6.82e-7 against 2.67e-8; 1.288¹⁴ ≈ 34 predicted).
- **It turns into a specification.** An oracle closer that captures a fraction φ of the all-distinct part gives raw
  ×1.116 at φ = 0.90 and ×0.992 at φ = 0.95 — so **a κ₃ closure must capture ≥ 93%**.
- **And the best oracle closer on a cheap basis captures 19.5%.** Re-fitting the t- and D-coefficients captures 0.16%,
  because the missing part is **orthogonal to both** (corr ±0.03) — the D-coefficient fits to 1.000 (0.98–1.03), i.e.
  D is already exact and what is missing is a different direction, not "a bit more D". A physically motivated star
  ansatz Tⱼₖₘ ≈ Sym(uⱼ C0ₖₘ) → (Mu)ᵢ·diag(MC₀Mᵀ)ᵢ captures 18.8% (10.6% at layer 8 → 26.7% at layer 14: the tensor does
  become more star-like with depth, but nowhere near 93%); all five basis vectors together 19.5%.
- **Why κ₄ can be closed and κ₃ cannot.** The κ₄ scale-mixture closure works in this very codebase because κ₄ enters
  the readout at **second order**; κ₃ **is** the readout, so the tolerance is an order of magnitude tighter.
- **Verdict — the statement one level above §54–§59:** the κ₃ state must carry Θ(n³) information forward, and the
  chain's cost is the price of that information. There is no cheap state not because we searched badly for a
  representation, but because the nonlinear step reads two slices that are **not closed**, and what is missing is
  **orthogonal** to everything computable in O(n²).
- **Honestly not closed:** a closure with a *larger* state (several extra O(n²) objects, i.e. a tensor sketch) — but that
  is representing the tensor again and returns to §54–§59; and predictability from observables that are not in the
  state at all.
  **→ The larger-state door was then tested in its most natural form (exact young sources + a slice-state tail, the
  "Slice-Markov tail") and closed: §66.** The jumped old tier with exact resets: §63.
- ⚠️ **The open puzzle this leaves:** thirteen entrants sit at 10.5–18% of budget with raw **better** than ours. If the
  κ₃ state must carry Θ(n³), then either they do not propagate κ₃ the way this chain does, or their representation of
  it escapes the argument in a way none of our five attempts did. We do not know which.

#### 60. Scalar level tuning is below the grader's resolution (2026-09-28, measured on the grader)
k3n = k3m + `feedmult 0.8` + `k22c 0.0`, two scalars fitted on 10 torch MLPs. Torch ×1.0089 (9/10) →
flopscope 100 MLPs ×1.0030 (66/100) → **grader 5.3977e-9 against k3m's 5.3906e-9, i.e. 0.13% WORSE**
(submission 332872, 100/100, 0 failures, util 24.834%). The local gain was real (66/100, p ≈ 0.001) and it
still did not survive the change of test set.
**Consequence — the rule to apply from now on:** under grader schema 2.0 the ranked number is a mean over
**50 MLPs we never see**, and the observed local→grader discrepancy on two submissions that differ by 0.3% is
**≈0.4–0.5%**. So **a lever worth less than ~2–3% is not worth a submission**, and no amount of scalar tuning
on our baked set can be trusted below that. Structural levers (a term added, a rank raised) are a different
class and are not covered by this rule — but none of the live ones is affordable (§61).

#### 61. Every live accuracy lever costs more than it is worth at 24.8% utilisation (2026-09-28)
`experiments/2026-09-28_accuracy-free/`. At the floor the cost-free stack is ×1.11 and the extra levers take it
to ×1.1122 (10/10 MLPs). None of it ships at our actual utilisation:
- `k4k3` (×1.0396, the best of them) costs **≈4 n³ products per source per layer** — four times the leg
  transports, i.e. it would roughly triple the budget;
- `thin384` costs ≈+5 points of B for ×1.0079; `pathD="full"` and `allages` add a path product per source
  per layer (≈+12.8 points) for ×1.025 and ×1.081;
- `k4b` alone IS free (O(n²) with `k4bparts=("sd",)`) but is **monotonically harmful on the shippable base**
  (k4b = 5 → ×0.977): it only pays together with all-age path births, because with `e3age=1` the deeper κ₄
  births are inconsistent with a κ₃ that carries no path content at those ages.
So the entire accuracy stack is **conditional on the cost breakthrough**, and the cost breakthrough is §54–§59.

#### 63. Jumping the old tier k layers with a slice rule between checkpoints — closed for every a0 < 10 (2026-09-29, Mac)
`experiments/2026-09-29_stride/RESULTS.md`; `chain.py stride=(a0, k, mode)`, self-check `(4,1)` = base to 2.2e-15.
The idea (the ideation workflow's completeness critic, the one class all seven lenses missed): legs are moved one
layer at a time only because every source is read at every layer; since all W are known, the old tier could be read
exactly only at checkpoints and **jumped** through one shared operator J = M_{c+k}⋯M_{c+1}, with the exact
repeated-index rule (state-closure §1, everything but the all-distinct term) supplying its (t, D) in between and an
exact reset at every checkpoint. Projected util ×0.65–×0.79, and it would have softened the MST tolerance 5–10×.
**Measured (MLP 0; the verdict is far outside the ±15% per-MLP spread):** (4,2,mst) **×3.51**; (4,2,Donly — legs
still move, only the old D pairings skipped) ×2.49; (6,2,Donly) ×1.49; (8,2) ×1.18–1.22; (10,2,mst) ×0.9945.
**Mechanism:** the all-distinct content is load-bearing **per source, per step**, for every source younger than ~10
layers — the loss is not cumulative, one un-reset step already costs ×1.5+. Only content older than ~10 layers has
become slice-concentrated, and that tier is ≈17.5% of source-layers. **Consequence:** MST (a0 = 2–5, no reset at
all) is predicted to fail by far more than ×1.05, so the cloud prompt for it was withdrawn — but the cloud had already run it and it failed as predicted (§66). (10-MLP
confirmation of the curve was queued but not run.)

#### 64. A post-hoc or learned corrector of the final-layer error — nothing to learn (2026-09-27, Mac)
`experiments/2026-09-27_residual/RESULTS.md`. The residual of the chain at layer 15 (10 baked MLPs): **coherent
(per-MLP mean) share of the MSE 0.46%**; correlation with every local feature ≤ 0.03 in absolute value (α −0.016,
σ −0.024, var −0.024, skewness −0.029, κ₄/var² +0.011, the prediction −0.019, the last step's change −0.013), so
R² ≤ 0.001; and **corr(residual at layer 14, residual at layer 15) = 0.004** — each layer's random W scrambles the
error vector. The residual is a random projection of accumulated upstream error, not a function of anything local;
71% of it sits on neurons with α > 1 (where ReLU is the identity and the residual is the propagated mean's error).
Independently confirmed by 504aldo (forum 18218, row 10: features correlate < 0.09). Only better propagation, at
every layer, reduces it.

#### 65. Terms re-opened at the multiplier floor and found WORSE on accuracy, not merely unaffordable (2026-09-28, Mac)
`experiments/2026-09-28_accuracy-free/RESULTS.md`. Everything once rejected as "does not pay" (a cost verdict at 25%
of budget) was re-measured with cost ignored, on top of the ×1.11 cost-free stack (10 MLPs, base 2.0844e-8):
**`aug` (ARC's dense AUGMENT diagrams) ×0.463, 0/10** — it double-counts what we derived independently (`c0d`,
`pairx`, `k31pat`, the path births); **`e3` births ×0.944, 0/10** (this corrects §45); `edge2` (second-order
Edgeworth) ×0.981; `tet=4` ×0.999; `q22x` ≈1.000; the full κ₄ birth parts (§47) ×1.0006. And a methodological
catch: **the exact correction leg (`corr="split1024rr"`) is WORSE than `split384rr`** — the randomized rank is
non-monotone past ~192 and acts as a regulariser. (Beware the `corr="full"` arms of `2026-09-28_ceiling/ceiling_n3`:
`chain.py` takes a separate branch there where the path/age switches are inactive — two different configs returned
identical numbers in a shorter runtime; those rows are not measurements.) And **`k4b` birth depth**, O(n²) and
therefore free, is ×1.0878 inside the stack but **monotonically harmful on the shippable base** (k4b 5 → ×0.977): it
pays only together with all-age path births.

#### 66. The Slice-Markov tail (MST) — §62's "larger state" door, closed by the cloud (2026-09-29, cloud)
`experiments/2026-09-29_mst/RESULTS.md` (branch `cloud/mst`, merged; `mst.py` monkeypatches `chain.py`, which was
not edited). The most natural form of "(t, D) plus more state": sources younger than a0 carried exactly (legs kept);
at age a0 a source folds its exact slices into a tail state (τ, Δ), and from then on the tail moves **only by the
exact repeated-index rule** — no reset, ever. The brief's hypothesis was that old content is more slice-like, so the
old tail's own all-distinct share f would be well below §62's 28.8%.
**Refuted directly:** f = **27.5–28.2% on layers ≥ 8, flat in a0 ∈ {2,3,4,5}** (10 MLPs, pooled 27.76%, n = 320;
per-MLP means 26.7–29.3%, so the spread between MLPs is smaller than the distance to any useful threshold — the first
3-MLP screen gave 27.2–28.0%; the cloud re-measured on the full 10 after the merge, commit 3573cc7); the 35–61% seen on
layers 1–3 in §62 is a shallow-layer effect (few sources), not a young/old difference.
**End-to-end (10 MLPs, base = k3l_t with deadcut off, 2.3057e-8):** a0 = 2 ×18.75 (util 9.57%, below the floor),
3 ×12.38, 4 ×8.14, 5 ×5.33, 6 ×3.69, 8 ×1.89, 10 ×1.20, 12 ×1.016; modelled adjusted at best 6.380e-9 (a0 = 12,
×0.988 of its own base, below the grader's resolution) — **dominated by k3m (5.3906e-9) at every a0.**
**Why monotone although f is constant:** restricting the rule to old sources does not shrink the error per
transfer (it stays ~28%); it shrinks the *number* of approximate transfers each source suffers. So cost and accuracy
are governed by the same parameter in the same direction — no compromise exists.
**Anchors (all exact):** transport ≤1e-10 (hard `raise`); a0 = 99 == base ×1.0000; a0 = 1 == §62's `td` mode to the
last digit on the same 3-MLP clean config (6.8521e-7, ×27.164; §62's ×25.6 is the 10-MLP figure); cost model at a0 = None == `cost.util`. The a0 = 1 anchor first gave ×28.06 and
caught a real bug: the tail's Δ must have **zero diagonal** (the diagonal of a transported (2,1) slice is already
carried by τ; the repeated-index rule is derived with D_jj = 0, else j = k is counted twice).
**Relation to §63 (Mac, same day):** the two are complementary — §63 *resets* the tier exactly every k layers and
still loses ×1.5–×3.5 below a0 ≈ 10; MST never resets and loses ×1.2 even at a0 = 10. The Mac's stride test had
predicted MST's failure before its numbers arrived. Together with §62: the cost axis is closed by a principle
**tested on its own weakest point.**


#### 67. The port battery's dead arms: age-dependent cuts, a0 = 4 bands, E2 below 2.5, un-orthonormalised normal equations (2026-09-29/30, cloud #8)
`experiments/2026-09-29_port-battery/RESULTS.md` §1, §3 (branch `cloud/port-battery`, merged 0c080b7). All on 10 MLPs,
exact gates.
- **E4, a harsher matched cut on old legs only** (rows at the gating, their D over alive_old², birth columns from
  age 4 on; young legs at −2.25): (4, −2.0) raw ×1.0557 (0/10), projected ×1.018; (4, −1.5) ×1.4407 → ×1.274. [U]
  Mechanism = §22's consistency read backwards: a second threshold makes a neuron dead for some sources and alive
  for others, and the old tier carries 69% of ‖t‖ (§16).
- **Bands entering at a0 = 4** (×1.07–1.23 raw) and **(5, 384)** (×1.154): the transported span is too small that young.
- **E2 below θ = 2.5:** 2.0 restore ×1.0854, sep ×1.1616 (the cut starts removing rows whose tail weights are still
  reliable); 2.25 no better than 2.5. The optimum 2.5 is confirmed on the production path (k3q: 2.0 ×1.097 adjusted).
- **The band with the normal equations and no re-orthonormalisation:** κ(PᵀP) = κ(P)² reaches 2e9 at layer 15 → leg
  error 100% on MLP 3 (×2.09 there, ×1.12 mean). The QR projection has no such problem (the production choice).
- **E3 arm a at −3.0** (receiver rows filled from the age-1 source only): ×1.0102 — for α < −3 the first-order
  Edgeworth correction of the post mean is 1–2× the Gaussian mean itself, so the dead neurons' own readout is their
  largest term; arm b (repeated-index fill-in) recovers 80%.

#### 68. The tails hold none of the accuracy gap — the error is born in the kink core (2026-09-29/30, cloud #9A)
The ×1.8 to top-5 was attributed to the Edgeworth series "breaking down in the tails" (#8 measured tail corrections of
28–79% of d2 at α > 2.5 and 1–2× the Gaussian mean at α < −3, and both consistency cuts that removed tail terms
*improved* raw). The exhaustive test says that reading was wrong.
- **Attribution.** Exact post-activation means substituted for one regime at a time, 10 MLPs, base B9 = 2.1967e-08:
  dead (α < −2.5) **×0.9953**, saturated **×0.4915**, kink **×0.2095**, all **×0.2065**. Inside the kink:
  core (|α| < 1) **×0.3356**, saturating shoulder ×0.5234, dead shoulder ×0.9255. So ~79% of the final error travels
  through the intermediate post means and is born **where the ReLU is most nonlinear and the series most reliable**,
  not at the shoulders. The saturated regime is a *carrier* (d1 ≈ 1), not a source. Per channel (MC oracles, MLPs 0
  and 1, two seeds): exact *kink* moments reproduce exact moments everywhere in every channel (joint ×0.117/×0.128
  pooled; var ×0.64/×0.82, κ₄ ×0.78/×0.78, κ₃ ×0.88/×0.88); **exact dead-tail moments are neutral (0.99–1.00)** and
  **exact saturated variance is HARMFUL (×1.02–1.09)** — exact but inconsistent with the chain's own covariance rows.
- **The truncated series does not break down.** Fed exact moments, the chain's first-order Edgeworth readout is
  accurate to ~1e-12 of layer MSE in BOTH tails and ~1e-10 in the kink (= §31's 9.7e-11). A proper-density tail form
  (shifted lognormal matched to μ, var, κ₃) is **50–100× worse** in the tails and 200× worse in the kink (it cannot
  carry κ₄). Second order (k₃²) helps only in the kink (×0.8) and hurts in the tails.
- **The exhaustive screen.** 149 arms = every term × tail × θ ∈ {2.0, 2.5, 3.0}: **18 survive, 131 killed**. The
  saturated side is *structural* (×1.02 … ×9000; **E2's transpose refuted at ×54.7** — D_ij reaches Cov(y_i,y_j)
  through b(1)_j = d1_j ≈ 1); the dead side is *inert* (best cuts ≤ 0.27% each on the screen, **0.17% together on 32
  fresh MLPs** — the winner's curse of a 149-arm selection). 45 pairs of the top 10: **no consistency pair** — every
  finished pair equals the product of its parts to 0.9999–1.0011.
- **What survived is not a tail lever: marg2** (§69, a positive). **Mechanism of the whole picture:** every measured
  gain of #8 and #9A is *consistency between coupled channels* — the matched cut removes a dead neuron from row and
  column together; E2 needs its covariance and leg channels together; marg2 must be Gaussian in Q11+Q21+Q22 alike;
  exact saturated variance is bad precisely because it is inconsistent. The chain is hurt where two channels that
  must share a term do not.
Files: `experiments/2026-09-30_tail-accuracy/` (RESULTS.md, `ta.py`, `grid.py`, `t1_attrib.py`, `t2_readout.py`, logs).

#### 69. Strassen level 5 is dead on wall time and memory, not on the meter (2026-10-01, Mac)
`strassen_levels=5` alone had been a **no-op** all along: `_Strassen.__init__` breaks out of the recursion at
`m < 128`, so L4 and L5 wrappers are bit-identical (3 MLPs: same FLOPs 4.4074e11, same raw, same 19375 calls). With
the floor lowered (`strassen_min=64`, leaf 32×32, 7⁵ = 16807 groups), 10 MLPs on top of k3q:
**util 19.725% → 18.520% (×0.9389), raw ×0.9991, adjusted ×0.9380** — and **wall 21.2/21.7 s → 45.9/46.9 s (×2.2)**
with **peak RSS 7.21 GB** of an 8 GB cap. Residual is fine (0.367/0.385 s vs 0.361/0.393 s; +12% calls). The grader
runs at ≈3.0× the local wall (k3q 21.7 s local → 63.7 s graded), so L5 projects to ≈140 s against the **120 s cap**.
**Mechanism:** a 32×32 leaf has no BLAS efficiency left — ×2.2 wall bought ×0.936 FLOPs. §56's "L5 is closed by wall
time" is now a measurement, with the memory wall as a second, independent blocker. The ×0.938 would have been the
largest single remaining lever (4.13e-9 → 3.87e-9); it cannot ship.
Files: `experiments/2026-10-01_k3r/runs/` (`k3r_L5real_n10.json`, `l5_rss.txt`, `L4_n3.json`, `L5_n3.json`).

#### 70. A learned corrector of the chain's own moments: the predictable part of the error is the part that does not matter (2026-10-01, Mac)
The one public mechanism we did not have (mliston, forum 18170: a frozen model trained offline on thousands of random
MLPs predicting the closure's error from its own propagating moment state, raw ×0.81–0.83 in Phase 1) was tested in
one afternoon on the two 1e7 MC oracles of MLPs 0 and 1, base B9. It is **not** 504aldo's dead end #10 nor §57 — those
killed *output-level* corrections from *final-layer* features; this corrects the moments **mid-chain**, where §68 says
79% of the error is born. It still fails, and the mechanism is the useful part.
- **The signal is real.** From layer 8 the moment error reproduces across independent oracles (corr 0.94 var / 0.99 κ₃
  / 0.99 κ₄), and a *linear* model on 20 smooth per-neuron features the chain already computes reaches, at L15,
  **R² = 0.51 (κ₃), 0.56 (κ₄)**, 0.10 (var) in sample (p/n bias ≈2%; two oracles agree to 0.015).
- **The harness was gated first**: `mcorr` with the exact error reproduces §68's oracle arm (×0.206/0.230 on MLP 0,
  ×0.302/0.320 on MLP 1). From that gate: a partial move toward truth helps **monotonically** (s = 0.25 → ×0.59);
  the channels work only **together** (joint ×0.21–0.30 vs 0.69–1.09 alone, and exact variance alone is ×1.09 =
  *harmful* on MLP 1); correcting from layer 8 on beats correcting everywhere (×0.189 vs ×0.206).
- **A magnitude ceiling promised a lot and was falsified.** An OLS predictor leaves √(1−R²) of the error, so matching
  magnitudes gives s_eq = 1 − √(1−R²) = 0.051/0.300/0.337 per channel; the exact correction at those strengths gives
  **×0.70 on both MLPs** (and ×0.53 if var were as predictable as κ₃). The real predictor does not come close.
- **Online it diverges** out of sample (NaN) and is ×1.5–2.0 *worse* in sample: fitted on the uncorrected trajectory,
  applied to a trajectory it then moves, the linear model extrapolates and runs away.
- **Offline (no feedback), in sample, where the fit is perfect, it buys ×0.93–0.98** — 2–7%, i.e. the magnitude
  ceiling overstates by ≈6×. Out of sample the coefficients do not transfer at all.
- **The mechanism (the result to keep).** Decompose the exact error as e = **P**e + (I−**P**)e with **P** the
  projection onto those 20 smooth features, and substitute each part alone:

  | removed | MLP 0 | MLP 1 |
  |---|---|---|
  | all of e | ×0.206 | ×0.302 |
  | **P**e (smooth, predictable) | **×1.015** | **×0.957** |
  | (I−**P**)e (rough, orthogonal) | **×0.269** | **×0.450** |

  **The rough part alone recovers almost the whole ceiling; the predictable part buys nothing.** Cross-checked by a
  second code path (offline in-sample arms, a different target: ×1.019/×0.938). So the damaging component of the
  moment error is precisely what a per-neuron-state model cannot see — §62's "orthogonal to everything O(n²)" and
  504aldo's "high-rank, not predictable from live-state features", now measured mid-chain at the one place where the
  direction looked most promising.
- **The pair-level extension, tested 2026-10-01 — and the mechanism that closes the family.** The "onto a closed
  door" caveat above was **wrong**: the chain already materialises C0, D and K22 every layer, so ROW REDUCTIONS of
  them are O(n²), i.e. free against n³. Added 17 such features (|C0| row sums, ΣC0², ΣC0³, Σ|C0|^1.5, max|C0|, the
  same for D by row and column, and for K22) to the 20 own-moment ones (`pairfeat.py`, MLP 0, R² in sample):

  | layer | var: own → own+pair (ceiling) | κ₃ | κ₄ |
  |---|---|---|---|
  | 8 | 0.071 → 0.095 (0.685) | 0.141 → 0.162 | 0.288 → **0.384** |
  | 12 | 0.094 → 0.124 (0.904) | 0.408 → 0.438 | 0.512 → 0.563 |
  | 15 | 0.095 → **0.130** (0.943) | 0.512 → 0.533 | 0.561 → **0.641** |

  κ₄ gains materially, **the variance channel barely moves** — and it is the variance that carries the oracle
  leverage (§68: exact var ×0.64/×0.82 against κ₄'s ×0.78). **Why it cannot be predicted per neuron:**
  Var(z_i) = Σ_jk W_ij W_ik Cov(y_j, y_k), so the chain's error in neuron i's variance is a **W-weighted quadratic
  form of the (unknown) n×n error matrix ΔC — a different linear functional for every i**. No per-neuron summary of
  the chain's state determines it; knowing it means knowing ΔC, i.e. having solved the problem. That is the
  mechanism behind both this entry and §62, and it closes the learned-corrector family for good.
Files: `experiments/2026-10-01_corrector/` (`RESULTS.md`, `moment_signal.py`, `xmlp.py`, `exact_check.py`,
`ceiling.py`, `offline.py`, `split_dir.py`); `chain.py` gained the `mcorr` hook (off by default, bitwise).


#### 71. A JOINT (variance, κ₄) corrector: the in-sample synergy is a fit to each network's own rough error (2026-10-02, Mac)
§70 removed the feature-predictable error one channel at a time; the exact correction, though, only works jointly
(×0.21–0.30 against 0.69–1.09 alone). So the joint version was tested (`experiments/2026-10-01_corrector/`, Part 4).
**In sample** (offline, own+pair projection at layers 1–15) the variance projection — harmful alone, ×1.03 / ×1.10 —
**helps together with κ₄**: var+κ₄ ×0.885 / ×0.900 against κ₄ alone ×0.916 / ×0.918 (gate: exact joint ×0.2060 /
×0.3024 = exact_check). **Out of sample it does not exist:** the k4loo protocol on 10 MLPs gives the variance error a
held-out R² of **0.004** (κ₄ 0.361), variance alone ×1.019 (0/10), and var+κ₄ ×0.980 (9/10) / ×0.987 (s = 1, 7/10)
against κ₄ alone ×0.974 (10/10) — the variance channel makes the corrector worse. Mechanism as in §70: neuron i's
variance error is wᵢᵀ ΔC wᵢ, a different functional of the unknown n×n matrix ΔC for every i, so per-neuron features
can only fit it network by network. The variance lever, if one exists, is ΔC itself (cloud #11).


#### 72. The variance channel is the off-diagonal post covariance, and nothing cheap carries it (2026-10-02, cloud #11)
`experiments/2026-10-02_cov-error/` (four 1e7 full post-covariance oracles, MLPs 0–1 × two seeds, layers 0–14).
**Mechanism, exact:** ΔC(l) depends on W_0..W_l only, so the next layer's variance error e_i = w_iᵀ ΔC w_i splits into a
coherent level (2/n)·tr ΔC (the diagonal; the chain sits at its own MSE optimum there — the exact diagonal hurts only
through its level, neutral with the level kept) and **e_off,i = w_iᵀ ΔC₀ w_i: mean 0, variance 8‖ΔC₀‖²_F/n², iid across
neurons** (verified 0.89–1.09). The off-diagonal carries **95%** of the next-variance error signal on both MLPs (90–105%
at every layer ≥ 3). This is the "rough residual" of §70 / #10 explained: orthogonal to every per-neuron feature.
**The money is real — as an upper bound:** exact off-diagonal substituted at layers 5/8/11/14 ≈×0.63 / ≈×0.80, at every
layer ≈×0.55 / ≈×0.68 (MLPs 0 / 1, noise-corrected two ways). **No structure carries it:** two reproducible
mean-aligned eigendirections hold 40–65% of ‖ΔC₀‖² and buy ×0.93–1.07 (the payoff appears from rank ~64 and lives in the
bulk); sparsity needs 20% of all pairs chosen by the oracle (top 1% buys nothing; the top pairs are large-|D|, large-|C₀|
pairs, not kink-core pairs); a linear model on 27 O(n²) matrices of the chain reaches R² ≤ 0.10 in sample / ≤ 0.06 out of
sample against a noise ceiling of 0.87–0.98; the proportional (ΔC₀ ∝ C₀) and per-neuron-gain models are neutral or
harmful. **65–76% of ΔC₀ at depth is carried** through earlier independent W (the born part, measured by running the
chain's step on the exact C_pre, is 22–30% and only 6–15% linear in local features). A correction would need the n×n
matrix itself — the pair-level n³ wall of §54–§62. Also measured: the chain is very sensitive to off-diagonal noise —
2e7-sample noise added at every layer costs ×3.2–3.9.


#### 73. The step's pair inputs D and K22 against Monte-Carlo oracles: wrong, but only worth fixing all at once (2026-10-02, Mac)
`experiments/2026-10-02_dslice/` (oracles of D = κ₃[i,i,j] and K22 = κ₄[i,i,j,j] at layers 9 and 12, MLPs 0 and 1, two
seeds; `chain.py` got a `K22_layers` oracle key, bitwise off). Cloud #11 had left the born covariance error unexplained
(the step on the exact input covariance still makes 22–30% of ΔC₀). **D makes 73–78% of that born error** (MLP 0), and
D is 7% wrong on the block the chain computes (corr 0.997, no level error); ≈85% of that 7% survives every tunable
approximation (thin rank, band, cuts, κ₄ birth depth) — it is the k = 3 truncation of the births and their inputs.
Exact D at two layers is worth ×0.885 / ×0.931 (complete), but **its parts are network-dependent**: the computed block
alone ×0.900 on MLP 0 and ×1.027 on MLP 1; the dead rows/columns alone ×0.993 / ×1.094. K22 is the closure's ansatz and
is 20% wrong (corr 0.62–0.70); pair features predict the true K22 out of sample better than the ansatz (R² 0.45–0.61 vs
0.35–0.45), but **exact K22 at two layers is ×0.970 on MLP 1 and ×1.102 on MLP 0**. The errors of the step's inputs
compensate one another and the rest of the state — the consistency mechanism of §68 seen input by input — so any
correction of one input alone, learned or exact, is a coin toss across networks. Also re-priced: `pathD="full"` with
path births at the first readout costs one product per layer (not per source, as §61 had it with all-age paths) and
gives raw ×0.983 on 10 networks (8/10) — but that product spans all ≈850 hubs of the newest source, +2.2% of the FLOPs
with Strassen (+3.8% plain): net zero, §61's verdict stands.

#### 74. The next cumulant order: worth ×3–7 at small width, but no partial carrier of it pays (2026-10-02/03, Mac)
`experiments/2026-10-03_korder/` (RESULTS §1–§4). Question (the user's): what method do the top-3 use that we lack?
ARC's reference ladder against Monte-Carlo at small width: our chain = k=3 AUGMENT level; **k_max=4 (the full κ₄
tensor; same Edgeworth budget as k=3 AUGMENT) is ×3 better at n=128/L=8 and ×6–7 at n=256/L=8** — consistent with
c_K (L/n)^K, ≈×10 at our L/n. Its statistics substituted into our chain pay only jointly (κ₄ diagonal alone ×0.74
upstream, all ×0.46). Our κ₄ diagonal is 19–42% wrong vs k=4 — a pattern error, mostly the (2,1,1) class of
κ₄post, which is the κ₃ tensor handed to κ₄ at every ReLU (T3 = 2 var_a d₁d₂ d₁d₁ κ₃(a,b,c), corr 0.62–0.81) and then
inherited. Carriers tried, each with the gate k4memo 3 = 3.7533e-08 when off: κ₄-birth readers at any depth (k4b,
×0.995–1.07); T3 one step (×0.89 / ×1.00 at n=128 / 256, network-dependent); T3 inherited through "doubled legs"
riding W∘W (×1.11 / ×1.06; one network ×1.49); the feed fed with k=4's true PRE (2,1,1) slice (the closure's v·var_a·C_bc form explains R² 0.30–0.38 of it) ×0.870 — ×0.754 on one network, ×1.059 on the other. Each improves its input and none moves the final error
consistently: **the next order is a consistent set, not a term** — the same wall as §73 and §3.5's principle, met one
order up. Even the last layer alone (no downstream) would be worth ×0.572 at n=256 with k=4's var, t and κ₄ diagonal
(both networks ×0.57) — and the transfer evaluated at the last layer only gives ×1.001: those statistics are
projections of inherited joint objects. At 1024×16 with Monte-Carlo oracles (`experiments/2026-10-03_lastlayer/`):
the exact last-layer var + t + K4d are worth ×0.76 / ×0.82, but the feature-predictable half of the t and K4d errors
(R² ≈ 0.5) buys ×0.99–1.01 even in sample — R² is not the payoff, even where nothing propagates. Exact k=4 costs ≈L²n⁴ (far over budget at 1024×16), so the leaders, if they are in it, have something we
have not found.

#### 75. The young κ₄ trees: the missing accuracy FOUND (×0.70 raw at 1024), but in our representation it costs what it buys (2026-10-04, Mac)
`experiments/2026-10-04_r1/`, `_r2/`, `_r3/` (plan R of `notes/PLAN-2026-10-04.md`). **Positive half (accuracy):** a full
factorial of k=4's statistics substituted into our chain (8 channels, 8 nets at n = 128 and 256) puts the core at K4 +
K31; ARC's κ₄ split by birth block shows the closure's residual sits in the blocks born in the last ≤ 2 layers; split by
graph type, the indispensable piece is the **3+2 tree, κ₃ handed on through a covariance edge** (12 d₁C d₂ κ₃(j,k,l)
d₁d₁) — §74's carriers had carried the self-variance transfer of the repeated-index correction block instead, which is
secondary. The trees built from the chain's own state (`chain.py k4young`, `young4.py`; = ARC's blocks to 1e-16,
legs = factored to 1e-14) and read into K4d + K31 beyond the closure's pattern: n = 256 ×0.62–0.67 on 3 of 4 nets;
**1024×16 ×0.701 on 10/10 baked MLPs** (K4d only ×0.824; with the band ×0.72). **Negative half (cost, METERED):** the
production form (`young4.Young4Chain`: only what k3z holds; the 3+2's cross-layer pairing Dx taken from the existing D
pairing with the single legs before their transport, D = Dx·W one product per age) — K4d only, 3+2 age 1: ×0.905
raw, +7.40% FLOPs → ×0.972 (3.63e-9); ages 1–2: ×0.848, +14.79% → ×0.973. The killer is the LAST layer: k3z computes
no D pairing there, the 3+2 needs one (+4.27% per age), and the last layer's 3+2 can neither be dropped (×0.848 →
×0.886) nor stand alone (×1.02). Cheaper subsets are worse than k3z (path + star alone ×1.05; the full path quadratic
form ×0.834 at ≈+24%). K31's n×n output costs two products per generation and never pays. Verdict: R closed by its
kill rule (projection < 3.55e-9 required). [U] The leaders reach this raw at 0.6–0.8 of our FLOPs, so their base is
cheaper than ours — the κ₃ state of §62 again.

#### 76. The old κ₃ tier cannot be made cheap in our representation: not truncated, not t-only, not in fewer dimensions (2026-10-04, Mac)
`experiments/2026-10-04_r4/RESULTS.md`. Motivation: many teams now sit at our raw at 10–13.5% of the budget, and our
young tier alone (≈2.5e11 FLOPs) is J2W's whole bill — the band members (≈37% of ours) are the suspect. On the
production accuracy configuration, 10 baked MLPs: dropping only the 3 OLDEST sources costs ×1.17 with the amplitude
tail (×1.27 without); 6 sources ×1.85 / ×2.47; 8 ×3.78 / ×6.25 — the amplitude form removes half the damage, an
order too little. Keeping the old tier in the readout t but taking the (2,1) slice D from the young sources only
(dropping the members' compressed pairing, −11.7% FLOPs) is ×9.25 (the 4 oldest out of D: ×1.07). Reading the
members through their OWN optimal (per-layer SVD) basis: r' 384 ×1.00–1.02, 320 ×1.03–1.04, 256 ×1.11–1.15, 128
×2.7–3.1 — the old tier lives in ≈350–450 dimensions even when the basis is the best one. With 504aldo's F72–F76
(row-space law, flat column-importance spectrum, capped atoms dead) every compression of the factorised κ₃ is now
measured dead by two teams: the cheap tier of the board runs a different expansion, not a compressed one.

---

### 77. The young-κ₄ reader ported to production (k4a) — correct, worth ×0.92 raw at +5.1% FLOPs, and over the residual-call cap ON THE GRADER: 16/100 networks zeroed (R6, 2026-10-06/07; FINAL)
`experiments/2026-10-06_k4a/RESULTS.md`, `src/estimator_k4a.py`. R3's "no last-layer pairing" variant was never priced
(step 3: ×0.879 raw on 10/10 at +6.25% metered → 3.49e-9, under the 3.55e-9 bar), so it was ported: the D pairing
restructured against ungated snapshots two layers back (M → Dx(l, l−1) = M·Wg(l−1) → D = (Dx∘d1(l−1) + newborn)·Wg(l);
gates: bitwise k3z when off; D equal to k3z's to 5e-6 with the reader off; FLOPs +4.9%), plus the Young4Chain reader
(pathdiag, star, 3+2 for ages 1–2). Three findings:
1. **The reader's projection coefficient needs rows production does not compute.** The 3+2 exists only on the rows of D
   (alive and not saturated); the coefficient of the projection off the closure's pattern 3·var², taken over the
   zero-filled vector, shifts every neuron's K4d coherently and reverses the gain (torch emulation ×1.023 on 4/10; the
   production run ×1.08 on MLPs 0–2). Taken over the uncut rows only it recovers to ×0.899 (10/10); with the full
   coefficient ×0.886; the thin remainder and the feed's q·mᵀ in Dx are worth nothing (×0.886 vs ×0.879). Production
   with the corrected coefficient reproduces torch per network (×0.973 / 0.944 / 0.880 vs 0.985 / 0.941 / 0.859).
2. **The residual cap binds, not the FLOPs.** The four small products per layer (two newborn contractions and the two
   recovery products, Strassen depth 4) add 5,581 flopscope calls per network (+23%: 23,911 → 29,492); k3z sits at
   0.31 / 0.331 s of the 0.400 s cap on the grader (local 0.35), and k4a reads 0.49–0.56 s locally. A lower Strassen
   depth cuts the calls and raises the FLOPs past what the reader buys (depth 2: 3.65e-9; plain: 3.66e-9, worse than
   k3z's 3.74e-9 display boundary). No depth satisfies both caps.
3. **Mechanism:** §54–§59's "the construction costs what it saves", one level down — the accuracy is paid for not in the
   FLOPs of the construction but in its call count against the residual budget. Any carrier that needs a few extra
   dense products per layer is priced in calls as well as FLOPs; at ~120 calls per Strassen product and ~1,200 calls of
   headroom, that is ten products per network, not four per layer.

4. **Measured on the grader (probe 334486, 2026-10-07, not designated):** 84/100 completed, **16/100 zeroed by
   `RESIDUAL_WALL_TIME_EXHAUSTED`** (0.401–0.479 s). Residual median 0.393 s vs k3z's 0.310 on the same networks (+0.084 s,
   ×1.27 ≈ the +23% calls), p90 0.416, max 0.479; FLOPs ×1.0509 as measured locally; wall max 112 s of 120. The failures
   sit every fifth network for a stretch — one slow worker — so the margin to design for is the slowest worker's. The
   100-network local run gave the accuracy: ×0.924 raw, better on 95/100 (`RESULTS.md` §6); the grader cannot show it
   because the zeroed networks dominate the mean.

**Final.** k4a is not designated (0-failed rule); R6 is closed. The first probe (334476) failed the grader's smoke test for
an unrelated reason — a snapshot buffer sized for 1024 × 16 only, while the grader's smoke test runs another shape —
found with a local flopscope-client/server pair and a shape grid, fixed and re-run the same night (`RESULTS.md` §7.1–7.3;
the shape grid is now part of the package discipline).

## The boundary, stated once

> **Current statement (2026-09-29, k3m = 5.3906e-9 graded, util 24.835%).** Accuracy: the chain at unlimited cost
> is ≈1.72e-8 raw (the ×1.11 stack × a further ×1.1122 of combined levers, `experiments/2026-09-28_accuracy-free/`), so the cost-free floor is ≈1.724e-9
> adjusted — rank 1–2 *if* util could reach 10%. Cost: every cheap representation of the κ₃ state is closed
> (§54–§59), by a principle (§62), including its "larger state" form (§66) and the jump (§63). What is still open:
> **engineering** savings that keep the representation exact — the matched dead-side cut (§48 revised, projected
> ×0.918) and the transported band basis (E1: accuracy survives; cost ×0.87 only without the coefficient rotation,
> float32 conditioning open). The residual is not learnable (§64); scalar tuning is below the grader's resolution
> (§60). The leaders reach 10.5% util with raw *better* than our unlimited-cost chain — so their method is not a
> cheaper version of ours (`notes/INTEL-forum-2026-09-29.md`).


> **Superseded on 2026-09-24 for the current method.** The paragraphs below describe the Part-I method.
> The "full cumulant-tensor chain near 92% of budget" was then built at 36% of budget: exact chain, memo22 κ₄,
> thin E, workspace Strassen L3. It gave 1.4e-8 adjusted locally, 5× better than Part I.
> **The current boundary is in `FRONTIER.md`:** κ₄ accuracy (a per-step deficit in the nonlinear step) and the
> O(L²) cost of the per-source representation.

The method is at a **characterised local optimum**. The remaining ~14× to top-10 requires the full
cumulant-tensor chain, which our own cost model puts near **92% of budget even with Strassen**.
**It is a cost barrier, not an accuracy barrier**: ARC's reference already reaches raw 4.1e-8, and
top-10 at the floor needs only 3.5e-8. Ten independent attempts to buy that cost reduction are
catalogued above; all of them closed.

Anyone attacking this next should start from the two facts that survived every test: the error lives in
the **propagated mean**, and the ingredients **multiply** — so a cheaper cumulant transport is worth
more than any additional term in the expansion.
