# METHOD — our estimators, derived from scratch

> **Two methods live in this file.**
> - **Part I (§0–§7)**: the Edgeworth-readout + regenerated-source method of submission **331944** (7.07e-8,
>   `src/estimator.py`). This method is closed; the v3 write-up is written around it.
> - **Part II (§8–§14, at the end)**: **the current method** (from 2026-09-24). It is our own re-derivation and
>   implementation of the SIMPLE k=3 cumulant chain of Wu et al. / ARC's reference (`mlp_kprop`), plus three
>   additions of ours: memoryless κ₄ transport, thin correction legs, and a buffered workspace-Strassen layout.
>   `src/estimator_k3b.py` became submission **331962** (1.915e-8). `src/estimator_k3c.py` is the current
>   candidate (≈1.4e-8).
>   Part II is **not** "no-reference": its algorithm is ARC's MIT reference and must be attributed as such.
>
> The Part I text below is kept as it was.

Everything here is our own derivation, verified against Monte Carlo or against the baked N=1e9
ground truth. **No constant is fitted to any data**, and a sweep confirmed that fitting one does not
help (see `NEGATIVE-RESULTS.md` §10). Nothing is derived from another participant's submission —
that matters, because prize verification requires source authenticity (`SPEC.md`).

Implementation: [`../src/estimator.py`](../src/estimator.py).
Shipped configuration: `n_corr_terms=1, edgeworth=True, n_sources=4, strassen_levels=1,
cov_correction=True, skew_clip=1.0`.

---

## 0. The one measurement that set the direction

Oracle-injection at layers 0..14 only, with the scored layer always computed by us
(`experiments/2026-09-16_kprop-arc/logs/state_split3.txt`). Reference: plain k=2 = 5.1494e-6,
full k=3 = 4.2509e-8 (121×).

| upstream mean | upstream covariance | our Edgeworth readout | final_layer_mse | vs k=2 |
|---|---|---|---|---|
| ours | ours | no | 5.1494e-6 | 1.00× |
| k=3's | ours | no | 2.5538e-7 | 20.2× |
| ground truth | ours | no | 2.2229e-7 | 23.2× |
| ground truth | ours | yes | 1.8388e-7 | 28.0× |
| k=3's | k=3's | no | 2.4293e-7 | 21.2× |
| k=3's | k=3's | yes | 6.4131e-8 | 80.3× |
| **ground truth** | **k=3's** | **yes** | **2.9370e-8** | **175×** |

Readings that drove every later decision:
1. A perfect **final readout alone is worth 1.08×** — fixing only the last layer is pointless, which
   kills any scheme that computes the final-layer cumulant by a cheap backward pass.
2. The three ingredients **multiply**: our readout is worth 1.08× on a bad state and 3.8× on a good one.
3. **The reference's residual is upstream, not in its readout** (corrected 2026-09-23 — this line used to
   say "our readout beats the reference's", which is an unfair comparison). In these arms the "readout" is
   our first-order Edgeworth term fed the **reference chain's exact diagonal κ₃** (see `state_split3.py`),
   and the 175× arm also has *ground-truth* upstream means. So: given exact upstream means, a first-order
   readout reaches 2.94e-8, below full k=3's 4.25e-8. On the reference's *own* upstream state our
   first-order term is **1.5× worse** than the reference's nonlinearity step (6.41e-8 vs 4.25e-8;
   `edgeworth_probe.py` agrees: 6.1e-8 vs 4.1e-8 at layer 15) — presumably higher-order terms, not isolated.
4. A **mean-only** method caps at 23–28×. Top-10 needs the covariance corrected too.

---

## 1. The ReLU mean is exact under a Gaussian closure; its leading error is one Edgeworth term

For `z ~ N(mu, sigma^2)`, `a = mu/sigma`, the k=2 update `mu_post = mu*Phi(a) + sigma*phi(a)` is exact.
The general cumulant rule is `Delta(cumulant) = sum_k (1/k!) E[d^k f] kappa_k`, and with
`E[d^3 ReLU] = -a*phi(a)/sigma^2` the first correction is

```
mu_post = mu*Phi(a) + sigma*phi(a) − kappa3_diag * a * phi(a) / (6 * sigma^2)
        = mu*Phi(a) + sigma*phi(a) − s * sigma * a * phi(a) / 6      with s = kappa3_diag / sigma^3
```

**Verified exactly** (`experiments/2026-09-16_kprop-arc/edgeworth_mc_check.py` → `logs/edgeworth_mc_check.txt`,
re-run 2026-09-23 because the script behind an earlier "40M-sample MC, 61–87%" figure was not kept). Test family
`z = mu + sigma*(G-k)/sqrt(k)`, `G ~ Gamma(k)`, whose ReLU mean has a closed form via the incomplete gamma function;
that closed form itself agrees with 4e7-sample MC to <1 s.e. at four points. The term removes **96–99%** of the
Gaussian-closure mean error at |skew| ≤ 0.1 (the network's regime: measured |skew| ≤ 0.09) and 86–97% at skew
0.2–0.3, for `a` in [-2, 2], `a != 0`. It vanishes identically at `a = 0`, and the residual scales as `skew^2`
(a 4th-cumulant / `kappa3^2` effect), as a first-order term should.

`kappa3_diag[i] = kappa3_pre[i,i,i]` is a **vector**, not the `n x n x n` tensor the reference carries.

The same rule at 2nd order gives the covariance correction used in §4, and at 4th order
`Delta mu = (1/24)(a^2-1) phi kappa4 / sigma^3`, which was implemented and rejected on cost
(`NEGATIVE-RESULTS.md` §4).

## 2. Where the diagonal third cumulant comes from: sources, not transport

A third-cumulant **source is born at every ReLU layer** and is then transported forward, so the
pre-activation at depth `l` carries `l` of them. Trying to transport the diagonal alone fails
(`NEGATIVE-RESULTS.md` §2). Instead each source is **regenerated from the mean and covariance the
chain already carries**, via the leading connected Wick diagram with derivative coefficients

```
d1 = Phi(a)            d2 = phi(a)/sigma          (both checked against the reference wick.py to 0.00e+00)
kappa3_born[a,b,c] ~= d2_a d1_b d1_c C_ab C_ac + (2 cyclic permutations)
```

which is the CP form `3 * Sym( sum_p d2_p * e_p (x) u^p (x) u^p )`, `u^p = diag(d1) C[:, p]`.
Validated at width 1024: correlation **0.998** with the true diagonal at layer 1, magnitude 0.918.

## 3. Transport, and why it is two matmuls per source per layer

A source born at ReLU layer `m` reaches the pre-activation of layer `l` through
`T = W^l diag(d1^{l-1}) ... diag(d1^{m+1}) W^{m+1}`. With `A = T` and `B = T U`, `U = diag(d1^m) C^m`:

```
kappa3_pre[i,i,i] = 3 * sum_a d2^m_a * A[i,a] * B[i,a]^2
```

`A` starts as the identity, so a freshly born source needs no matmul for it. Per layer each live
source costs `A <- W^T A` and `B <- W^T B`, then both are gated by `d1` elementwise.
**Window `J = 4`**: accuracy saturates at J = 4–5 and degrades beyond (the leading diagram is a poor
model for old sources). Raw gain over plain k=2: 1.61× / 2.37× / 3.16× / 3.94× at J = 1/2/3/4.

## 4. The covariance correction — and the split that makes it work

From the same rule at 2nd order, with `D21[i,j] = kappa3[i,i,j]`:

```
Delta cov[i,j] = (1/2) ( d2_i d1_j D21[i,j] + d1_i d2_j D21[j,i] )
D21 = 2 * (A*d2*B) @ B^T   +   ((B*B)*d2) @ A^T
       \_____term 1_____/       \____term 2____/
```

**Two structural choices, both measured, neither fitted:**
- **Only the off-diagonal is applied.** Diagonal only = 0.22× (harmful), off-diagonal only = 1.82×,
  both together = 0.56×. The diagonal keeps the exact rectified-Gaussian variance. Applying both at
  once had masked each effect and led to an earlier wrong conclusion — see `NEGATIVE-RESULTS.md` §5.
- **Only term 1 is kept.** It measures *better* alone (2.15× at J=4) than with term 2 (1.82×) and
  costs half as much.

## 5. Strassen-Winograd on the source transport

The organisers confirmed on the forum that Strassen-Winograd expressed as flopscope operations is
permitted with no recursion limit, and flopscope genuinely bills the op tree: measured ratios
**0.88 / 0.78 / 0.70 / 0.65** at 1/2/3/4 levels, max relative deviation 3.6e-6 in float32 — three
orders below the accuracy the estimator delivers (`experiments/2026-09-23_multisource/strassen_test.py`).

**Depth is capped by residual time, not by the meter.** Each level multiplies Python-level `fnp`
calls by ~7, and the grader allows 400 ms of non-flopscope time per MLP. Level 2 measured 0.325 s on
20/20 MLPs (81% of the cap) on a machine faster than the grader's single physical core. **Level 1 is
the shipped setting**, running at 0.06–0.08 s.

## 6. Results

Full public `mini` split, 100 MLPs, baked N=1e9 ground truth, 0/100 failures throughout.

| configuration | adjusted | final MSE | utilization | graded |
|---|---|---|---|---|
| starter covariance propagation (re-run on our pipeline: 4.0504e-6) | 4.050e-7 | 4.050e-6 | 2.35% | — |
| our k=2 + tetrachoric (Sept baseline; **corrected**, was mis-quoted as 4.058e-6) | 3.8502e-7 | 3.8502e-6 | 2.35% | — |
| + Edgeworth mean, 1 source | 2.2767e-7 | 2.2767e-6 | 3.82% | **2.2219e-7** (id 331885) |
| + 3 sources | 1.16775e-7 | 1.16775e-6 | 9.10% | **1.1360e-7** (id 331892) |
| + 4 sources, Strassen | 1.02115e-7 | 9.87674e-7 | 10.34% | **9.9016e-8** |
| **+ off-diagonal covariance correction** | **7.1794e-8** | **4.79346e-7** | **14.98%** | **7.0710e-8** (id 331944) |

**5.36× adjusted / 8.03× raw over our k=2 baseline, 5.64× / 8.45× over the kit's covariance propagation;**
the grader agreed with the local number to 1.5–3.0%, four times in a row, and was better every time.
Ablation on 100 MLPs: tetrachoric off costs 3.1% final / 14.6% all-layers (`experiments/2026-09-23_writeup-ablations/`).

## 7. Cost model (for anyone extending this)

One `(n,n)@(n,n)` product costs `n*n*(2n-1) = 2.147e9` FLOPs, `1.89e9` with one Strassen level.
The k=2 base (the symmetric covariance einsum) is `5.179e10` = 2.35% of `B = 2**41`.
Per layer at `J = 4`: 11 products — 2 for the newest source (its `A` is free), 3 for each older one
(transport `A`, transport `B`, one `D21` product). Total 14.98% of budget.

**The multiplier floors at 0.1 below 10% of budget**, so above the floor every FLOP costs score
linearly. At 14.98% we pay 0.1498 instead of 0.1 — running the same accuracy under the floor would
score 4.79e-8 and put us inside the public top 100 (rank 100 was 5.08e-8 on 2026-09-23). Every
attempt to get there is catalogued in `NEGATIVE-RESULTS.md` §6–§9.

---
---

# PART II — the k=3 cumulant chain (current method, from 2026-09-24)

Code: [`../src/estimator_k3b.py`](../src/estimator_k3b.py) (submission 331962) and
[`../src/estimator_k3c.py`](../src/estimator_k3c.py) (current candidate). Experiments and every number:
[`../experiments/2026-09-24_k3-chain/RESULTS.md`](../experiments/2026-09-24_k3-chain/RESULTS.md). Paper:
`references/Wu2026_mlp_cumulant_propagation_2605.05179.pdf`.

**Attribution (must appear in the write-up).** §8 is the algorithm of Wu et al. (arXiv:2605.05179), available as
ARC's MIT-licensed reference `mlp_kprop` (SIMPLE kind, k=3, factor=True). We read its nonlinear step, wrote it
out as explicit formulas, and re-implemented it ourselves in a different representation (§8.3). We checked the
formulas slice by slice to 1e-15 and the whole chain to 1e-16. **§9–§11 are ours.** No code comes from other
participants. 504aldo's public repo describes a similar "7→4 products with thin residuals" trick; we derived
ours independently (§10), and the write-up should say that it exists.

## 8. The algorithm

### 8.1 Linear step (exact)
μ ← Wᵀμ, C ← WᵀCW. Every factor of κ₃ is multiplied by Wᵀ. The reference keeps κ₄ as one radial scalar a4 and,
after the linear step, uses the average metric E[WWᵀ] = 2I.

### 8.2 Nonlinear step (SIMPLE k=3)
With w(k,p) = E[∂ᵏ ReLU(Z)ᵖ] for Z ~ N(μ_i, σ²_i), d1 = Φ(μ/σ), d2 = φ(μ/σ)/σ:
- **Single-index power moments** (p = 1..4), with κ₃ and κ₄ Edgeworth terms:
  `P_p = w(0,p) + w(3,p)·t/6 + w(4,p)·K4d/24`, where t = κ₃[iii] and K4d = κ₄[iiii].
- **Pair power cumulants:**
  `Q(p,q) = a1b1·C0 + ½a1b2·Dᵀ + ½a2b1·D + ½a2b2·C0² + ¼a2b2·K22`,
  with a_k = w(k,p)_i, b_k = w(k,q)_j, C0 = the covariance with ZERO diagonal, D = κ₃[i,i,j] (the (2,1) slice),
  and K22 = κ₄[i,i,j,j].
- **Leonov–Shiryaev conversion:**
  - `K21 = Q21 − 2P1·Q11`
  - `K22 = Q22 − 2P1_j·K21 − 2P1_i·K21ᵀ − 2Q11² − 4P1_iP1_j·Q11`
  - K2, K3 and K4 on the diagonal come from the P's.
- **Reference κ₄:** `a4 = 3(ΣK4_iiii + Σ_{i≠j}K22_ij)/(n(n+2))`. At the next layer it is read back as
  κ₄[iiii] = 4·a4 and κ₄[iijj] = 4·a4/3.
- **κ₃ (1,1,1) part.** It has three pieces:
  - pass-through, with every factor gated by d1;
  - a tree birth `Sym(U ⊗ 3I ⊗ U·diag(d2))` with U = diag(d1)·C0;
  - a correction block `Sym(V ⊗ I ⊗ I)` with `V = diag(ΔK3) + 3ΔK21ᵀ`. This block makes the repeated-index slices
    (3,) and (2,1) exact. Here ΔK3 and ΔK21 are the new cumulant minus what the pass-through and the tree already
    produce.
- **Readout:** the post-ReLU mean P1, i.e. a one-step Edgeworth readout from (μ, σ², κ₃ᵢᵢᵢ, κ₄ᵢᵢᵢᵢ).

### 8.3 Our representation: sources
Each birth at layer m creates one source with four parts:
- **A** = T, the transport from the birth (identity at birth; this is the "hub" leg);
- **B** = T·U, the tree leg;
- **E** = T·V, the correction leg;
- a fixed hub vector **g** = d2 at birth.

κ₃ = Σ_sources [Sym(B ⊗ 3A ⊗ B·diag g) + Sym(E ⊗ A ⊗ A)]. Only the slices the next step needs are ever
formed:
- `t_i = 3 Σ_s Σ_p g_p A_ip B_ip² + Σ_s Σ_p E_ip A_ip²`
- `D = Σ_s [L1·Bᵀ + L2·Aᵀ + L3·Eᵀ]`, with `L1 = 2A∘B∘g`, `L2 = B²∘g + ⅔E∘A` and `L3 = ⅓A²`.

After each nonlinear step every leg row is gated by d1 (Hadamard). Cost: 6 dense n×n products per source-layer
(3 transports + 3 for D). That comes to ≈660 products ≈ 64% of B; the reference itself runs at 148%.

## 9. Memoryless κ₄ transport ("memo22") — ours, 1.13×
The radial scalar throws away the (2,2) structure. We carry the post-ReLU κ₄ diagonal and the K22 matrix. Through
the next linear step we keep only the **direct pairings**:
`M = Wsqᵀ·(K22 + diag K4)·Wsq`, `K4d_next = 3·diag(M) − 2·(Wsq∘Wsq)ᵀ·K4`, `K22_next = offdiag(M)`, with
Wsq = W∘W.

The other pairings carry random signs over the fresh W and average to zero; an MC check confirms the transport is
unbiased. Cost: 2 products per layer. Result on 3 MLPs: 3.7533e-8 against 4.2509e-8 radial, better on all 3.
It is "memoryless" because it keeps no per-source κ₄ legs.

## 10. Thin correction legs — ours, 6→4 products per source-layer, lossless
The (2,1) correction splits **exactly**:

  `ΔK21 = diag(α)·C0·diag(d1) + R`, with `α = w(1,2) − 2·P1·d1`.

The first piece is the Gaussian a1b1·C0 term of K21 that the tree does not produce. Its contribution to V,
`3(diag(α)C0 diag(d1))ᵀ = 3B·diag(α)`, is a column rescaling of the tree leg, so after transport it stays
3B_t·diag(α). **The first piece costs nothing.** The remainder R is numerically low-rank. We use a randomized range
finder:
1. 24 Gaussian probes, seeded by `mlp.seed`, plus one power step: Y = R·(Rᵀ·(R·Ω)).
2. Q = qr(Y).
3. Carry the thin leg H = RᵀQ (n×24), transported like any leg, and the fixed Ut = Qᵀ, so that E_R = 3H·Ut.

In the readout:
- E = A·diag(ΔK3) + 3B·diag(α) + 3H·Ut is formed element-wise; the one thin product is cheap.
- L1 gains A²∘α.
- L2 gains ⅓A²∘ΔK3.
- The R part of D is `(A²·Utᵀ)·Hᵀ`, a thin product.

Measured on 3 MLPs:
- exact split (`freefull`): 3.7533e-8, identical to the base;
- R dropped: ×10.5 worse;
- R at rank 16 by SVD: 3.7742e-8;
- **range finder 16+8: 3.7679e-8** (×1.004).

The flopscope port matches torch at full rank to 5e-7 in float32. That self-check caught a double-counted factor 3.

## 11. Making it run: buffered layout and workspace Strassen (residual-time discipline)
flopscope arrays are immutable and the grader zeroes any MLP whose untracked time exceeds 0.4 s. So:
- all dense legs live in two ping-pong buffers (2·Smax, n, n), with slot 2s = B_s and slot 2s+1 = A_s. Thin legs
  have their own ping-pong buffer. The per-source vectors g, ΔK3 and α are rows of preallocated (Smax, n) arrays;
- every per-source operation is **one batched flopscope call per layer**, writing `out=` into preallocated
  workspaces (`fnp.empty`, never `fnp.zeros`);
- transports and the D pairings go through **batched Strassen–Winograd** (`_Strassen`). Each level has
  preallocated buffers; the 7+7 combinations and the recombination are written with out= into quadrant views.
  Python-call counts therefore do not grow ×7 per level. Billing ratio vs plain matmul: 0.88 / 0.78 / 0.70 / 0.64
  / 0.61 at levels 1–5. `chunk` bounds the workspace (memory ∝ 7^level·chunk);
- in k3c the base products (C_pre and the κ₄ M) also go through Strassen, via `mm()`.
The pitfalls behind these rules are listed in RESULTS.md §4.

## 12. Cost model of the k=3 chain
S sources are alive at layer l (one per previous layer):
- **transports:** 2S−1 dense products (the newborn's A is Wᵀ itself, a free copy);
- **D pairings:** 2S dense products, skipped at the last layer;
- **base:** 2 (C) + 2 (κ₄ M) products per layer;
- **thin terms:** about 2% of a dense product per source.

Over L = 16 that is Σ(2l−1) + Σ2l + 64 ≈ 500 products ≈ 49% of B plain; with Strassen L3, **36.3%**.
k3b (6 products per source) was 70.6% plain and 50.1% at L3. The per-source representation is O(L²) products. At
L = 16 it cannot get below ~25–30% even at Strassen L5.

## 13. Results of Part II

| | adjusted | final MSE | util | notes |
|---|---|---|---|---|
| reference SIMPLE k=3, radial κ₄ (3 baked MLPs) | — | 4.2509e-8 | 148% | reproduced exactly by `chain.py` |
| + memo22 κ₄ (3 MLPs) | — | 3.7533e-8 | — | ours, §9 |
| k3b plain, 100 MLPs | 2.5037e-8 | 3.5447e-8 | 70.6% | 0 fails, residual ≤0.07 s |
| **k3b Strassen L2, 100 MLPs** | **1.9802e-8** | 3.5446e-8 | 55.9% | **graded 1.915e-8 (id 331962), rank 89** |
| k3b Strassen L3 (3 MLPs) | 1.879e-8 | — | 50.1% | packaged, never submitted (superseded) |
| k3c L3 (3 MLPs) | 1.369e-8 | 3.77e-8 | 36.3% | wall 10 s, residual 0.15 s |
| k3c L4/chunk 8 (3 MLPs) | 1.253e-8 | 3.77e-8 | 33.2% | **100 MLPs: 4 fails on residual under load → not shipped** |
| k3c L3, 100 MLPs | 1.2335e-8 | 3.5835e-8 | 34.4% | residual max 0.215 s → not shipped |
| **k3d L3, 100 MLPs** | **1.2260e-8** | 3.5833e-8 | 34.2% | batched base products + cached workspaces; residual max 0.137 s; packaged |
| k3e L3 (§15), 100 MLPs | 9.7622e-9 | 3.0224e-8 | 32.3% | + C0·D diagram + scale-mixture κ₄ closure; **graded 9.562e-9 (332066), rank 82** |
| k3f L3 (§16), 100 MLPs | 8.5916e-9 | 2.6547e-8 | 32.4% | + marginal coefficients + K31 + κ₄→κ₃ feed (P2); **graded 8.286e-9 (332128)** |
| **k3h L4 (§17), 100 MLPs** | **7.6733e-9** | 2.6549e-8 | 28.9% | k3f + Strassen L4, sum-of-products pairings; **graded 7.401e-9 (332142)** |
| k3i v2 (§18), 100 MLPs | 6.9242e-9 | 2.3221e-8 | 29.82% | + path births at the first readout, star κ₄ birth, C0³/C0∘K22; **graded 6.728e-9 (332170)** |
| k3j (§19), 100 MLPs | 6.1137e-9 | 2.3267e-8 | 26.27% | + dead-row cut α < −3.0, rectangular Strassen; **graded 5.939e-9 (332247)** |
| k3l (§20), 100 MLPs | 5.7387e-9 | 2.1837e-8 | 26.27% | + K31 × 0.65; **graded 5.5984e-9 (332253)** |
| **k3m (§19), 100 MLPs** | **5.5209e-9** | 2.2172e-8 | 24.894% | dead-row cut at −2.5; **graded 5.3906e-9 (332403) — BEST, designated** |
| k3n (§21), 100 MLPs | 5.5045e-9 | — | 24.892% | + feed × 0.8, K22 pattern without C0²; **graded 5.3977e-9 (332872), 0.13% worse** |

## 14. What limits Part II (details in `FRONTIER.md`)
- **Accuracy.** κ₃ is right: MC correlation 0.998, relative error 6%. **κ₄ is not**: at depth the chain carries about half the κ₄
  level. The nonlinear step loses ~4.5% of the (2,2) slice per layer, and the loss compounds. With exact κ₄ the final
  MSE would be ×1.57 better; with exact κ₃+κ₄, ×2.1 better, which is the leaders' raw level.
- **Cost.** The per-source family is O(L²n³). Top-10 needs adjusted ≤3.5e-9. That means raw ≤3.5e-8 at ≤10% of
  budget (our raw is already about there), or raw ~1.7e-8 at ~20%.

## 15. The edge × κ₃ diagram and the scale-mixture κ₄ closure — ours, ×1.13 at −2% budget (k3e)
Found with a Monte-Carlo oracle (`experiments/2026-09-24_k3-chain/RESULTS.md` §7).

**(a) The missing per-step diagram.** In every pair power cumulant add the second-order term
  `Q(p,q) += ½ (a2 b3 Dᵀ + a3 b2 D) ∘ C0`   (a_k = w(k,p)_i, b_k = w(k,q)_j, D = κ₃[iij]).
It is the cross term C0∂ᵢ∂ⱼ × ½κ₃[iij]∂ᵢ²∂ⱼ of the joint Edgeworth exponent. Fed exact (MC) inputs, the SIMPLE step
loses 2.3/3.5/4.5% of the post (2,2) level at layers 6/10/14; with this term it returns 0.999/0.999/1.004 of MC
(corr 0.974 → 0.999). The covariance and the (2,1) slice stay exact. It costs O(n²) elementwise work.

**(b) Why it did not help alone, and the closure that makes it work.** The memo22 transport gets the per-neuron κ₄
PATTERN wrong: MC correlation 0.23–0.53. It uses W∘W and drops the signed structure. For a Gaussian scale mixture,
κ₄ᵢ = 3v·σᵢ⁴, where σᵢ² is the pre-activation variance; the MC κ₄ᵢ correlates 0.86 with σᵢ⁴ at depth. So we keep from
the transport only the LEVEL (one scalar v per layer) and set
  `K4dᵢ = 3v·varᵢ²`,   `K22ᵢⱼ = v·(varᵢ varⱼ + 2 C0ᵢⱼ²)`   (the scale-mixture forms).
With the right pattern, raising the level through (a) helps: 10 baked MLPs 3.52e-8 → 3.10e-8 (×1.136; production thin
E ×1.129), 9/10 better.

**(c) The levels in O(n²).** For M = Wsqᵀ X Wsq with X = K22post + diag(K4post) and u = Wsq·1:
- `sum(M) = uᵀXu` exactly;
- `tr(M) = Σ_ab X_ab G_ab` with `G = Wsq Wsqᵀ`, taking the exact diagonal `G_aa = Σᵢ W⁴_ai` and the mean-field
  off-diagonal `G_ab ≈ u_a u_b / n`;
- `mean K4d = 3 tr(M)/n − (2/n) Σ_a G_aa K4post_a`;
- `ΣK22 = sum(M) − tr(M)`.

This reproduces the exact levels to 4 digits (3.0974e-8 vs 3.0975e-8), so the two dense κ₄ products per layer are
gone: 34.2% → 32.3% of B.

Closed on the way (NEGATIVE-RESULTS §34–§38): pure scale-mixture higher orders (wrong sign), marginal Edgeworth
coefficients, the (3,1) slice, higher Gaussian orders, K22², t·C0, κ₃² blocks.

## 16. The rest of the first-order scale-mixture set — k3f (×1.14 at +0.07% budget)
With the κ₄ closure from §15 in place, the scale mixture predicts three more first-order terms. **They only pay
together** (RESULTS §8):
- **marg:** marginal Edgeworth correction of the pair Wick coefficients,
  a_k → w(k,p) + w(k+3,p)·t/6 + w(k+4,p)·K4d/24 (k ≤ 2);
- **K31:** the κ₄ (3,1) slice of the closure, κ₄[iiij] = 3v·varᵢ·C0ᵢⱼ, entering Q(p,q) as
  (1/6)(a3 b1 K31 + a1 b3 K31ᵀ). Together with marg it cuts the one-step covariance error 0.3% → 0.1%. Alone,
  each is neutral or harmful (§36, AUGMENT §26).
- **κ₄→κ₃ feed (P2):** all-distinct κ₃ births from the κ₄ (2,1,1) slice. With the closure
  κ₄[aabc] = v(varₐC_bc + 2C_abC_ac), the ½ d2ₓ d1ᵧ d1_z κ₄[xxyz] birth splits into two parts:
  - v × tree (P1; slightly harmful, not shipped);
  - Sym(h ⊗ D1C0D1) with h = v/2·d2∘var.

  In the source representation, J(D1C0D1)Jᵀ = B diag(d1_birth) Aᵀ. With m = A h and q = rowsum(B∘d1_birth∘A):
  `t += 3 m∘q` and `D += 2 diag(m) B diag(d1_birth) Aᵀ + q mᵀ`. The first D term rides the existing L2·Aᵀ product;
  everything else is O(n²). At birth, P2's (2,1) slice 2(h d1)ᵢC0ᵢⱼd1ⱼ moves into α, so R is unchanged. The MC
  check (`mc_feed.py`) confirms that the closure's contracted feed matches MC (corr 0.97–0.98). This channel was
  named as the main lever of 504aldo's public chain (forum #18218); our form is derived from the closure.
- 10 baked MLPs: k3e 3.1496e-8 → 2.9722e-8 (marg+K31) → **2.7989e-8 (+P2)**, all 10 better. 100 public MLPs:
  **8.5916e-9**, 100/100 better than k3e.
- Residual trims in k3f (bitwise identical): t/6 and K4d/24 computed once; unused Wick coefficients skipped; `Wt·C`
  rides the leg transport in the free newborn-A slot (−1 Strassen call per layer).

## 17. Strassen L4 through sum-of-products pairings — k3h (×1.12, same accuracy)
- **Constraint.** Residual time is ≈20 µs per flopscope op, whatever the batch shape. The breadth-first Strassen
  workspace grows as (7/4)^d·n²·chunk. So "deeper" means either more calls (a smaller chunk: residual) or more
  memory. A plain L4 at chunk 8 costs +0.1 s of residual.
- **Observation.** The D pairings need only D = Σ_k L_k R_kᵀ, never the individual products. Strassen is bilinear,
  so the K leaf products can be added before the recombination:
  - leaf: M[g] = Σ_k LA[g,k] LB[g,k], computed as `matmul` into a scratch buffer and `sum` over k;
  - the scratch is a view of the level L−2 region, which is free while the leaf runs (it is 8/7 of the leaf size);
  - `einsum` would be billed the same but is 70× slower in flopscope.

  The product buffer M then has no K axis and the recombination runs once.
- **Buffers.** Each level has X1, X2 (chunk slots) and X3 (one slot):
  - the transports use LA = X3, LB = X1, M = X2;
  - the pairings use LA = X1, LB = X2, M = X3.

  L4 at chunk 16 fits in ≈2.7 GB (L3: 2.05 GB) with L3's number of calls.
- **Result.** 100 MLPs: 28.90% of B (k3f 32.36%), MSE unchanged (per-MLP ratio 0.997–1.003), residual +0.03 s,
  wall 17 s locally (≈55 s expected on the grader), RSS 3.95 GB.
- **Limit at n = 1024.** The recursion stops at 64×64 leaves (`m < 128`), so "L5" = L4.

## 18. Path births of the augmented term set at the first readout, star κ₄ births, augmented pair diagrams — k3i
Found by running ARC's own reference in `kind=AUGMENT, factor=True` (Wu et al.'s "augmented" algorithm; MIT code):
MLP 0 2.0563e-8, MLP 1 1.9893e-8 against our k3f 2.6608e-8 / 2.5780e-8, i.e. ×1.3 more accurate (at ~2.4× SIMPLE's
cost). Against the MC oracle its κ₃ diagonal is exact to the oracle's own noise, and its variance is better; its κ₄
diagonal is NOT better than our σ⁴ closure (corr 0.87 vs 0.86). So the lever is the κ₃ step (RESULTS §10).
- **(a) Path births.** ARC's factored augment adds, besides the SIMPLE tree, every O(n)-rank (1,1,1) "path" x–p–z
  with one heavy leg (κ₃ (2,1) slice, κ₄ (2,2) slice or a doubled edge). All of those whose other edge is C0 share the
  hub leg A and the tree leg B, so together they are ONE extra leg:
  `Sym(T·Y ⊗ 3A ⊗ B)`, `Y = d1 Dᵀ d3 + d2 D d2 + ½ d2 K22 d3 + d2 C0² d3` (hub weights inside; row = tensor index,
  column = hub; D, K22, C0 = pre-activation slices at birth; checked against ARC's code to 5e-16).
  Its (2,1) birth slice is Y∘U (U = D1 C0) and is removed from the correction leg. Only the FIRST readout matters
  (age ≤1 keeps ~90% of the all-ages gain): at the next layer X = Wᵀ Y is one more matrix in the leg transport;
  `t += 3 rowsum(A∘B∘X)` for the newest source, and (X∘A)Bᵀ + (X∘B)Aᵀ ride the existing L1, L2 pairings (free).
  Then X is dropped. Cost: +1 dense product per layer (+0.9% of B).
- **(b) Star-diagonal κ₄ birth** of the newest source, `K4d += 4 rowsum(d3_birth ∘ A ∘ B³)` (O(n²); on k3f alone
  ×1.017, with the paths ×1.04). The feed level v is then taken from the K4d mean after this birth (as in the torch chain).
- **(c) The augmented pair diagrams we lacked:** `(1/6) a3b3 C0³` and `¼ a3b3 C0∘K22` in every Q(p,q) (O(n²)). With (a)
  our pair-slice term set equals ARC's augmented one (`get_all_terms_iso`, printed in RESULTS §10).
- 10 baked MLPs (torch): k3f 2.7989e-8 → paths(≤1, t-only) + (b) + C0∘K22 **2.4919e-8** (10/10 better) → + merged D
  + C0³ **2.4441e-8** (×1.145). flopscope `src/estimator_k3i.py`: switches off = bitwise k3h; vs torch 2–5e-7.
  100 public MLPs (t-only version): **7.0233e-9 @29.81%**, final 2.3561e-8, 98/100 better than k3h, residual 0.176/0.203 s.

## 19. The dead-row cut — k3j (α < −3.0) and k3m (α < −2.5): cost, not accuracy
A neuron with α = μ/σ < thr is dead: its gate d1 ≈ 0, so every leg's row for it is ≈ 0 after gating. **k3j zeroes those
rows exactly** in every leg after gating and in D, and the next transport runs over **alive rows only** (rectangular
Strassen, the row count padded to a multiple of 16); C and the path leg stay full. Cost model: transports × alive(l−1),
pairings × alive(l)². At −3.0: util 29.82% → 26.27%, final MSE +0.2% (`experiments/2026-09-25_cost/`). The cloud's cost
frontier (`experiments/2026-09-26_cost-frontier/`) found the optimum **−2.5** (cliff between −2.5 and −2.0) → **k3m**:
util 24.894%, raw ×1.014 (torch 10 MLPs 2.3135 → 2.3462e-8), adjusted ×0.962, 99/100 better → **graded 5.3906e-9**.
`src/estimator_k3m.py` is a standalone copy of k3l with `deadcut: float | None = -2.5`; `deadcut=-3.0` is bitwise k3l.

## 20. K31 × 0.65 — k3l (a consistent-truncation level, not a free fit)
The closure's κ₄ (3,1) slice K31 sits at 0.74–0.78 of MC; ARC's AUGMENT, a consistent truncation, sits at 0.54. The
chain's optimum multiplier is 0.6–0.7 (flat), all 10 MLPs better at 0.7 → **0.65** (100 MLPs adjusted 6.1137 → 5.7387e-9, graded
5.5984e-9). Every other level was re-scanned on the k3l base and found at its optimum (RESULTS §11 of the chain folder).

## 21. k3n — two fitted scalars, and why k3m stays designated
k3n = k3m + κ₄→κ₃ feed × 0.8 + the K22 pattern without its C0² term (`feedmult=0.8, k22c=0.0`). Torch 10 MLPs ×1.0089 →
flopscope 100 MLPs ×1.0030 (66/100) → **grader ×0.9987 (worse)**. The local→grader discrepancy is ≈0.4–0.5% on a 50-MLP
mean, so **a lever worth < 2–3% cannot be validated by a submission**, and fitted constants transfer at ≈⅓ strength
(NEGATIVE §60). k3m (332403) is the submission to designate; k3n is recorded, not shipped further.

## 22. The dead-side birth-COLUMN cut — the transpose of §19 (research, 2026-09-29; port pending)
At a source's birth every column p of its legs is multiplied by that parent's B, g, E, feed weight h or path leg; for a
parent with α_p far out on either side d2_p ≈ 0, so the column can be removed **exactly** (`chain.py colcut`). Measured
(`experiments/2026-09-29_colcut/`, 10 MLPs): the symmetric cut is bad (|α| > 3: ×1.12 while removing 0.03% of the
d2-weighted mass) and **all of the damage is on the saturated side** — κ₃ passes *through* a near-identity neuron, so its
column carries transported content. **The dead side alone at −2.5: raw ×0.9928 on 10/10.**
**Mechanism = consistency:** a neuron whose row is gated (§19) should not keep birthing κ₃ on the column axis; the two
cuts must match. It predicted that matched row + column cuts beat row-only at every threshold, growing with the gated
fraction — confirmed at all five (−3.0 … −2.0: matched/row = 0.9985, 0.9970, 0.9928, 0.9790, 0.9475). The interaction
moves the optimum from −2.5 to **−2.25, projected ×0.918 (≈4.95e-9)**, util modelled not metered.
**Port scope:** per-source widths differ, so the legs become one wide (n × Σw) matrix in which transport and pairing
stay single products; a two-class (young/old) shortcut captures only 29–54% of the saving.

## 23. The transported band basis for old sources (research, 2026-09-29; alive on accuracy)
k3k's shared subspace (NEGATIVE §54) re-fitted its basis every layer, and the entry update ate the saving. The
alternative: **transport** the basis P like a leg (Wᵀ(PZ) = (WᵀP)Z, so the coefficients Z never change) and project
each source once, on entry at age a0 (`chain.py ssp=(a0, r, 0, 0)`). E1 (`experiments/2026-09-29_band/`, 10 MLPs):
raw +0.69% at (6,448), +2.46% at (6,384), +2.15% at (5,448), +10.9% at (6,320) (no spectral gap near 320). Priced with
`cost.py`: cost-neutral if the coefficients are rotated every layer (×1.02 at (6,448)); **×0.87 at (6,384) without the
rotation** (util 20.45%). Open: float32 conditioning of an un-orthonormalised P over ~10 random transports (a lazy QR
every k layers) — the cloud's port battery (`notes/CLOUD-BRIEF.md`).

## 25. k3o — the matched cut on the production path: the legs stored transposed as one wide stack of hub rows
§22's saving needs per-source column widths, which k3m's batched `(2S, n, n)` leg stack cannot hold. k3o
(`src/estimator_k3o.py`, `experiments/2026-09-30_k3o/`) stores the legs **transposed**:
`XT = [B_0ᵀ; …; B_{S-1}ᵀ; A_0ᵀ; …; A_{S-1}ᵀ]`, source s owning a block of w_s hub rows in each region, every per-hub vector
(g, dK3, α, feed h, d1 at birth, Utᵀ) in the same row order. The transpose is what makes it cheap:
- the leg transport becomes one row-stack product `XT' = XT W` over the alive rows of the previous layer — a
  right-shared Strassen (`rshared`, the mirror of `shared`) over row chunks, whose padding rows must be exactly zero
  (Strassen mixes quadrants, so garbage would leak);
- the D pairing `D = L Xᵀ = LTᵀ XT` is one contraction over all hub rows (the existing `paired_sum` on row chunks);
- per-source blocks are contiguous row ranges, so gathers write contiguous outputs; the newborn's Bᵀ waits in its own
  buffer and is gathered between the regions at its first transport, and its hub leg becomes Aᵀ = W[P, :] (no product);
- C and the newest path leg ride a separate full-W product (their dead rows are not zero), as in k3m's cut branch.
With `colcut=None` k3o reproduces k3m to float32 rounding (≤1.3e-6, FLOPs ×1.0020); with the cut it reproduces the torch
`colcut` to 2–6e-7 at every tested threshold. Utilisation and score: `experiments/2026-09-30_k3o/RESULTS.md`.

## 26. k3p — the transported band basis on the production path
Old sources (age ≥ a0 = 5) live as legs P·Z in one shared basis P (n × r, r = 448): P rides the leg transport as r
extra rows Pᵀ (gated by d1, contracted over the alive rows like a leg), the coefficients Z never change
(Wᵀ diag(d1)(P Z) = (Wᵀ diag(d1) P) Z); members are materialised every layer by one right-shared product
[Z_Bᵀ; Z_Aᵀ] Pᵀ and their D pairing is compressed, (L_mᵀ [Z_Bᵀ; Z_Aᵀ]) Pᵀ. A source enters at the end of the layer before
it turns a0, projected on the gated P through a QR of P (Z = R⁻¹ Qᵀ X for A, B and the thin H); P is never
re-orthonormalised and Z never rotated (cloud #8: float32-exact with the QR projection). The first basis is the dominant
r-subspace of the first entrant's [A B] (range finder, one power step). Production = the cloud's torch `bandz` to
2–7e-7; 100 MLPs 4.6839e-9 (×0.893 of k3o, 100/100), residual +0.05 s (≈4 extra Strassen calls per band layer).
`experiments/2026-09-30_k3p/`.

## 27. k3q — the saturated-row cut of the D pairing with the κ₄ level restored (E2)
Rows i of D_ij = κ₃[i,i,j] of neurons saturated now (α_i > 2.5) are not computed and are zero; the κ₄ level the next
layer reads only through Ssum = uᵀ K22post u is restored by dS = Σ_{i sat} u_i cf22_i (D_full v)_i,
v = u ∘ (cf12 − 2 P1 cf11), with D_full v assembled from the legs by matrix-vector products (O(S n²)) and added to the
next Ssum. **An accuracy gain, not a trade:** raw ×0.94 (the naive cut's damage was entirely the κ₄ level). Mechanism
(cloud #8): a saturated row reaches the step only through the pass-through part of K21 and the Edgeworth-corrected
tail coefficients coef(2,1) = d2 + d5 t/6 + d6 K4d/24, whose correction is −28…−79% of d2 at α > 2.5 — the marginal
Edgeworth series is at the edge of validity in both tails, and removing rows that enter only through unreliable tail
weights is §22's consistency again. Production = the cloud's torch `satrow + satrow_k4="sep"` to 2–7e-7; 10 MLPs on top
of k3p ×0.899 (10/10). `experiments/2026-09-30_k3q/`.

## 24. Where the method stands (2026-09-29)
Accuracy at unlimited cost ≈1.72e-8 raw (the ×1.11 stack × a further ×1.1122 combination, both in `experiments/2026-09-28_accuracy-free/`; floor 1.724e-9
if util reached 10%). Cost: every cheap *representation* of the κ₃ state is closed, by one mechanism (NEGATIVE §54–§59)
and then by a principle (§62: (t, D) is not a sufficient state; the missing all-distinct part, ~28% per layer, is
orthogonal to everything O(n²)), including its larger-state form (§66) and the jumped tier (§63). What remains are
**exact engineering savings** — §22 and §23 — and Strassen L5 if util falls below ~17% (§56 was a wall-time veto).

