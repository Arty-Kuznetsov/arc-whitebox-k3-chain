# 2026-10-14 — the skew law and a "k = 2.5" closure

Question (item 1 of the 10-13 plan): the coherence law (`../2026-10-13_coherence-law`) found that a Gaussian closure's
error is ~65% learnable per neuron because the true pre-activation κ₃ of a neuron is almost a function of its own mean.
(a) How exact and how universal is that law? (b) Can a chain that does not carry κ₃ carry it instead as that function,
with constants fitted offline, and how much of the k = 3 chain's gain does that buy?

## 1. The skew law (`skew_law.py` → `res/skew_law.json`)

Per layer and network, least squares κ₃ᵢ ≈ a(l) + c(l)·μᵢ over the neurons; R² noise-corrected from two independent
Monte-Carlo seeds (energies as cross products); synthetic shapes from the coherence-law files (10 networks, 6 at 512),
public 1024 × 16 networks 0 and 1 from the two 1e7 oracles. Layer 0 is the input layer (κ₃ = 0).

| shape | R²(κ₃ ~ μ) layer 1 → 4 → last | c(1)·n | c(last)/c(1) | CV of c(l) across networks, layers 1–8 / last |
|---|---|---|---|---|
| 256 × 8 | 0.50 → 0.80 → 0.84 | 13.8 | 0.80 | 6–13% / 13% |
| 256 × 16 | 0.53 → 0.80 → 0.84 | 14.4 | 0.64 | 6–8% / 32% |
| 256 × 32 | 0.54 → 0.80 → 0.76 (min 0.71) | 14.2 | 0.53 | 6–15% / 77% |
| 512 × 16 | 0.50 → 0.80 → 0.87 | 14.0 | 0.48 | 3–10% / 25% |
| 1024 × 16 (public, 2 nets) | 0.49 → 0.81 → 0.90 | 13.7 (20 nets: 13.94) | 0.54 | 0.5–5% / 0.5% |

- The intercept a(l) is zero to the noise at every layer and shape: the skew of a neuron is proportional to its mean.
- **c(1)·n ≈ 14 at every width**: layer 1's slope (the first non-Gaussian layer) is a constant over the width; it then decays with
  depth (to 0.5–0.8 of its first value). The same law in standardised form (γ = κ₃/σ³ against α = μ/σ) fits slightly
  better (R² 0.85–0.91 at the last layer).
- The slope is a property of the shape, not of the network: its spread across networks is a few per cent at 1024 × 16
  and grows at small widths and at depth (finite-n fluctuations), which is what a universal-constant closure needs.
- **The first-layer slope is derived [D], not only measured.** Input N(0, I), W ~ N(0, q/n), q = 2, so the layer-0
  pre-activations have variance s² = q and pairwise correlations ρⱼₗ with E ρ² = 1/n. The layer-1 skew is
  κ₃ᵢ = Σⱼ Wⱼᵢ³ κ₃(h) + 3 Σⱼ≠ₗ Wⱼᵢ² Wₗᵢ κ(hⱼ, hⱼ, hₗ) + (all distinct). By Mehler's expansion,
  κ(hⱼ, hⱼ, hₗ) = s³ (½φ(0) ρ + B₂ ρ² + …) with B₂ = ½φ(0) − φ(0)³ = 0.13598 (from Cov(X², Y) − 2E[X] Cov(X, Y), with
  E[R'] = ½, E[R''] = φ(0), E[(R²)'] = 2φ(0), E[(R²)''] = 1). Regressing on μᵢ = E[R] s Σⱼ Wⱼᵢ over the neurons: the
  diagonal term gives 3q κ₃(R)/(n E[R]) (κ₃(R) = 0.32646 for a unit Gaussian), the linear-in-ρ term has zero mean,
  the ρ² term gives 3q B₂/(n E[R]) through E ρ² = 1/n, and the all-distinct term is uncorrelated with μ. So
  **c(1)·n = 3q·s²·(κ₃(R) + B₂)/E[R] = 12 · 0.46244 / 0.39894 = 13.91** (the diagonal term alone: 9.82).
  Measured: **13.94 ± 0.14** (standard error; SD 0.62; 1024 × 16, 20 networks, single-seed 2·10⁶ MC oracles 10–29), 13.98 (512 × 16, 6 networks, CV 3%),
  13.82 / 14.43 / 14.21 (256 × 8 / 16 / 32, 10 networks each, CV 7–8%, pooled 14.15 ± 0.2). Deeper layers: c(l)·n
  decays (to 6.8–11 at the last layer), a recursion we have not derived.
- Why: Proposition 3 (write-up §6.2, `../2026-10-13_prop3`): the next layer's κ₃ diagonal is dominated by the
  partial-trace term 3s²·wᵢᵀv, with v almost parallel to the post-activation mean vector m (cos 0.92–0.99), and
  μᵢ = wᵢᵀm through the same column. Consistent with the R²: the partial trace's share of the next diagonal's energy
  is q_v/(1+q_v) = 0.82 at n = 256 (prop3 table), times cos²(v, m) ≈ 0.97 → R² ≈ 0.80; measured 0.80–0.84 at layers ≥ 4.
- The same argument gives the (2,1) slice: D_ij = κ₃[i,i,j] ≈ t_j/3 for i ≠ j (rows identical, rank one). Measured
  inside our k = 3 chain (config `full`, 1024 × 16, layers ≥ 2): corr(D_ij, t_j/3) over i ≠ j = **0.81**.

## 2. The closure inside the chain (`skew_chain.py` → `res/skew_chain_<shape>.json`, `_final.npz`)

Our float64 chain in config `full` (= ARC's SIMPLE k = 3); at every layer the κ₃ the step reads is replaced through the
online hook `mcorr` (the chain gained a "D" key for this, absent = bitwise the old path; gate `ablate.py k4memo 3` =
3.7533e-08 after the edit). Arms:

- `full` — the chain's own t and D (k = 3);
- `zero` — t = D = 0 (the κ₃-free reference, κ₄ kept as the same radial scalar);
- `clos_own` — t = a + cμ, D_ij = t_j/3, (a, c) per layer fitted on THIS network's Monte Carlo (upper bound of the form);
- `clos_univ` — the same with (a, c) averaged over OTHER networks of the shape (deployable: offline constants);
- `clos_univ_t` — `clos_univ` on the diagonal only, D = 0;
- `tlaw_Down` — t = the law, the chain's own D; `town_Dr1` — the chain's own t, D = t_j/3 (which approximation loses).

Truth: 1024 × 16 public networks 0–9 against the 1e9 ground truth (constants from the MC moment oracles of networks
10–29); synthetic shapes against the two-seed MC of the coherence law (constants leave one network out).

### Results (`res/skew_chain_run1.txt`, `res/skew_cs.txt`)

Final-layer MSE, mean over networks; "×zero" = ratio of the means; "kept" = mean over networks of
(zero − arm)/(zero − full), the share of the k = 3 chain's κ₃ gain an arm keeps; n/N = networks better than `zero`.

| shape | L/n | full (k = 3) | zero (κ₃ = 0) | **clos_univ** | ×zero | kept | n/N | clos_own kept | clos_univ_t (D = 0) | tlaw_Down | town_Dr1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024 × 16 | 1/64 | 3.92e-8 | 4.11e-6 | **1.44e-6** | 0.35 | **64%** | 10/10 | 64% | ×1.05, −9% | 80% | 75% |
| 512 × 16 | 1/32 | 3.54e-7 | 1.64e-5 | **5.99e-6** | 0.37 | **63%** | 6/6 | 65% | ×1.08, −13% | 77% | 72% |
| 256 × 8 | 1/32 | 9.35e-7 | 3.46e-5 | **1.17e-5** | 0.34 | **65%** | 10/10 | 67% | ×0.68, 30% | 77% | 75% |
| 256 × 16 | 1/16 | 2.77e-6 | 6.49e-5 | 3.59e-5 | 0.55 | 35% (−107…84%) | 9/10 | 46% | ×1.33 | 57% | 47% |
| 256 × 32 | 1/8 | 1.39e-5 | 1.16e-4 | 1.71e-4 | **1.47** | −210% | 3/10 | −42% | ×1.89 | −118% | diverges (×189) |

The `zero` arm's error is 1–12% below ARC's own k = 2 (covariance propagation) at every shape (`summary_c2_*`: 4.45e-6,
1.86e-5, 3.67e-5, 6.86e-5, 1.18e-4; ratios 0.925, 0.883, 0.943, 0.945, 0.989), so it stands for "a Gaussian closure" below.

**What the closure's residual still holds** (`skew_cs.py` adds the arms to the coherence-law files as estimators `cs` =
clos_univ and `cz` = zero; `coh_analyze.py --est cs|cz`, same truth, same 8 features, leave one network out):

| shape | zero: coherent | **zero: learnable** | closure: coherent | **closure: learnable** |
|---|---|---|---|---|
| 1024 × 16 | 25% | **64%** | 1.6% | **0.2%** |
| 512 × 16 | 27% | **65%** | 4.2% | **−1.8%** |
| 256 × 8 | 31% | **60%** | 4.0% | **1.7%** |
| 256 × 16 | 25% | **62%** | 7.3% | **1.3%** |
| 256 × 32 | 26% | 62% | 4.9% | unstable (ridge blows up on one network; in sample 22%) |

## 3. Verdict

1. **The learnable part of a Gaussian closure's error is the skew slaved to the mean, and nothing else** [M]. Where the
   closure is stable (L/n ≤ 1/32: three shapes, 26 networks, every one improved) it removes 63–65% of the κ₃-free
   chain's error, equal to that chain's learnable share (60–65%), and leaves a residual with no learnable part (−2…+2%)
   and almost no coherent level (1.6–4.2%) — the k = 3 chain's own profile (coherence law: ≤ 4% / ≤ 4%). Offline constants
   are as good as each network's own (64% vs 64% at 1024 × 16).
2. **What is left is the incoherent part, and it is most of the score** [M]. At 1024 × 16 the closure is ×2.9 better
   than κ₃ = 0 but the k = 3 chain is ×37 better than the closure: the slaved level is two thirds of a Gaussian
   closure's MSE and a small fraction of the log-gain of carrying κ₃. Carrying the rest needs the κ₃ state itself
   (Propositions 1–2: it is random-projection noise with no per-neuron predictor).
3. **Both slices are needed** [M]: the diagonal law alone (D = 0) is worse than κ₃ = 0 at four of five shapes; the two
   approximations cost about the same (law for t with the true D: 77–80% kept; true t with D = t/3: 72–75%).
4. **The closure fails with depth relative to width** [M]: 35% kept at L/n = 1/16 (spread −107…84%), worse than no κ₃
   at 1/8 (×1.47; with each network's own constants ×1.36), and the rank-one D with the chain's own t diverges there.
   256 × 8 and 512 × 16 (both L/n = 1/32) behave alike, so L/n, not the width, is the variable. Our reading [U]: the
   law's residual (10–20% of the κ₃ variance, the non-rank-one part of D — corr(D, t/3) falls from 0.81 at 1024 × 16 to
   0.67 at 256 × 32) is an inconsistent κ₃ that compounds layer by layer, the consistency principle of write-up §3.5
   again; κ₃ = 0 is at least consistent.
5. **Cost** [D]: the closure needs no κ₃ state — t is O(n) per layer and D = 1·tᵀ/3 enters the pair step elementwise, so
   it adds O(n²) per layer to covariance propagation's O(n³). We did not build a standalone "k = 2.5" estimator: the arm
   is our chain with its κ₃ replaced (the chain still computes the unused state).

**For whom this matters.** Anyone running a Gaussian-closure (covariance / mean-field) estimator on networks with
L/n ≤ 1/32 gets ×2.7–3 lower error for two offline constants per layer and no new state; anyone training a per-neuron
corrector on such an estimator learns, at best, this law. Beyond L/n ≈ 1/16 the law does not survive inside the chain.

## 3b. The partial-trace projection (2026-10-14 evening; arm `ptrace`, `res/skew_chain_run2.txt`)
Proposition 5 (`../2026-10-14_vm`) suggests a closure with no fitted constant: the pre-activation κ₃ projected on its
Isserlis pairing part, κ₃ ≈ Sym(I ⊗ v)/n, i.e. t = 3v/n, D_ab = v_b/n with v = t + D.sum(0) (the chain's own). Kept share of
the k = 3 gain (linear) / share of the log gain ln(zero/full):

| shape | law (clos_univ) | ptrace | law, log | ptrace, log |
|---|---|---|---|---|
| 1024 × 16 | 63.7% | 72.0% (10/10) | 22.6% | 28.4% |
| 512 × 16 | 62.9% | 72.7% (6/6) | 26.2% | 33.2% |
| 256 × 8 | 65.1% | 71.9% (10/10) | 30.0% | 35.3% |
| 256 × 16 | 34.5% | 51.0% (10/10) | 18.8% | 23.6% |
| 256 × 32 | −210% | −176% (1/10) | −18% | −47% |

The law keeps the part of v along m; the whole partial trace adds 7–10 points; what remains, ≈ 72% of the log gain at
1024 × 16, is the all-distinct part of κ₃ (Proposition 3). Not standalone: v is read from the chain's full κ₃. The re-run
reproduced every earlier arm bit for bit (the JSONs only gained the `ptrace` arm).

## 4. Files
- `skew_law.py` → `res/skew_law.json` (§1). `skew_chain.py baked | synth <n> <L>` → `res/skew_chain_<shape>.json` +
  `_final.npz` (final-layer predictions per arm and network); log `res/skew_chain_run1.txt` (1024 × 16, 256 × 16,
  512 × 16, 256 × 32) and `res/skew_cs.txt` (256 × 8 + the learnable-residual test). Runtime: ≈ 15 min in all.
- `skew_cs.py <tag>` adds `cs_*` / `cz_*` to `../2026-10-13_coherence-law/res/*.npz`; `coh_analyze.py --est cs|cz`
  (writes `res/summary_cs_*.json`, `summary_cz_*.json` there).
- `../2026-09-24_k3-chain/chain.py`: the online hook `mcorr` may return "D" (2026-10-14); absent = bitwise the old path,
  gate `ablate.py k4memo 3` = 3.7533e-08.
