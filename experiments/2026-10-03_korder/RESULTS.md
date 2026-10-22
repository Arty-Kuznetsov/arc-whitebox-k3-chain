# What are we missing? The next cumulant order, measured at small width (2026-10-02, evening)

**Why.** The leaders' raw error (1.1–1.6e-8) is better than our chain's cost-free ceiling (≈1.87e-8 projected), their
per-network errors are a near-constant ×0.57–0.80 of ours (CV 7–9%), and their improvement is about the same on the
all-layer mean as on the final layer (marius ×0.60 / ×0.57) — a better step at every layer, not a late fix
(`notes/INTEL-peers-2026-10-01.md` §5.1). The forum has nothing new since 09-22. Wu et al.'s error law is
c_K (L/n)^K, and our own measured ladder at 1024×16 was k1 3.3e-4 → k2 4.5e-6 → k3 4.1e-8 (×74, ×110 per order): the
next order is huge, and the leaders sit a small step into it. So: how big is the next order, and in which pieces?

**Instrument.** `korder.py`: random He-init ReLU MLPs at width n, depth L (the challenge's convention), Monte-Carlo
ground truth at 1e7 samples (cached in `res/`), ARC's reference (`mlp_kprop`) at k_max = 2, 3, 4 (SIMPLE / AUGMENT,
factored for k ≥ 3), and our float64 chain: `full` (must equal ARC's k=3 SIMPLE — the gate) and `B9m2` = the research
base (matched cut, E2, marg2; no band at this width). `subst.py`: our chain with ARC k=4's pre-activation statistics
substituted at layers 1 … L−1−SKIP. `k4bsweep.py`: the chain's existing κ₄-birth readers.

## 1. The ladder (final-layer MSE vs MC; two networks each)
| | k2 SIMPLE | k3 SIMPLE (= our `full`, gate exact) | k3 AUGMENT | **our B9m2** | **k4 SIMPLE** | k4 / ours |
|---|---|---|---|---|---|---|
| n=64, L=8 (one net, 1e6 MC) | 3.0e-4 | 4.1e-5 | 1.7e-5 | — | 5.6e-6 | — |
| n=128, L=8 (L/n = 1/16) | 1.4e-4 | 9.4e-6 | 4.5e-6 | **3.18e-6** | **1.10e-6** | **×0.34** |
| n=256, L=8 (L/n = 1/32) | — | 8.0e-7 | 3.58e-7 | **3.55e-7** | **6.1e-8** (≈4.7e-8 after the MC noise 1.4e-8) | **×0.13–0.17** |

- Our chain sits at the level of ARC's k=3 AUGMENT (at both widths), as at 1024×16.
- **k_max = 4 is ×3 better at L/n = 1/16 and ×6–7 better at 1/32**; the ratio shrinks with L/n as the (L/n)^K law
  says, so at our 1/64 the full next order would be worth roughly ×0.1. The leaders' ×0.57 is a small part of it.
- In the reference, k=3 AUGMENT and k=4 SIMPLE use the SAME Edgeworth diagram budget (`get_vec_cond`: k_max − 1 +
  augment = 3 for both); the only difference is that k=4 tracks the **full κ₄ tensor**, while k=3 AUGMENT keeps κ₄ as
  Sym(A⊗I) (no all-distinct part, an additive K22). The gain is the fourth cumulant, carried completely.
- Cost: ARC's factored k=4 stores κ₄ as Σ_r Sym(A_r⊗B_r) with A, B of shape (n, n, r), r growing with depth —
  ≈ L²n⁴ (the paper's bound). At n = 1024 that is far outside the budget: the question is what part is cheap.

## 2. Which channels carry it (`subst.py`, n=128, L=8; × of our chain's final MSE)
| substituted from k=4 | at all layers 1–7 (incl. the last readout) | upstream only (1–6; SKIP=1) | 1–5 (SKIP=2) |
|---|---|---|---|
| var | ×0.795 | ×0.849 | ×0.939 |
| full pre-covariance C | ×0.792 | ×0.803 | ×0.846 |
| t (κ₃ diagonal) | ×0.863 | ×0.964 | ×0.971 |
| D (κ₃ (2,1) slice) | ×0.999 | ×0.999 | ×0.961 |
| t + D | ×0.727 | ×0.851 | ×0.898 |
| **K4d (κ₄ diagonal)** | ×0.776 | **×0.739** | ×0.868 |
| K22 (κ₄ (2,2) slice) | ×0.981 | ×0.981 | ×1.036 |
| K4d + K22 | ×0.798 | ×0.758 | ×0.961 |
| t + D + K4d + K22 | ×0.521 | ×0.617 | ×0.812 |
| all (C, t, D, K4d, K22) | ×0.344 (= k4S: the last readout is k=4's — trivially) | **×0.457** | ×0.735 |

- As at 1024×16 with Monte-Carlo oracles: each channel alone is worth little, together a lot — the consistency of the
  next order is the point.
- **One step of our own chain on k=4-quality inputs already gives back ×1.33** (all: ×0.344 → ×0.457), two steps ×2.1:
  most of the gap is born in every step, and the single largest upstream channel is the **κ₄ diagonal** (×0.74).
- The chain's existing κ₄-birth readers do not reach it (`k4bsweep.py`): k4b = 1…L with all four tree parts
  ×0.995–1.07 at n=128 and ×1.00–1.01 at n=256; the cost-free levers of cloud #12 B (allages, k4k3 0.5) ×0.94–0.97.

## 3. The κ₄ diagonal: what our closure misses (`k4diag.py`, `k4classes.py`, `k211.py`; n = 128 / 256, L = 8)
- Against k=4's κ₄ diagonal on our own trajectory, our closure is **19–42% wrong (rms), corr 0.75–0.86, slope
  0.78–0.98, coherent share 0–10%**: a PATTERN error, not a level (var is right to ≤1.5%, t to 4–12%). The chain's
  κ₄-birth readers at full depth (k4b = L, all tree parts) do not move it (corr 0.78–0.86).
- k=4's K4d(l) split by the index class of κ₄post(l−1) it comes from (n=128): the **level** is the (2,2) class (+0.8 …
  +1.0 of the mean) plus the diag (+0.06 … +0.19) — the odd classes average to zero, as the parity rule says; the
  **pattern** (share of the centered variance): **(2,1,1) 35–78%**, (2,2) 17–35%, all-distinct 10–25%, (3,1) 1–7%,
  diag ≈ 0. The bivariate classes (diag, (3,1), (2,2) — O(n³) to project) correlate only 0.48–0.71 with the truth;
  adding (2,1,1) → 0.87–0.95.
- The (2,1,1) slice of κ₄post is **not a fresh Gaussian birth**: the doubled-hub tree (Leonov–Shiryaev coefficient
  2[Φ(1−Φ) − φ(αΦ+φ)]) correlates −0.13 … +0.20 with it at the slice level. It is inherited — e.g. the (2,2) class
  of layer l−1 projects into the (2,1,1) slice of layer l (Σ_pr W_pa² W_rb W_rc K22post(p,r)), which is O(n³) to
  carry for one step; the (2,1,1)→(2,1,1) step needs the n³ slice itself (n⁴ per layer).

- Nor is it the (2,2) class carried two steps (`k22to211.py`, n=128): the projected K22post(l−1) explains 7–16% of
  the pre-activation (2,1,1) slice at l (corr 0.26–0.40), and the closed-form two-step carrier correlates 0.00–0.20
  with the (2,1,1) part of K4d(l+1). The (2,1,1) content is inherited through the (2,1,1)→(2,1,1) and higher-class
  paths — the n⁴ part of κ₄.

## 4. The κ₃ → κ₄ transfer, found — and why carrying it does not pay (`k3to211.py`, `t3k4test.py`, `zdiag.py`)
- **Where the (2,1,1) content comes from.** Expanding y = μ + d₁δz + ½d₂(δz² − var) + … and applying Leonov–Shiryaev
  to κ(y_a, y_a, y_b, y_c) gives, besides the Gaussian doubled-hub tree, terms linear in the pre-activation cumulants.
  Against k=4's post (2,1,1) slice (n=128): the **κ₃ transfer through the self-variance**,
  T3 = 2·var_a d₁_a d₂_a d₁_b d₁_c κ₃pre(a,b,c), correlates **0.62–0.81** (slope 1.07–1.6); the gated pre (2,1,1)
  slice G = d₁_a² d₁_b d₁_c κ₄pre(a,a,b,c) 0.38–0.78 (rising with depth); **T3 + G 0.81–0.90**, slope ≈ 1.0; with the
  D-edge and edge×κ₃ terms 0.84–0.92. The κ₄ pattern our closure lacks is, to first order, the FULL κ₃ tensor
  (all-distinct part included) handed to κ₄ at every ReLU and then inherited.
- **One-step carrier** (`chain.py t3k4="exact"`: T3's projection on the next K4d through the sources' legs, one
  pairing-size product per source): K4d corr with k=4 0.80 → 0.84–0.86; final MSE ×0.893 at n=128 (×0.809 / ×1.028),
  ×0.997 at n=256 (×1.016 / ×0.982). Mean-field form ("mf", O(n w)): ×0.941 / ×1.008.
- **Inherited carrier** (`chain.py zlegs="nodiag"`: per source, "doubled legs" Z = (2/3)·w·h∘X for both CP families
  of the source — the doubled index gated by d₁² and riding W∘W, the singles the source's own legs, the b = c part
  excluded because it belongs to the (2,2) class the closure already carries): K4d level right, corr ≈ 0.85 (= the
  one-step carrier: the W∘W path of the inheritance adds nothing), final MSE **×1.112** at n=128 (×0.877 / **×1.493**)
  and **×1.058** at n=256; at half strength ×0.923 / ×0.983. Without the b = c exclusion the level is +29…+58% wrong.
- **The feed's input.** The κ₄ → κ₃ feed reads the PRE-activation (2,1,1) slice as the closure's v·var_a·C_bc; that
  form explains only R² 0.30–0.38 of k=4's slice (`feedslice.py`; its level v matches k=4's diagonal). The feed itself
  matters (off: ×1.40 at n=128, ×1.09 at n=256). Feeding it k=4's true slice instead (`feedoracle.py`, `chain.py
  dense3`: a dense κ₃ tensor riding the chain, small widths only; gate exact when off): **×0.870 (×0.754 / ×1.059)**;
  half strength ×0.999; added on top ×0.964 — the missing 62–70% of the slice is real content, and alone it is again
  a coin toss across networks.
- Every partial κ₄ improvement measured here raises the κ₄ diagonal's agreement with k=4 a little and does not move
  the final error, or moves it differently on different networks — the consistency mechanism again (§3.5 of the
  write-up, NEGATIVE §73): k=4's ×3–7 comes from all channels being of the same order at once.

## 5. The last layer alone (`lastlayer.py`; no downstream, so no consistency issue)
k=4's pre-activation statistics substituted at the LAST layer only (× of our chain's final MSE):
| | var@L | t@L | K4d@L | t+K4d@L | **var+t+K4d@L** |
|---|---|---|---|---|---|
| n=128, L=8 | ×0.965 | ×0.863 | ×1.057 | ×0.905 | ×0.809 (×0.778 / ×0.858) |
| n=256, L=8 | ×0.832 | ×0.891 | ×0.877 | ×0.748 | **×0.572 (×0.576 / ×0.568)** |
A method whose last-layer var, t and κ₄ diagonal were of the next order would be at the leaders' level — consistently
across networks. The κ₃ → κ₄ transfer at the last layer only (`chain.py t3k4_last`) does not supply it: ×0.966 /
×1.001 (its K4d corr stays ≈ 0.80); the last layer's statistics are projections of the inherited joint objects.

## 6. The inherited transfer with the exact propagators (`k4lastQ.py`, in k=4's own world, n=128, last layer)
The T3 pieces born at every earlier layer, carried to the last layer's pre-activations by the exact gated propagator
(the doubled index through Q∘Q — all split paths — or through the product of the layers' W∘W, as the doubled legs do):
their sum correlates **0.79 (Q∘Q) / 0.77 (W∘W)** with the part of k=4's K4d(L) that the bivariate (2,2)+diag classes
of the last step leave unexplained — the right pattern at about a third of its size (slope 3.5). But against k=4's
K4d(L) itself, the closure's var² pattern alone already has R² **0.71 / 0.74** (two networks), and adding the inherited
transfer raises it to **0.74 / 0.74** (Q∘Q; with the bivariate part too, 0.76 / 0.75). The transfer is collinear with
the scale-mixture pattern; the quarter of the κ₄ diagonal that the closure misses is something else — not the
transfer, not the bivariate classes, not the split paths.

## 7. A second shared factor? (`twofactor.py`, n=128) — no
A "two-factor scale mixture" (the one-factor closure plus a shared rank-1 component λ b⊗b⊗b⊗b) would be an O(n²)
closure, consistent across all κ₄ slices by construction. Against k=4: the one-factor form explains 0.78–0.91 of
K22's energy (falling with depth); the residual's top singular direction holds only 14–24% of the residual, its b⁴
predicts the κ₄-diagonal residual at corr 0.37–0.57, and b aligns weakly (≤ 0.41) with C's top eigenvector or μ.
The residual is spread over many directions: no single shared non-Gaussian factor to add.

## Reading [U] and what to measure next
The "explicit method we have not applied" is, on this evidence, **the fourth cumulant carried consistently** — what
ARC's k_max = 4 does at n⁴ cost. The leaders cannot be running it at 1024×16; they must have a cheap partial version.
By the parity rule (`../2026-10-02_parity/`), the next layer's κ₄ diagonal is Σ W_ai W_bi W_ci W_di κ₄post_abcd: only
its diag and (2,2) index classes carry a coherent level (what our closure models); the (3,1), (2,1,1) and all-distinct
classes add zero-mean projections whose realised values the closure cannot know. Two of those classes are BIVARIATE
(the diag and (3,1) slices of the post-activation κ₄ depend on pairs (z_a, z_b) only, like the pair expansion) and
their projections cost O(n³) per layer — affordable. Next:
1. ✅ Done (§3): the κ₄ diagonal's pattern is mostly the INHERITED (2,1,1) class, not births and not the bivariate
   classes alone.
2. (a) ✗ the (2,2)→(2,1,1)→K4d two-step carrier explains almost nothing. Open: (b) exact class transport for the last
   1–2 layers with the closure behind it; (c) a factored κ₄ carrier whose legs ride the κ₃ legs (the (2,1,1) content
   as Sym(A⊗A⊗B⊗B)-type terms built from the sources' own legs, all ages) — measure how much of k=4's κ₄ diagonal and
   of its final-MSE gain each recovers. A scheme worth ×0.6–0.7 at O(n³) per layer would be the leaders' level.
3. The ladder at n=512, L=8 (L/n = 1/64, our ratio; k=4 ≈ 25 min per network) — on the cloud.
