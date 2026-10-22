# When can a per-neuron learned correction work? The coherent share of the error across network shapes (2026-10-13)

**Question.** Write-up §6.4 (Proposition 1) says the next layer's per-neuron errors of a moment-propagation estimator
are random projections of this layer's joint errors through independent weights: a level shared by all neurons (the
coherent part) plus noise that is independent across neurons. A model that corrects a neuron's moments from that
neuron's own state can learn the coherent part and nothing of the noise. In Phase 2 (1024 × 16) the coherent part of
the final error is 0.46% of the MSE and every per-neuron corrector failed (ours, 504aldo's row 10). In Phase 1
(256 × 32) mliston's learned corrector of the closure's centering error worked, ×0.81–0.83 (forum 18170).

**Prediction, written before any run (2026-10-13 14:20).** For one estimator, ARC's SIMPLE k = 3 (our chain's config
'full'):
1. Odd orders have no coherent level at any shape: the per-neuron errors of the pre-activation mean and of the κ₃
   diagonal have z = mean/SE ≈ N(0, 1). Even orders (the variance) carry a level.
2. The coherent share of the final post-activation-mean error, and the share a per-neuron model learns out of
   sample, both grow with depth and with L/n. At 1024 × 16 both are ≈ 0 (≤ 2%); at 256 × 32 the learnable share is
   ≥ 10%, the size of mliston's gain.
3. The learnable share tracks the coherent share: a per-neuron model learns the level and little else.
**Falsifiers.** If at 256 × 32 the learnable share is ≈ 0 while mliston's corrector worked there, Proposition 1's
consequence does not explain their success (their GRU would then use information we do not model). If odd orders
carry a level, Proposition 2 fails at that shape.

**Instrument.** `coh.py` (data) and `coh_analyze.py` (analysis). Synthetic shapes: random He-init networks, two
Monte-Carlo seeds per network (noise = half their difference, subtracted from energies). 1024 × 16: the ten public
networks with their 1e9-sample ground truth (no Monte-Carlo needed).

## Pilot (2026-10-13 14:20; 2 networks per shape; `res/pilot.log`, `res/summary_*.json`)
| shape | L/n | final MSE | coherent share of the final post-mean error | learnable out of sample (leave one out) | variance error, coherent share at the last layer | mean and κ₃ errors |
|---|---|---|---|---|---|---|
| 256 × 32 | 1/8 | 2.6e-5 | 2.4% (0.7%, 4.2%) | 2.2% | 30% (8% at layer 8) | ≈ 0 at every layer |
| 1024 × 16 (public, 1e9 truth) | 1/64 | 4.1e-8 | 0.3% | ≈ 0 | — | — |
Prediction 1 holds at 256 × 32 (odd orders carry no level; the variance's level grows with depth). Prediction 2 fails on
the pilot: at 256 × 32 only ≈ 2% of the final error is learnable from a neuron's own state, not ≥ 10%.

**What mliston's corrector actually used (forum 18170, re-read 2026-10-13).** The anchor is a cumulant chain compared at
layer 22 with the sample mean of a 6–8.6k-sample antithetic cloud; the GRU (≈52k parameters, a few thousand training
MLPs) predicts the closure's centering error from the moment state **plus sample-shape statistics of that cloud**, and
cut raw MSE by 17–19% on 20 MLPs. Sample statistics carry information about the JOINT error, which Proposition 1 does
not forbid; so their success is not a counterexample and needs no coherence argument. Our earlier reading in
`notes/INTEL-peers-2026-10-01.md` §5.6 ("at depth 32 the truncation bias is coherent and learnable") was a guess [U]
and the pilot does not support it.

## Full run (2026-10-13 15:25; `res/summary_c_*.json`, `res/summary_c2_*.json`, `coh_analyze.py --est c|c2`)
10 networks per shape (6 at 512 × 16); 1024 × 16 = the ten public networks with 1e9 truth; the rest two MC seeds of 2e6
(512: 4e6), all energies as cross products of the two seeds. "Learnable" = leave-one-network-out ridge on 8 smooth
features of the neuron's own state (α, α², α³, log σ, (log σ)², α log σ, t/σ³, α t/σ³).

| shape | L/n | k = 3: final MSE | k = 3: coherent | k = 3: learnable | k = 2: final MSE | k = 2: coherent | **k = 2: learnable** |
|---|---|---|---|---|---|---|---|
| 1024 × 16 | 1/64 | 3.9e-8 | 0.3% | 0.0% | 4.4e-6 | 26% | **66%** |
| 256 × 8 | 1/32 | 9.3e-7 | 2.1% | 1.4% | 3.7e-5 | 35% | **66%** |
| 512 × 16 | 1/32 | 3.5e-7 | 1.2% | 4.1% | 1.9e-5 | 28% | **66%** |
| 256 × 16 | 1/16 | 2.8e-6 | 0.9% | 1.5% | 6.9e-5 | 26% | **64%** |
| 256 × 32 | 1/8 | 1.4e-5 | 3.0% | 3.4% | 1.2e-4 | 27% | **64%** |
Parity at every shape (k = 3, coherent share at the last layer): pre-activation mean 0.1–0.7%, κ₃ 0.1–0.6%; the
variance 4–23% (it grows with depth: 256 × 32 from ≈ 0 at layer 4 to 19–30% at the last layer).

**Which features carry k = 2's learnable error** (`feat_ablation.py`, leave one out): intercept alone 25% (1024 × 16) /
20% (256 × 32); α alone 56% / 45%; a cubic in α 65% / 53%; + log σ 66% / 64%; log σ alone 27% / 33%; the κ₃ and κ₄
readout patterns alone ≈ the intercept. The learnable part is a shared level plus a smooth function of α = μ/σ.

**Why** (`feat_ablation` + Proposition 3, `../2026-10-13_prop3`): the TRUE pre-activation κ₃ of a neuron is almost a
function of its own mean: corr(κ₃ᵢ, μᵢ) = +0.87 … +0.93 at the last layer and (1, μ, μ², σ) explain 79–88% of its
variance (noise-corrected, every synthetic shape). Mechanism: the next diagonal's κ₃ is dominated by the partial-trace
term 3s²·wᵢᵀv (Proposition 3), and v is almost parallel to the post-activation mean vector m, cos(v, m) = +0.92 … +0.99
(prop3, n = 128 / 256), while μᵢ = wᵢᵀm through the same column. A Gaussian closure drops κ₃, so its error is a
predictable function of the neuron's own (μ, σ); a k = 3 chain carries κ₃ and what is left is the incoherent noise of
Proposition 1.

**Verdict on the pre-registered prediction.** (1) holds at every shape. (2) fails: the learnable share does not grow
with L/n; it is set by the truncation ORDER (k = 2 ≈ 65% at every shape, k = 3 ≤ 4% at every shape). (3) holds for
k = 3 (learnable ≈ coherent); for k = 2 the learnable share exceeds the coherent one, because a function of α is learnable,
not only the level. The law, as measured: **a per-neuron learned correction can remove about two thirds of a Gaussian
closure's error at any of these shapes, and almost nothing of a chain that carries κ₃.** That explains why learned
correctors helped Gaussian-closure anchors and failed on every k ≥ 3 chain in Phase 2, without a depth argument.
