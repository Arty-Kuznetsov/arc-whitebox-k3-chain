## 6. The boundary: four mechanisms that decide what is possible here

We think this section and §7 are the contribution. §3 is a sequence of terms that pay; this section is a map of why
the remaining ideas do not, and the map is made of measurements. Each mechanism is stated as a rule, with the
prediction it makes about ideas we never tried. Over three weeks we catalogued seventy-six negative results with numbers
(the repository's `notes/NEGATIVE-RESULTS.md`; we cite its entries as N1–N76, which the repository's own notes write as
§1–§76); the four mechanisms below account for every one that failed for a reason more
interesting than a bug or a budget.

### 6.1 A cost model that prices the saved product and not the construction says yes every time

Every attempt below has the same shape. Some object in the chain costs `k` dense n³ products per layer. A compression
exists that would carry the same information in `q·k` products, `q < 1`. The cost model says the saving is `(1−q)·k`,
and the accuracy measurement says the compression is affordable. **Then the construction that produces the
compressed object turns out to cost at least as much as the saving.** Five independent instances, each closed by its
own measurement [M]:

| # | what was to be compressed | modelled saving | what the construction cost | net |
|---|---|---|---|---|
| N54 | old sources onto a shared subspace (band 6:320, *re-rotated every layer*) | 26.3% → 21.4% of budget, ×1.20 | the per-layer entry update (Gram applications, QR, `eigh`, coefficient rotation ≈ 4–7 products per band source-layer) | **27.78% measured, i.e. worse than the 26.62% baseline** |
| N55 | the path leg at rank ρ = 256 (`pathrank`) | −0.7 points of B for +0.66% MSE | the rank-ρ basis: `YΩ` + `QᵀY` (2q) on top of `WᵀQ` + `(WᵀQ)Z` (2q) = **4q = 1.0 units at ρ = 256**, before a `qr` worth 0.125 | break-even *before* the QR; measured on the meter at rank 384: **×1.0495 of the dense leg** |
| N58 | the old κ₃ tail as one rank-n block | 24.14% → 12.69–13.87%, ≈ **8.8 units per layer**, and a rank-n block provably **exists** (n = 64, converged to 4.29e-4) | one Gram-form fitting pass is `8·n·R·r` flops = **8.0 units**, so the budget is *one pass*; 400 passes reach 4.32% **and are still falling** | **≈400× gap**, and no warm start: adding one source to a converged block throws the error 2.9% → **48.7%**, because the CP optimum **is not nested** |
| N59 | the same block by unbiased sampling (O(n²), no passes, nested by construction) | the budget is 3–5% relative error | the estimator's own variance: relative error ≈ κ/√r with **κ = (Σ‖term‖)/‖T‖ = 62.08** | **194% at r = n**; 5% would need r ≈ 1.5·10⁶ columns against the 15 360 the tail contains |
| N76 | the old tier read through a smaller per-layer optimal basis (r′ < 448), or truncated by age, or its D taken from the young sources only | r′ = 256 would save ≈8% of the FLOPs; dropping the three oldest sources 11% of the member-layers | the content itself: reading the members in 256 dimensions costs ×1.11–1.15 raw and in 128 ×2.7–3.1; the three oldest sources dropped ×1.17; D from the young sources only ×9.25 | **the band at rank 448 and entry age 5 is the cheapest exact-enough carrier in this family** (both knobs are sharp optima, §3.6) |

The five share one cause: the accounting was incomplete. A compression has two costs, transporting the compressed
object and producing it, and only the first is what a "this would be cheaper" estimate naturally counts. The second is
paid every layer, because the object being compressed is regenerated every layer by the nonlinearity.

**The counter-example that confirms the rule.** One compression of the old sources *does* ship (§3.6): a basis of
fixed rank 448 that **rides the transport**: it is moved by the same products as the legs and never re-built,
re-rotated or re-orthonormalised, so its construction cost is paid once per source, at its entry. Its condition number
grows ≈2.5× per layer (to 7·10⁴ by the last entry) and that is harmless, because nothing ever needs it orthonormal again
[M]. The difference from N54 is exactly the term the rule says to price.

**The prediction we would hand the next person.** Before building any compression of a propagated object, price the
*basis*, per layer. For a projection onto `r` directions the basis needs at least one pass over the object (`q` units)
plus one to express the object in it (`q`), plus orthogonalisation; for a CP-type re-parameterisation it needs a
solver, and N58 shows the solver has no warm start. The saving must beat **all** of that, or the basis must ride the
transport for free.

**Budgets other than the meter.** Exact arithmetic savings obey a different rule, and we report it because the
grader enforces it. One more Strassen-Winograd level everywhere (leaf 64 → 32) bills ×0.939 of the FLOPs, changes the output by ~1e-6, and doubles the wall time (×2.2; ≈140 s projected against the 120 s cap) at a peak RSS of 7.21 GB of
8 GB (N69) [M]: a 32×32 leaf has no BLAS efficiency left. Pricing each contraction separately by **FLOPs saved per
second of wall added** (0.49 %/s for the leg transport down to 0.27 %/s for the D pairing) and deepening only where it
pays recovered most of that ×0.939 on five sites, inside the cap (grader wall 98.8 s max); the next site measured at
0.094 %/s (both of its products have a 448-wide side, so its level-5 leaves are 14 wide) and was not taken [M].

### 6.2 The principle underneath the four: the third cumulant is not closed on the slices the step reads

The four instances in §6.1 are failures of *representations*. The question that sits under all of them is whether
the κ₃ tensor needs to be represented at all, since the nonlinear step reads only two of its slices, the diagonal
t and the (2,1) slice D. We measured that directly.

With the leg transport M = Wᵀ·diag(d₁) (verified to reproduce the chain's own transport to ≤ 5.1e-16), the next
layer's diagonal splits exactly, by coincidence pattern of the three indices, into a t-term (Σ<sub>j</sub> M<sub>ij</sub>³ t<sub>j</sub>), a D-term
(3 Σ<sub>j≠k</sub> M<sub>ij</sub>² M<sub>ik</sub> D<sub>jk</sub>), and an **all-distinct** term Σ<sub>j,k,m distinct</sub> M<sub>ij</sub> M<sub>ik</sub> M<sub>im</sub> T<sub>jkm</sub>, which we obtain as a
remainder without ever forming n³ (closed forms checked against brute force to 6e-17 and 4.3e-16). Then:

| quantity | measured (n = 1024) |
|---|---|
| share of the next diagonal carried by the all-distinct term | **28.8%** per layer (layers ≥ 8; 27.2–31.2%) |
| a chain whose κ₃ state is (t, D) only, end-to-end | **raw ×25.6** (compounding: 1.288¹⁴ ≈ 34) |
| fraction of the all-distinct term a closer must capture to stay within 3% raw | **≥ 93%** |
| best oracle closer on any O(n²) basis we could construct | **19.5%** |
| of which re-fitting the t- and D-coefficients | **0.16%** — the missing part is orthogonal to both (corr ±0.03); the D-coefficient fits to 1.000 |

So a (t, D) state is insufficient, and what is missing is **orthogonal to everything computable in O(n²)**. That is a
statement one level above §6.1: the κ₃ state must carry Θ(n³) information forward, and the cost of this chain is the
price of that information. The contrast that explains why our κ₄ closure (§3.2) works and a κ₃ closure cannot is
worth one line: κ₄ enters the readout at second order, whereas κ₃ *is* the readout, so its tolerance is an order of
magnitude tighter.

**Why 28.8%, derived.** The share is not a property of our arithmetic; it follows from how the gated κ₃ tensor's
energy is spread over index patterns.

> **Proposition 3 (the share the slices do not carry).** Let G = T ∘ (d₁ ⊗ d₁ ⊗ d₁) be layer l's κ₃ tensor with the
> gates on its legs, v<sub>a</sub> = Σ<sub>b</sub> G<sub>abb</sub> its partial trace, and G<sub>AD</sub> its entries with three different indices. With
> W(l+1) i.i.d. N(0, 2/n) and independent of G, the part of the next diagonal that (t, D) do not determine has, for every
> neuron,
>
> share² = E‖AD‖² / E‖t′‖² = 6‖G<sub>AD</sub>‖²<sub>F</sub> / (6‖G‖²<sub>F</sub> + 9‖v‖²). [D]
>
> *Proof.* t′<sub>i</sub> = Σ<sub>abc</sub> w<sub>a</sub>w<sub>b</sub>w<sub>c</sub>G<sub>abc</sub> with w the i-th column of W(l+1). Isserlis' theorem for the sixth moment
> of a Gaussian vector gives 15 pairings: the 6 that match the two triples give 6s⁶‖G‖², the 9 that pair inside each
> triple give 9s⁶‖v‖². Terms with three distinct indices survive only the 6 matching pairings and are uncorrelated with
> every term that repeats an index. ∎

*Measured* (random networks of depth 16, the dense tensor built from the chain's own sources, the share taken with the
network's own next weights): the ratio of measured to predicted share is 0.94 ± 0.15 at n = 64, 1.00 ± 0.12 at 128 and
0.99 ± 0.06 at 256 (layers ≥ 8), the scatter shrinking as 1/√n [M]. On the true κ₃ tensor, estimated by Monte Carlo
(two seeds of 2·10⁶ samples, energies as cross products), the ratio is 0.94 ± 0.10 at n = 64, 1.01 ± 0.13 at n = 128 and 0.97 ± 0.04 at n = 256:
the share is a property of the network family, not of our chain [M] (`experiments/2026-10-14_vm` §9). Most of the tensor's energy is all-distinct (72% →
88% from n = 64 to 256), but the partial trace weighs more still and grows faster (9‖v‖²/6‖G‖² = 2.1 → 4.6), so the
share falls only slowly with width: 46% → 40% over that range and 28.8% at 1024. No practical width closes κ₃ on its
slices. And v points almost exactly along the post-activation mean vector m, cos(v, m) = +0.92 … +0.99 [M], which has a
consequence for the next section: the dominant part of a neuron's next κ₃ is a multiple of its own next mean
(`experiments/2026-10-13_prop3`).

**The larger-state door, closed by the same principle.** If the slices are not enough, a richer O(n²) state might
be: we let the old tail evolve as a Markov chain of its own slices (`MST`, N66). Its all-distinct share is 27–28%, so the
old tail is exactly as non-slice as the young, and the chain it gives is dominated by ours at every age threshold [M].

**Independent replication.** While we were measuring this, team 504aldo published on the public forum, with
their code under MIT, a catalogue of 25 dead ends for the same chain at the same shape (AIcrowd Discourse topic
18218, 2026-09-10). Their rows match ours almost one for one: a memoryless closure on the (2,1)/(3,) slices is 45×
worse, "the lost part is 18–23% of D21 energy per layer, high-rank, not predictable from live-state features
(+0.02 R²)" (our 28.8%); a CP-rank cap on the merged tensor needs 8n columns for 3–14% D21 error where 2% is needed,
with "flat column importance" (our N58); learned output corrections see features correlated < 0.09 with the error
(ours ≤ 0.03); and their row 21 states our mechanism in their own words: "forming a leg from the basis costs what
transporting it costs; this is THE wall". Two teams, different code, different ground-truth instruments, the same
walls. We think that turns "we could not" into "this family of representations cannot", which is the claim we make.

**What neither team explains**, and we say so: 504aldo observe that their bill *without* the old tier is 0.150·B,
and that the top three entries sat at 0.149, 0.152 and 0.164·B on the same day with better raw error than either of
us. Whatever carries the leaders' old-source content costs them approximately nothing. It is not a compression of
the source representation, both catalogues rule that out, and we do not know what it is. [U] The public board of
2026-10-16 says what that is worth: teams at about our raw error (1.95–2.05e-8) sit at 10–19% of the budget, one of
them at rank 11 on the 10% floor. Our raw error at that floor would score 2.03e-9, about 10th place against our 55th;
the cost of carrying κ₃ alone is worth some forty places [M].

### 6.3 Coherence decides whether a cheap correction can help at all

**Prior work.** The mechanism of this section was published first, in Phase 1 at depth 32, by jamesrahenry (AIcrowd Discourse 18097; summarised in the Phase-1 technique census, 18157, item 1c): the chain is an error-compensating system in which zero-mean perturbations damp while coherent injected corrections amplify, at a measured ~16:1. We rediscovered it independently at depth 16 and credit it here. What this section adds is the Phase-2 measurement — a variance-level parabola ~1e-5 wide, the incoherent/coherent ratio 44 ≈ √n — and its second, less obvious use in (b) below, where the same argument explains why a same-sign sum *cannot* be sampled.

The scored quantity is a mean over n = 1024 neurons of a quantity built from same-sign contributions. That one
fact sets the value of every correction we might add, through the distinction between an error that is
**coherent** across neurons (a smooth function of smooth per-neuron quantities) and one that is
**incoherent** (independent per neuron). The coherent part is paid in full; the incoherent part is discounted
by √n. Two measurements pin it down, and they point in opposite directions, which is the useful part.

**(a) The variance channel: the chain is already at the bottom of a parabola 1e-5 wide.** Substituting the
Monte-Carlo oracle's exact per-neuron pre-activation variance at every layer is worth ×1.54 on the final MSE,
so the channel is large. But scaling the variance by a constant, `var ← var(1+c)`, gives

> final MSE ≈ 2.4e-8 + 5·c², minimum at **c ≈ 0 to within 3e-5**; c = 1e-4 costs **×2.7**, c = 1e-3 costs ×200.

The *level* is therefore already correct to about 1e-5 relative, and every low-parameter correction of the
*pattern* we tried is **worse than doing nothing**: a per-layer level −0.59×, level plus a log-variance term
−0.54×, a seven-feature per-layer least squares −0.535×, with the fit explaining only **R² = 0.098** of the
error. The reason is coherence: a smooth function of (α, log var, κ₄/var², …) is smooth across neurons, so it
acts coherently and is charged at the coherent rate, while the part of the error it captures is incoherent and
worth √n less. Measured directly: an rms pattern error of 1.8e-3 at depth costs the same as a *coherent* shift
of 4.1e-5, a ratio of **44 ≈ √n**.

The same argument explains a result we had previously mis-read as evidence of error cancellation: that
substituting an "exact" covariance made things worse (N50). It did, because the substitution was fed by a
4·10⁶-sample oracle whose own noise, 6.6e-4 relative, is *coherent* in exactly this sense.

**A methodological trap worth one paragraph**, because it cost us an hour and it will cost the next person
more. A multiplicative correction measured on the base trajectory is **stale from the next layer on**: once
the variance is touched, the chain's own variance moves, so a factor calibrated as `var_true/var_chain` no
longer lands on `var_true`. At a stiffness of ×2.7 per 1e-4, that drift alone reads as a ×0.658 "result" for
an arm that should be the ×1.54 ceiling. Only corrections that are a **function of the current state** can be
tested, which is also the only kind production could compute.

**(b) The κ₃ tail: same-sign in the readout, nearly orthogonal in the norm that matters.** The tail of old
sources carries ≈69% of ‖t‖ at depth and its terms are same-sign, which is why every *truncation* of it fails
coherently: dropping the light terms of a same-sign sum loses their whole mass. That observation is what made
unbiased importance sampling look like the answer: an unbiased estimator keeps the mass in expectation, and
its error is a random fluctuation rather than a systematic loss. It fails anyway, for this reason:

> sampling variance is governed by the **Frobenius norm of the tensor**, not by the diagonal readout, and in
> that norm the tail's terms are nearly orthogonal: **κ/√R = 0.50**, against 0.008 for perfect coherence and
> 1.0 for full orthogonality.

Same-sign `t` and near-orthogonal `T` are not in conflict: `t` is one diagonal of the tensor, while sampling
damages all of it. **So "the terms add coherently" is a statement about a particular contraction, and it does
not transfer to a different norm.** We had the right mechanism and applied it in the wrong norm; the
measurement corrected us.

### 6.4 The next layer's per-neuron errors are random projections of this layer's joint errors

The third boundary explains the second family of failures: everything we tried that *learns* a correction of the
chain's per-neuron moments. It is a small theorem, stated after a cloud session had measured the object it is about.

> **Proposition 1 (per-neuron variance errors are random projections).** Let `ΔC(l) = C_chain(l) − C(l)` be the error
> of the chain's post-activation covariance at layer `l`, and `ΔC₀` the same matrix with its diagonal removed. Let
> `w_i` be the i-th column of `W_{l+1}`, with i.i.d. N(0, 2/n) entries. The chain computes the next layer's
> pre-activation variances exactly from its own covariance, so its error for neuron i is
>
> e_i = w_iᵀ ΔC w_i = Σ_j w_ji² ΔC_jj + w_iᵀ ΔC₀ w_i.
>
> Conditioned on ΔC: the first term has mean (2/n)·tr ΔC, the same for every neuron, with a fluctuation of relative
> size 1/√n; the second has mean zero and variance 8‖ΔC₀‖²_F / n²; and the e_i are independent across neurons. [D]
>
> *Proof.* ΔC(l) is a function of `W_0, …, W_l` only, so it is independent of `W_{l+1}`. Given ΔC, the quadratic form
> in the Gaussian vector w_i has mean (2/n)·tr ΔC, all of it from the diagonal because E[w_ji w_ki] = 0 for j ≠ k; the
> off-diagonal part has variance Σ_{j≠k} 2·(2/n)²·ΔC_jk² = 8‖ΔC₀‖²_F / n²; the columns w_i are independent given ΔC,
> hence so are the e_i. ∎

*Measured.* On network 0's real matrices the ratio of the observed off-diagonal energy to that prediction is 0.89–1.09 at every layer
and its mean is at most 0.05 of its rms at 14 of 15 layers (0.10 at one; 1/√n = 0.03). Against four 10⁷-sample oracles
of the full post-activation covariance (two per network, noise subtracted with their difference), the off-diagonal
term carries **95% of the energy of the next layer's variance error** (90–105% at every layer ≥ 3, on both networks;
Figure 5) [M].

**Consequence 1: no per-neuron model can see the error that matters.** The only per-neuron features correlated with
`w_iᵀ ΔC₀ w_i` are quadratic forms of the neuron's own weight column against
an n×n matrix, such as its own variance `w_iᵀ C w_i`, which sees only the part of ΔC₀ proportional to C (5–14% of the signal, Consequence 3). A model of the
moment error on summaries of the neuron's state sees the coherent level, that small part, and noise
[D]. This is what we had measured *before* we
understood it [M]: removing the projection of the exact moment error onto 20 smooth per-neuron features buys ×1.015 /
×0.957, while removing the remainder buys ×0.269 / ×0.450 of a ×0.206 / ×0.302 joint ceiling (N70); a joint variance +
κ₄ corrector looks synergistic in sample (×0.885 against ×0.916 for κ₄ alone) and has a held-out variance R² of
**0.004** across ten networks; out of sample the variance channel makes it worse (N71).

**Consequence 2: the one exception, and why.** The κ₄ diagonal's error is not purely a projection: it has a level that
depends on the neuron's environment, and O(n²) row reductions of the matrices the chain already holds (C0, D, K22) see
it: held-out R² 0.36, and the shipped corrector is worth ×0.988 on the grader, exactly its out-of-sample local value
(§3.7) [M]. Own-moment features alone see nothing (N70) [M].

**Consequence 3: the reserve is real, and diffuse.** Substituting the exact off-diagonal covariance at every layer
would be worth **×0.55–0.68** (an oracle upper bound, noise-corrected two ways) [M]. Nothing cheap carries it [M]:
- **low rank:** two eigendirections reproducible across oracle seeds at 0.99, aligned with the post-mean vector, hold
  40–65% of ‖ΔC₀‖² at depth and buy ×0.93–1.07; the payoff appears only from rank ~64 and lives in the bulk;
- **sparsity:** the top 1% of entries buy nothing; two thirds of the payoff needs the top 20% of all pairs, chosen
  by the oracle; the top pairs are large-|D|, large-|C0| pairs, not kink-core pairs;
- **prediction:** a linear model of ΔC₀ on 27 O(n²) matrices of the chain (every term of its pair expansion, six
  candidate diagrams it lacks, its own C and C², gain-type features) explains **≤ 10% in sample and ≤ 6% out of
  sample** against a split-half noise ceiling of 0.87–0.98; proportional (ΔC₀ ∝ C0) and per-neuron-gain models are
  neutral or harmful;
- **where it comes from:** 65–76% of ΔC₀ at depth was *carried* through earlier, independent weights; the part born
  at the current step (the step run on the exact input covariance) is 22–30% and only 6–15% linear in local features.

<figure><img src="figures/fig_v4_projection.png" alt="Figure 5"><figcaption>Figure 5. Left: the identity of §6.4 on real matrices — observed off-diagonal energy of the next layer's variance error over its prediction (8/n)‖ΔC₀‖²_F (network 0, one oracle seed, every layer), and the off-diagonal share of that error's noise-subtracted energy (both networks, two seeds; layers 0–2 are below the oracle's noise floor). Right: the payoff (raw error ×, oracle substitution at layers 5, 8, 11, 14) of the best rank-r part of the exact off-diagonal correction; dotted lines are the full correction. The two dominant directions carry 40–65% of the energy and nothing of the payoff.</figcaption></figure>

**Consequence 4: the step's pair inputs are wrong, but only worth fixing together.** The covariance error that the
step gives birth to (22–30% of ΔC₀ at depth; with *every* input exact the step itself is accurate to ~0.1%) comes from
the step's own inputs, and we checked the two n×n ones against oracles [M]. The κ₃ (2,1) slice D, carried by the source
representation, is 7% wrong on the entries the chain computes (correlation 0.997, no level error) and makes 73–78% of
the born error; exact D at just two layers is worth ×0.885 / ×0.931 on two networks. But **any part of it alone is a
coin toss**: the computed block alone is ×0.900 on one network and ×1.027 on the other. The closure's κ₄ (2,2) slice
K22 is 20% wrong (correlation 0.62–0.70); pair features predict the true K22 out of sample better than the closure
(R² 0.45–0.61 against 0.35–0.45), and the exact K22 at two layers is ×0.970 on one network and ×1.102 on the other.
About 85% of D's error survives every tunable approximation of the representation; it is the k = 3 truncation of the
births. The chain's errors form a compensating set, and a correction of one member, exact or learned, can push the
rest out of balance. That is §3.5's consistency principle seen from the other side: what made removing terms on both
sides *gain* accuracy is what makes fixing one input alone *lose* it.

> **Proposition 2 (parity).** The chain's error in the next layer's κ₃ diagonal of neuron i is the cubic projection
> Σ_{abc} W_ai W_bi W_ci Δκ₃(l)_abc of the post-activation κ₃ error through the same independent weights. Conditioned on
> Δκ₃ it has mean zero for every neuron, and the errors are independent across neurons. [D]
>
> *Proof.* As in Proposition 1, Δκ₃(l) is independent of `W_{l+1}`; a cubic form in a zero-mean Gaussian vector has zero
> mean because every odd moment of w_i vanishes, whatever the coefficient tensor; independence across i follows from the
> independence of the columns. ∎

So where the variance error inherits a coherent level from tr ΔC, the κ₃ error has none, and its mean over neurons
should be zero up to its own standard error. *Measured.* Against the per-neuron 10⁷-sample moment oracles of two networks, z = mean/standard error of the κ₃-diagonal
error has mean z² = 0.93 and max |z| = 2.29 over 28 (layer, network) pairs, exactly zero-mean noise, while the
variance error sits 3–9 standard errors from zero at every layer ≥ 8 [M]. The mean is the first-order case, and the
output shows it: the coherent (per-network mean) part of the final-layer error is 0.46% of its MSE, and the residuals
of two consecutive layers correlate at 0.004 (each
random W scrambles the error vector), so no post-hoc correction of
the output has anything to learn either (N64; 504aldo's row 10 found the same) [M]. A per-neuron feature can see the κ₃ error only
through the forms the identity allows: a cubic form against the chain's own κ₃ (its t, 21–30% of the error's signal at
the last layer) and a linear one against the partial trace m_a = Σ_b Δκ₃_abb (the pre-activation mean, 9–17%); ≈40%
together at depth [M]; and N70 showed that removing the visible part of the moment errors buys nothing. Everything
else needs the joint object: n² numbers for the variance, n³ for κ₃. That is §6.2's wall again, seen from the accuracy
side. The chain is also very sensitive to that object: off-diagonal noise at the level of a 2·10⁷-sample estimate,
added at every layer, costs ×3.2–3.9 [M], one more reason no sampling hybrid of this chain can work at this
budget.

*Measured, random directions.* The sharper test of both propositions is to read the same post-activations along
directions that have nothing to do with the network. We replaced W_l by a matrix of i.i.d. N(0, 2/n) entries, read the
chain's variance and κ₃ along its columns, and estimated the same projections by Monte Carlo (10⁷ samples, two seeds,
the same samples for the real columns; networks 0 and 1, layers 4, 8, 11, 14). If the per-neuron errors are projections
through independent columns, the real columns' errors must be distributed as the random directions' are, and they are:
for κ₃ both means are zero (|z| ≤ 2.0, the two means within 1.7 standard errors of each other) and the noise-subtracted
spreads agree to 0.96–1.12 over the eight layer–network pairs; for the variance both carry the same coherent level (the
means within 1.6 standard errors) and the spreads agree to 0.98–1.03 at layers 8–14 of both networks (layer 4's variance
error is below the oracle's noise floor); the standardised 1st and 99th percentiles agree within 0.1–0.4 on heavy-tailed
distributions on both sides. The per-neuron errors are i.i.d. draws from the law of the chain's error along a random
direction, with that law's variance — not only without a coherent level [M] (`experiments/2026-10-07_randdir`).

**When can a per-neuron correction work? The truncation order decides, not the depth.** Proposition 1 says a
model of a neuron's error from that neuron's own state sees the coherent level and whatever is a function of the
state. For our chain that is nothing. For a Gaussian closure it is most of the error, and Proposition 3 says why: the
closure drops κ₃, and a neuron's true κ₃ is almost a function of its own mean (corr(κ₃ᵢ, μᵢ) = +0.87 … +0.93 against
Monte-Carlo at the last layer of every shape below, 79–88% of its variance explained by (μ, σ)), because both are
projections through the same weight column and v ∥ m. We measured the learnable share, leave one network out, for ARC's
SIMPLE k = 2 and k = 3 on the same networks [M]:

| shape | L/n | k = 2: learnable share | k = 2: coherent share | k = 3: learnable share | k = 3: coherent share |
|---|---|---|---|---|---|
| 1024 × 16 (ten public networks, 10⁹ truth) | 1/64 | **66%** | 26% | 0.0% | 0.3% |
| 512 × 16 (6 networks) | 1/32 | **66%** | 28% | 4.1% | 1.2% |
| 256 × 8 / 16 / 32 (10 each) | 1/32 … 1/8 | **64–66%** | 26–35% | 1.4–3.4% | 0.9–3.0% |

The share does not grow with depth; it is a property of what the estimator carries. A function of α = μ/σ alone
carries most of the Gaussian closure's learnable part (α: 56%, a cubic in α: 65% at 1024 × 16); the readout patterns of
a missing κ₃ or κ₄ level alone carry no more than the intercept. Across all shapes the odd orders keep no level
(coherent share of the pre-activation mean and κ₃ errors ≤ 0.7%), the variance's level grows with depth (to 19–30% at
256 × 32). This also settles a question we had misread: the learned corrector that worked in Phase 1 (mliston, forum
18170, raw MSE −17 to −19% on 20 networks) took as input, besides the chain's moments, the shape statistics of a
Monte-Carlo cloud, which carry joint information; it is consistent with Proposition 1 and needs no depth argument
(`experiments/2026-10-13_coherence-law`).

The learnable part has a closed form. A neuron's true pre-activation κ₃ is proportional to its mean, κ₃ᵢ ≈ c(l)·μᵢ,
with zero intercept and R² rising from 0.5 at layer 1 (the first layer whose pre-activations are not Gaussian) to
0.71–0.90 deeper, at every shape [M]. The
first-layer slope can be computed. Expanding the joint cumulants of the layer-0 activations in their correlations ρ
(Mehler's formula), the diagonal terms give 9.82/n, the ρ² terms add 4.09/n through E ρ² = 1/n, and the rest is
uncorrelated with μ, so for He weights c(1)·n = 12(κ₃(R) + B₂)/E[R] = 13.91, with R the ReLU of a unit Gaussian and
B₂ = φ(0)/2 − φ(0)³ [D]. We measure 13.94 ± 0.14 (standard error) on twenty 1024 × 16 networks, 13.98 at width 512 and 14.2 at 256 [M].
Deeper, the slope decays to 0.5–0.8 of c(1) and stays a property of the shape (a few per cent of spread across networks
at 1024 × 16). The same argument makes the (2,1) slice nearly rank one, D<sub>ij</sub> ≈ t<sub>j</sub>/3 (correlation 0.81 inside
our chain at 1024 × 16). We put this law into the chain in place of its κ₃: t = a + cμ with constants averaged over
other networks of the shape, D = 1·tᵀ/3, no κ₃ state, O(n²) per layer. The reference is κ₃ = 0, whose error is 1–12%
below that of ARC's k = 2 at every shape [M]:

| shape | L/n | κ₃ = 0 | skew-law closure | k = 3 chain | share of k = 3's gain kept | learnable in the closure's residual |
|---|---|---|---|---|---|---|
| 1024 × 16 | 1/64 | 4.1e-6 | 1.44e-6 | 3.9e-8 | 64% (10/10 networks) | 0.2% |
| 512 × 16 | 1/32 | 1.6e-5 | 6.0e-6 | 3.5e-7 | 63% (6/6) | −1.8% |
| 256 × 8 | 1/32 | 3.5e-5 | 1.2e-5 | 9.3e-7 | 65% (10/10) | 1.7% |
| 256 × 16 | 1/16 | 6.5e-5 | 3.6e-5 | 2.8e-6 | 35% (9/10) | 1.3% |
| 256 × 32 | 1/8 | 1.2e-4 | 1.7e-4 | 1.4e-5 | worse than κ₃ = 0 (3/10) | — |

At L/n ≤ 1/32 the law removes 63–65% of the Gaussian closure's error on all 26 networks, the share the per-neuron
model learned, and what it leaves has nothing learnable in it (−2 … +2%). The learnable two thirds of a Gaussian
closure's error is this law. The remainder is the incoherent part of Proposition 1, and it carries the score: our k = 3
chain is still ×37 more accurate than the closure at 1024 × 16. The law also has a range. At L/n = 1/16 it keeps 35%
of the gain with a wide spread across networks; at 1/8 it is worse than κ₃ = 0, even with each network's own
constants; and the diagonal alone (D = 0) is worse than κ₃ = 0 at four of the five shapes. An approximate κ₃ that the
rest of the chain does not match compounds layer by layer, the consistency rule of §3.5 again
(`experiments/2026-10-14_skew`).

**Why the skew follows the mean.** With the next column w i.i.d. N(0, q/n) and independent of the layer, Isserlis gives
the slope of the next κ₃ on the next mean over the neurons exactly in expectation, c(l+1) = (3q/n)·⟨v, m⟩/‖m‖², with
R² = cos²(v, m)·q<sub>v</sub>/(1 + q<sub>v</sub>) with q<sub>v</sub> = 9‖v‖²/(6‖G‖²) of Proposition 3, and D′ ≈ t′/3 (one pairing instead of three), where vₐ = Σ<sub>b</sub> κ₃(hₐ, h<sub>b</sub>, h<sub>b</sub>) =
Cov(hₐ, ‖h − m‖²) is the post-activation partial trace [D]. On Monte-Carlo truth the predicted slope matches the measured
one at every layer to a median 0.6–1.6% at 1024 × 16 and 2.4–4.6% at widths 256–512 [M]. The first-layer value splits
into 9.82 from the projected per-neuron skews and 4.09 = 6(1 − 1/π) from the input's radius, which a positively
homogeneous network passes on as an exact scale mode: inputs on the sphere give 9.68 and 9.24, and the Gaussian–sphere
difference equals 3σ̄²<sub>w</sub> (the μ²-weighted mean variance) at every layer to within 0.024 [D, M]. The rest of the alignment is made inside the network.
An amplitude mode along the mean takes over the post-activation covariance with depth (the mean's Rayleigh quotient grows
from 1.2 to 101 times the mean variance), v is the neurons' response to it (cosine 0.93–0.99), and a ReLU neuron's
response to a common rescaling is exactly its mean by Euler's theorem; together cos(v, m) = 0.92 → 0.99 [M]
(`experiments/2026-10-14_vm`). Projecting the chain's κ₃ on its partial trace alone (t = 3v/n, D<sub>ab</sub> = v<sub>b</sub>/n) keeps
72–73% of the k = 3 gain at L/n ≤ 1/32 but only 28% of its log gain at 1024 × 16: about 72% of what carrying κ₃ buys in
log terms lives in the all-distinct part of Proposition 3. Positive homogeneity is what the law needs: with leaky ReLU
the radius test and the alignment behave as with ReLU, while with SiLU Proposition 5 still holds (median deviation 4.5%) but
cos(v, m) falls to 0.75 and the law's R² to 0.47 at layer 15 (256 × 16) [M]. Two write-ups posted on 2026-10-15, after these measurements
(repository history 2026-10-14) and before our release, reach related results independently: il.miscusi (forum 18248)
uses the same scale mode (covariance + εmmᵀ, κ₃ + 2ε·Sym(C ⊗ m), κ₄ + 4ε·Pair₃(C ⊗ C)) and removes it inside an estimator;
kai_leal_gonzalez (forum 18246) states a conditional form of the decomposition behind the slope identity (the next κ₃
given the weight column's projection on the mean).

The same measurement can be run on any estimator from its outputs alone, because the pre-activation mean is linear:
its truth is Wᵀ times the true means of the previous layer. `scripts/error_anatomy.py` does that on the ten public
networks with their 10⁹-sample truth. On the starter kit's covariance-propagation example it finds a final error 26%
coherent and 66% learnable out of sample, with the order-one parity holding (z² = 0.98 over 150 layer–network pairs);
on our designated estimator, 0.6% coherent and nothing learnable (z² = 1.11) [M].

### 6.5 The instrument has a resolution, and it is coarser than most of our levers

One more boundary, of a different kind. Under the Phase-2 grader the ranked number is a mean over **50** held-out MLPs,
with a further 50 sealed. Two scalar levels of our own method, re-tuned on 10 of our baked MLPs, gave in sequence
[M]:

| stage | value |
|---|---|
| 10 baked MLPs, float64 research chain | **×1.0089**, better on 9/10 |
| 100 baked MLPs, the production estimator | ×1.0030, better on **66/100** |
| **the grader** (332872 vs 332403) | **×0.9987, worse** |

The local gain was real and still reversed sign on a different draw of networks. We take the observed discrepancy,
**≈0.4–0.5%**, as the instrument's resolution: **a fitted lever worth less than about 2–3% cannot be validated by a
submission.** The converse also held, and it is the reason the rest of our numbers can be trusted: levers that are
*structural* (a term added or removed by a derivation, a cut matched on both sides, an exact arithmetic saving that
moves the output only by float rounding) transfer, and the last ten ports, each validated on our 100 public networks
before shipping, landed on their local projections to the printed digit (k3o projected 5.12 → graded 5.1127e-9, …,
k3z 3.735 → 3.7357e-9) [M]. The one learned component (§3.7) was validated the honest way: trained on 10
networks, measured on 87 others it had never seen (×0.988), then graded (×0.988); its coefficients were never fitted to
leaderboard feedback.
