## 3. The method

The estimator is a cumulant-propagation chain closed at the third cumulant. **Its skeleton is the SIMPLE k = 3
algorithm of Wu et al. (arXiv:2605.05179) and their MIT reference implementation `mlp_cumulant_propagation`**, which we
re-derived from the paper and re-implemented from scratch for flopscope; §3.1 states exactly what we took. Everything
after §3.1 is ours unless a credit says otherwise. Every component is a constructor switch, and switching it off
reproduces its predecessor: bitwise in the production estimator, and to float rounding (≤ 1e-6) against the float64
research chain. Each subsection ends with the graded submission that first shipped it.

| step | component (section) | 100 public MLPs, local | graded (50 public of 100) | util |
|---|---|---|---|---|
| k3b | the reference chain in our source representation + workspace Strassen (3.1) | 1.98e-8 | 1.915e-8 (331962) | 55.9% |
| k3e | scale-mixture κ₄ closure + the C0·D diagram (3.2) | 9.76e-9 | 9.562e-9 (332066) | 32.3% |
| k3f | the rest of the first-order scale-mixture set (3.3) | 8.59e-9 | 8.286e-9 (332128) | 32.4% |
| k3h | Strassen level 4 through sum-of-products pairings (A.1) | 7.67e-9 | 7.401e-9 (332142) | 28.9% |
| k3i | the augmented term set where it is cheap (3.4) | 6.92e-9 | 6.728e-9 (332170) | 29.8% |
| k3j → k3m | dead rows out of every contraction (3.5) | 5.52e-9 | 5.3906e-9 (332403) | 24.8% |
| k3o | the matched dead-side row + column cut (3.5) | 5.25e-9 | 5.1127e-9 (333028) | 23.8% |
| k3p | old sources in a basis that rides the transport (3.6) | 4.68e-9 | 4.5644e-9 (333029) | 20.7% |
| k3q | saturated rows out of D, κ₄ level restored — E2 (3.5) | 4.24e-9 | 4.1279e-9 (333276) | 19.7% |
| k3r | marg2, the pair expansion's one k = 2 defect (3.5) | 4.17e-9 | 4.0579e-9 (333307) | 19.7% |
| k3t | Strassen level 5 on three contractions (A.1) | 4.06e-9 | 3.9554e-9 (333357) | 19.2% |
| k3v | the learned κ₄-diagonal corrector (3.7) | (87 fresh MLPs ×0.988) | 3.9102e-9 (333368) | 19.3% |
| k3w, k3x | level 5 on two more contractions (A.1) | 3.90e-9 | 3.8069e-9 (333398) | 18.7% |
| k3y | symmetric C_pre, explicit triangular R⁻¹ (A.1) | 3.85e-9 | 3.7573e-9 (333402) | 18.4% |
| **k3z** | **Winograd's 15-addition form; level 5 inside the C_pre halves (A.1)** | **3.83e-9** | **3.7357e-9 (333534)** | **18.3%** |

**Algorithm 1 — one layer of the designated estimator, with the subsection of each line.** State carried from layer
l−1: the post-activation mean μ and covariance C; the young κ₃ sources s = (A_s, B_s, H_s, g_s); the old ones as
coefficients Z_s in the band basis P; the post-activation κ₄ diagonal and (2,2) level.

```
1  μ ← Wᵀμ;  C_pre ← Wᵀ(CW);  var = diag C_pre;  C0 = C_pre − diag(var);
   α = μ/√var                                                          §3.1
2  dead neurons (α < −2.5) leave every contraction; the young legs and Pᵀ
   move one layer: Xᵀ ← (gated Xᵀ)·W                              §3.5, §3.6
3  the old sources' legs, materialised: (A, B) = P·(Z_A, Z_B)               §3.6
4  read κ₃: tᵢ = 3 Σ_s Σ_p g_p A_ip B_ip² + thin, feed and path terms;
   D = Σ_s (L1·Bᵀ + L2·Aᵀ) + …; rows of D with α > 2.5 removed and the κ₄
   level they carried restored by one matrix–vector product       §3.1, §3.5
5  close κ₄ in O(n²): level v from sum and trace identities;
   κ₄ᵢᵢᵢᵢ = 3v·varᵢ² (+ the newest source's star birth, + the learned
   correction at layers 8–14);  K22 = v(varᵢvarⱼ + 2C0ᵢⱼ²);
   K31 = 0.65·3v·varᵢC0ᵢⱼ                                          §3.2–§3.4, §3.7
6  the ReLU step: power moments with Edgeworth terms in (t, κ₄); pair
   power cumulants with the C0·D, K31, C0³, C0∘K22 diagrams and the marg2
   coefficients; Leonov–Shiryaev → post-activation μ (layer l's output)
   and C                                                           §3.1–§3.5
7  the source that reaches age 5 enters the band: Z = R⁻¹QᵀX, P = QR   §3.6
8  birth: hubs = neurons with α ≥ −2.5; B = diag(d₁)·C0 on those columns;
   g = d₂; correction leg = column rescaling of B + rank-24 thin leg;
   the path leg, read once at the next layer                       §3.1, §3.4
```

### 3.1 What we took from the reference, and the representation we put it in

**Taken from Wu et al. (credit, not contribution).** The linear step is exact: μ ← Wᵀμ, C ← WᵀCW, and every factor
of κ₃ is multiplied by Wᵀ. At each ReLU, with w(k,p) = E[∂ᵏ ReLU(Z)ᵖ] for Z ~ N(μᵢ, σᵢ²): single-index power
moments with Edgeworth terms, P_p = w(0,p) + w(3,p)·t/6 + w(4,p)·κ₄ᵢᵢᵢᵢ/24 with t = κ₃ᵢᵢᵢ; pair power cumulants
Q(p,q) = a₁b₁C0 + ½a₁b₂Dᵀ + ½a₂b₁D + ½a₂b₂C0² + ¼a₂b₂K22 (a_k = w(k,p)ᵢ, b_k = w(k,q)ⱼ, C0 the covariance without its
diagonal, D = κ₃ᵢᵢⱼ, K22 = κ₄ᵢᵢⱼⱼ), converted to cumulants by Leonov–Shiryaev; a third-cumulant birth at every
layer (the tree diagram Sym(U ⊗ 3I ⊗ U·diag(d₂)), U = diag(d₁)C0, plus a correction block that makes the repeated-
index slices exact); and a one-step Edgeworth readout of the post-ReLU mean. The reference keeps κ₄ as one radial
scalar. It reaches raw MSE 4.25e-8 on our first three networks at 148% of the budget; our re-implementation reproduces
every slice of it to 1e-15 [M].

**Our representation: sources, and only the slices the step reads.** Each birth at layer m creates a *source*: a hub leg A (the transport since birth; the identity at birth), a tree leg
B = A·U, a correction leg, and a fixed hub weight g = d₂ at birth. The κ₃ tensor is never formed. The next step reads only its diagonal t and its (2,1) slice D, so
only those are built:

    t_i = 3 Σ_s Σ_p g_p A_ip B_ip² + (correction-leg terms),      D = Σ_s [L1·Bᵀ + L2·Aᵀ + (thin terms)],
    L1 = 2 A∘B∘g,  L2 = B²∘g + …

After each step every leg row is gated by d₁. The correction leg splits **exactly** into a column rescaling of the tree
leg, 3B·diag(α) with α = w(1,2) − 2P₁d₁, which is free because it survives transport unchanged, and a remainder that is
numerically low-rank, carried as a thin leg of rank 24 found by a seeded randomized range finder (×1.004 against the
exact split; R dropped entirely costs ×10.5) [M]. Every per-source operation is one batched flopscope call per layer
into preallocated buffers; that is what keeps the non-flopscope ("residual") time under the grader's 0.4 s cap.

### 3.2 A closure that makes the fourth cumulant free — and the diagram it needs

The fourth cumulant enters the readout and the pair cumulants. Transporting its (2,2) structure ("memo22": the direct
pairings of Wsqᵀ(K22 + diag K4)Wsq, Wsq = W∘W) beats the reference's radial scalar (3.75e-8 against 4.25e-8, 3/3
networks) but costs two dense products per layer and gets the per-neuron **pattern** wrong (correlation with a
Monte-Carlo oracle 0.23–0.53) [M]. Two findings replace it.

**(a) The missing per-step diagram.** Fed exact (Monte-Carlo) inputs, the SIMPLE step loses 2.3 / 3.5 / 4.5% of the
post-ReLU (2,2) level at layers 6 / 10 / 14. The missing term is the cross term of the joint Edgeworth exponent between
the edge C0 and the κ₃ (2,1) slice:

    Q(p,q) += ½ (a₂b₃ Dᵀ + a₃b₂ D) ∘ C0,

and with it the step returns 0.999 / 0.999 / 1.004 of the oracle (correlation 0.974 → 0.999) [M]. It costs O(n²).

**(b) The closure.** For a Gaussian scale mixture, κ₄ᵢ = 3v·σᵢ⁴; the oracle's κ₄ᵢ correlates 0.86 with σᵢ⁴ at depth.
So we keep **one scalar v per layer** and set

    κ₄ᵢᵢᵢᵢ = 3v·varᵢ²,      K22ᵢⱼ = v·(varᵢ varⱼ + 2C0ᵢⱼ²)

(the scale-mixture forms), with v obtained in O(n²) from the exact identities sum(M) = uᵀXu and tr(M) = Σ X_ab G_ab
(u = Wsq·1, G = Wsq Wsqᵀ with an exact diagonal and a mean-field off-diagonal), which reproduce the transported levels
to four digits. The two dense κ₄ products per layer are gone [M].

**They only pay together.** C0·D on the memo22
chain is *worse* (4.64e-8 against 3.75e-8, three networks); the closure alone is ×1.07; together, with the closure's
K22 pattern, ×1.136 on ten networks (3.52e-8 → 3.10e-8), better on 9/10, at −2% of the budget [M]. With the wrong
pattern, raising the κ₄ level by the correct diagram makes the error larger; with the right pattern it removes it.
→ **k3e, graded 9.562e-9.**

### 3.3 The rest of the first-order scale-mixture set

With the closure in place, the scale mixture predicts three more first-order terms, and again **they only pay
together** (each alone is neutral or harmful) [M]: the marginal Edgeworth correction of the pair Wick coefficients,
a_k → w(k,p) + w(k+3,p)·t/6 + w(k+4,p)·κ₄/24; the closure's κ₄ (3,1) slice, κ₄ᵢᵢᵢⱼ = 3v·varᵢ·C0ᵢⱼ, entering Q as
⅙(a₃b₁K31 + a₁b₃K31ᵀ) (together with the marginal coefficients it cuts the one-step covariance error 0.3% → 0.1%); and
the **κ₄ → κ₃ feed**: all-distinct κ₃ births from the closure's (2,1,1) slice, which in the source representation is
Sym(h ⊗ D1C0D1) with h = (v/2)·d₂∘var and rides the existing products (`t += 3 m∘q`, `D += 2 diag(m) B diag(d₁) Aᵀ + q mᵀ`
with O(n²) extras). The contracted feed matches Monte-Carlo at correlation 0.97–0.98. **The κ₄ → κ₃ channel was named
as the main lever of 504aldo's public chain (AIcrowd forum 18218); our form is derived from the closure.** Ten networks
3.15e-8 → 2.80e-8 (10/10); 100 networks ×0.880, 100/100 better [M]. → **k3f, graded 8.286e-9.**

### 3.4 The augmented term set, where it is cheap

Wu et al. also give an "augmented" algorithm; their reference in that mode is ×1.3 more accurate than k3f on our first
two networks at ≈2.4× SIMPLE's cost. Against the oracle its κ₃ diagonal is exact to the oracle's noise, while its κ₄
diagonal is *not* better than our σ⁴ closure (correlation 0.87 vs 0.86), so the lever is the κ₃ step [M]. We took the
part of it that is cheap in our representation: every (1,1,1) "path" birth whose other edge is C0 shares the hub and
tree legs, so all of them together are **one extra leg** Sym(T·Y ⊗ 3A ⊗ B) with
Y = d₁Dᵀd₃ + d₂Dd₂ + ½d₂K22d₃ + d₂C0²d₃ (checked against the reference code to 5e-16), and only its **first readout** matters (age ≤ 1 keeps ~90% of the all-ages gain), so it costs one product per layer; plus the star-diagonal κ₄ birth
of the newest source (O(n²)) and the two augmented pair diagrams we lacked, ⅙a₃b₃C0³ and ¼a₃b₃C0∘K22 (O(n²)). Ten
networks ×1.145 (10/10); 100 networks 98/100 better [M]. → **k3i, graded 6.728e-9.** One level of the closure was then
set to a consistent truncation rather than fitted: the closure's K31 sits at 0.74–0.78 of the oracle, the augmented reference, a consistent truncation, at 0.54, and the chain's optimum is a flat 0.6–0.7 (×0.65 shipped; 98/100) [M].

### 3.5 Consistency: remove a term on both sides, or not at all

This principle moved the score most in the last week of the work. A neuron that the chain treats as absent on one
axis must be treated as absent on every axis it appears on; a term whose input is unreliable should go with that
input. Its Phase-1 precursor is hyojun_kwon's finding that a learned one-step operator 40% more accurate than the
Gaussian closure diverges when iterated, because the closure's value is that it is a stable iterate (AIcrowd forum
topic 18154); what follows is the same phenomenon seen from the side of removing terms. Three instances, each **an accuracy gain obtained by removing computation** [M]:

- **Dead neurons.** A neuron with α = μ/σ < −2.5 has d₁ ≈ 0, so every leg row for it is ≈ 0 after gating: we zero those
  rows exactly and run the next transport over alive rows only (rectangular Strassen, padded to the recursion's
  quantum): util 26.3% → 24.8% for +1.4% raw (k3m, graded 5.3906e-9). The transpose of that cut, the dead-side
  **birth columns** of every new source, is also removable exactly, and the two cuts must match: the matched row +
  column cut beats the row-only cut at every one of five thresholds, by a margin that grows with the gated fraction
  (matched/row = 0.9985, 0.9970, 0.9928, 0.9790, 0.9475 at −3.0 … −2.0). Production needs per-source column widths, so
  the legs are stored **transposed**, as one wide stack of hub rows, which turns the transport into one row-stack
  product and the D pairing into one contraction. 100 networks ×0.950 adjusted, **100/100** (k3o, graded 5.1127e-9).
- **Saturated rows of D (E2).** Rows of D for neurons saturated now (α > 2.5) reach the step only through the tail
  coefficients coef(2,1) = d₂ + d₅t/6 + d₆κ₄/24, whose Edgeworth correction is −28 … −79% of d₂ there; the marginal series is at the edge of validity. Removing those rows naively costs ×1.34 at 2.5, and **all** of that damage is the
  κ₄ level the next layer reads through Ssum = uᵀK22u; restoring that level by a matrix-vector product
  (dS = Σᵢ uᵢ cf22ᵢ (D v)ᵢ) turns the cut into an accuracy **gain**: raw ×0.943 (k3q, graded 4.1279e-9).
- **marg2, the pair expansion's one defect.** Keeping the marginal correction of the k = 2 Wick coefficient only in
  the dead tail (α ≤ −2) and using the plain Gaussian w(2,p) elsewhere (in Q11, Q21 *and* Q22 together; in one power
  cumulant alone it is ×1.09–1.11 worse) is worth ×0.984 on 32 fresh networks (cloud session #9A) and ×0.981 on ten
  locally, at zero FLOPs (k3r, graded 4.0579e-9). A later generative screen of seven next-order pair diagrams found
  exactly two that help, C0·K31 and D², and both are **substitutes** for marg2 (×1.008 together with it): the pair
  expansion has one defect, in its k = 2 leg, and marg2 is its cleanest form.

### 3.6 Old sources in a basis that rides the transport

Sources older than a0 = 5 layers no longer carry their own legs. They live as P·Z in one shared basis P (n × 448):
Pᵀ rides the leg transport as 448 extra rows (gated and contracted over alive rows like any leg), and since
Wᵀdiag(d₁)(PZ) = (Wᵀdiag(d₁)P)Z the coefficients Z **never change**. A source enters once, projected on the current P
through a QR (Z = R⁻¹QᵀX); P is never re-orthonormalised and Z never rotated. The members are materialised each layer
by one product for the readout, and their part of D is paired in the compressed space, (L_mᵀ Zᵀ)Pᵀ. 100 networks
×0.893 adjusted, **100/100** better (k3p, graded 4.5644e-9) [M]. Both band knobs sit at sharp optima (rank 320 / 384 /
448 / 512: ×1.228 / 1.032 / 1 / 1.033; entry age 4 / 5 / 6: ×1.125 / 1 / 1.016) [M]. Why this compression works where
the re-fitted subspace of §6.1 did not is the subject of that section: the basis is never rebuilt.

### 3.7 A learned correction of the κ₄ diagonal

**The idea is mliston's** (AIcrowd forum 18170, Phase 1): a model trained offline on self-generated networks to predict
the closure's error from its own propagating state, frozen into the estimator as constants. No code of theirs was seen
or used. Our version and what we learned testing it:

- From the neuron's **own** moments the damaging error is invisible (§6.4); the κ₄ diagonal is the exception because
  its error has an environment-dependent level, which **O(n²) row reductions of matrices the chain already holds**, C0, D and K22, can see. The model is a ridge regression (λ = 1e-3) of the relative κ₄ error on 39 features
  (21 functions of the neuron's own moments, 17 such row reductions, the layer index), applied at layers 8–14 with
  shrinkage 0.5 (not tuned); 118 numbers, trained on the production trajectory of 10 public networks against our own
  Monte-Carlo oracles (two at 10⁷ samples, eight at 2·10⁶).
- **Two networks are not enough** (fit on one, apply to the other: ×1.00–1.08); ten are, and thirty are not better
  (held-out R² 0.353 vs 0.354) [M].
- **The validation ladder**, each step less optimistic than the one above it [M]: in sample ×0.92 → ten-fold leave-one-
  network-out ×0.974 → online ×0.968 → through the production estimator ×0.980 → **87 networks never seen ×0.988
  (69/87)** → **the grader ×0.988** (raw 2.0559e-8 → 2.0312e-8).
- The same construction for the variance or κ₃ **hurts**, in sample and out (§6.4). → **k3v, graded 3.9102e-9.**

### 3.8 Exact arithmetic

Strassen–Winograd as flopscope operations (deepened only where a contraction pays for its wall time), the symmetric
C_pre computed as three blocks of four, an explicit triangular inverse where flopscope bills `solve` as an LU, and
Winograd's 15-addition form change the FLOP bill and move the output only by float rounding. They are engineering, and
they are in Appendix A.1 with their measurements; here only the record: k3t, k3w, k3x, k3y and k3z together are ×0.931 of
the ×0.693 from 5.39e-9 to 3.74e-9, and each landed on the grader as projected [M].
