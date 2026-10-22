# Does the κ₃ tensor need to be represented at all? (t, D) as the state: measured

**Date:** 2026-09-28, cloud session, branch `cloud/state-closure`.
**Question (not asked in five attempts):** §54–§58 all tried to represent the κ₃ **tensor** more cheaply. Nobody
asked whether it has to be represented at all: `nonlin_slices` reads only **t and D**. If (t, D) were closed under
one layer, all the per-source machinery (legs 12.8% + pairings 10.6% of the budget) would collapse into an O(n²)
state, util would go under the floor, and at the ceiling accuracy that is ≈1.92e-9.

## Short answer

**No, and the failure is quantitative, not only formal. Moreover, the missing part cannot be closed cheaply either:
it is orthogonal to everything available in O(n²) (§5a). The cost axis is closed by a principle.**

| measurement | result |
|---|---|
| share of the next κ₃ diagonal NOT determined by (t, D) | **28.8%** per layer (layers ≥8, 3 MLPs, n=1024) |
| of these, D carries | ≈95% ‖t‖ |
| of these, t carries | ≈0.9% ‖t‖ |
| chain with a (t, D) state, end-to-end, 10 MLPs | **raw ×25.6** (6.82e-7 vs the base 2.66e-8) |

But the same measurement gives a **specification for a closer**, and that is constructive:

| share of the "all-distinct" part captured by the closer | raw ×base (10 MLPs) |
|---|---|
| 0 (pure (t,D) state) | ×25.6 |
| 0.90 | **×1.116** |
| 0.95 | **×0.992** |
| 1.0 (self-check) | ×1.000 |

**Threshold ≈93%: a κ₃ closer has to capture ≥93% of the missing part to stay within 3% raw.**

## 1. Why (t, D) are certainly not closed

All legs are carried linearly, so with **M = Wᵀ diag(d₁)** the whole κ₃ tensor is carried as
T → (M,M,M)·T (this holds for the tree, for the correction leg E, for path and for the P2 feed: all of them are
sums of symmetrised terms, trilinear in the legs). Splitting the contraction by index coincidence:

```
t⁽ˡ⁺¹⁾_i = Σ_j M_ij³ t_j                     ← contained in t
         + 3 Σ_{j≠k} M_ij² M_ik D_jk          ← contained in D
         + Σ_{j,k,m pairwise distinct} M_ij M_ik M_im T_jkm   ← NOT contained in (t, D)
```
The third term is taken **as a remainder**, without building the n³ tensor: the full contraction is the chain's own
readout after the transport. The same for the (2,1)-slice, with u = M_i, w = M_k:
```
D⁽ˡ⁺¹⁾_ik = Σ_j u_j² w_j t_j + (u²)ᵀ D w + 2 (u∘w)ᵀ (D u)  + (all-distinct)
          = [ (M∘M)∘t + (M∘M)D + 2 M∘(M Dᵀ) ] Mᵀ + (all-distinct)
```

## 2. Self-checks (without them no number below is usable)

| check | result |
|---|---|
| closed form for t == brute force over the index-coincidence masks (n=48) | **6.0e-17** |
| the three masks partition the contraction with no remainder | **5.3e-16** |
| closed form for D == brute force (n=40) | **4.3e-16** |
| **M reproduces the chain's own leg transport** | **≤5.1e-16** on all layers |
| mode `exact` (we substitute our own (t,D), but exact) reproduces the base | **×1.000** |
| mode `td1.0` (we capture 100% of the all-distinct part) reproduces the base | **×1.000** |

⚠️ **The first run gave 138% and was wrong.** The transport self-check showed 1.4 instead of ~1e-15: I was taking
the W of layer l instead of W_{l+1}. Between the readout at l and the readout at l+1 there are the gating d₁⁽ˡ⁾ and the
transport **W_{l+1}**. The 138% figure was thrown out; the script now has a hard `raise` if the transport check is
worse than 1e-10.

## 3. Per-layer measurement (`closure.py`, n=1024, production configuration without path births)

| layers | ‖all-distinct part‖ / ‖t⁽ˡ⁺¹⁾‖ |
|---|---|
| 1–3 (few sources) | 35–61% |
| ≥8 | **28.8%** (range 27.2–31.2%, 3 MLPs × 7 layers) |

The share carried by **D** is ≈95% of the norm; by **t**, only ≈0.9%. So almost all of the predictable part lives in
the (2,1)-slice, and the diagonal on its own carries almost nothing.

Note **how much SMALLER this is than the naive estimate.** If the terms of T were generic, the "all-distinct" part
(n³ summands against n² for D and n for t) would dominate by a factor of ≈n. The measured 28% means that the tensor's
structure is strongly concentrated on the two slices. That is why the question was worth asking, and that is why the
answer is a specification and not "off by orders of magnitude".

## 4. End-to-end: does (t, D) carry through to layer 15 (`tdchain.py`, 10 MLPs)

The sources are still carried by the real chain, but their contribution is **ignored**: the slices fed into the
nonlinear step are carried forward only by the two-slice rule, plus each layer's birth is computed exactly (the birth
is O(n²) from C0, D, t, κ₄, so a real (t,D) chain can do this too). So the raw below is what a chain with an O(n²)
state would actually score.

| mode | raw (10 MLPs) | ×base |
|---|---|---|
| base (this configuration) | 2.6659e-8 | 1.000 |
| **(t, D) state** | **6.8220e-7** | **×25.6** |
| + oracle closer at 90% | 2.9755e-8 | ×1.116 |
| + oracle closer at 95% | 2.6433e-8 | ×0.992 |

28.8% per layer compounds roughly multiplicatively over 14 transports (1.288¹⁴ ≈ 34, measured 25.6).

The base is 2.6659e-8 and not 2.3135e-8 because the diagnostic runs on the configuration **without path births**:
with `e3age=1` the path leg is dropped at age 2, which is not a transport, and the remainder would be contaminated.
The structural question does not depend on this: every comparison is made within one configuration.

## 5. What this changes

**The direct answer to both questions: (t, D, C, κ₄) is not sufficient as a state, and a block fitted to the two
slices does not carry through to layer 15: ×25.6.** This also closes the softer hope of §58: fitting a block to the
slices is cheaper than fitting it to the tensor, but such a block is already wrong after one layer.

The idea translates into a measurable target, closing the all-distinct part (28.8% of the readout) to ≥93%, and
**this target has been tested and is not reached**: see §5a, the best oracle closer on a cheap basis gives 19.5%, and
refitting the coefficients gives 0.16%, because the missing part is orthogonal to both the t term and the D term.
The form "do not represent the object, model its contribution" is exactly the form in which the κ₄ closure
(scale-mixture) works in this project; the difference is that κ₄ enters the readout at second order, while κ₃ **is**
the readout, so the tolerance is an order of magnitude tighter, and there turned out to be no margin.

## 5a. Closure tried, and it does not work: the all-distinct part is ORTHOGONAL to everything cheap (`fitclose.py`)

The 93% threshold was tested directly. The fit is an **oracle** fit: least squares onto the true carried diagonal,
per layer, with refitted coefficients and with the exact t and D terms in the basis. This is an **upper bound for any
closer on this basis**, not a realistic model.

| basis | capture of the all-distinct part (layers ≥8, 3 MLPs) |
|---|---|
| refitting only the coefficients of the (t,D) terms | **0.16%** (0.0–0.7%) |
| + "star" T_jkm ≈ Sym(u_j C0_km) → (Mu)ᵢ·diag(MC₀Mᵀ)ᵢ | **18.8%** (10.6–26.7%) |
| + all five basis vectors | **19.5%** |
| **needed** | **93%** |

Two conclusions, and the second matters more than the first:

1. **Refitting the coefficients gives nothing (0.16%), because the all-distinct part is orthogonal to both terms:**
   corr(remainder, D term) = ±0.03 across all layers. At the same time the coefficient of the D term fits to **1.000**
   (0.98–1.03), i.e. the D term is already exact, and what is missing is not "a bit more D" but a different direction.
   (The coefficient of the t term wanders from −2.7 to +2.7, but the t term is 0.9% of the norm, so it is simply
   undetermined; this is not a signal.)
2. **The physically meaningful ansatz gives 18.8% and hits a ceiling:** the fifth basis vector adds 0.7%.
   The capture grows with depth (10.6% at layer 8 → 26.7% at layer 14), i.e. with depth the tensor does become more
   "star-like", but even an extrapolation does not get close to 93%.

On the curve of §4, a capture of 19% corresponds to roughly ×15 in raw, so a cheap closure does not change the
verdict.

**This closes the cost axis by a principle, not by five accidents.** Before this there were five independent
measurements of the form "the construction costs as much as it saves" (§54–§59). Now a statement one level up is
added to them: **the κ₃ state has to carry Θ(n³) information forward, and the cost of the chain is exactly the price of
that information.** There is no cheap state, not because we searched badly for a representation, but because the nonlinear
step reads two slices that are **not closed**, and what is missing is **orthogonal** to everything that can be
computed in O(n²).

**What remains open (honestly).** This does NOT close a closure with a larger state: for example, if several
additional O(n²) objects are carried (a sketch of the tensor), the capture may grow. But that is again a
*representation* of the tensor, i.e. a return to §54–§59, where the cost of the construction is what kills the idea.
And this does NOT close the possibility that the all-distinct part is predictable from observables that our state
does not contain at all.

## 6. What is measured, what is derived

**Measured:** §3, §4 and §5a in full (torch float64, baked MLPs, n=1024). **Derived and checked by brute force:**
both closed forms of §1 (§2). **Not a model:** there is no cost model here at all; the util ≈5% for an O(n²) state is
taken from `cost_merge.py` of the previous session and is given only as motivation.
**Not touched:** `chain.py` did not change (the diagnostics work through a monkeypatch), `CLAUDE.md`,
`notes/*.md`, `src/` were not edited; nothing was submitted to AIcrowd.

## 7. Files

| file | what |
|---|---|
| `closure.py` | decomposition of the contraction, brute-force self-checks, per-layer measurement of the "all-distinct" share |
| `tdchain.py` | the (t,D) chain end-to-end + an oracle closer with a given capture share |
| `fitclose.py` | §5a: oracle fit of the all-distinct part on a cheap basis (upper bound for a closer) |
| `logs/` | `closure_m3.txt/.jsonl`, `td_n10.txt`, `tdchain*.jsonl` |
