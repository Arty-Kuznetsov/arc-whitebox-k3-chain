# The birth-column cut — NEGATIVE §48 revisited: the dead side is a lever, the saturated side is not

**Date:** 2026-09-29, Mac. **Scripts:** `scan.py` (symmetric cut), `sides.py` (one-sided cuts).
**Switch:** `chain.py colcut` (added today; default `None`; `ablate.py k4memo 3` still 3.7533e-08 exactly).
**Base:** the shippable k3m configuration in torch (`deadcut=-2.5`), 10 baked MLPs = **2.3462e-08**.
Self-checks: `colcut=None` and `colcut=1e9` (every column kept) both reproduce 2.3462e-08 exactly.

## The idea, and why it was worth re-opening

A source born at layer s has one column per **parent** neuron p (A is born as the identity, B = diag(d1)·C0,
and the column's weight is g_p = d2_p = φ(α_p)/σ_p). A parent far from the ReLU kink — dead (α_p ≪ 0) or
saturated (α_p ≫ 0) — has d2_p ≈ 0, yet its column is transported every layer for the rest of the source's life.
It is the **transpose of the dead-row cut** (`deadcut`), the one cost lever of the last week that worked.

NEGATIVE §48 had looked at it once, only at |α| < 3, found a cost-weighted saving of ≈7% and called it "not worth
it" on 24 Sep — when bigger levers still existed. That is a cost/benefit verdict made in a different regime, not
an accuracy verdict, so it was re-opened. The per-layer α distributions (`2026-09-27_residual/dump10.npz`) said
the modelled saving grows much faster than the removed birth weight:

| symmetric cut | columns dropped (lifetime-weighted) | modelled util | d2-weighted birth mass removed |
|---|---|---|---|
| \|α\| > 3.0 | 8.2% | 23.0% | 0.03% |
| \|α\| > 2.5 | 12.8% | 22.0% | 0.23% |
| \|α\| > 2.0 | 19.5% | 20.5% | 1.18% |
| \|α\| > 1.5 | 29.1% | 18.3% | 4.8% |

## 1. The symmetric cut is much worse than the weight proxy predicts (`scan.py`)

| colcut | raw | ×raw | better | modelled util | projected score |
|---|---|---|---|---|---|
| — (base) | 2.3462e-8 | 1.000 | — | 24.84% | 5.391e-9 |
| \|α\| > 3.0 | 2.6328e-8 | **1.122** | 1/10 | 23.00% | 5.60e-9 (×1.04) |
| \|α\| > 2.5 | 3.2747e-8 | 1.396 | 0/10 | 21.97% | 6.66e-9 |
| \|α\| > 2.0 | 5.0070e-8 | 2.134 | 0/10 | 20.47% | 9.48e-9 |
| \|α\| > 1.5 | 1.4912e-7 | 6.356 | 0/10 | 18.32% | 2.53e-8 |

Removing **0.03%** of the d2-weighted birth mass costs **12%** of raw — a ~400× amplification of the proxy.
So something in those columns is not proportional to d2.

## 2. The split by side answers why (`sides.py`)

| arm | ×raw | better | columns dropped | modelled util | **projected score** |
|---|---|---|---|---|---|
| base | 1.0000 | — | 0 | 24.84% | 5.391e-9 |
| **DEAD side, α < −3.0** | **0.9985** | 9/10 | 4.1% | 23.92% | 5.185e-9 (×0.962) |
| **DEAD side, α < −2.5** | **0.9928** | **10/10** | **6.4%** | **23.41%** | **5.045e-9 (×0.936)** |
| DEAD side, α < −2.0 | 1.0252 | 1/10 | 9.7% | 22.67% | 5.046e-9 (×0.936) |
| DEAD side, α < −1.5 | 1.4031 | 0/10 | 14.5% | 21.60% | 6.58e-9 |
| SATURATED side, α > 3.0 | **1.1234** | 1/10 | 4.1% | 23.92% | 5.83e-9 (×1.08) |
| SATURATED side, α > 2.5 | 1.3948 | 0/10 | 6.4% | 23.40% | 7.08e-9 |

**The whole cost of the symmetric cut is on the saturated side.** The saturated side alone at α > 3 reproduces
the symmetric cut's ×1.12 almost exactly; the dead side alone at the same threshold is ×0.9985.

## 3. Mechanisms (two, one per side)

**Saturated parents carry transported κ₃, not births.** A saturated neuron has ReLU ≈ identity, so the κ₃
arriving at it passes **through** unchanged. The part of a source that is *not* the tree birth — the (2,1)
correction leg E = A·diag(dK3) + 3B·diag(α) + ER, with α here the Wick coefficient w₁₂ − 2K₁d₁ — does not vanish
with d2 on that side. §48's premise, "for |μ/σ| large all hub weights (d2, α, ΔK3) are ≈ 0", **is true only on the
dead side.** The d2-weighted mass was the wrong proxy because it measures births, and a saturated column's value
is its transported content.

**Dead parents carry nothing, and cutting them makes the method *more* consistent.** With `deadcut=-2.5` a
neuron with α < −2.5 already has its output gated to exactly 0 and its **row** dropped from every contraction —
but until today its **column** still birthed third-cumulant content at weight d2_p. That is an inconsistency: the
neuron is treated as dead on one axis and alive on the other. Cutting the column at the **same threshold** removes
it, and raw improves on **10 of 10 MLPs** (×0.9928). The optimum sits exactly at the row cut's threshold (−2.5); one
step further (−2.0) goes the wrong way on 9/10. [U] The improvement is small enough that it could also partly be
the regularisation effect we saw with the randomized correction rank (`2026-09-28_accuracy-free/`, §1); the
consistency reading is the one that predicts the optimum's location, which is why we prefer it.

## 4. What this is worth, and what is still a model

**Projected ×0.93–0.94 on the score, above the grader's resolution** (NEGATIVE §60: ~2–3%), from a cut that
*improves* accuracy. It is the largest cost lever since `deadcut` itself.

**The utilisation is modelled, not measured** — lifetime-weighted fraction of dropped columns × the ≈90% of the
budget that scales with columns (legs 12.79% + D pairings 10.62%). The production estimator stores all sources
in one batched (2S, n, n) stack, so per-source column widths do not batch as-is. A port that pays needs the legs
laid out as one wide (n × Σ widths) matrix: then the transport Wᵗ·[legs] and the D pairing [L]·[legs]ᵗ are each a
**single** rectangular product whose cost scales with the total column count (the rectangular Strassen from the
dead-row cut already exists). That is the next engineering step, and only its flopscope measurement can confirm
the projection.


## 5. The mechanism made a prediction, and it held at every threshold (`pairs.py`)

If the dead-column gain comes from removing an **inconsistency** between the two axes (a neuron gated to 0 as a
row but still birthing κ₃ as a column), then a **matched** cut — row and column at the same threshold t — must beat
the row cut alone at **every** t, and the gain must **grow** as more neurons are gated. Measured, 10 MLPs:

| t | row only | matched | matched / row | better | matched / k3m | dead columns |
|---|---|---|---|---|---|---|
| −3.00 | 2.3135e-8 | 2.3101e-8 | **0.9985** | 9/10 | 0.9846 | 4.1% |
| −2.75 | 2.3174e-8 | 2.3105e-8 | **0.9970** | 8/10 | 0.9848 | 5.1% |
| −2.50 | 2.3462e-8 | 2.3293e-8 | **0.9928** | **10/10** | 0.9928 | 6.4% |
| −2.25 | 2.4670e-8 | 2.4151e-8 | **0.9790** | 9/10 | 1.0294 | 7.8% |
| −2.00 | 2.7855e-8 | 2.6393e-8 | **0.9475** | **10/10** | 1.1250 | 9.7% |

**Both predictions confirmed**: matched < row-only at all five thresholds, and the benefit grows monotonically
(0.15% → 0.3% → 0.7% → 2.1% → 5.3%). This is the kind of evidence we trust most in this project — a mechanism that
predicts an untested regime, then the regime measured.

**The interaction moves the optimum.** Alone, the row cut is best at −2.5 (that is why k3m ships −2.5). Matched
with the column cut, the score optimum shifts to **−2.25**:

| t | raw / k3m | row util (model) | columns dropped | util (model) | **projected** | × k3m |
|---|---|---|---|---|---|---|
| −3.00 | 0.9846 | 26.34% | 4.1% | 25.33% | 5.41e-9 | 1.004 |
| −2.75 | 0.9848 | 25.59% | 5.1% | 24.36% | 5.21e-9 | 0.966 |
| −2.50 | 0.9928 | 24.84% | 6.4% | 23.34% | 5.03e-9 | 0.933 |
| **−2.25** | 1.0294 | 23.92% | 7.8% | **22.15%** | **4.95e-9** | **0.918** |
| −2.00 | 1.1250 | 23.02% | 9.7% | 20.92% | 5.11e-9 | 0.948 |

Row utilisation from the calibrated cost-frontier model rescaled to the graded 24.835% at −2.5; column saving applied
to the 94% of the budget that scales with the column count. **Both utilisation columns are modelled**; the port is
what measures them.

## 6. Port scoping — where the saving lives and what layout it needs

Lifetime-weighted dead-column saving by birth layer (dead side, 10 MLPs):

| born | 0–2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dead frac at −2.25 | ≤1% | 3.1% | 6.0% | 9.0% | 11.3% | 13.7% | 15.4% | 17.1% | 18.4% | 20.2% | 22.1% | 23.4% | 24.4% |
| share of the saving | 1.6% | 4.0% | 7.0% | 9.6% | 10.8% | 11.7% | 11.5% | 10.9% | 9.8% | 8.6% | 7.1% | 5.0% | 2.6% |

The saving is spread flat over sources born at layers 4–13: the oldest sources live longest but are born at the kink
(no dead parents), the youngest have many dead parents but short lives. **A two-width-class scheme (full width for
old sources, one common narrow width for young ones) captures only 29–54%**, because the common width is set by
the class's least-dead member.

**So the port needs per-source widths**, i.e. the legs as **one wide `(n × Σ_s w_s)` matrix** instead of the batched
`(2S, n, n)` stack. That layout is natural for this algebra rather than a contortion:
- the transport `Wᵀ·X` never touches the column index — columns are a pure batch axis, so all sources are **one
  rectangular product** `(n × n_alive)(n_alive × Σw)` over the dead-row-compacted rows (the rectangular Strassen of
  k3j already exists; pad Σw to ×16);
- the D pairing `Σ_s L1_s B_sᵀ + L2_s A_sᵀ` is a contraction **over** the column index, so it is **one product**
  `[L](n × Σw) · [R]ᵀ(Σw × n)` and the sum over sources happens inside the contraction;
- the readout `t += 3 Σ (A∘g∘B²)` becomes one element-wise op and one row sum over `(n × Σw)`, with the per-column
  weights concatenated into length-Σw vectors.
The residual cost is governed by the op count (≈20 µs/op), which this layout does not raise: the batched ops become
2-D ops one for one. The only per-source work left is the thin correction `H_s·Ut_s` with `Ut_s` now `kk × w_s`
(≈2% of a product each). **The binding risk is correctness of a core-layout refactor, not residual time** — and the
torch `colcut=(t, inf)` arm gives a per-MLP reference to check it against.
