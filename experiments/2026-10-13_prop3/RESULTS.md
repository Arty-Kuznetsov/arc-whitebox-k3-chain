# Proposition 3: the share of the next κ₃ diagonal that (t, D) do not carry, derived (2026-10-13)

**Statement.** The next layer's κ₃ diagonal is t'ᵢ = Σₐbc wₐ w_b w_c Gₐbc with w = W_{l+1}[:, i] i.i.d. N(0, s²), s² = 2/n,
independent of G = T ∘ (d₁ ⊗ d₁ ⊗ d₁) (the chain's κ₃ with layer l's gates on its legs). By Isserlis,
E[t'ᵢ²] = s⁶ (6‖G‖²_F + 9‖v‖²) with vₐ = Σ_b Gₐbb, and the all-distinct part is uncorrelated with the rest,
E[ADᵢ²] = 6 s⁶ ‖G_AD‖²_F. So for every neuron, and for the norm over neurons,
**share = ‖AD‖/‖t'‖ ≈ √( 6‖G_AD‖² / (6‖G‖² + 9‖v‖²) ) = √( f_AD / (1 + q_v) )**, f_AD = ‖G_AD‖²/‖G‖², q_v = 9‖v‖²/(6‖G‖²).

**Test** (`prop3.py`; random He networks, depth 16, the chain configuration of `../2026-09-28_state-closure/closure.py`;
the dense tensor built from the chain's sources with `chain._k3_dense`; the share measured with the network's own
W_{l+1} exactly as `closure.py` does). Self-checks: the dense tensor's diagonal equals the chain's t (≤ 4.6e-16); its full
contraction equals the next t (≤ 5.1e-15).

| n | (layer, net) pairs, layers ≥ 8 | measured | predicted | ratio, mean (sd) | f_AD | q_v | cos(v, m) |
|---|---|---|---|---|---|---|---|
| 64 | 21 | 45.9% | 49.0% | 0.94 (0.15) | 72% | 2.1 | — |
| 128 | 21 | 44.1% | 44.2% | 1.00 (0.12) | 81% | 3.3 | +0.92 … +0.98 |
| 256 | 14 | 39.7% | 40.0% | 0.99 (0.06) | 88% | 4.6 | +0.98 … +0.99 |
| 1024 (state-closure, measured) | 21 | 28.8% | ≈ 32% (q_v, f_AD extrapolated) | | | | |

**Reading.** The formula is exact in expectation and the realized share concentrates around it like 1/√n. The 28.8%
at n = 1024 is therefore not a property of our chain's arithmetic: it is set by how the gated κ₃ tensor's energy splits
between all-distinct entries (most of the energy, 72% → 88% as n grows) and the partial trace v, whose weight q_v grows
with n faster still. The share falls slowly with width (≈ n^-0.15 over 64–256), so no practical width closes κ₃ on
(t, D). And v is almost parallel to the post-activation mean vector m (cos +0.92 … +0.99, closer to 1 as n grows): the
dominant part of the next κ₃ diagonal is a multiple of the next pre-activation mean, which is why a Gaussian closure's
error is learnable per neuron (`../2026-10-13_coherence-law`).
