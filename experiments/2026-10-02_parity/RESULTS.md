# The parity rule of the random-projection theorem (2026-10-02, evening, Mac)

**Question.** Write-up §6.4 / NEGATIVE §72: the next layer's per-neuron variance error is the projection
w_iᵀΔC w_i of the post-activation covariance error through the independent column w_i — a coherent level (2/n)·tr ΔC
plus i.i.d. noise. The same identity holds for κ₃ with a cubic form, Σ W_ai W_bi W_ci Δκ₃_abc, and there it predicts
something sharper [D]: **odd Gaussian moments vanish, so the κ₃-diagonal error has no coherent level at all** — its
mean over neurons is zero up to its own standard error. Untested until now (it was cloud #12's optional task C).

**Instrument** (`parity.py`, ~1.5 min per MLP): the float64 chain at B9 (`pb.matched(-2.5, satrow=2.5,
satrow_k4="sep")`) against the two 10⁷-sample per-neuron moment oracles of MLPs 0 and 1
(`../2026-09-24_k3-chain/oracles/mc_oracle_<m>.npz` and `_s777`), averaged. Per layer
z = mean_i(e_i) / √(mean_i(e_i²)/n): ≈ N(0,1) without a coherent level, large with one. The chain's var and t are
exactly projections of its own post-activation C and κ₃ (no look-ahead), so the theorem applies to their errors; its
κ₄ diagonal is a closure (3v·var²), not a projection, and is shown only for contrast.

## Result [M] (`res/parity_0.json`, `res/parity_1.json`)
| | MLP 0 | MLP 1 |
|---|---|---|
| **κ₃ (odd): z per layer 2–15** | 0.66, −0.48, −1.70, 1.09, 0.09, 0.08, 0.98, 2.29, 0.69, 2.08, 0.69, −0.29, 0.11, −0.26 | 0.33, −0.88, −0.38, −0.07, −0.37, −0.84, −0.67, 0.20, −0.45, 0.56, 1.25, 1.57, −1.49, −0.67 |
| κ₃: mean z², max \|z\| | 1.17, 2.29 | 0.69, 1.57 |
| **variance (even): z at layers ≥ 8** | 2.9 → 8.6 | 5.1 → 8.3 |
| κ₄ (closure, not a projection): z at layers ≥ 8 | −6.2 … −9.6 | −6.8 … −10.7 |

Pooled over the 28 (layer, network) pairs: **mean z² = 0.93, max |z| = 2.29 — the κ₃ error's mean is exactly as
large as zero-mean noise predicts**, while the variance error carries a coherent level 3–9 standard errors from zero
at depth, as the even-order projection predicts. The κ₄ closure's error has a large coherent level (consistent with
§6.4 Consequence 2: the κ₄ diagonal is the one channel a learned per-neuron correction could fix [U]).

**What per-neuron features see of the κ₃ error** (no intercept; R² = explained / noise-subtracted signal energy):
| layer | 8 | 10 | 12 | 15 |
|---|---|---|---|---|
| along the chain's own t (a cubic form against its own κ₃), MLP 0 / 1 | 0.061 / 0.082 | 0.138 / 0.111 | 0.228 / 0.226 | 0.213 / 0.301 |
| along the pre-activation mean μ (a linear form: the part along m_a = Σ_b Δκ₃_abb) | 0.037 / 0.042 | 0.079 / 0.071 | 0.129 / 0.126 | 0.087 / 0.166 |
| both | 0.073 / 0.111 | 0.176 / 0.125 | 0.313 / 0.291 | 0.397 / 0.407 |

So up to ≈40% of the κ₃ error at depth is visible to per-neuron features, exactly through the two forms the identity
allows (the part of Δκ₃ along the chain's own κ₃, and its partial trace along μ); NEGATIVE §70 already showed that
removing the feature-visible part of the moment errors buys nothing (×1.015 / ×0.957).

**Reading.** The parity rule is confirmed on both networks: even-order moment errors inherit a coherent level from
the joint error's partial traces, odd-order ones inherit none. Write-up §6.4's "prediction" paragraph moves from [D]
to [M] for the zero-mean part.
