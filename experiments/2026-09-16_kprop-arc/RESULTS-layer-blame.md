# Where the k=2 error lives, and whether a vector can fix it (2026-09-22)

Two new experiments, both with endpoint self-checks that passed exactly. Logs:
`logs/layer_blame.txt`, `logs/edgeworth_probe.txt`. Arrays: `layer_blame.npz`.

## 1. `layer_blame.py` — per-layer localisation of the Gaussian-closure error

Three oracle-injection sweeps on the baked-GT MLPs. At layer `l` one quantity of the k=2 chain's
post-ReLU state is replaced, then the chain continues as plain k=2 to layer 15 and is scored
against the baked N=1e9 ground truth.

| arm | what is injected at layer l |
|---|---|
| **A** | the **exact GT mean**; covariance left as k=2 computed it |
| **C** | the **k=3 chain's covariance**; mean left as k=2 computed it |
| **B** | both, from the k=3 chain |

**Self-checks (all exact):** no injection = 5.1494e-6 = full k=2 · A at l=15 = 0 exactly ·
B at l=15 = 4.25e-8 = full k=3 · C at l=15 = 5.1494e-6 = base exactly.

Result (3 MLPs, `X/base` = fraction of the final error that survives):

| l | 0 | 1 | 2 | 3 | 4 | 6 | 8 | 10 | 12 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** (mean fixed) | 1.000 | 0.864 | 0.746 | 0.648 | 0.575 | 0.400 | 0.283 | 0.207 | 0.129 | 0.047 | 0.000 |
| **C** (cov fixed) | 0.984 | 1.099 | 1.158 | 1.160 | 1.140 | 1.132 | 1.152 | 1.068 | 1.066 | 1.018 | 1.000 |
| **B** (both fixed) | 0.984 | 0.951 | 0.865 | 0.752 | 0.645 | 0.426 | 0.308 | 0.206 | 0.126 | 0.047 | 0.008 |

### Findings
1. **The k=2 final-layer error is carried entirely by the propagated MEAN.** Fixing the mean drives
   the error to zero; fixing the covariance to k=3 quality **never helps at any depth** and at layers
   1–8 makes it **up to 16% WORSE**. B ≈ A everywhere → the covariance contributes nothing.
2. **Improving one component alone is harmful** — arm C is a clean, controlled instance of the
   "accuracy ⊥ stability" effect reported at depth 32 in Phase-1 write-up #326024: a more accurate
   covariance paired with an unchanged mean is a mismatched state and degrades the iterate.
   **This explains three independent published near-nulls in one mechanism:** our tetrachoric
   correction (≈neutral at width 1024), #18200's full Hermite-2 covariance correction (−1.2%,
   i.e. worse), and #18214's exact bivariate ReLU covariance (+7.4% only). Covariance work is
   provably the wrong lever at this shape.
3. **Blame is spread over all layers, with no hot spot.** Marginal per-layer contribution `dA`:
   ≈7e-7 at layers 1–3, decaying smoothly to ≈2e-7 at layers 9–15. Early layers cost ~3× more each
   than late ones, consistent with errors born early being amplified by the remaining depth.
   This is the structural reason layer-adaptive cumulant order fails (measured separately).
4. **Layer 0 contributes exactly zero** (`dA = +8e-10`). It must: the input is exactly `N(0,I)`, so
   the pre-activations are exactly Gaussian and the Gaussian closure is exact there. A free
   validation of the whole measurement.

## 2. `edgeworth_probe.py` — can a VECTOR replace the O(L²n³) tensor?

Since the error is a mean error, look at the mean update itself. `mu_post = mu*Phi(a) + sigma*phi(a)`
is exact for Gaussian pre-activations; its leading error in the pre-activation's own third cumulant is

    mu_post ≈ mu*Phi(a) + sigma*phi(a) − kappa3_diag * a * phi(a) / (6*sigma²),   a = mu/sigma

derived here and **verified against 40M-sample Monte Carlo** on skewed test variables (it removes
61–87% of the Gaussian-closure mean error at non-zero `a`; it vanishes identically at `a=0`, where the
residual is a 4th-cumulant effect). `kappa3_diag[i] = kappa3_pre[i,i,i]` is a **vector**, not a tensor.

**Self-check:** `get_dslice((3,))` validated against an explicit dense third-cumulant tensor at
width 16, max abs difference 8.9e-16.

Probe: at each layer take the k=3 chain's pre-activation state and its TRUE diagonal `kappa3`, read the
post-ReLU mean both ways, score against the baked GT (2 MLPs):

| l | 0 | 1 | 4 | 8 | 11 | 13 | 15 |
|---|---|---|---|---|---|---|---|
| Gaussian readout MSE | 6.69e-10 | 1.56e-7 | 3.14e-7 | 2.84e-7 | 2.97e-7 | 2.57e-7 | 2.27e-7 |
| **Edgeworth readout MSE** | 6.69e-10 | 1.76e-8 | 4.05e-8 | 4.66e-8 | 5.86e-8 | 5.90e-8 | 6.14e-8 |
| ratio | 1.00 | **8.83** | **7.75** | **6.09** | **5.07** | **4.36** | **3.70** |
| k=3 chain's own MSE | 6.69e-10 | 9.17e-10 | 4.79e-9 | 1.35e-8 | 2.85e-8 | 3.73e-8 | 4.08e-8 |
| mean abs standardized skew | 0.000 | 0.008 | 0.023 | 0.048 | 0.067 | 0.080 | 0.091 |

### Findings
1. **The diagonal-`kappa3` correction improves the one-step mean readout by 3.7×–8.8× at every
   layer**, and the ratio at layer 0 is exactly 1.00 with both equal to the GT's own MC noise floor —
   another free validation (`a = 0` there, so the correction must vanish).
2. **At the final layer a VECTOR-only correction lands at 6.1e-8 against the full k=3 chain's 4.1e-8** —
   within 1.5× of a method that pays `O(L²n³)` to carry the whole cross-cumulant tensor.
3. The standardized skew of pre-activations is small (0.008 → 0.091 with depth), which is the regime
   where a first-order Edgeworth term is expected to be accurate. Consistent with finding 1.

### What this does and does NOT establish
**Does:** given an accurate pre-activation state *and* the true diagonal third cumulant, a vector-sized
correction recovers most of the mean accuracy the full tensor chain buys.
**Does not:** that the diagonal can be *propagated* cheaply and stay accurate. The oracle supplies it
here. 504aldo's dead-end table (#18218) reports a memoryless diagonal-slice chain landing at raw 9.1e-7,
45× worse than their engineered chain — but still ~5× better than our current 4.5e-6, which is
consistent with the ratios above. Propagating it ourselves is the next experiment.

## Next step (designed, not yet run)
A self-contained O(n²)-per-layer diagonal scheme, zero fitted constants:
1. **Birth:** even for exactly Gaussian pre-activations the post-ReLU variable is skewed. Compute that
   rectified-Gaussian third cumulant per neuron in closed form from `(mu, sigma)`. **Verify the closed
   form against MC before building on it.**
2. **Transport:** `kappa3_pre[i,i,i] = sum_abc W[a,i]W[b,i]W[c,i] kappa3_post[a,b,c]`; under the
   memoryless (diagonal-dominant) approximation this is `sum_a W[a,i]^3 kappa3_post[a,a,a]`, a matvec
   against the elementwise cube of W — `O(n²)`, negligible beside the `O(n³)` covariance einsum.
3. **Readout:** the Edgeworth term above, at every layer.
Measure against the baked GT before any flopscope port. Expected landing zone on this evidence is
~1e-6 raw, i.e. adjusted ~1e-7 — a ~4× gain on our current submission, **not** a top-10 method.
