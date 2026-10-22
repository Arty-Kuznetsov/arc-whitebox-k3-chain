# The learned moment corrector — task 1 (the signal), run on the Mac 2026-10-01

**What this is.** The decisive first measurement of `notes/CLOUD-BRIEF.md` #10, run here because both 1e7 MC oracles
of MLP 0 were already on this machine (now committed in `experiments/2026-09-24_k3-chain/oracles/`). No baking, no
training: just *is the chain's per-neuron moment error predictable from the chain's own cheap state?*

Base: **B9** = k3m + matched cut −2.5 + E2 (satrow 2.5, sep) in torch float64 (`chain.py`, `record=True`), MLP 0 of the
baked public set. Oracles: `mc_oracle_0.npz` (default seed) and `mc_oracle_0_s777.npz`, both N = 1e7, independent.
Script: `moment_signal.py`. Alignment checked: corr(chain `mu_pre`, oracle `mean`) = 1.0000 at layers 5, 10, 15.
Layer 0 is skipped — `mu_pre` is 0 for every neuron there and there are no sources, so every feature is degenerate.

## The error of each channel: how big, and how much of it is real

`rel` = rms(chain − oracle) / rms(chain). `corr(eA, eB)` = correlation of the error measured against the two
independent oracles — **for two noisy measurements of the same truth this is σ²_signal/(σ²_signal+σ²_noise), i.e. the
ceiling on any R²**. The part of the apparent "error" that does not reproduce across seeds is oracle noise and nothing
can predict it.

| channel | rel error (L8 / L15) | corr(eA,eB) L4 / L8 / L15 | mean over L1–15 |
|---|---|---|---|
| variance | 0.1% / 0.2% | 0.21 / 0.69 / **0.94** | 0.56 |
| κ₃ (the t diagonal) | 4.3% / 6.7% | 0.47 / 0.91 / **0.99** | 0.73 |
| κ₄ (K4d) | 17.6% / 29.7% | 0.78 / 0.95 / **0.99** | 0.86 |

So from about layer 8 the chain's moment error is **almost entirely deterministic structure, not oracle noise** — at
the last layers 94–99% of it reproduces across independent oracles. That alone is new: it says there is something
there to predict.

## How much of it a linear model on the chain's own state already explains

21 per-neuron features the chain has at that point (α and its powers, φ(α), Φ(α), αφ, (α²−1)φ, σ, var, skew, excess
kurtosis, μ, t, K4d, Σ_j W²_{l+1,ij}, the column norm of W_l, the α-rank, and three cross terms). Ordinary least
squares with an intercept, **in sample** — with p/n = 21/1024 the optimistic bias is ≈2%, and the two oracles give
nearly identical R² (e.g. κ₃ at L15: 0.512 / 0.526), which is itself evidence the fit is on real structure.

| layer | var R² | κ₃ R² | κ₄ R² |
|---|---|---|---|
| 4 | 0.025 | 0.021 | 0.156 |
| 8 | 0.071 | 0.141 | 0.288 |
| 12 | 0.094 | 0.408 | 0.512 |
| 15 | 0.095 | **0.512** | **0.561** |
| mean L1–15 | 0.062 | 0.223 | 0.297 |

Against the ceilings above, at the last layers this captures ≈**52% of the explainable κ₃ error**, ≈**57% of κ₄**,
and only ≈**10% of the variance error**. Restricted to the kink core (|α| < 1, where NEGATIVE §68 says the error is
born) the picture is the same but weaker: var 0.07–0.15, κ₃ 0.16–0.20, κ₄ 0.17–0.33 at the late layers.

## Reading, and what it does and does not establish

1. **The track is alive.** A *linear* model on features the chain already computes explains half of the explainable
   κ₃ and κ₄ error at the layers that matter. A tree or a small network, with pair-level features and more of the
   state, should do better. This is the opposite of 504aldo's dead end #10 (final-layer features, corr < 0.09) and of
   our own §57 residual analysis (≤ 0.03) — and the difference is exactly the one #9A predicted: the signal is
   **mid-chain**, not at the output.
2. **The leverage is inverted, and that is the risk.** The channel that is easiest to predict (κ₄, rel error 20–30%,
   R² 0.56) has the *smallest* oracle ceiling (×0.78), and the channel with the biggest ceiling (variance, ×0.64) has
   a rel error of only 0.1–0.2% and is the hardest to predict (R² ≈ 0.10). So R² is **not** the number that decides
   this — the payoff through the chain is, and that is task 2.
3. **Not yet established:** anything out of sample. Everything above is one MLP, fitted and scored on itself. The
   decisive test is: fit on MLP 0, apply to MLP 1 *through the chain*, measure the final-layer raw (and the reverse).
   MLP 1's two oracles are baking on the Mac now (`experiments/2026-10-01_k3r/runs/queue3.sh`).

## Files
`moment_signal.py` (the measurement), `dbg.py` (the scale/correlation dump that caught the first version's singular
design matrix — six layer-level features are constant *within* a layer and made the ridge return β ≈ 0).


---

# Task 2 (2026-10-01, afternoon) — the payoff, and what it costs to collect it

## 0. The harness gate (mandatory, `exact_check.py`)

`mcorr` returning the EXACT error (chain value − oracle value, computed from the LIVE state) is identical to
substituting the oracle value, i.e. cloud #9A's arm. Measured here: MLP 0 **×0.206** (seed A) / ×0.230 (seed 777),
MLP 1 **×0.302** / ×0.320 — #9A's single-seed values were ×0.235 / ×0.215. **The harness reproduces the known
ceiling**, so everything below means what it says.

Two things fell out of the gate that matter more than the gate itself:

| exact correction, s = | MLP 0 | MLP 1 |
|---|---|---|
| var+κ₃+κ₄, 1.00 | **0.206** | **0.302** |
| var+κ₃+κ₄, 0.50 | 0.318 | 0.388 |
| var+κ₃+κ₄, 0.25 | 0.594 | 0.629 |
| var only, 1.00 | 0.693 | **1.090** |
| κ₃ only, 1.00 | 0.942 | 0.915 |
| κ₄ only, 1.00 | 0.753 | 0.809 |
| joint, layers 8–15 only | **0.189** | 0.256 |
| joint, layers 12–15 only | 0.446 | 0.452 |

1. **A partial move toward the truth helps monotonically** (0.25 → ×0.59). Partialness is not the problem.
2. **The channels only work together**: jointly ×0.21–0.30, separately 0.69–1.09, and **exact variance alone is
   HARMFUL on MLP 1 (×1.09)** — the consistency mechanism of NEGATIVE §68 again, now on the whole-neuron variance.
3. Correcting from layer 8 on is *better* than correcting everywhere (×0.189 vs ×0.206 on MLP 0).

## 1. The ceiling: what a corrector of our measured accuracy is worth (`ceiling.py`)

An OLS predictor removes the projection of the error onto the feature space and leaves a residual of magnitude
√(1−R²) that is **orthogonal** to the features. A uniform exact correction at strength s leaves (1−s)·error in the
**same** direction. Matching magnitudes gives s_eq = 1 − √(1−R²). With the measured late-layer R² (var 0.10,
κ₃ 0.51, κ₄ 0.56 → s_eq 0.051 / 0.300 / 0.337):

| arm | MLP 0 | MLP 1 |
|---|---|---|
| **s_eq per channel** | **×0.707** | **×0.696** |
| s_eq, layers 8–15 only | ×0.743 | ×0.719 |
| s_eq without the variance channel | ×0.760 | ×0.740 |
| variance alone at s_eq | ×0.950 | ×0.962 |
| **all three at s = 0.30 (var as predictable as κ₃)** | **×0.529** | **×0.571** |

**In score terms** (graded k3r: raw 2.0560e-8 × util 19.74% = 4.058e-9):
raw ×0.70 → **≈2.84e-9 (rank ≈6–7)**; raw ×0.55 → **≈2.23e-9 (rank ≈4–5)**. So the prize is real and it is large,
**and the lever is named: the variance channel.** Raising var's R² from 0.10 to κ₃'s 0.51 is the difference between
×0.70 and ×0.53.

⚠️ s_eq matches the residual's **magnitude**, not its direction. §57 says direction decides (coherent corrections are
paid in full, incoherent ones damped by √n), so this is an idealisation — an upper bound on a magnitude argument, not
a measured payoff. What a real predictor collects is §2.

## 2. The real predictor, online — unstable (`xmlp.py`)

Per-layer OLS fitted on MLP A, applied ONLINE on MLP B through `chain.py mcorr`:

| | fit 0 → apply 1 | fit 1 → apply 0 |
|---|---|---|
| s = 0.10 | diverged (1e11×) | diverged (1e7×) |
| s = 0.25 … 1.00 | NaN | diverged |
| **in sample, s = 1.00** | **×1.97** (MLP 0) | **×1.52** (MLP 1) |

**The online correction runs away.** The fit is made on the *uncorrected* trajectory; correcting layer 1 moves
layer 2's state off that distribution, the linear model extrapolates, the correction grows. The in-sample arms do not
diverge but are still ×1.5–2.0 *worse* — so this is not only a generalisation failure.

Two real bugs in this harness were found and fixed before these numbers were trusted: the features were not
standardised (lstsq's rank truncation is scale-dependent, so the coefficients extrapolated wildly), and
`K4d/var²` overflowed float64 at dead neurons sitting on the 1e-12 variance clamp. Re-running task 1 with the clip
left its R² unchanged (L15: 0.512 κ₃, 0.561 κ₄), so §1's ceilings stand.

## 3. Offline — the test that separates the two failure modes (`offline.py`)

Predict from the RECORDED uncorrected trajectory, then substitute those fixed arrays through the `oracle` path: no
feedback. If offline helps, the predictor is useful and only the online loop is unstable (fixable by training in the
loop or by damping). If offline also hurts, the prediction itself is the problem — and the reason would be §68's
consistency mechanism: a per-channel statistical fit with R² 0.10 / 0.51 / 0.56 necessarily produces an
*inconsistent* triple, and this chain punishes inconsistency.

| fit → apply | s = 0.25 | s = 0.50 | s = 1.00 |
|---|---|---|---|
| 0 → 1 (out of sample) | NaN | NaN | NaN |
| 1 → 0 (out of sample) | NaN | NaN | NaN |
| **0 → 0 (in sample)** | ×0.982 | ×0.979 | ×1.019 |
| **1 → 1 (in sample)** | ×0.952 | **×0.925** | ×0.938 |

**Offline, in sample, where the fit is as good as it can possibly be, the corrector buys ×0.93–0.98 — 2–7%, not the
×0.70 the magnitude ceiling promised.** Out of sample the per-layer OLS coefficients do not transfer at all (20
features fitted on 1024 points of a single MLP, unregularised). The ceiling of §1 overstates the achievable payoff by
about **6×**, so §1 must be read as what it is: a magnitude argument, now falsified by measurement.

## 4. WHY — and this is the result worth keeping (`split_dir.py`)

Decompose the exact per-neuron moment error at each layer as e = **P**e + (I−**P**)e, where **P** projects onto the
span of the chain's own 20 smooth per-neuron features (plus an intercept, orthonormalised by QR), and substitute each
part alone, offline, with no feedback:

| removed from the moments | MLP 0 | MLP 1 |
|---|---|---|
| all of e (the ceiling) | ×0.206 | ×0.302 |
| **Pe only** — the smooth, feature-predictable part | **×1.015** | **×0.957** |
| **(I−P)e only** — the rough, orthogonal part | **×0.269** | **×0.450** |

**Removing the rough part alone recovers almost the whole ceiling; removing the predictable part alone buys nothing.**
(Cross-check: §3's offline in-sample s = 1.00 arms, computed by a different code path from a different target — the
two-oracle average instead of a single oracle — gave ×1.019 / ×0.938 against ×1.015 / ×0.957 here.)

**The damaging component of the chain's moment error is precisely the part that is NOT a smooth function of the
neuron's own state.** A corrector reading per-neuron state features — a ridge, a GBM, mliston's GRU, anything — aims
at the component that does not matter. That is why R² = 0.5 on κ₃ and κ₄ translates into a 5% gain instead of 30%,
and it is the same wall as NEGATIVE §62 ("the missing part is orthogonal to everything O(n²)") and 504aldo's dead end
#10 ("high-rank, not predictable from live-state features, +0.02 R²") — now measured **mid-chain**, at the one place
#9A said the error lives and where the direction therefore looked most promising.

## 5. Verdict

**The learned moment corrector is closed for per-neuron state features**, with a mechanism and a cross-checked
measurement, in one afternoon and with no baking. What is NOT closed: a predictor that sees **pair-level** objects
(the neuron's covariance row, its K22 row, the interaction of its incoming weights with the previous layer's state) —
those live in the rough directions. But materialising per-pair information is exactly what §54–§59 priced at n³, so
that door opens onto the cost wall we already closed.

**What this leaves:** the exact-engineering track (E3, ≈×0.96) is the only sure lever left, and the accuracy gap has
to be attacked analytically — a *missing term* is a structured, rough correction of exactly the kind smooth features
cannot represent, which makes the generative screen of `notes/CLOUD-BRIEF-10-01-analytic-archive.md` (task 2 there:
what pair diagram is the expansion MISSING, rather than what can it drop) the better use of the cloud.



---

# Part 3 (2026-10-01, evening, Opus 5.5) — the κ₄ PATTERN with pair features: in sample ×0.92, two MLPs fail, TEN MLPs work

§70 closed the corrector for **own-moment** features. Two things it had not tested: (a) **pair-level** features (row
reductions of C0, D, K22 — O(n²), i.e. free, because the chain materialises those matrices every layer; §70's own
caveat calling this "the n³ wall" was wrong), and (b) **more than two training networks**.

## 1. The κ₄-only projection split (`k4split.py`, in sample)
| removed (κ₄ only) | MLP 0 | MLP 1 |
|---|---|---|
| all of e (the κ₄ ceiling) | ×0.7529 | ×0.8094 |
| projection on the 21 **own** features | ×0.9967 | ×0.9578 |
| projection on **own + 17 pair** features | **×0.9160** | **×0.9183** |

Unlike §70 (all channels, own features: nothing), the **pair** features carry the part of the κ₄ error that matters.

## 2. Two MLPs cannot tell a pattern from an idiosyncrasy (`k4xmlp.py`)
Fit on one, apply offline to the other: per-layer OLS explodes (×1e24); a pooled model gives ×1.000–1.077. The
dominant coefficients (C0² row sum +0.52, |C0|^1.5 row −0.39, |C0| row +0.28, …) alternate in sign over collinear
features — one network's cancellations.

## 3. Ten MLPs: leave-one-out (`k4loo.py`)
Oracles for MLPs 2–9 baked (2e6, one seed — for a training target the per-neuron noise averages over 1024 neurons),
0–1 at 1e7. Pooled ridge on layers 8–15, relative κ₄ error target, fitted on 9, applied OFFLINE to the 10th, for all 10:

| model | held-out R² (feature space) | held-out raw × through the chain |
|---|---|---|
| **38 own+pair (+layer), λ = 1e-3, s = 0.5** | **0.361** | **mean ×0.9739, median ×0.9750, better 10/10** |
| 38 own+pair, s = 1.0 | | mean ×0.9718, 8/10 |
| 8 small, s = 0.5 | 0.316 | ×0.9871, 9/10 |
| 8 small, s = 1.0 | | ×0.9927, 5/10 |

**The first out-of-sample accuracy gain from a learned model in this project.** The two-MLP failure was a sample-size
failure. ⚠️ Still **offline** (corrections computed from the recorded trajectory); the online test — the step that
killed §70's corrector — is running (`k4online.py`).

## 4. What §70 now says, precisely
- **Closed:** correcting the moments from the neuron's **own** state (any model) — the damaging error is invisible to it.
- **Open and positive offline:** correcting the **κ₄ diagonal only** from **own + pair-level** features with a model
  trained on ≥9 networks: ×0.974 out of sample, 10/10. Production cost: 17 row reductions of n×n matrices per layer
  (O(n²), ≈0.03% of the budget) + a 39-coefficient dot product; the binding cost is **residual** (≈1000 extra
  flopscope calls ≈ 30 ms), with k3t at 0.365 s local max.


## 5. The other two channels — closed in sample (`chansplit.py`)
The κ₄ test repeated for the variance and κ₃ (one channel at a time, projection on own and own+pair features):

| channel | removed | MLP 0 | MLP 1 |
|---|---|---|---|
| variance | all of e | ×0.693 | **×1.090** (exact variance alone is harmful — the §68 consistency effect) |
| variance | P_own e / P_own+pair e | ×1.048 / ×1.031 | ×1.107 / ×1.100 |
| κ₃ | all of e | ×0.942 | ×0.915 |
| κ₃ | P_own e / P_own+pair e | ×1.066 / ×1.095 | ×1.015 / ×1.015 |

**For the variance and κ₃ the smooth, feature-predictable part of the error HURTS when removed, even in sample.** Only
κ₄ benefits (×0.916 / ×0.918). Consistent with cloud #10's mechanism: κ₄ enters the readout at second order and its
error carries an environment-dependent *level* that the pair row reductions see, while the variance and κ₃ errors are
rough. The corrector stays κ₄-only; its only lever left is more training networks (oracles for MLPs 10–29 baking).


---

# Part 4 (2026-10-02, night, Opus 5.5) — the channels JOINTLY: an in-sample synergy that does not generalise

## 1. The test that was never run (`jointsplit.py`, in sample, offline, own+pair projection removed at layers 1–15)
§5 above removed one channel at a time and found var and κ₃ harmful. But the exact correction is strongly synergistic
(joint ×0.21–0.30 against 0.69–1.09 per channel), so one-at-a-time could have been the wrong test for a joint corrector.

| arm (P_own+pair e removed) | MLP 0 | MLP 1 |
|---|---|---|
| exact var+κ₃+κ₄ (**gate**: exact_check gave ×0.206 / ×0.302) | **×0.2060** | **×0.3024** |
| var+κ₃+κ₄ | ×0.9174 | ×0.8702 |
| κ₃+κ₄ | ×1.0121 | ×0.9530 |
| **var+κ₄** | **×0.8846** | **×0.9004** |
| var+κ₃ | ×1.0398 | ×1.0450 |
| var+κ₃+κ₄, layers 8+ only | ×0.9415 | ×0.9227 |
| P_own only, all three | ×1.0146 | ×0.9573 |
| (κ₄ alone, §1 of Part 3) | ×0.9160 | ×0.9183 |
| (var alone, §5) | ×1.0309 | ×1.0996 |

In sample the variance projection, harmful alone, **helps together with κ₄** on both MLPs (×0.966 / ×0.981 on top of κ₄).

## 2. Out of sample it does not survive (`jvloo.py`, the k4loo protocol: pooled ridge λ = 1e-3 on 38 own+pair features
+ layer, fitted on 9 MLPs, applied offline to the held-out one, 10 MLPs)

| layers 8–15 | held-out R² | held-out raw × (10 MLPs) |
|---|---|---|
| κ₄, s = 0.5 (**gate** = k4loo's ×0.9739) | 0.361 | **×0.9739**, 10/10 |
| var, s = 0.5 | **0.004** | ×1.0190, **0/10** |
| var+κ₄, s = 0.5 | — | ×0.9799, 9/10 |
| var+κ₄, s = 1.0 | — | ×0.9872, 7/10 |

**The variance error has no cross-network structure in these features (held-out R² 0.004)**; the in-sample synergy was
a fit to each network's own rough error. Adding the variance channel makes the corrector *worse* out of sample. This
is §70's mechanism once more — the variance error of neuron i is wᵢᵀ ΔC wᵢ, a different functional of the unknown
matrix ΔC for every neuron — now confirmed for the joint corrector too. Closed; the variance lever, if any, is ΔC
itself (cloud #11).

## 3. Extending the κ₄ corrector to the early layers does not pay either (`jvloo.py`, layers 2–15)
In sample the early layers looked valuable (§1: all three channels at layers 1–15 ×0.917 / ×0.870 against layers 8+
×0.942 / ×0.923). Out of sample, a pooled κ₄ model over layers 2–15 has held-out R² **0.248** (0.361 for 8–15) and
gives **×0.9822 (9/10)** against **×0.9739 (10/10)** for 8–15: the early-layer κ₄ error has a different structure and
pooling it in degrades the late-layer fit. The shipped range (8–14) stays. (The remaining layers-2–15 variance arms
were stopped by hand — with a held-out var R² of −0.003 they cannot change the verdict.)
