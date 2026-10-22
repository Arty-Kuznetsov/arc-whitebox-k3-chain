# k3v — the learned κ₄-diagonal corrector in production (2026-10-01)

`src/estimator_k3v.py` = k3t + a correction of the κ₄ diagonal `K4d` at layers 8–14 (the last layer has no C0, D or
K22 in production — the pair machinery is skipped there). The research leading here is in
`../2026-10-01_corrector/RESULTS.md` Part 3; the idea of an offline-trained corrector reading the closure's own moment
state is **mliston's (forum topic 18170, Phase 1)** — no code of theirs was used or seen.

## The model
A pooled ridge regression of the **relative** κ₄ error, (K4d_chain − κ₄_oracle) / mean|K4d_chain|, on 39 features
the step already holds: 21 functions of the neuron's own moments, 17 **O(n²) row reductions of C0, D and K22**, and
the layer index. Trained on the **production trajectory** (k3v itself with recording on — no train/serve skew) of the
first 10 mini MLPs, loaded from the dataset with the runner's own weights and seeds, against Monte-Carlo oracles
(MLPs 0–1 at 1e7, 2–9 at 2e6). λ = 1e-3 chosen by leave-one-MLP-out R² (0.354). Applied with shrinkage s = 0.5,
not tuned. The 118 numbers live in the estimator file as plain lists (the package holds only `estimator.py`).

## Validation, in increasing honesty
| test | result |
|---|---|
| torch, in sample, κ₄-only projection on own+pair features | ×0.916 / ×0.918 (upper bound) |
| torch, 2 MLPs cross | ×1.00–1.08 — **two networks are not enough** |
| torch, 10-MLP leave-one-out, offline | ×0.9739, 10/10 |
| torch, online, 4 held-out, layers 8–15 / 8–14 | ×0.9634 / ×0.9681, 4/4 |
| **production estimator, 10-MLP leave-one-out** | **×0.9803 mean, 8/10** |
| **production, 100-MLP run, the 87 non-breached OUT-OF-SAMPLE MLPs (10–99)** | **adjusted ×0.9878, 69/87 better** |

Each step down the table was less optimistic than the one above it, and the last is the honest number: **≈1.2%**.
The direction is solid (69/87, binomial p ≈ 1e-8); the magnitude is what it is.

## Gates
- corrector off = **bitwise k3t, equal FLOPs** (1024×16 and 128×8);
- **`whest validate` passes** (the shipped k3t does not — the small-width Strassen bug, fixed here);
- FLOPs ×1.0005; the corrector's own residual cost **+4 ms** (median over 93 MLPs).

## ⚠️ The machine, again
The 100-MLP run had **7 residual breaches** (max 0.615 s) — clustered in time (run positions 1–4, 47, 50, 51), with
wall spikes up to 49.8 s on the first four. At the time the Mac was running Apple's XProtect scanner (96% of a core),
`sysmond` (98%), Telegram, a trading bot and **another Claude Code project's session**. The corrector's intrinsic cost on
the 93 clean MLPs was +4 ms; the breaches are load. **Local residual on this shared Mac is unreliable at ±50–100 ms;
the grader's telemetry is the measurement** (k3t: 0.2676 s max).

## Submitted — 333367 REJECTED, 333368 GRADED **3.9102e-9**
333367 was rejected by the grader's static scan ("Prohibited file in submission") for an `import numpy` inside the
never-executed recording branch. Fixed (the hook stores flopscope copies; the training script converts them),
verified bitwise identical in output to the rejected file, and resubmitted as **333368 = 3.9102e-9** (projected 3.91):
**raw 2.0312e-8 = ×0.9880 of k3t's graded raw — the 87-MLP out-of-sample estimate was ×0.9873.** The learned corrector
performs on the grader's hidden networks exactly as measured locally out of sample. Residual max 0.2766 s (+9 ms vs
k3t), wall max 81.2 s, 0 failed.


## What limits the corrector (measured 2026-10-01, late)
- **Not the data.** Retrained on 30 networks (oracles for MLPs 10–29 baked): held-out R² **0.353**, the same as with 10
  (0.354). The 10-MLP model stays in production (validated on 87 MLPs and on the grader).
- **Partly the model class.** 5-fold cross-validation by whole networks on the 30 (`nonlin_ceiling.py`): ridge
  **0.359**, a small MLP 39→16→1 **0.417**, 39→32→1 **0.423** — consistent in every fold (+0.05…+0.07). In flopscope
  such an MLP costs two matmuls and one tanh per layer, i.e. nothing. But the payoff grows far more slowly than R²
  (R² 0.35 bought raw ×0.988), so the MLP corrector is worth an estimated **+0.3%** — below the grader's resolution.
  Recorded as an available small lever, not built.
