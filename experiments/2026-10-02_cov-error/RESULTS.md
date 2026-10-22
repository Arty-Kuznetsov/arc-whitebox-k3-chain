# Cloud #11 — the structure of the covariance error ΔC (`notes/CLOUD-BRIEF.md`, 2026-10-01 late) — RESULTS

Branch `claude/laughing-hopper-61l0bn` (the session's designated branch; the brief's `cloud/cov-error` name was not
used) from `origin/main` 00d5c67. Torch chain, float64; FLOPs and MSE only (CLOUD.md §2). Harness: the tail-accuracy
one (`ce.py` wraps `../2026-09-30_tail-accuracy/ta.py`, results in `res/`). Base for everything: **B9** =
`pb.matched(-2.5, satrow=2.5, satrow_k4="sep")`.

## 0. Gates (`logs/gate_*.txt`, `logs/selfcheck_hooks.txt`)
| gate | required | measured on this branch |
|---|---|---|
| `ablate.py k4memo 3` | 3.7533e-08 | **3.7533e-08** (again after the `postc`/`prec` hooks: 3.7533e-08) |
| `ablate.py k3l_t 10` | 2.3135e-08 | **2.3135e-08** (per MLP 0/1/2/7/9 = the Mac's to every printed digit) |
| B9, 10 MLPs | 2.1967e-08 | **2.19667e-08** |
| hooks `postc` = `prec` = identity (and a recording `postc`) | bitwise B9 | **bitwise** on MLPs 0 and 1 (all 16 layers of post means equal); diag(Wᵀ C W) of the post C the hook sees = the next pre-variance to 4.7e-15 |
| `cov_oracle.py` vs the same random stream recomputed in float64 (4e4 samples) | ≪ 1e7 noise (~3e-4 of var) | max \|ΔC\| / mean var **3.1e-6** |

`chain.py` gained two hooks, both off by default: `postc(l, C) -> C` on the post-activation covariance after layer l
(the only thing that feeds the next layer, through C_pre = Wᵀ C W) and `prec(l, C_pre) -> C_pre` at the top of the
layer, before every other use.

## 1. The oracles (`cov_oracle.py`, `logs/oracle_*.txt`)
C_exact(l) = Cov(y_l), the full 1024×1024 post-activation covariance at layers 0..14 (every layer that feeds a next one),
plus post means, MLPs 0 and 1, seeds 2101 and 2102, 1e7 samples each (≈35 min per oracle on 3 threads here; checkpoints
every 1e6 survived three container restarts). Estimator: (y − ref)ᵀ(y − ref) per 20000-sample chunk in float32 GEMM,
float64 across chunks, ref = the baked 1e9 post mean. **Committed** (force-added, float32, compressed, 16 MB each):
`oracles/cov_<m>_s<seed>_L5-8-11-14.npz` (layers 5, 8, 11, 14 + post means). The full float64 all-layer files
(126 MB each) stay local; rebuild: `python cov_oracle.py <m> 1e7 <seed>`.
Checks: post means vs the baked 1e9 GT 6.6e-8 (L0) → 9.5e-9 (L15), = the Mac's 1e7 oracles; the variance implied by the
covariance, var_pre(l+1) = diag(W_{l+1}ᵀ C(l) W_{l+1}), differs from the Mac's two variance seeds by 6.3e-4 relative rms
— exactly the seed-to-seed spread between those two (6.3e-4 = √(4/1e7)); common-mode (all-neuron) offsets ≤1.3e-4, the
same size as between the Mac's own two seeds.

## 2. Task 2 — the attribution
### 2.0 The mechanism, exactly (`theory_check.py`, `res2/theory_0.json`)
C_chain(l) and C_exact(l) are functions of W_0..W_l only, so ΔC(l) is **independent of W_{l+1}**. With w_i the i-th
column of W_{l+1}: e_i(l+1) = w_iᵀ ΔC w_i = e_diag,i + e_off,i, and
- e_off,i = w_iᵀ ΔC₀ w_i has mean 0, variance 8‖ΔC₀‖²_F/n², independently across neurons i;
- e_diag,i has mean (2/n)·tr ΔC — **one number for every neuron** (a coherent level) — plus a 1/√n fluctuation.

Verified on the real matrices: ‖e_off‖² / ((8/n)‖ΔC₀‖²_F) = 0.89–1.09 at every layer; mean(e_off)/rms(e_off) = ±0.03
(= 1/√n); at depth e_diag is 94–97% its coherent level, which equals (2/n)·tr ΔC to 1%.
**So the "rough residual" of the variance channel (#10 task 1, NEGATIVE §70) is, by a theorem, the random projection
of the off-diagonal post-covariance error through the next layer's weights**: iid noise across neurons, orthogonal to
every per-neuron feature except quadratic forms of the neuron's own weight column. No per-neuron model could see it.

### 2.1 (a) The energy split (`t2.py split`, `res2/split_<M>.json`; rendered by `tables.py t2`)
Next layer's variance error e(l+1) on the chain's own B9 trajectory, split into diag = Σ_j W_ji² ΔC_jj and off = the
rest; two covariance seeds averaged, oracle noise subtracted with their difference ((e_A − e_B)²/4); the total is
cross-checked against the 4-seed variance oracle (Mac's two + these two) in brackets.

**MLP 0** — signal energies (noise-subtracted) of the next layer's variance error, summed over neurons; `diag` = Σ_j W_ji² ΔC_jj, `off` = the rest; `2x` = twice the cross term; shares are of the total signal.

| layer l→l+1 | rel. rms e (all) | all: diag | all: off | all: 2x | all: total (4-seed var) | noise / ‖e‖² | core: diag | core: off | core: 2x |
|---|---|---|---|---|---|---|---|---|---|
| 0→1 | 3.1e-04 | -2.89e-06 | -1.28e-05 | 3.56e-07 | -1.53e-05 (-1.33e-06) | 1.08 | -2.49e-06 | -2.00e-05 | 1.59e-06 |
| 1→2 | 3.2e-04 | -1.29e-06 | -8.31e-06 | -1.05e-06 | -1.06e-05 (2.69e-06) | 1.10 | -8.52e-07 | -5.98e-06 | -3.82e-07 |
| 2→3 | 3.4e-04 | -2.98e-07 (-10%) | 4.09e-06 (133%) | -7.21e-07 (-23%) | 3.07e-06 (1.03e-05) | 0.96 | -2.20e-07 (-4%) | 6.91e-06 (114%) | -6.44e-07 (-11%) |
| 3→4 | 3.9e-04 | 7.08e-07 (3%) | 2.32e-05 (99%) | -4.95e-07 (-2%) | 2.34e-05 (1.83e-05) | 0.64 | 3.66e-07 (5%) | 7.40e-06 (101%) | -4.17e-07 (-6%) |
| 4→5 | 4.2e-04 | -6.35e-07 (-3%) | 2.38e-05 (105%) | -4.49e-07 (-2%) | 2.28e-05 (2.30e-05) | 0.55 | -3.25e-07 (-3%) | 1.07e-05 (103%) | 3.00e-08 (0%) |
| 5→6 | 5.1e-04 | -3.61e-07 (-1%) | 3.30e-05 (102%) | -2.66e-07 (-1%) | 3.24e-05 (2.89e-05) | 0.38 | -1.70e-07 (-2%) | 1.04e-05 (101%) | 7.24e-08 (1%) |
| 6→7 | 6.2e-04 | 8.58e-08 (0%) | 4.46e-05 (99%) | 5.06e-07 (1%) | 4.52e-05 (4.10e-05) | 0.25 | 3.50e-08 (0%) | 1.48e-05 (100%) | -5.34e-08 (-0%) |
| 7→8 | 7.4e-04 | 2.93e-07 (1%) | 4.28e-05 (99%) | 3.05e-07 (1%) | 4.34e-05 (4.31e-05) | 0.17 | 1.06e-07 (1%) | 1.08e-05 (102%) | -3.62e-07 (-3%) |
| 8→9 | 9.0e-04 | 2.20e-06 (4%) | 5.02e-05 (97%) | -6.04e-07 (-1%) | 5.18e-05 (5.10e-05) | 0.12 | 7.44e-07 (7%) | 1.14e-05 (104%) | -1.14e-06 (-10%) |
| 9→10 | 1.0e-03 | 3.83e-06 (6%) | 5.86e-05 (91%) | 1.82e-06 (3%) | 6.42e-05 (6.61e-05) | 0.09 | 1.26e-06 (11%) | 1.12e-05 (95%) | -7.34e-07 (-6%) |
| 10→11 | 1.3e-03 | 6.86e-06 (8%) | 7.41e-05 (90%) | 1.32e-06 (2%) | 8.23e-05 (7.95e-05) | 0.06 | 2.05e-06 (13%) | 1.34e-05 (84%) | 4.80e-07 (3%) |
| 11→12 | 1.4e-03 | 6.42e-06 (8%) | 7.26e-05 (90%) | 1.39e-06 (2%) | 8.05e-05 (7.84e-05) | 0.05 | 1.80e-06 (15%) | 1.05e-05 (89%) | -5.00e-07 (-4%) |
| 12→13 | 1.6e-03 | 8.63e-06 (9%) | 8.70e-05 (94%) | -2.92e-06 (-3%) | 9.27e-05 (9.14e-05) | 0.04 | 2.02e-06 (21%) | 9.83e-06 (100%) | -2.02e-06 (-21%) |
| 13→14 | 1.9e-03 | 8.15e-06 (6%) | 1.19e-04 (92%) | 1.66e-06 (1%) | 1.29e-04 (1.28e-04) | 0.02 | 1.97e-06 (14%) | 1.42e-05 (98%) | -1.63e-06 (-11%) |
| 14→15 | 2.0e-03 | 8.60e-06 (8%) | 1.03e-04 (93%) | -6.02e-07 (-1%) | 1.11e-04 (1.09e-04) | 0.02 | 1.96e-06 (17%) | 1.14e-05 (100%) | -1.99e-06 (-17%) |
| **Σ all** | | 4.03e-05 (5%) | 7.15e-04 (95%) | +2.56e-07 (0%) | 7.55e-04 (7.70e-04) | 0.42 | | | |
| **Σ core** | | 8.25e-06 (7%) | 1.17e-04 (100%) | -7.70e-06 (-7%) | 1.17e-04 (1.42e-04) | 0.76 | | | |

**MLP 1** — signal energies (noise-subtracted) of the next layer's variance error, summed over neurons; `diag` = Σ_j W_ji² ΔC_jj, `off` = the rest; `2x` = twice the cross term; shares are of the total signal.

| layer l→l+1 | rel. rms e (all) | all: diag | all: off | all: 2x | all: total (4-seed var) | noise / ‖e‖² | core: diag | core: off | core: 2x |
|---|---|---|---|---|---|---|---|---|---|
| 0→1 | 3.1e-04 | 1.76e-08 | -1.95e-05 | -3.72e-07 | -1.98e-05 (-7.71e-06) | 1.10 | 2.01e-08 | -1.99e-05 | -4.70e-07 |
| 1→2 | 3.2e-04 | 1.16e-07 (3%) | 3.75e-06 (98%) | -2.94e-08 (-1%) | 3.83e-06 (5.16e-07) | 0.97 | 9.83e-08 | -1.34e-06 | -1.25e-07 |
| 2→3 | 3.4e-04 | -8.81e-08 (-1%) | 8.07e-06 (106%) | -3.81e-07 (-5%) | 7.60e-06 (1.32e-05) | 0.91 | -5.32e-08 (-1%) | 3.72e-06 (103%) | -4.41e-08 (-1%) |
| 3→4 | 3.6e-04 | 6.27e-08 (0%) | 1.33e-05 (102%) | -2.76e-07 (-2%) | 1.31e-05 (1.58e-05) | 0.78 | 4.24e-08 (1%) | 6.77e-06 (101%) | -1.03e-07 (-2%) |
| 4→5 | 4.2e-04 | -1.14e-07 (-0%) | 2.36e-05 (99%) | 4.05e-07 (2%) | 2.39e-05 (2.47e-05) | 0.56 | -6.23e-08 (-1%) | 9.18e-06 (100%) | 4.33e-08 (0%) |
| 5→6 | 5.0e-04 | 2.10e-07 (1%) | 3.15e-05 (98%) | 4.74e-07 (1%) | 3.22e-05 (3.06e-05) | 0.37 | 9.27e-08 (1%) | 1.01e-05 (100%) | -7.91e-08 (-1%) |
| 6→7 | 5.8e-04 | 8.79e-07 (2%) | 3.47e-05 (97%) | 1.97e-07 (1%) | 3.58e-05 (3.52e-05) | 0.29 | 3.31e-07 (3%) | 1.18e-05 (96%) | 1.30e-07 (1%) |
| 7→8 | 7.1e-04 | 1.79e-06 (4%) | 4.22e-05 (95%) | 4.98e-07 (1%) | 4.45e-05 (4.64e-05) | 0.22 | 6.34e-07 (6%) | 9.99e-06 (92%) | 2.05e-07 (2%) |
| 8→9 | 8.8e-04 | 2.89e-06 (5%) | 5.64e-05 (93%) | 1.55e-06 (3%) | 6.09e-05 (5.86e-05) | 0.13 | 9.00e-07 (8%) | 9.66e-06 (88%) | 3.62e-07 (3%) |
| 9→10 | 1.0e-03 | 5.08e-06 (8%) | 6.01e-05 (93%) | -4.03e-07 (-1%) | 6.47e-05 (6.53e-05) | 0.10 | 1.54e-06 (13%) | 1.09e-05 (90%) | -3.80e-07 (-3%) |
| 10→11 | 1.2e-03 | 4.50e-06 (5%) | 7.95e-05 (95%) | -5.88e-07 (-1%) | 8.35e-05 (8.08e-05) | 0.07 | 1.16e-06 (12%) | 8.89e-06 (91%) | -2.62e-07 (-3%) |
| 11→12 | 1.4e-03 | 6.07e-06 (6%) | 9.94e-05 (95%) | -3.74e-07 (-0%) | 1.05e-04 (1.05e-04) | 0.05 | 1.65e-06 (10%) | 1.51e-05 (93%) | -5.78e-07 (-4%) |
| 12→13 | 1.7e-03 | 8.23e-06 (6%) | 1.27e-04 (93%) | 1.98e-06 (1%) | 1.37e-04 (1.39e-04) | 0.03 | 2.05e-06 (12%) | 1.71e-05 (98%) | -1.75e-06 (-10%) |
| 13→14 | 1.9e-03 | 9.38e-06 (6%) | 1.46e-04 (96%) | -3.16e-06 (-2%) | 1.53e-04 (1.54e-04) | 0.03 | 2.08e-06 (11%) | 1.78e-05 (98%) | -1.67e-06 (-9%) |
| 14→15 | 2.2e-03 | 4.65e-06 (3%) | 1.61e-04 (98%) | -2.12e-06 (-1%) | 1.64e-04 (1.65e-04) | 0.02 | 1.00e-06 (8%) | 1.27e-05 (97%) | -6.20e-07 (-5%) |
| **Σ all** | | 4.37e-05 (5%) | 8.68e-04 (95%) | -2.60e-06 (-0%) | 9.09e-04 (9.27e-04) | 0.38 | | | |
| **Σ core** | | 1.15e-05 (9%) | 1.22e-04 (95%) | -5.34e-06 (-4%) | 1.29e-04 (1.45e-04) | 0.74 | | | |

**At every layer ≥ 3 the off-diagonal part carries 90–105% of the signal, on both MLPs; summed: off 95% / diag 5% (all
neurons), off 95–100% (kink core); the cross term is ≈0.** Layers 0–2 are below the 2e7 noise floor (signal ≈ 0).

### 2.2 (b) The chain payoff (`t2.py arm`, `res2/arm_*.json`)
At the layers of SET (s4 = 5, 8, 11, 14 as in the brief; all = 0..14) the post covariance is substituted on the LIVE
trajectory (absolute values) — PART all / diag / off; diagr / allr = the same with the chain's own diagonal LEVEL kept.
x = raw / B9 on that MLP. P = two seeds averaged (2e7), A / B = one seed each. Two independent noise corrections:
2P − (A+B)/2 (Richardson in the noise variance) and P / R, with R = the reset noise control (the off-diagonal set to the
base trajectory's own + (C_A − C_B)/2 at the same layers: same noise, same reset dynamics; its self-check R0 = base
alone gives x1.0000 on both MLPs). N = (C_A − C_B)/2 ADDED at every layer without reset — it accumulates, so it
over-states the noise cost (x3.2–3.9 at all layers) and is NOT used for corrections.

| MLP | layers | part | P (2e7) | A | B | 2P − (A+B)/2 | noise ctrl R (reset) | P / R | noise ctrl N (accumulating) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | s4 | off | 0.6757 | 0.7357 | 0.7154 | 0.6259 | 1.0814 | 0.6249 | 1.1895 |
| 0 | s4 | diag | 1.0880 | 1.1279 | — | — | — | — | 1.0298 |
| 0 | s4 | all | 0.7056 | 0.8078 | 0.7192 | 0.6477 | — | — | 1.2274 |
| 0 | s4 | diagr | — | 1.0033 | — | — | — | — | — |
| 0 | s4 | allr | — | 0.7362 | — | — | — | — | — |
| 0 | all | off | 0.6235 | 0.7339 | 0.6693 | 0.5455 | 1.1211 | 0.5562 | 3.2037 |
| 0 | all | diag | 1.1087 | 1.1829 | 1.0634 | 1.0943 | — | — | 1.4213 |
| 0 | all | all | 0.6563 | 0.8647 | 0.6635 | 0.5485 | — | — | 3.7808 |
| 0 | all | diagr | — | 1.0119 | — | — | — | — | — |
| 0 | all | allr | — | 0.7348 | — | — | — | — | — |
| 1 | s4 | off | 0.8486 | 0.9336 | 0.8747 | 0.7931 | 1.0530 | 0.8059 | 1.1919 |
| 1 | s4 | diag | 1.0469 | 1.0565 | — | — | — | — | 0.9992 |
| 1 | s4 | all | 0.8613 | 0.9527 | 0.8821 | 0.8052 | — | — | 1.1910 |
| 1 | all | off | 0.7715 | 0.9665 | 0.7886 | 0.6655 | 1.1116 | 0.6941 | 3.7656 |
| 1 | all | diag | 1.0752 | 1.0984 | 1.0568 | 1.0727 | — | — | 1.0570 |
| 1 | all | all | 0.8010 | 0.9961 | 0.8152 | 0.6964 | — | — | 3.8625 |

**KILL RULE: not triggered** — off-diagonal only, s4, P: **×0.676 (MLP 0), ×0.849 (MLP 1)**; noise-corrected
(the two corrections agree): **≈×0.625 / ≈×0.80** at s4 and **≈×0.55 / ≈×0.68** at all layers. These are
ORACLE-SUBSTITUTION UPPER BOUNDS (METHOD-LESSONS rule 2), not estimates of any lever.
Further facts (MLP 0, one seed unless stated; `res2/`):
- **The exact diagonal is harmful (×1.05–1.18) only through its LEVEL**: with the chain's own level kept (diagr) it is
  neutral (×1.003 / ×1.012) and allr = off (×0.736 / ×0.735). The chain's post-variance level sits ~4e-4 relative above
  the truth at depth, at its own MSE optimum (NEGATIVE §57, #10 task 1); the rough part of the diagonal is worth nothing.
- **Single-layer resets** (off, A): l5 ×0.998, l8 ×0.909, **l11 ×0.830**, l14 ×0.916; s4 (×0.736) ≈ all (×0.734).
- **Channel split** (s4, off, A; `t2.py armch`, both = plain bitwise): via the next layer's VARIANCES only ×0.839, via
  its PAIR structure C₀_pre only ×0.805, both ×0.736 (l11: ×0.944 / ×0.859 / ×0.830). Both channels matter.

## 3. Task 3 — the structure of the off-diagonal ΔC₀ (`t3.py`, `res3/`)
### 3.1 (a) Rank — present in the energy, absent in the payoff (`spec`, `rank`)
| layer | signal share of ‖ΔC₀_P‖² | rank-1 / 4 / 16 / 64 share (P) | same for the pure noise matrix | top-2 eigvec reproducibility A↔B | top-1 vs post-mean direction |
|---|---|---|---|---|---|
| MLP 0 L5 | 0.61 | 0.09 / 0.19 / 0.36 / 0.68 | 0.03 / 0.12 / 0.34 / 0.73 | 0.89, 0.82 | 0.72 |
| MLP 0 L8 | 0.87 | 0.23 / 0.39 / 0.56 / 0.80 | 0.06 / 0.17 / 0.44 / 0.81 | 0.97, 0.96 | 0.75 |
| MLP 0 L11 | 0.95 | 0.29 / 0.54 / 0.68 / 0.87 | 0.09 / 0.25 / 0.53 / 0.86 | 0.98, 0.98 | 0.73 |
| MLP 0 L14 | 0.98 | 0.41 / 0.65 / 0.78 / 0.92 | 0.12 / 0.29 / 0.58 / 0.90 | 0.99, 0.99 | 0.75 |
| MLP 1 L5 | 0.60 | 0.09 / 0.19 / 0.37 / 0.68 | 0.03 / 0.12 / 0.35 / 0.74 | 0.89, 0.83 | 0.72 |
| MLP 1 L8 | 0.86 | 0.19 / 0.40 / 0.57 / 0.81 | 0.06 / 0.17 / 0.44 / 0.82 | 0.97, 0.97 | 0.70 |
| MLP 1 L11 | 0.95 | 0.29 / 0.58 / 0.72 / 0.89 | 0.11 / 0.26 / 0.55 / 0.88 | 0.99, 0.99 | 0.71 |
| MLP 1 L14 | 0.98 | 0.38 / 0.67 / 0.79 / 0.93 | 0.17 / 0.34 / 0.62 / 0.91 | 0.99, 0.99 | 0.73 |

Payoff of the best rank-r part of the LIVE off-diagonal correction (s4, P):

| MLP | r = 1 | 4 | 16 | 64 | 256 | full (P) |
|---|---|---|---|---|---|---|
| 0 | 1.002 | 0.927 | 0.897 | 0.764 | 0.691 | 0.676 |
| 1 | 1.066 | 1.061 | 0.985 | 0.830 | 0.837 | 0.849 |

At depth ΔC₀ has TWO eigenvalues 8–12× above the noise edge (reproducible across seeds at 0.99, aligned with the post-
mean vector at 0.64–0.75 and with the top eigenvectors of the chain's own C at ~0.73): rank 1 holds ~40% of the energy
at L14. **They carry nothing**: rank 1 ×1.002 / ×1.066, rank 4 ×0.927 / ×1.061. The payoff appears only from rank
~64 on and saturates at 64–256 — i.e. it lives in the bulk, whose rank-r energy curve is the noise matrix's own shape.

### 3.2 (b) Sparsity and who the pairs are (`sparse`, `who`)
Top-q off-diagonal entries by |ΔC| selected on one seed, values from the other (cross-fit), s4:
| MLP | top 1% (sel A / B) | top 5% | top 20% | all (one seed A / B) |
|---|---|---|---|---|
| 0 | 0.968 / 1.056 | 0.868 / 0.877 | 0.763 / 0.768 | 0.736 / 0.715 |
| 1 | 1.056 / 1.036 | 1.008 / 0.962 | 0.838 / 0.883 | 0.934 / 0.875 |

The top 1% buys nothing, 5% little, 20% about two thirds. The top pairs are NOT kink-core pairs (core × core enrichment
×0.01–0.25 in the top 1%); they are enriched in large |D| (×2.0–3.8) and large |C₀| (×1.4–3.2). No cheap identity.

### 3.3 (c) Entrywise predictability from O(n²) objects (`feat`, `featborn`, `gain`, `../prop.py`)
Design: the eight terms of the chain's own Q11 (C0, D, C0², K22, K31, C0·D, C0·K22, C0³ with B9's coefficients — their
sum reproduces the chain's K11 to 1e-10), six candidates it lacks (C0⁴, C0·K31, D², D·Dᵀ, d1d1ᵀ, √(var var)), the chain's
own post C and C², and five gain-type features K∘(x_j + x_k) for x = α, log var, skewness, kurtosis, |α|. 27 features,
523k pairs per layer.
| layer | R² in sample MLP 0 / 1 | out of sample 0→1 | noise ceiling (split-half) | BORN part, MLP 0: R² in / ceiling |
|---|---|---|---|---|
| 5 | 0.033 / 0.029 | 0.025 | 0.61 | 0.057 / 0.81 |
| 8 | 0.049 / 0.050 | 0.029 | 0.87 | 0.089 / 0.94 |
| 11 | 0.072 / 0.061 | 0.057 | 0.95 | 0.123 / 0.98 |
| 14 | 0.097 / 0.067 | 0.051 | 0.98 | 0.146 / 0.99 |

**At most 10% in sample and ≤6% out of sample, against a ceiling of 0.87–0.98** — ΔC₀ is measured well and is not a
function of anything the chain holds. Three structured models, each with an oracle-fitted payoff arm:
- **proportional** ΔC₀ ≈ β(l) C₀ (`prop.py`, β from the variance oracles): explains 5–14% of the variance-error signal;
  applied as increments ×1.001 / ×1.028, half ×0.999 / ×1.010, last layer only ×1.021 / ×1.015 — nothing, or harm;
- **per-neuron gain** ΔC₀ ≈ ΔK + KΔ (n free parameters per layer, fitted on the oracle): 6–26% of the signal energy;
  payoff ×1.007 / ×1.069 (harmful);
- **low rank** (3.1): ~40–65% of the energy at r ≤ 4, payoff ×0.93–1.07.

### 3.4 Born at the step vs carried (`born`)
The chain's own step at layer l run on the EXACT C_pre(l) (input and target from the same seed, so the linear part of
the sampling noise cancels — noise drops ~10×): the error born at that step is **22–30% of ‖ΔC₀‖² at layers ≥ 11, the
carried part 65–76%** (both MLPs). The carried part has passed through one or more independent random W (the theorem of
§2.0 at matrix level), which is why resetting every third layer (s4) is as good as every layer, and why it cannot be
a function of the current layer's state. The born part alone is only 6–15% linear in the local features.

## 4. Verdict
**The off-diagonal post-covariance error is the variance channel, and it is real money — but it is diffuse.**
1. Mechanism (exact): the next layer's variance error is 95% the random projection w_iᵀΔC₀w_i of the off-diagonal error
   through independent weights — the "rough residual" of #10 / §70, now explained; the diagonal part is a coherent level
   the chain already sits optimally on.
2. Payoff, UPPER BOUNDS from oracle substitution: exact off-diagonal at layers 5/8/11/14 **≈×0.625 (MLP 0), ≈×0.80
   (MLP 1)**; at every layer ≈×0.55 / ≈×0.68 (noise-corrected two ways). Comparable to #10's exact core variance.
3. Structure: none that carries the payoff. Low rank exists (two reproducible mean-aligned directions, ~40–65% of the
   energy) and is worth ×0.93–1.07; sparsity needs ~20% of all pairs chosen by the oracle; the best O(n²) linear model
   reaches R² ≤ 0.10 in sample / ≤ 0.06 out of sample against a ceiling of 0.87–0.98; per-neuron gain and proportional
   models do not pay. 65–76% of the error at depth is carried through earlier independent W.
4. **Cheapest correction it suggests: none.** No predictor of ΔC₀ from O(n²) objects exists in this data; a correction
   would need the n×n matrix itself, i.e. the pair-level information of the n³ wall (§54–§59, §62).
**Closing number for the write-up:** the variance channel's remaining reserve (exact off-diagonal ×0.55–0.68, all layers)
is 95% a random projection of an off-diagonal covariance error of which ≤6% is predictable out of sample from anything
the chain holds, and whose low-rank part (≈40–65% of its energy) buys nothing — closed with a mechanism.
