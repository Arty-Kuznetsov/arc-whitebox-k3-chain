# Cost levers after k3i (2026-09-24 night →)

## 1. Is a dead-row cut billable? (`rect_probe.py`, flopscope, n=1024, n'=832)
| op | billed (× one n³ product) | wall | residual |
|---|---|---|---|
| square W @ X | 1.000 | 3.4 ms | 0.10 ms |
| rect W[:, :832] @ X[:832, :] (views) | 0.812 (= 832/1024) | 1.3 ms | 0.07 ms |
| take, W columns (n×832) | 0.002 | 3.8 ms | 0.15 ms |
| take, 8 legs' rows (8×832×n) | 0.013 | 3.3 ms | 0.03 ms |
| stack of 8 legs over alive rows | 6.496 (vs 7.996) | 12 ms | 0.43 ms |

Rectangular products are billed exactly in proportion, and gathers are billed as numel (≈free). So dropping dead
rows (α<−3; torch +0.3% MSE, cost model −13.5% of per-source products, RESULTS §10 of the k3-chain folder) is a real
saving. Plan (not implemented): after each nonlinear step permute the neuron order so that alive neurons come first
(one `take` of the legs, replacing the d1 gating multiply; `take` of the next Wᵀ's columns), run the next transport
over the first n' rows and the D pairings as n'×n' blocks, pad n' to a multiple of 16 for Strassen L4, and generalise
`_Strassen` to rectangular (m×k)(k×m) blocks. Un-permute the means per layer.

Machine note (2026-09-24 20:09): RAM 103 GB, ~88% free, swap 2.8/4 GB used, load ≈5 from other projects. No more
ARC AUGMENT reference runs (40–55 GB each) unless essential; torch chain and flopscope probes only, one at a time.

## 2. Rectangular workspace Strassen (`rect_strassen.py`, test `rect_strassen_test.py`; n=1024, L4, chunk 4)
Same buffers as the square k3h/k3i `_Strassen`, used through views [..., :a, :b]; all dims must be divisible by 16.
| call | rel. err vs float64 | billed / plain rectangular product | residual per call |
|---|---|---|---|
| shared, n×n' @ n'×n, 5 legs, n' = 1024 / 832 / 768 | 5e-6 | 0.606 / 0.607 / 0.608 | 2.6 ms |
| paired_sum, n'×n @ n×n', 6 pairs, n' = 1024 / 832 / 768 | 5–6e-6 | 0.607 / 0.612 / 0.614 | 2.5–2.7 ms |
Works; the Strassen saving is kept on the cut dimensions, residual and memory unchanged.

**Integration plan (k3j = k3i + dead-row cut), gather version (no global permutation):**
- alive set at layer l: α_l = μ_pre/σ ≥ −3, padded with the least-dead neurons to a multiple of 16 → index `idx_l`;
- transport at l+1: `take` the legs' rows `idx_l` (one gather of the stack) and `Wt` columns `idx_l`, rectangular
  `shared`; keep C transported in full (separate call) unless dropping its dead rows is measured harmless
  (torch: legs+D only +0.3%, with C +0.7%);
- D pairings at l+1: `take` rows `idx_{l+1}` of the left and right stacks, rectangular `paired_sum` → n'×n', then
  scatter back to n×n with two `take`s through an index map that sends dead neurons to an appended zero row/col;
- t readout stays on all rows. Self-checks: cut disabled → bitwise k3i; vs torch `deadcut=-3` at small n.

## 3. k3j = k3i + dead-row cut (`src/estimator_k3j.py`, check `k3j_check.py`)
- `deadcut=None` → **bitwise k3i** on 1024×16 (the rectangular Strassen is bitwise the square one on full blocks).
- vs torch `chain.py deadcut` (n=128, depth 8, thin=128): threshold −3 (0–2 dead per layer) 2–6e-7; **stress test at
  threshold −1.0 (up to 40/128 cut): 2.5–4e-7 while the cut itself moves the outputs by 2e-4–4e-3** → the rectangular
  transport, the alive×alive pairings and the scatter-back are exact.
- 1024×16 (seed 11, direct harness): **26.62% of B vs k3i 29.86% (−10.8%)**; residual 0.332 s vs 0.286 s (+0.046 s:
  gathers, the separate full C/path transport call, the masks); wall 21.1 vs 18.8 s; final max |k3j−k3i| 3e-6 rel.
- 100-MLP local run started 2026-09-25 ~00:30 → `k3j_n100.json` (expected ≈6.2e-9 local if MSE +0.3%).
  If the residual is too high, put C back into the cut stack (torch: +0.4% more MSE, one Strassen call per layer less).
- **100 MLPs (`k3j_n100.json`): adjusted 6.1137e-9 (k3i 6.9242e-9, ×1.133), final 2.3267e-8 (per MLP 0.995–1.008 of
  k3i), util 26.27% (25.5–27.0%), 0 fails, residual 0.208 / 0.251 s (k3i 0.199 / 0.245).** Packaged
  `submission-20260924-201225.tar.gz` (sha256 a3231c67…); subprocess 3 MLPs bit-identical, residual 0.29–0.36 s.

## 4. Shared subspace for old sources on the k3j configuration (torch, 10 MLPs) + cost model (`cost_model.py`)
- torch baseline k3j config (`ablate.py k3j_t` = k3i2 + deadcut −3): **2.4516e-8**; shared subspace 5:384
  (`k3j_ss_5_384`): **2.4887e-8 (+1.5%)**.
- Cost model (units of n³ products, billing calibrated on k3i: 0.636; MLP 0 dead fractions): k3i 29.82% (fit), k3j
  25.98% (measured 26.27%, so the model is within ~1%); **k3j + shared 5:384 → 21.9%, 4:384 → 20.9%, 6:320 → 21.4%,
  5:512 → 25.5%, 4:512 → 25.1%** (rank 512 buys almost nothing: materialisation + rotation grow with r).
  Expected net for 5:384: util ×0.844 at MSE ×1.015 → ≈×1.17 on the adjusted score (≈5.2e-9 local).
- **Not measured (stopped 2026-09-25 01:50 at the user's request):** 4:512, 4:384, 5:512, 6:320 — the batch loop
  never started them (zsh does not word-split `$pair`; use `${=pair}`). Configs exist in `ablate.py`.
- Production design still open: the torch arm re-projects on the top-r eigenvectors of the band's Gram every layer
  (after entry the legs stay in span(P) exactly, so only the entry step matters); production needs an entry update
  of a transported basis P (range finder on the entering legs' residual, r×r eigh, coefficient rotation r²n per leg).
- Per-layer, MLP 0: k3j still has ≈0.65–0.74× AUGMENT's advantage concentrated at layers 4–9 (AUGMENT/k3j excess
  error), 0.85–0.88 at 10–15 → the next accuracy lever is early-layer (paths at all ages or AUGMENT consistency terms).

## 5. Shared-subspace variants on the k3j configuration (torch, 10 MLPs; base `k3j_t` 2.4516e-8)
| band a0:r | final MSE | Δ | util (cost model) | adjusted factor vs k3j |
|---|---|---|---|---|
| 5:384 | 2.4887e-8 | +1.5% | 21.92% | 0.856 |
| **6:320** | **2.4833e-8** | **+1.3%** | **21.43%** | **0.836** |
| 5:320 | 2.6169e-8 | +6.7% | 20.26% | 0.832 |
| 6:256 | 2.6403e-8 | +7.7% | 20.11% | 0.834 |
| 4:384 | 2.6649e-8 | +8.7% | 20.90% | 0.874 |
There is a cliff below r≈320–384 (as in #18218). **Choose 6:320.** These use the ideal torch arm (top-r eigenvectors of
the band's Gram recomputed every layer); the entry update of a production basis is tested next (§6).
Runner: `experiments/2026-09-24_k3-chain/run_pairs.sh cfg1 cfg2 ...` (two at a time, correct zsh splitting).

## 6. Production-like entry update (torch `ss_iter`, band 6:320, 10 MLPs)
Transported basis P (gated like a leg); at each layer the band basis = top-r of the band Gram found by `ss_iter` subspace
iterations started from [transported P | 32 random], then Rayleigh–Ritz (instead of an exact eigh).
| | final MSE |
|---|---|
| exact eigh (ideal arm) | 2.4833e-8 |
| **q = 1** | **2.4876e-8** (+0.2% vs exact) |
| q = 2 | 2.4837e-8 (= exact) |
Small-n smoke test (r=64/128): q=3 = exact to 1e-14. **One or two cheap iterations per entry suffice.**

### Production design for k3k (band a0 = 6, r = 320), to implement
- State: band basis P (n×r, transported by Wᵀ with the dead-row cut, rows gated by d1 like a leg); per band source fixed
  coefficients α_s = Pᵀ-coords of A_s, β_s of B_s (r×n each; buffer (2Smax, r, n) ≈ 39 MB); thin legs H_s unchanged
  (projected on P at each basis update); band = oldest sources = the first 2·S_band leg slots.
- Per layer: transport P (rect Strassen n×n'·n'×r); transport only the young legs (slots 2S_band..); materialise the band
  legs into their slots X[:2S_band] = P @ coef (rect Strassen n×r·r×n) so t, E, L1, L2, feed code is unchanged;
  D pairings: young part as now; band part Z = Σ L1 βᵀ + L2 αᵀ (paired_sum, n'×n·n×r) then Z @ P[alive]ᵀ.
- Entry (the source reaching age 6, transported as young this layer): Gram-apply
  G X = P(M(PᵀX)) + A_new(A_newᵀX) + B_new(B_newᵀX) + 9 Σ H Hᵀ X with M = Σ_band (ααᵀ + ββᵀ) (r×r, kept);
  X0 = [P_t | deterministic probes], q = 1–2, QR, Rayleigh–Ritz (352×352 eigh) → P'; rotate old coefficients by
  R = P'ᵀP (r×r·r×n per leg), new ones α_new = P'ᵀA_new, β_new = P'ᵀB_new; update M; project H onto P'.
- Self-checks: band off → bitwise k3j; vs a torch arm with the same deterministic start → float noise.
- Expected: util 26.3% → ≈22% (model 21.4% + entry iterations), MSE +1.3–1.5% → ≈×1.15–1.18 (≈5.2e-9 local).
- **Torch production mirror** (`chain.py ssp=(a0, r, q, over)`: transported basis, deterministic start
  [transported P | first 32 columns of the entrant's A] (the entrant's first r+32 at the first entry), q iterations,
  Rayleigh–Ritz, all band legs projected): **6:320, q=1: 2.4820e-8 (+1.2%)**; q=2: 2.4840e-8 → the port target is
  `ablate.py k3k_t`.

## 7. k3k = k3j + shared-subspace band, flopscope port (`src/estimator_k3k.py`, check `k3k_check.py`) — CORRECT BUT NOT CHEAPER
- `band_age=None` → bitwise k3j; vs the torch mirror `ssp=(4, 64, 1, 16)` at n=128 (depth 10, thin=128): 3–6e-7 while
  the band itself moves the outputs 2.4e-5–7.4e-5 → the basis update, coefficient rotation, materialisation and
  compressed pairings are exact.
- **1024×16 (seed 11): 27.78% of B vs k3j 26.62% — MORE expensive; direct residual 0.413 s vs 0.276 s.**
- Why (per entry layer, units of n³ products): the entry update runs in plain (1.0-billed) products: two Gram
  applications on 352 vectors (A_e, B_e both ways: 2.75), 2 QR (0.42), eigh (0.18), coefficient rotation r²n per leg
  (≤1.8 at 18 legs), new coefficients P'ᵀ[B_e, A_e] (0.63), Mnew (0.2), P transport (0.3), Z·Pᵀ (0.25) → ≈4–7 per
  layer, against a saving of ≈1.2 per band source-layer (transports + full pairings − materialisation − compressed
  pairing, all at 0.61 billing). At layers 6–9 (1–4 band sources) the overhead exceeds the saving. `cost_model.py`
  had assumed ≈1.5 for the entry.
- Best case with q=0 (one Gram application), Strassen for the big entry products and a delayed band start: ≈−2.9
  points of B (26.3 → 23.4%) at +1.2% MSE ≈ ×1.11, plus ~2 more Strassen calls per layer of residual. **Parked**:
  worth doing only after the accuracy levers, or if a cheaper entry update is found (e.g. q=0 validated in torch).

## 8. k3l = k3j + K31 × 0.65 (`src/estimator_k3l.py`)
`k31_scale=1.0` → bitwise k3j. **100 MLPs (`k3l_n100.json`): 5.7387e-9 @26.27%, final 2.1837e-8 (per MLP 0.868–1.011 of
k3j, 98/100 better), 0 fails, residual 0.207/0.239 s.** Packaged `submission-20260925-075409.tar.gz` (sha256 094cd84c…),
subprocess 3 MLPs bit-identical, residual 0.30–0.33 s. Expected grader ≈5.57e-9.
