# k3o — the matched dead-side cut on the production path (port of `2026-09-29_colcut`)

**Date:** 2026-09-29 night (Mac). **Estimator:** `src/estimator_k3o.py` (new, standalone). **Scripts:** `smoke.py`,
`k3o_check.py` (fs + torch parts), `opprof.py` (per-op FLOPs/overhead vs k3m), `runs/` (whest wrappers + JSON).

## What k3o is

k3m + the birth-column cut at the same threshold as the dead-row cut (`deadcut = colcut = t`): a source born at layer
s keeps only the hubs (parents) p with α_p ≥ t. Mechanism and torch evidence: `../2026-09-29_colcut/RESULTS.md` §5
(matched beats row-only at every threshold; −2.5: raw ×0.9928 on 10/10; −2.25: raw ×1.0294 vs k3m, modelled util
22.15% → projected ×0.918).

**The layout change that makes the saving real.** k3m keeps all legs in one batched `(2S, n, n)` stack, so per-source
column widths cannot be exploited. k3o stores the legs **transposed**, as one wide stack of hub rows
`XT = [B_0ᵀ; …; B_{S-1}ᵀ; A_0ᵀ; …; A_{S-1}ᵀ]` (source s owns w_s rows in each region; every per-hub vector — g, dK3, α,
feed h, d1 at birth, Utᵀ — lives in the same row order). Then:
- the leg transport is ONE row-stack product `XTᵀ' = XTᵀ W` contracted over the alive rows of the previous layer
  (a new **right-shared** Strassen, `_Strassen.rshared`, the mirror of `shared`, over row chunks of ≤ n rows);
- the D pairing `D = L Xᵀ = LTᵀ XT` is ONE contraction over all hub rows (the existing `paired_sum` on row chunks);
- the newborn's Bᵀ rows wait in a buffer and are gathered between the two regions at its first transport; its hub leg
  A = I[:, P] becomes Aᵀ = W[P, :] with no product;
- C and the newest path leg ride a separate full-W product (their dead rows are not zero), as in k3m's cut branch.
Only k3m's production path is kept (closure, fused base, memo κ₄ levels). The per-source thin parts (E's 3 H Ut,
the (A²Utᵀ) thin pairing factor, the feed's m and q) run as short per-source loops (5·S small ops per layer).

## Self-checks

| check | result |
|---|---|
| k3o(colcut=None) vs k3m, 128×6 (thin 128) | max rel diff per layer ≤ 4.0e-7 (float32 rounding) |
| k3o(colcut=None) vs k3m, **1024×16** | max rel diff per layer ≤ **1.3e-6**; FLOPs **×1.0020** (gathers billed 4×numel, the newborn's transposed copies) |
| k3o matched cut vs torch `chain.py` (k3m config, `corr='split128'` = exact legs, `deadcut=t, colcut=(t, inf)`), 128×8 | **2–6e-7 per layer at t = −2.5, −1.5, −1.0**, while the column cut's own effect on the final layer is 4.4e-5 / 9.6e-4 / 1.3e-3 (100–2000× larger) — the production cut is the torch cut |
| row-only cut, same comparison | 2–5e-7 per layer |

`k3o_check.py fs` (1024×16 part) is the reproducible form of the first row of the 1024 check.

## Flopscope measurements

### 10-MLP screen (`runs/run10.sh`; the first 10 MLPs of `mini` = the torch baked 10), production path

| arm | raw (final MSE) | × k3m | better | util | × k3m | **adjusted** | **× k3m** | better | residual mean/max (local) |
|---|---|---|---|---|---|---|---|---|---|
| k3m (re-run today, same machine state) | 2.3460e-8 | 1 | — | 24.886% | 1 | 5.8379e-9 | 1 | — | 0.243 / 0.279 s |
| **k3o matched −2.5** | 2.3313e-8 | **0.9937** | 9/10 | 23.788% | **0.9559** | 5.5454e-9 | **0.9499** | **10/10** | 0.245 / 0.264 s |
| k3o matched −2.25 | 2.4158e-8 | 1.0298 | 1/10 | 22.702% | 0.9122 | 5.4844e-9 | 0.9394 | 10/10 | 0.252 / 0.273 s |

- **Raw reproduces the torch prediction on the production path:** ×0.9937 vs torch ×0.9928 at −2.5; ×1.0298 vs ×1.0294 at
  −2.25.
- **Utilisation is metered now, and the model was optimistic:** ×0.956 at −2.5 against the modelled ×0.940 (23.34/24.84),
  ×0.912 at −2.25 against ×0.892. Part of the gap is the layout's own +0.2% (gathers billed 4×numel, the newborn's
  transposed copies); the rest is the modelled column share (94% of the budget) being too high.
- **−2.25 vs −2.5 is ×0.989 on adjusted — below the grader's resolution (§60).** −2.5 is the primary: it improves raw
  AND cost, the consistency mechanism's best point; −2.25 pays with 3% raw on 9/10 MLPs. Both are candidates for the two
  designations (only the better one on the private re-evaluation counts).
- **Residual time is unchanged by the layout.** Measured against k3m re-run in the same machine state: 0.245/0.264 vs
  0.243/0.279 s (k3m's 09-26 run read 0.190/0.197 — machine load, not the estimator). The extra ~1 200 flopscope calls
  per MLP (per-source thin loops, gathers, views; `opprof.py`) cost flopscope *overhead* (+88 ms), which is not
  residual time: on one 1024×16 MLP the residual reads 0.280 (k3o) vs 0.269 s (k3m).

### 100 MLPs (`runs/run100_src.sh` → `runs/k3o_n100.json`) — `src/estimator_k3o.py` with its defaults (−2.5 / −2.5)

| | adjusted | final MSE | all-layers | util | fails | residual mean/max | wall mean/max (local) |
|---|---|---|---|---|---|---|---|
| k3m (09-26, `k3l_cut25_n100.json`) | 5.5209e-9 | 2.2172e-8 | 8.3627e-9 | 24.894% | 0 | 0.197 / 0.227 s | 19.4 / 20.4 s |
| **k3o** | **5.2464e-9** | 2.2046e-8 | 8.3295e-9 | **23.792%** | **0** | 0.228 / 0.248 s | 20.4 / 21.6 s |
| ratio | **×0.9503, better on 100/100** | ×0.9943 (79/100) | ×0.9960 | ×0.9557 (per MLP 22.91–24.70%) | | (load: k3m read 0.243/0.279 today) | |

**Expected on the grader ≈5.12e-9** (5.3906e-9 × 0.9503; the same via grader/local 0.976: 5.1205e-9) — a 5% gain,
above the grader's ~0.5% resolution.

## Package check — `submission-20260929-173629.tar.gz`

- `whest validate` passed (82 ms, the 4-wide / 2-deep probe MLP).
- The tarball's `estimator.py` is **byte-identical** to `src/estimator_k3o.py`: sha256 `cdcea273c58290055995e4141c5ab8fa9b9db172a635f05c9fbb01a6ec2bb957`
  (the manifest records the same) — the exact file measured on the 100 MLPs above.
- `--runner subprocess`, 3 MLPs (`k3o_pkg_subproc_n3.json`): per-MLP adjusted **bit-identical** to the local 100-run
  (5.8475e-9, 5.0520e-9, 5.4919e-9), 0 failures, util 23.883%, residual 0.318 / 0.332 / 0.318 s (the subprocess runner
  reads ≈2× the grader; k3m's package read 0.290 / 0.357 / 0.380 and graded 0.153–0.178 s), wall 20.6–22.1 s/MLP.
- Submit (the user, with their key):
  `uv run --project external/whest-starterkit whest submit submission-20260929-173629.tar.gz --description "k3o: k3m + matched dead-side column cut at -2.5, transposed wide leg layout" --watch`
