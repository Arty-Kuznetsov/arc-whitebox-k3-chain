# k3p — the transported band basis on the production path (port of the cloud's `bandz`, port battery task 1–2)

**Date:** 2026-09-29 late night (Mac). **Estimator:** `src/estimator_k3p.py` = k3o + `band=(a0, r, init)`.
**Scripts:** `smoke.py`, `k3p_check.py` (fs + torch parts), `opprof.py` (per-op profile vs k3o), `runs/`.
**Source of the design:** cloud session #8 (`cloud/port-battery`, commits fbc1bd5 → edafaa4, `chain.py bandz`):
float32 band exact with the QR projection and NO re-orthonormalisation / coefficient rotation (cond(P) ≈ 3e4 at depth,
legs within ≤ 2e-5 of float64); honest price (6,448) pre entry ×0.887 util; (5,448) with the range-finder init the
best projected so far (×0.878 on k3m).

## The construction, as ported (METHOD §26)

Sources of age ≥ a0 ("members", always the oldest ones) live as legs = P Z in one shared basis P (n × r). In k3o's
transposed layout:
- **Pᵀ rides the leg transport** as r extra rows of the gathered stack `[Pᵀ; Bᵀ of the non-members; newborn Bᵀ; Aᵀ of the
  non-members]` — the basis is gated by d1 like a leg (its dead columns are exactly zero) and contracted over the alive
  rows with them. No extra Strassen call.
- **Members are materialised** every layer by one right-shared product `[Z_Bᵀ; Z_Aᵀ] Pᵀ` (their coefficient rows are
  copied into one contiguous stack first) for the readout and the element-wise pairing partners; their thin legs
  H = P Z_H by one batched product.
- **Their D pairing is compressed:** `D_m = (L_mᵀ [Z_Bᵀ; Z_Aᵀ]) Pᵀ` on the alive × alive block (a paired_sum over the
  members' hub rows into an alive × r block, then one product with the alive columns of Pᵀ), added to the non-members'
  dense block before the scatter.
- **Entry ("pre"):** at the end of the layer before a source reaches age a0 (after the gating), it is projected on the
  gated P through a QR of P: `Z = R⁻¹ Qᵀ X` for its A, B and thin H (one product `[Bᵀ; Aᵀ] Q`, two LU solves). P is never
  re-orthonormalised and the coefficients never rotated. The first entrant defines P: `init="rr"` = the dominant
  r-subspace of its [A B] (range finder, one power step, probes from the MLP's seeded generator) or `init="cols"` = its
  first r hub columns.
- Per-hub vectors and Z rows are indexed by the global hub offset, so members are always the leading rows [0, o_m) and
  non-members the rows [o_m, WB) — no re-layout when a source enters.

## Self-checks

| check | result |
|---|---|
| k3p(band=None) vs k3o | **bitwise, FLOPs equal** (128×8 and 1024×16) |
| k3p band vs the cloud's torch `bandz` (k3m config, `corr='split128'`, deadcut −2.5, colcut None, `k=None, proj='qr', entry='pre', init='cols'`, float32 band), 128×8 | **2–7e-7 per layer** at (a0, r) = (3, 64), (4, 48), (3, 96), while the band's own effect on the final layer is 1.0e-3 / 6.8e-4 / 2.3e-4 — the production band is the measured band |
| r = n = 128 on a 128-wide MLP | singular R: the dead rows make P rank-deficient when r approaches the alive count — irrelevant at r = 448 with ~860 alive, but r must stay well below the alive count |

`init='rr'` cannot be compared digit for digit (the torch probes come from another generator and the torch A is not
masked by the column cut); it is judged on the 10/100-MLP screens.

## Residual time (the constraint to watch)

The band adds ≈4 Strassen calls per band layer (materialisation, the entry product, the compressed pairing, the D_m
product), ≈90 flopscope calls each, ≈20 µs each: on one 1024×16 MLP the residual reads **0.376 s vs k3o 0.294 s**
(`opprof.py 5 448 rr`), i.e. ≈0.31 s on the local runner and ≈0.24 s on the grader (0.75×) against the 0.4 s cap.
Replacing the two small products (entry `[Bᵀ; Aᵀ] Q`, `Y Pᵀ`) by plain matmuls saves 16 ms but costs +0.45% of B
(≈2% of the score) — **not done**; the Strassen versions stay.

## Flopscope measurements

### 10-MLP screen of four bands on top of k3o (`runs/run10.sh`, `summarize.py`), production path

| arm | raw | × k3m | util | × k3o | **adjusted** | × k3m | **× k3o** | better than k3o | residual mean/max (local) | wall max |
|---|---|---|---|---|---|---|---|---|---|---|
| k3m | 2.3460e-8 | 1 | 24.886% | 1.046 | 5.8379e-9 | 1 | 1.053 | — | 0.243 / 0.279 s | — |
| k3o (matched −2.5) | 2.3313e-8 | 0.9937 | 23.788% | 1 | 5.5454e-9 | 0.9499 | 1 | — | 0.245 / 0.264 s | 20.5 s |
| **band (5,448,rr)** | 2.3869e-8 | 1.0174 | **20.689%** | **0.870** | **4.9383e-9** | **0.8459** | **0.8905** | **10/10** | 0.338 / 0.360 s | 21.8 s |
| band (6,448,rr) | 2.3363e-8 | 0.9958 | 21.477% | 0.903 | 5.0180e-9 | 0.8595 | 0.9049 | 10/10 | 0.342 / 0.374 s | 21.8 s |
| band (5,512,rr) | 2.3270e-8 | 0.9919 | 21.929% | 0.922 | 5.1032e-9 | 0.8741 | 0.9203 | 10/10 | 0.346 / 0.396 s | 22.7 s |
| band (6,512,rr) | 2.3296e-8 | 0.9930 | 22.539% | 0.948 | 5.2511e-9 | 0.8995 | 0.9469 | 10/10 | 0.334 / 0.364 s | 21.8 s |

- **(5,448) wins**, as the cloud projected (×0.878 on k3m there; ×0.846 here on top of the matched cut). It pays +2.4%
  raw for ×0.870 util — every arm is better than k3o on all 10 MLPs. **Default of `src/estimator_k3p.py`.**
- The metered util agrees with the cloud's model (5,448 pre: model ×0.887 of k3m's cost structure; here ×0.870 on k3o).
- **Residual time is the price:** +0.09 s mean on the local runner (0.338 vs 0.245 s), i.e. ≈0.25–0.30 s expected on
  the grader (0.75×) against the 0.4 s cap. Where it comes from: ≈20 µs per flopscope call, and the band adds ≈4
  Strassen calls (≈90 calls each) per band layer. A cProfile of one predict (`pyprof.py`) shows the residual is
  flopscope's per-call wrapper time outside its own timed region, not our Python — only fewer calls reduce it.
  Reordering Strassen's combinations (paired strided adds) would save ≈9% of the calls, bitwise-neutral; not done yet.
- Guard added: the band is disabled when r > n/2 or r is not Strassen-divisible (small MLPs, e.g. the validate probe).

### 100 MLPs (`runs/run100_src.sh` → `runs/k3p_n100.json`) — `src/estimator_k3p.py` with its defaults

| | adjusted | final MSE | util | fails | residual mean/max | wall mean/max (local) |
|---|---|---|---|---|---|---|
| k3m (09-26) | 5.5209e-9 | 2.2172e-8 | 24.894% | 0 | 0.197 / 0.227 s | 19.4 / 20.4 s |
| k3o | 5.2464e-9 | 2.2046e-8 | 23.792% | 0 | 0.228 / 0.248 s | 20.4 / 21.6 s |
| **k3p** | **4.6839e-9** | 2.2633e-8 | **20.693%** | **0** | 0.279 / 0.311 s | 20.9 / 22.0 s |

**k3p = ×0.8928 of k3o and ×0.8484 of k3m, better on 100/100 against both.** Raw ×1.0266 of k3o, util ×0.8698.
**Expected on the grader ≈4.57e-9** (5.3906e-9 × 0.8484).
**Memory** (`memprobe.py`, peak RSS of a fresh process over two predicts): k3m 5.58 GB, k3o 4.56 GB, **k3p 4.50 GB** — the
8 GB cap is far (the whest runner's 14–19 GB RSS is the loaded dataset).

## Package check — `submission-20260929-190455.tar.gz`

- The tarball's `estimator.py` is byte-identical to `src/estimator_k3p.py` (sha256 `e3e0c1a187f1e08256d9fef5a40a1aad855b1941a1b3f180f9e2600aa8d4242c`,
  unchanged since commit 4e4cb12 = the file measured above); `whest validate` passed.
- `--runner subprocess`, 3 MLPs (`k3p_pkg_subproc_n3.json`): MLPs 0–1 bit-identical to the local run, **MLP 2 zeroed by
  the subprocess runner's residual cap (0.432 s)**. The subprocess runner reads ≈2× the grader (k3m's package read
  0.290/0.357/0.380 s there and graded 0.153–0.178 s), so the grader is expected at ≈0.21–0.24 s — but the margin to the
  0.4 s cap is now ≈1.7× instead of k3m's 2.2×, and one grader failure would zero an MLP (≈0.9 adjusted for it).
- **Risk handling:** the leaderboard keeps the best submission, so submitting k3p is safe for the displayed rank and
  its telemetry measures the grader's residual for this code; a *designated* submission must not carry this risk.
  Calls per region of one predict (`opcount.py`): Strassen 51% (rshared 26%, paired_sum 13%, paired 12%), `_nonlin` 23%
  (~266 calls/layer, mostly (n,)-vector ops), the rest 26%. Next: cut the calls (batched Wick coefficients, paired
  Strassen combinations — both bitwise-neutral) before the final candidates.
