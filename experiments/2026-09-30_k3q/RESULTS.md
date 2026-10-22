# k3q — the saturated-row cut of the D pairing with the κ₄ level restored (port of the cloud's E2 "sep")

**Date:** 2026-09-29/30 night (Mac). **Estimator:** `src/estimator_k3q.py` = k3p + `satrow=θ`.
**Source:** cloud session #8, commit aa1efe8: on the matched base, E2 **improves raw** — exact restoration
("restore") 3.0 ×0.9917 (10/10), 2.5 ×0.9534 (9/10); the production form ("sep") 3.0 ×0.9845, **2.5 ×0.9402,
projected ×0.877**.

## The construction, as ported

- Rows i of D_ij = κ₃[i,i,j] of neurons saturated NOW (α_i > θ) are not computed: the pairing's left factor loses
  those columns (in k3o's transposed layout: `take(LT, idr, axis=1)` with idr = the alive, non-saturated neurons —
  the tail of the α-descending order from the saturated count rounded down to the Strassen quantum; the few saturated
  rows that block still holds are zeroed by the row mask). The pairing becomes rectangular (n_r × npc), for the
  non-members' dense part and the band members' compressed part alike, scattered back through a row map and the usual
  column map. Everything downstream (the nonlinear step, the births) sees the cut D, as in torch.
- **The κ₄ level restoration ("sep")** — the next layer reads K22post only through Ssum = uᵀ K22post u (u = row sums of
  W_{l+1} ∘ W_{l+1}), so the cut rows' contribution to that level is restored by
  dS = Σ_{i sat} u_i cf22_i (D_full v)_i, v = u ∘ (cf12 − 2 P1 cf11) (the marginally corrected pair coefficients, now
  returned by `_nonlin`), D_full = the dead-masked, zero-diagonal D before the cut. D_full v is assembled from the legs
  by matrix-vector products — LTᵀ(XT v), the members' LMᵀ(ZS(Pᵀv)), the thin Σ thinw_s(H_sᵀ v), the feed q (m·v) —
  minus the diagonal term; O(S n²). dS is added to the next layer's Ssum, which is exactly torch's
  K22post += k22_fix off the diagonal (tr(M)'s mean-field off-diagonal part follows it automatically).

## Self-checks

| check | result |
|---|---|
| k3q(satrow=None) vs k3p, 128×8 | **bitwise, FLOPs equal** |
| k3q vs the cloud's torch `satrow=θ, satrow_k4="sep"` (k3m config + matched cut, `corr='split128'`, no band), 128×8 | **2–7e-7 per layer** at (t, θ) = (−2.5, 1.5), (−2.5, 1.0), (−1.5, 1.0), while the cut + restoration moves the final layer by 1.0e-3 / 3.8e-3 / 3.8e-3 and the κ₄ restoration alone by 4.7e-4 / 1.0e-3 / 1.1e-3 — both parts of the production E2 are the torch E2 |
| FLOPs on 128×8 | ×0.99 / ×0.975 / ×0.978 of the no-cut run (small MLP; the saving is the saturated share of the pairing) |

## Flopscope measurements

**1024×16 (one synthetic MLP):** satrow=None bitwise k3p with equal FLOPs; FLOPs ×0.972 / ×0.957 / ×0.939 of k3p at
θ = 3.0 / 2.5 / 2.0. First version cost +60 ms of residual per predict; two 100 MB temporaries of the level correction
(LT∘XT, LM∘XM) moved into free buffers and the matrix-vector products rewritten as row-vector products → ≈+37 ms
(+≈800 flopscope calls; residual is ≈20–40 µs per call and noisy at ±20 ms on this machine).

### 10-MLP screen on top of k3p (`runs/run10.sh`), production path

| θ | raw | × k3p | util | × k3p | **adjusted** | **× k3p** | × k3m | better than k3p | residual mean/max |
|---|---|---|---|---|---|---|---|---|---|
| — (k3p) | 2.3869e-8 | 1 | 20.689% | 1 | 4.9383e-9 | 1 | 0.8459 | — | 0.338 / 0.360 s |
| 3.0 | 2.3509e-8 | 0.9849 | 20.036% | 0.9685 | 4.7098e-9 | 0.9537 | 0.8068 | 10/10 | 0.351 / 0.381 s |
| **2.5** | **2.2512e-8** | **0.9431** | **19.725%** | 0.9534 | **4.4394e-9** | **0.8990** | **0.7604** | **10/10** | 0.361 / 0.393 s |
| 2.0 | 2.8040e-8 | 1.1747 | 19.333% | 0.9345 | 5.4180e-9 | 1.0971 | 0.9281 | 4/10 | 0.354 / 0.381 s |

The production E2 reproduces the cloud's torch curve (sep: 3.0 ×0.9845, 2.5 ×0.9402, 2.0 ×1.16 raw) on top of the band.
**Default θ = 2.5.** Expected on the grader ≈5.39e-9 × 0.760 ≈ 4.1e-9 (the 100-MLP run decides).

### 100 MLPs (`runs/run100_src.sh` → `runs/k3q_n100.json`) — `src/estimator_k3q.py` with its defaults (sha 8ebea33e…, commit 4f20c58)

| | adjusted | final MSE | util | fails | residual mean/max | wall mean/max (local) |
|---|---|---|---|---|---|---|
| k3m (09-26) | 5.5209e-9 | 2.2172e-8 | 24.894% | 0 | 0.197 / 0.227 s | 19.4 / 20.4 s |
| k3p | 4.6839e-9 | 2.2633e-8 | 20.693% | 0 | 0.279 / 0.311 s | 20.9 / 22.0 s |
| **k3q** | **4.2391e-9** | 2.1479e-8 | **19.735%** | **0** | 0.282 / 0.336 s | 20.8 / 21.8 s |

**k3q = ×0.9050 of k3p (100/100), ×0.7678 of k3m (100/100)**; raw ×0.9490 of k3p, util ×0.9537. **Expected on the grader
≈4.14e-9.**

## Package — `submission-20260929-201406.tar.gz` (NOT submitted yet)
- The tarball's `estimator.py` is byte-identical to `src/estimator_k3q.py` (sha256 `8ebea33e8a544ebb145c553dc87ec02352113ceb5bf28130763b0811e260062c`).
- `--runner subprocess`, 3 MLPs (`k3q_pkg_subproc_n3.json`): **2 of 3 zeroed by that runner's residual cap (0.400, 0.408 s;
  MLP 1 0.361 s, bit-identical).** That runner reads ≈2× the grader (k3m: 0.29–0.38 there → 0.153–0.178 graded), so the
  grader is expected at ≈0.21–0.25 s — but the margin is thin. **Hold the submit until 333029 (k3p) is graded and its
  per-MLP residual telemetry is read:** if k3p's grader max ≲ 0.26 s, submit k3q; if higher, cut calls first
  (`_nonlin` batching; CURRENT STATE).
