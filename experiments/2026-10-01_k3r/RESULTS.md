# k3r — marg2 (the Gaussian second-order pair coefficient) and the honest Strassen L5 test

**Date:** 2026-10-01 (Mac). **Estimator:** `src/estimator_k3r.py` = k3q + two switches.
**Source of marg2:** cloud session #9A (`experiments/2026-09-30_tail-accuracy/RESULTS.md` §2c/§4) — the only lever the
149-arm tail screen produced, and it is not a tail lever.

## 1. marg2 — what it is

coef(k, p) = w(k,p) + w(k+3,p)·t/6 + w(k+4,p)·K4d/24 are the marginally corrected pair Wick coefficients. marg2 keeps
the k = 1 correction always, and the **k = 2** correction only where α = μ/σ ≤ −2.0 (the dead tail); elsewhere
coef(2, p) = the pure Gaussian w(2, p), **in Q11, Q21 and Q22 alike**. The consistency is the whole point: the cloud
measured Gaussian-in-Q11-only at ×1.09 and Q21-only at ×1.11 (*worse*), all three together ×0.984 — K21 = Q21 − 2 P1 Q11
is a difference of two large terms and the k = 2 coefficient must be the same in both. The E2 level restoration keeps
its **fully corrected** coef(2,2), exactly as in the cloud's arms. O(n), **zero products — it removes terms.**

## 2. Self-checks

| check | result |
|---|---|
| k3r(`marg2=None, strassen_min=128`) vs k3q, 1024×16 | **bitwise, FLOPs equal** (438744688353 both) |
| the same at 128×8 (thin=128, no band) | **bitwise, FLOPs equal** |
| `ablate.py k4memo 3` after the `chain.py mcorr` edit | 3.7533e-08 ✓ |

⚠️ **The first version was NOT bitwise** (max|Δ| 7.2e-6 at 1024×16) because I had regrouped the coefficient sum as
`w + (c₅·t/6 + c₆·K4d/24)` instead of the original left-to-right accumulation — a float32 last-bit difference,
amplified over 16 layers. The fix keeps the original accumulation and selects with `c·keep + w·drop`, which is exact
on **both** sides (×1.0 and ×0.0 are exact, and adding 0.0 is exact).

## 3. Measurements on top of k3q

### 10-MLP screen (`runs/screen.sh`)
| arm | raw | × k3q | util | better | adjusted |
|---|---|---|---|---|---|
| k3q | 2.2512e-8 | 1 | 19.7248% | — | 4.4394e-9 |
| **marg2 = −2.0** | **2.2074e-8** | **0.9805** | 19.7248% (×1.00000) | **9/10** | **4.3527e-9** |
| marg2 everywhere | 2.2127e-8 | 0.9829 | 19.7248% | 9/10 | 4.3632e-9 |
| k3q re-run (`strassen_levels=4` wrapper) | 2.2512e-8 | 1.0000 | 19.7248% | 0/10 | 4.4394e-9 (identical — runner self-check) |

The cloud's torch value for the same arm was ×0.9811 on its screen set, so the production port reproduces it to 0.06%.

### 100 MLPs (`runs/run100_src.sh` → `runs/k3r_n100.json`), the packaged defaults
| | adjusted | raw | util | fails | residual mean/max (local) |
|---|---|---|---|---|---|
| k3q | 4.2391e-9 | 2.1479e-8 | 19.7352% | 0 | 0.282 / 0.336 s |
| **k3r** | **4.1702e-9** | 2.1130e-8 | 19.7358% | 0 | **0.280 / 0.296 s** |

**raw ×0.9838, 91/100 better, util ×1.00003.** The 100-MLP ratio lands on the cloud's 32-fresh-MLP value (×0.9842)
rather than on the 10-MLP screen (×0.9805) — as it should, the screen being the smaller sample.
**Expected on the grader ≈4.06e-9** (k3q local 4.2391e-9 → graded 4.1279e-9, ratio 0.9738).
**Submitted as 333307** (`submission-20261001-081824.tar.gz`, estimator byte-identical to `src/estimator_k3r.py`,
sha `fe043bdf…`, `validate-package` ✓).

## 4. Strassen L5 — measured and closed (NEGATIVE §69)

`strassen_levels=5` alone had been a **no-op** since k3h: `_Strassen.__init__` breaks the recursion at `m < 128`, so
the leaf stays 64×64 whatever the level count says. L4 and L5 wrappers on 3 MLPs: **bit-identical** (FLOPs 4.4074e11,
raw 2.2594e-8, 19375 calls, all equal). The honest test needs the new `strassen_min` kwarg.

With `strassen_min=64` (leaf 32×32, 7⁵ = 16807 groups), 10 MLPs on top of k3q:

| | util | raw | adjusted | residual mean/max | **wall mean/max** | peak RSS |
|---|---|---|---|---|---|---|
| k3q (L4) | 19.7248% | 2.2512e-8 | 4.4394e-9 | 0.361 / 0.393 s | **21.2 / 21.7 s** | 4.5 GB |
| L5 | **18.5197%** (×0.9389) | ×0.9991 | **×0.9380** | 0.367 / 0.385 s | **45.9 / 46.9 s (×2.2)** | **7.21 GB** |

**Residual is fine; wall time and memory are not.** The grader runs at ≈3.0× the local wall (k3q 21.7 s local → 63.7 s
graded on 333276), so L5 projects to ≈140 s against the **120 s cap**, and 7.21 GB against the **8 GB** cap. A 32×32
leaf has no BLAS efficiency left: ×2.2 wall bought ×0.936 FLOPs. It would have been the largest single remaining
lever (4.13e-9 → 3.87e-9) and **it cannot ship**. `strassen_min` stays at 128.

Side measurement worth keeping: the Mac's residual reading of the *same* config varies **0.309–0.377 s** between runs
(3 MLPs, L4) — ±35 ms of noise. Only grader telemetry is a real residual measurement.

## Files
`runs/` — `screen.sh`, `run100_src.sh`, `queue2.sh`, `queue3.sh`, the wrappers `v_k3r_m2_m20.py`, `v_k3r_m2_all.py`,
`v_k3r_L5real.py`, `v_k3q_L4.py`, `v_k3q_L5.py`, and every JSON. `k3r_check.py` — the bitwise/off and L5 checks.
`peers/` — the leaders' per-MLP data behind `notes/INTEL-peers-2026-10-01.md`.
