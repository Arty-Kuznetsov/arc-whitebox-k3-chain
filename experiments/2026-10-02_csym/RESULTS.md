# csym — C_pre is symmetric: compute 3 of its 4 blocks (2026-10-02, night; candidate for k3y)

`C_pre = Wᵀ (C W)` is symmetric, but k3x computes the whole product (Strassen level 5 at the `cpre` site). Splitting the
OUTPUT into 2×2 blocks, the upper block row `[O11 O12] = Wᵀ[:h] (C W)` and `O22 = Wᵀ[h:] (C W)[:, h:]` are 3/4 of the
product, and `O21 = O12ᵀ` is a copy. Prototype: `estimator_csym.py` = `src/estimator_k3x.py` + switch `csym`
(0 = off = bitwise k3x; 1 = the split at the default Strassen depth 4; 2 = the split at level 5).

| synthetic 1024×16 (seed 11), vs k3x defaults | FLOPs | max rel move | wall of one predict (local, indicative) |
|---|---|---|---|
| csym = 0 | **bitwise, FLOPs equal** | 0 | — |
| **csym = 1** (two products at level 4) | **×0.99335** (util 18.860% → 18.735%) | 1.1e-6 | 39.1 → 38.2 s, 38.5 → 38.1 s |
| csym = 2 (the same at level 5) | ×0.99113 | 1.3e-6 | 38.5 → 40.3 s |
| 128×8 small config, csym = 1 | ×0.99465, finite | 2.7e-7 | — |

**csym = 1 is free: −0.67% of the FLOPs with no wall cost** (the level-5 `cpre` site it replaces cost wall, the
two level-4 half products do not). csym = 2 buys 0.22% more for +1.8 s local (0.12 %/s — worse than any L5 site
shipped), so not with k3x's wall. The diagonal blocks are symmetric again, but recursing would add Strassen
invocations (≈+100 flopscope calls each per layer) — residual we would rather keep.

## Other exact crumbs found in the same pass (per-op profile of k3r, `experiments/2026-10-01_e3/prof_summary.json`)
matmul 89.2% · add/multiply/subtract 5.4% (Strassen combinations and copies) · linalg.solve 2.25% + linalg.qr 1.93%
(the band entry) · reshape 0.49% (328 calls) · sum 0.37% · take 0.36%.
- **Band entry:** flopscope bills `solve` as LU + getrs (2r³/3 + 2r²·nrhs) although R is triangular, and the reduced
  QR forms Q at the same count again. An explicit R⁻¹ (2r³, once per entry) applied by a Strassen matmul to both
  right-hand sides would save ≈2.5e8 FLOPs per entry ≈ **0.6%** — but R's condition number reaches ~1e5 in float32
  (E3's divergence), so an explicit inverse is a numerical risk to measure first.
- **Winograd's 15-addition form** of the recursion: 21 instead of 22 elementwise ops per level → ≈0.25%.
- **reshape**: 0.49%, many call sites.


## binv — prototyped and folded into k3y (2026-10-02, ≈01:00)
`estimator_binv.py` (k3y + switch `binv`): one explicit `R⁻¹` per band entry (flopscope `inv`, 2r³) applied by the
right-shared Strassen to the entrant's rows, `Zᵀ = ([Bᵀ; Aᵀ] Q) R⁻ᵀ`, and as a plain product to its thin legs, instead
of two `solve` calls (each billed LU 2r³/3 + getrs 2r²·nrhs). Synthetic 1024×16 on top of csym: **FLOPs ×0.99410**, max
rel move 9.9e-7, wall 41.8 → 40.8 s. R's conditioning in float32 is the risk; the 100-MLP run compares every MLP's raw
with k3x's.

## k3y = k3x + csym + binv (`src/estimator_k3y.py`, both ON by default) — gate (`gate_k3y.py`)
| config | 1024×16 vs k3x | 128×8 vs k3x |
|---|---|---|
| csym=False, binv=False | **bitwise, FLOPs equal** | **bitwise, FLOPs equal** |
| csym only | FLOPs ×0.99335, max rel move 1.1e-6 | ×0.99465, 2.7e-7 |
| **defaults (csym + binv)** | **FLOPs ×0.98749** (util 18.86% → 18.62%), max rel move 1.15e-6, finite | ×0.99465 (no band at 128) |

`whest validate` passes; static import scan clean. 100-MLP local run launched ≈01:00 (the first, csym-only run was
stopped at minute 6 to fold binv in).

## Why the band entry cannot drop Q (mode "r" QR + semi-normal equations, which would save ≈0.9% more) — `condP.py`
Condition number of the transported basis P at its successive entries (synthetic 1024×16; the first number is the
range finder's input): seed 11 — 1.0, 11, 35, 98, 292, 657, 2.0e3, 5.5e3, 1.4e4, 3.5e4, **7.3e4**; seed 12 the same to
×1.5. It grows ≈×2.5 per layer (no re-orthonormalisation, by design). Any method that squares it (PᵀP) reaches 5e9 ≫
1/ε₃₂ = 1.7e7 by the last entries: closed. The explicit R⁻¹ of binv is a different matter — a triangular inverse is
far more accurate than cond(R)·ε suggests, and the gate measured its effect on the output at +0.04e-6.

## reshape, located (`reshape_sites.py`, k3y, one fresh predict, synthetic 1024×16)
flopscope bills `reshape` at numel(input) whether numpy returns a view or not. One fresh predict: **0.735% of FLOPs in
352 calls**, but 0.40% of it is the one-off Strassen workspace construction (`_Strassen.__init__`: 0.295% + the leaf
scratch 0.106%), which the cached workspace pays only on a worker's first MLP. **The recurring part is ≈0.33%**:
`rsh` (the 2-D → 3-D views of its operands, 0.119%, 126 calls), the D pairing's GL/GR views (0.04 + 0.02 + 0.02%),
the members' GLm (0.02%), the transport's GX (0.02%), and smaller ones. Removing it means pre-shaped buffers for
data-dependent sizes — ≈0.3% for fiddly changes at ~15 call sites; parked.


## k3y — 100 MLPs and the package (2026-10-02, ≈02:10)
| check | result |
|---|---|
| **100 MLPs** (`runs/k3y_n100.json`, `--no-residual-wall-time-limit`) | adjusted **3.8519e-9**, **0 fails** |
| vs k3x (`../2026-10-02_k3x/runs/k3x_dpair_n100.json`) | raw ×1.00009 (per-MLP max dev 0.28%), FLOPs ×0.98741, **adjusted ×0.98749, 100/100 better** |
| vs k3w | adjusted ×0.97381, 100/100 better |
| local wall (contended machine) | median 37.7 / max 40.0 s (k3x in its run: 38.5 / 46.7) — paired timing: k3y = k3x −0.07 s |
| local residual (contended) | median 0.393 / max 0.459 s (k3x 0.382 / 0.444): csym adds one Strassen invocation per layer, binv an inv + a product per entry → grader ≈0.73× → **≈0.34 s of 0.40** |
| package | `submission-20261001-210547.tar.gz` sha 344d2588…, estimator byte-identical (sha d28b27f4…), `validate-package` ok |

binv is numerically clean on all 100 real networks (the raw ratio's spread is the same 1e-6-output-move noise as
k3x's). **Projection:** ≈3.8576e-9 × 0.97381 ≈ **3.757e-9**. To be submitted once 333398 (k3x) has graded and shown
its wall (k3y adds none).

## Graded: **333402 = 3.7573e-9** (projected ≈3.76e-9 — the ninth port in a row on its projection)
raw 2.03132e-8 (= k3w's), util 18.44%, 0 failed; residual median 0.298 / **max 0.310 s** (+0.03 s over k3x: the extra
calls of csym's second product and binv); **wall median 92.2 s, max 97.6 s** (k3x 90.7 / 95.2). Margins left: ≈22 s of
wall, ≈0.09 s of residual.
