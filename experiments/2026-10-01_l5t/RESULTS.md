# k3t — Strassen level 5 on selected contractions only (2026-10-01)

**Why this exists.** Full L5 (every contraction) is dead on wall time ×2.2 and RSS 7.21 GB (NEGATIVE §69). The
per-call-site profile (`../2026-10-01_e3/L5_SELECTIVE.md`) measured, for each contraction, the FLOPs L5 saves and the
backend time it adds. Ranked by efficiency (% of total FLOPs saved per second of backend added): **transport 0.49,
ctransport 0.46, cpre 0.39**, then dpair 0.27, band_members 0.26, band_entry 0.10, the members' compressed pairing
0.03. So L5 goes only where it pays per second.

## Implementation (`src/estimator_k3t.py`)
- `_Strassen` gains a **per-call depth cap** (`self.cap`): the three recursions stop at `self.cap` instead of
  `len(self.bufs)`; the sum-leaf uses a scratch view built **only for the depths actually used** (L and the default
  depth) — building views for unused depths costs FLOPs because flopscope bills `reshape`, and that is exactly what
  broke the first version of the gate (output bitwise, FLOPs not equal).
- The instance is built with 5 levels and a 64 floor; the default cap is 4; each listed site raises it to 5 for its one
  call. The alive set is padded to 32 (so the transport's contracted axis divides by 2⁵); the extra padded neurons
  are dead with exactly zero leg columns, so only float32 rounding changes. The level-5 transports use 32-row chunks.
- `transport_l5=False` is **bitwise k3r with equal FLOPs** (1024×16 and 128×8).

## Measurements
| run | FLOPs / util | raw | adjusted | residual mean/max | wall | fails |
|---|---|---|---|---|---|---|
| 1024×16 synthetic, transport only | ×0.98251 | max rel move 1.2e-6 | — | 0.416→0.501 (contended) | +18% | — |
| 10 MLPs, **contended** (oracle bake running) | — | — | — | 0.393/0.407 and 0.406/0.437 | — | **4 and 7 residual breaches** |
| 10 MLPs clean, transport only | util ×0.98500 | ×0.9999 (6/10) | **×0.9849** | 0.331 / 0.350 | 24.2 / 25.5 s | 0 |
| **10 MLPs clean, transport + ctransport + cpre** | util ×0.97406 | ×0.9998 (7/10) | **×0.9738** | 0.330 / 0.342 | 26.5 / 27.0 s | 0 |

**The contended run is the warning to keep:** with the oracle bake running alongside, the local residual of the very
same estimator rose by ~20% and the local runner zeroed 4 and 7 of 10 MLPs. Residual is a property of the machine's
load, not only of the code — every timing decision for this project must be read on a quiet machine, and the grader's
own telemetry is the only real measurement.

**Expected on the grader:** residual ≈0.342 × 0.886 (the worst grader/local ratio seen, on k3r) ≈ **0.30 s** of 0.4;
wall ≈ 69.7 s × (27.0 / 22.3) ≈ **84 s** of 120; RSS ≈ 6.1 GB of 8; **adjusted ≈ 4.0579e-9 × 0.974 ≈ 3.95e-9.**

## 100 MLPs — `runs/k3t_n100.json`
adjusted **4.0623e-9** vs k3r 4.1702e-9 (**×0.9741**), raw ×0.9998 (58/100, float32 noise), util 19.7358% → 19.2286%,
**0 failures**, residual 0.311 / 0.365 s, wall 27.2 / 28.7 s.

## Graded — **333357 = 3.9554e-9** (projected 3.95e-9)
raw 2.0559e-8 (k3r: 2.0560e-8 — L5 is exact), 0 failed, **residual max 0.2676 s** (local 0.365 → ratio 0.733, back to
the usual 0.72–0.75), **wall max 82.1 s** of 120 (projected 84). Fifth port in a row on its projection.

⚠️ **Latent defect found after shipping:** the shipped file fails `whest validate` — with `transport_l5=True` by
default, `_Strassen.__init__` builds a leaf scratch for depth 4 from a region that does not exist when the width
allows fewer than 4 levels (the tiny contract-check MLP). The grader's smoke test uses a full-size MLP, so 333357
grades fine; the fix (`2 <= Ld <= L`, and resets to the default depth) is in `src/estimator_k3v.py`. Lesson: run
`whest validate`, not only `whest validate-package`, before every package.
