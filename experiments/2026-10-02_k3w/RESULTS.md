# k3w — Strassen L5 on a fourth site (the band members' materialisation), 2026-10-01 night

`src/estimator_k3w.py` = k3v + "band_members" in the default level-5 site list (ZS @ Pᵀ: the rank-448 contraction
divides by 32; rows in 32-blocks). The κ₄ corrector is k3v's (10-MLP model): a 30-MLP retrain left the held-out R²
unchanged (0.353 vs 0.354), so it was not swapped.

| check | result |
|---|---|
| k3w with k3v's site list vs k3v (1024×16, 128×8) | bitwise, FLOPs equal |
| `whest validate` · `scripts/prepkg_check.sh` | pass · clean |
| 1024×16 synthetic, band_members L5 | FLOPs ×0.98680, max rel move 9.8e-7 |
| **100 MLPs** (`--no-residual-wall-time-limit`) | adjusted 3.9556e-9, **0 fails**; vs k3v on 93 clean MLPs: raw ×0.99994, FLOPs ×0.98649, **adjusted ×0.98643** |
| residual median/p90/max (local, not enforced) | 0.325 / 0.366 / 0.392 s |
| wall median/max (local) | 32.3 / 34.5 s → ≈98 s on the grader (k3v: local 28.7 → graded 81.2, ×2.83) |

**Submitted as 333388** (`submission-20261001-185517.tar.gz`, sha 9999b78c…, full checklist). Expected ≈3.86e-9.

## Next: k3x (`src/estimator_k3x.py`)
+ "dpair" (the non-members' D pairing at level 5; E2's saturated-row block then starts at a multiple of 32): FLOPs
×0.98547 on top of k3w, max rel move 1.2e-6, defaults currently = k3w (bitwise). Estimated graded wall ≈110 s of 120.
**Decision rule: submit only if 333388's graded wall max ≤ 95 s.** Before submitting, set the default site list to
include "dpair", run the full pre-package checklist and a 100-MLP run with `--no-residual-wall-time-limit`.
