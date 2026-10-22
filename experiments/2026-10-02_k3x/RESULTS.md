# k3x — Strassen level 5 on the D pairing (the fifth L5 site), shipped by the wall rule (2026-10-02)

`src/estimator_k3x.py` = k3w + "dpair" in the default level-5 site list (the non-members' D pairing; E2's saturated-row
block then starts at a multiple of 32). The rule set on 10-01: ship only if k3w's graded wall max ≤ 95 s.
**333388 (k3w) graded 3.8576e-9 with wall max 92.2 s (median 84.2 s) → the rule fired.**

| check | result |
|---|---|
| static import scan · `whest validate` | clean · passes |
| gate (`gate.py`): k3x with k3w's sites vs k3w, 1024×16 and 128×8 | **bitwise, FLOPs equal** |
| gate: k3x defaults vs k3w, 1024×16 | FLOPs ×0.98547, util 18.86%, max rel move 1.15e-6, finite |
| **100 MLPs** (`--no-residual-wall-time-limit`, `runs/k3x_dpair_n100.json`; wrapper with the same kwargs) | adjusted **3.9007e-9**, **0 fails**; vs k3w: raw ×1.00007 (per-MLP max dev 0.36%: 1e-6 output moves on a 2e-8 MSE), FLOPs ×0.98611, **adjusted ×0.98614, 100/100 better** |
| local wall / residual in that run | median 38.5 / max 46.7 s; residual median 0.382 / max 0.444 s — **inflated by contention** (a hidden browser tab at ~2 cores, jvloo, the k3y run) |
| **paired wall timing** (`walltime.py`, alternating on one MLP, warm) | k3w 34.65 s, **k3x 38.36 s (+3.71 s)**, k3y 38.29 s |
| package | `submission-20261001-193115.tar.gz` sha bb7e0603…, estimator byte-identical (sha 03887a16…), `validate-package` ok |

**Grader projection.** k3w added +3.6 s locally and +12.7 s on the grader's median (×3.5 — L5 sites cost the grader
more than the generic ×2.8). So k3x: median ≈97 s, **max ≈103–106 s of 120**; residual ≈0.3 s; adjusted
≈3.8576e-9 × 0.9861 ≈ **3.80e-9**. **Submitted as 333398** (2026-10-02 ≈01:33 +05).

## Graded: **333398 = 3.8069e-9** (projected ≈3.80e-9 — the eighth port in a row on its projection)
raw 2.03156e-8 (×1.00013 of k3w), util 18.68%, 0 failed; residual median 0.272 / max 0.281 s; **wall median 90.7 s,
max 95.2 s** (p90 92.9) — 25 s under the cap. The grader charged the dpair site +6.4 s on the median for +3.7 s
locally (×1.7, against ×3.5 for band_members): per-site grader/local wall ratios are not constant, so the next rule
should be set on the measured graded wall, not on a ratio.
