# k3z = k3y + Winograd's 15-addition form + level 5 inside the C_pre halves (2026-10-02, afternoon)

The two exact FLOP crumbs left after k3y (`../2026-10-02_csym/RESULTS.md`), folded into `src/estimator_k3z.py`, both ON
by default:
- **winograd** — the Strassen recursion in Winograd's form: the 7 products of (A11, A12, S4, A22, S1, S2, S3) ×
  (B11, B21, B22, T4, T1, T2, T3) and the recombination o22 = m1+m6; o12 = o22+m5; o12 += m3; o22 += m7;
  o21 = o22−m4; o22 += m5; o11 = m1+m2 — 15 additions per level instead of 18;
- **csym = 2** — the two half products of the symmetric C_pre (k3y's csym) at Strassen level 5 (`sw.cap` raised
  around them, only when the level-5 workspace exists).

## Gate (`gate.py`, `gate_bitwise.py`; synthetic 1024×16 and 128×8, seed 11)
| config vs k3y | result |
|---|---|
| winograd=False, csym=1 (both WARM — a cold first predict also bills the workspace construction's reshapes, ≈0.4%) | **bitwise, FLOPs equal** at 1024×16 and 128×8 |
| defaults (winograd + csym=2) | **FLOPs ×0.99397**, max rel move 1.5e-6, finite; one warm predict's wall unchanged within noise |

`whest validate` passes; `scripts/prepkg_check.sh` clean.

## Memory (`rss_k3z.py`, one predict under `/usr/bin/time -l`)
Peak RSS **6.85 GB** (k3y 6.91 GB): the half products reuse the level-5 workspace that already exists.

## 100 MLPs (`runs/k3z_n100.json`, `--no-residual-wall-time-limit`)
(The first attempt was killed at minute 30 by the 30-minute default limit of background Bash tasks — no partial
output; re-run with a 2-hour limit, 16:26 → 17:35.)
| check | result |
|---|---|
| 100 MLPs | adjusted **3.8289e-9**, raw 2.0844e-8, util 18.368%, **0 fails** |
| vs k3y (`../2026-10-02_csym/runs/k3y_n100.json`) | raw ×1.00001 (per-MLP max dev 0.48% — float rounding of a 1e-6 output move against an error of ~3e-4 of the prediction; it averages out), FLOPs ×0.99396, **adjusted ×0.99401, 100/100 better** |
| local wall / residual (contended Mac) | median 36.4 / max 40.6 s (k3y in its run 37.7 / 40.0) · median 0.361 / max 0.476 s (k3y 0.393 / 0.459) |
| cost map, one network (`../2026-10-02_costmap/prof_k3z_n1.json`, namespaced copy bitwise k3z) | util 18.712% (k3y 18.824%), wall 37.8 s (37.9); C_pre's share of backend time 2.4% → 6.2% (its halves now at level 5), every other site slightly faster (Winograd) |
| package | `submission-20261002-123540.tar.gz` sha 9d3c9ae8…, `estimator.py` byte-identical to `src/estimator_k3z.py` (sha 4f48ae56…), `whest validate` + `validate-package` + static scan pass |

**Submitted 2026-10-02 17:36 as 333534.** Projection: 3.7573e-9 × 0.99401 ≈ **3.735e-9** (board: 3.7e-9).

## Graded: **333534 = 3.7357e-9** (projected ≈3.735e-9 — the tenth port in a row on its projection)
raw 2.03169e-8 (k3y 2.03132e-8: ×1.00018, float noise), util 18.328% (k3y 18.44%), 0 failed; residual median 0.310 /
**max 0.331 s** (k3y 0.298 / 0.310: +0.02 s, margin left ≈0.07 s); **wall median 91.2 s, max 98.8 s** (k3y 92.2 / 97.6 —
neutral, as on the laptop). Board: 3.7e-9. The write-up now designates 333534 (`writeup/switch_to_k3z.py`).
