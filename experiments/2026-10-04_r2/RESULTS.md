# R2 — one joint carrier: the young κ₄ trees, built from the chain's own state (2026-10-04)

Plan: `notes/PLAN-2026-10-04.md` (R2, kill rule: ×0.85 or better on BOTH nets at n = 256). Target from R1
(`../2026-10-04_r1/RESULTS.md`): the young (≤ 2 layers) κ₄ trees — path, star and above all the 3+2 — read into K4d and
K31 on top of the chain's own closure.

## The carrier (`chain.py` `k4young`; `young4.py`)
At every ReLU m the chain builds, from its OWN state, the κ₄ trees ARC's k = 4 would build there (factor_k4.py, SIMPLE
budget 3): path 12 d1 C d2 C d2 C d1, star 4 d1 C d3 (C d1)², 3+2 12 d1 C d2 κ₃(j,k,l) d1 d1 — C the chain's
pre-activation covariance, κ₃ the chain's sources. They ride the linear steps and the gates for `ages` layers and are read
into the pre-activation K4d and K31, minus their least-squares fit on the closure's pattern (3 var², 3 var_i C0_ij).
Two implementations, identical to 1e-14 (`../2026-10-04_r1/test_young_legs.py`): `impl="arc"` (ARC's FactoredTensor4,
small widths) and `impl="legs"` (dense n × w legs: T, U~ = T D1 C, the hub coupling G = 12 D2 C D2, frozen κ₃ legs on the
shared vertex; full width). The tree builder itself matches ARC's tree blocks to 1e-16 (`test_trees.py`).
Conventions measured: ARC's (full C edge, full κ₃ incl. repeated entries) **×0.566** vs "pure" (C0, all-distinct κ₃)
×0.923 on n = 128 net 0 — the trees' repeated-index content matters; ARC's is the default.

## Results (`r2young.py`; × of our chain's final MSE, MC noise subtracted)
| arm | n=128 net 0 / 1 / 2 | n=256 net 0 / 1 / 2 / 3 |
|---|---|---|
| **y2** (ages ≤ 2, K4d + K31) | **0.566 / 0.762 / 0.689** | **0.664 / 0.668 / 0.622 / 0.941** |
| y2K4only | 0.686 / 0.696 / 0.653 | 0.763 / 0.723 / 0.749 / 1.007 |
| y2K31only | 0.785 / 1.065 / 0.878 | — |
| y1 (age 1 only) | 0.578 / 1.007 / 0.624 | 0.789 / 0.798 / 0.709 / 0.977 |
| y3 | 0.527 / 0.753 / 0.789 | 0.696 / 0.656 / 0.590 / 0.948 |
| y2 without the 3+2 | 1.146 / 1.562 / 1.303 | — |
| y2 without the closure-pattern subtraction | 2.146 / 2.937 / 4.061 | — |
| y2, 3+2 coupled to κ₃ sources of age ≤ 1 / ≤ 2 only | 0.892 / 1.153 / 1.113 ; 0.645 / 0.888 / 0.875 | ≤ 2: 0.702 / 0.746 / 0.820 / 0.922 |

- **Kill rule R2: PASSED** (n = 256: ×0.664 and ×0.668 on the two nets of the plan; 3 of 4 nets ×0.62–0.67; net 3 ×0.94,
  the net whose exact K4+K31 is also weakest, ×0.79 in R1's factorial).
- The first carrier in this project that improves CONSISTENTLY across networks — every earlier κ₄ carrier (§74) was a
  coin toss. The chain's own state is good enough to build the trees: on n = 128 net 0 the chain-built carrier gives
  ×0.566 where k = 4's own young trees gave ×0.553.
- Necessary pieces: the 3+2 (without it worse than the base), the subtraction of the closure-like part (else the level
  is counted twice), and the 3+2 needs most of the κ₃ (coupling to the youngest source only loses the gain) — the
  production cost question; its K4d reader is a cross-layer D pairing (`young4.py` docstring).

Continued at full width in `../2026-10-04_r3/`.
