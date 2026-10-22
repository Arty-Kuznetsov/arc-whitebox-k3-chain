# mpair — Strassen level 5 on the band members' compressed D pairing: closed on wall (2026-10-02, night)

The last large contraction still at level 4 (≈7% of the FLOPs, inside the unnamespaced block): the members' D pairing
in the compressed space, `Ym = GLmᵀ ZS` over the 2·om member hub rows (n_r × r) and `Ym @ P̃ᵀ` (n_r × npc), r = 448.
Prototype `estimator_mpair.py` = k3y + site "mpair" (chunk rows rounded to 32, cap 5 around both products).

| synthetic 1024×16, vs k3y | result |
|---|---|
| mpair off | **bitwise, FLOPs equal** |
| mpair on | FLOPs **×0.99278**, max rel move 9.3e-7, finite |
| **paired wall** (alternating, warm, 3 reps) | k3y 37.92 s → mpair **45.56 s (+7.65 s)** |

**0.094 % of FLOPs per second of wall** — a fifth of the worst shipped L5 site (dpair 0.27 %/s, band members 0.26 %/s).
Both products have r = 448 as one dimension, so a level-5 leaf has a 14-wide side: BLAS efficiency is gone. On the grader
that is +13–27 s on k3y's 95 s maximum, i.e. at or over the 120 s cap. **Closed.** With k3x/k3y every contraction that
can afford level 5 has it; the wall budget is spent.
