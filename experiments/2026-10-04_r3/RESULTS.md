# R3, step 1 — the young-κ₄ carrier at the challenge's 1024 × 16 (2026-10-04, evening)

`r3full.py`: float64 research chain, research base B9 + marg2 (as R1/R2 and the 10-03 last-layer runs), the carrier
`chain.py k4young impl="legs"` (`young4.py`, gated against ARC's factored objects to 1e-14 — `../2026-10-04_r1/
test_young_legs.py`), baked MLPs 0–9 against the baked ground truth. × of the base's final-layer MSE. (Numbers from the run
logs, `logs/all_lines.txt`: four processes wrote the same per-MLP JSONs and some arms were lost to the write race —
use one file per (MLP, arm) next time.)

## Accuracy
| arm | what is read | pooled × | per MLP |
|---|---|---|---|
| **y2** | K4d + K31, ages ≤ 2, all trees | **×0.701 (10/10)** | 0.697 0.770 0.681 0.775 0.649 0.726 0.699 0.615 0.691 0.701 |
| y2c31 | K31 without the 3+2's expensive term | ×0.741 (3/3) | 0.754 0.756 0.711 |
| y2n31 | K31 from paths and stars only | ×0.800 (3/3) | 0.795 0.803 0.801 |
| y2K4 | K4d only | ×0.824 (10/10) | 0.835 0.862 0.807 0.900 0.740 0.933 0.817 0.752 0.816 0.793 |
| y2K4pd | K4d only, the path's coinciding-hub piece (O(n²)) | ×0.863 (3/3) | 0.872 0.881 0.835 |
| y2K4-32 | K4d only, the 3+2 alone | ×0.954 (2/3) | 0.911 1.017 0.932 |
| y1 | K4d + K31, age 1 only | ×0.815 (10/10) | 0.858 0.818 0.818 0.915 0.774 0.820 0.815 0.765 0.829 0.754 |
| y1c31 | age 1, cheap K31 | ×0.846 (3/3) | 0.888 0.805 0.845 |
| y1K4 | age 1, K4d only | ×0.892 (10/10) | 0.921 0.879 0.890 0.960 0.844 0.963 0.894 0.863 0.893 0.830 |
| y1K4pd | age 1, K4d only, path diagonal | ×0.910 (3/3) | 0.941 0.883 0.905 |

With the production band (bandz a0 5, r 448, rr; base with band ×1.016 of without): **by2 ×0.723, by2c31 ×0.747,
by2K4 ×0.839** (MLPs 0–2) — the gain survives the production accuracy configuration.

**The accuracy question is answered: the young κ₄ trees are worth ×0.70 of our raw error at full width, on every
network.** Applied to the shipped k3z (raw 2.03e-8) that is ≈1.42e-8 — the leaders' raw (marius 1.14e-8, J2W 1.50e-8).

## Cost (an ESTIMATE — METHOD-LESSONS rule 3: to be measured on the meter before any port decision is final)
Our whole chain counts ≈ 12.5 n³-matmul equivalents per layer (k3z: 4.03e11 FLOPs / 2.15e9 per 1024³ product / 15).
The carrier's readers, per young generation and layer: path K4d x G (1 n³; the coinciding-hub piece n²), star n²,
3+2 K4d needs the cross-layer D pairing Dx(a, r) = κ₃(z_r at birth; z_a, z_a now) — computing it as an extra pairing is
≈ 2.5 n³-eq, but restructuring the existing D pairing to use the single legs BEFORE transport gives Dx for free and
D(l) = Dx V for one n³ product per age; K31 grouped as [·] U~ᵀ + [·] Tᵀ = 2 n³ per generation. With Strassen L4 (×0.59):
| variant | n³ / layer | ≈ FLOPs | raw | ≈ adjusted |
|---|---|---|---|---|
| y2K4pd (restructured pairing) | 2 | +9% | ×0.863 | **×0.94** (3.7357e-9 → ≈3.53e-9, displayed 3.5e-9) |
| y1K4pd (age 1 only; the existing double leg buffer suffices) | 1 | +5% | ×0.910 | ×0.95 |
| y2K4 | 4 | +19% | ×0.824 | ×0.98 |
| y1 (K4d + K31, age 1) | 4 | +19% | ×0.815 | ×0.97 |
| y2c31 | 8 | +37% | ×0.741 | ×1.02 |
The carrier's value is roughly its price in our representation: a display step (3.7 → 3.5e-9), not a rank jump. The
leaders reach this raw at 0.58–0.79 of OUR FLOPs — their base representation is cheaper than ours.
**Superseded by step 2 below (metered):** the estimate missed the pairing the 3+2 needs at the last layer (+4.27% per
age); the best metered projection is ×0.972 (3.63e-9).

Production constraints: k3z peak RSS 6.85 of 8 GB, graded wall max 98.8 of 120 s; the D pairing has four code paths
(satrow / deadcut / Strassen L5 / band members compressed through P) that the restructuring must all follow.


## Step 2 — the production form, metered; the kill rule stops the port (2026-10-04, evening)
**The production form of the reader** (`young4.Young4Chain`, `chain.py k4young impl="chain"`): only what k3z already
holds — T, U~ from the legs of the chain's own source born at m (its column cut included), Dx from the chain's own D
pairing terms with the single legs at the birth layer (the restructured pairing). K4d only. n = 128: equal to the
exact reader (×0.672 / 0.738 / 0.729 vs ×0.690 / 0.740 / 0.736). At 1024 × 16 with the band, 10 MLPs:
| variant | raw (× of the band base) | FLOPs, METERED | projected score (× 3.7357e-9) |
|---|---|---|---|
| 3+2 ages 1–2, path diagonal + star ages 1–2 (bCy2K4pd) | ×0.848 (10/10) | +14.79% | ×0.973 → 3.64e-9 |
| 3+2 age 1 only (bCy2K4pd32a1) | ×0.905 (10/10) | +7.40% | ×0.972 → 3.63e-9 |
| … without the 3+2 at the last layer (nl) | ×0.886 / ×0.942 | — | worse |
| 3+2 at the last layer only, ages 1–2 / age 1 | ×1.016 / ×1.023 (3/10) | +8.54% / +4.27% | worse than k3z |
| path diagonal + star only, no 3+2 | ×1.047 (1/10) | ≈0 | worse than k3z |
| full path quadratic form (bCy2K4) | ×0.834 | ≈+24% (est.) | ≈×1.03 |

**Meter** (`meter/`): `estimator_k4meter.py` = k3z with the extra products inserted on zero buffers of their exact
shapes (outputs bitwise k3z): one D = Dx·W product per 3+2 age at every layer with a D pairing (≈0.22% each) and the
old sources' pairing at the LAST layer, where k3z pairs nothing (+4.27% per age) — the last-layer 3+2 cannot be
dropped (×0.848 → ×0.886) and cannot stand alone (×1.02). Synthetic 1024 × 16, k3z = 4.070984e11 FLOPs.

**Verdict: the kill rule (metered FLOPs ≤ +10% AND projection < 3.55e-9) fails for every variant — best 3.63e-9
(displayed 3.6e-9).** The port is not started. The young trees are where the leaders' accuracy is, but in our
representation the pairing they need costs as much as they buy; the consistency principle again decides the details
(the last layer's 3+2 pays only with the upstream ones).

## Step 3 — the variant step 2 never priced (2026-10-06): the 3+2 upstream only, no last-layer pairing
Step 2's table marked "without the 3+2 at the last layer (nl)" as *worse* on raw alone and left its FLOPs unmetered.
Priced today with the same meter (`meter/meter.py nolast`: the D = Dx·V recovery products on zero buffers, no last-layer
pairing; outputs bitwise k3z):

| variant (band base, 10 MLPs, pooled raw ×, better) | FLOPs, METERED | projected score (× 3.7357e-9) |
|---|---|---|
| 3+2 ages 1–2 upstream only + pathdiag + star (bCy2K4pdnl) | ×0.879 (10/10; per MLP 0.82–0.96) | **+6.25%** | **×0.934 → 3.49e-9** (display 3.5e-9, rank ≈ 29) |
| 3+2 age 1 upstream only (bCy2K4pd32a1nl) | ×0.928 (9/10) | +3.12% | ×0.957 → 3.58e-9 |
| (step 2) 3+2 age 1 with the last-layer pairing | ×0.905 (10/10) | +7.40% | ×0.972 → 3.63e-9 |

**The kill rule (metered FLOPs ≤ +10% and projection < 3.55e-9) passes for the first row.** R3's conclusion "every cheaper
variant is worse than k3z" was wrong for this row: it compared raw error, not the metric (METHOD-LESSONS rule 3 applies
to the comparison as much as to the pricing). Port target: k4a = k3z + the restructured D pairing (pair the current L
factors with the legs snapshotted two layers back; Dx = Dx2·Wr(l−1), D = Dx·Wr(l); members through P(l−2)) + the
Young4Chain reader (pathdiag, star, 3+2 for ages 1–2; no 3+2 at the last layer). Risks to control: residual time
(k3z 0.331 s max of 0.400 on the grader; +28 Strassen products per network add calls), wall (98.8 s of 120), RSS
(6.85 of 8 GB). The first graded k4a is a probe, not a designation.
