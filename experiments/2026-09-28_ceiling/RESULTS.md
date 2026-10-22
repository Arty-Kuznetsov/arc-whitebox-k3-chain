# If cost became free, how accurate can this chain get? — the ceiling of the merge→floor plan

**Date:** 2026-09-28, Mac. **Scripts:** `ceiling.py` (12-arm screen, 3 MLPs), `combo.py` (7 arms, 10 MLPs).
**Base:** k3l_t torch config, 10 baked MLPs = **2.3135e-08** (self-check printed by the base arm of both runs).

## Why this measurement

The score is `raw × max(0.1, C/B)`. The multiplier **floors at 0.1**, so if a fixed-size κ₃ state ever puts us
at or below 10% of budget, every further FLOP is free on the score and the right thing to do with it is **buy
accuracy back**. The ceiling of that whole plan is therefore `raw_best × 0.10`, and `raw_best` is what we have
never measured — every accuracy lever below was refused *on cost grounds*, never on accuracy grounds.

## The stack (10 MLPs, all measured)

| arm | raw | ×base | better than base | `raw × 0.10` |
|---|---|---|---|---|
| base (k3m's config in torch) | 2.3135e-8 | 1.0000 | — | 2.314e-9 |
| knobs (`feedmult 0.8`, `k22c 0.0`) | 2.2931e-8 | 1.0089 | **9/10** | 2.293e-9 |
| `thin192` (correction leg rank 24 → 192) | 2.2784e-8 | 1.0154 | 8/10 | 2.278e-9 |
| **`allages` (path births at every age)** | **2.1403e-8** | **1.0809** | **9/10** | 2.140e-9 |
| knobs + allages | 2.1208e-8 | 1.0909 | 9/10 | 2.121e-9 |
| knobs + allages + thin192 | 2.0897e-8 | 1.1071 | 9/10 | 2.090e-9 |
| **knobs + allages + thin192 + `deadcut` off** | **2.0844e-8** | **1.1099** | **9/10** | **2.084e-9** |

**The free-cost accuracy stack is ×1.11, and it improves 9 of 10 MLPs** — which also matters under grader
schema 2.0 (the ranked number is a 50-MLP mean, so breadth beats a mean won by a few MLPs).

Converted to the grader through the measured flopscope/torch ratio (k3l: graded raw 2.1293e-8 / torch 2.3135e-8
= **0.920**): the stack would land at **raw ≈ 1.92e-8**, so **at the 10% floor the score is ≈1.92e-9**.
Board today: #1 J2W 1.80e-9, #2 Luna 1.90e-9. **That is rank 2–3 with no new accuracy idea at all.**
Conservatively, dropping `allages` (the one lever whose cost does not obviously survive a merge) still gives
×1.026 → raw 2.11e-8 → **2.11e-9, rank 4–5.**

## Per-lever notes

- **`allages` (`e3age=None`) is the surprise: ×1.081 on 10 MLPs**, against the ×1.04 in the standing notes and
  ×1.026 on the 3-MLP screen. This is the fourth time in this project that a 3-MLP screen has mis-ranked a
  5–10% effect (NEGATIVE §51); 10 MLPs is the minimum for anything in this size class. It is also the *only*
  lever in the stack whose production cost is large (a path leg per source per layer ≈ +12.8% of budget in the
  current unmerged chain), so it is the one that has to be re-costed inside whatever merged representation wins.
- **`thin192` ×1.0154, saturating.** The return on the correction-leg rank is 24 → 48 → 96 → 192 =
  ×1.0135, ×1.0183, ×1.0231 on 3 MLPs. So the randomized rank-24 split, which looked like the biggest
  truncation in the chain, costs under 2%. The thin leg is O(n·kk) — this is nearly free even today.
- **`deadcut` off is worth only ×1.002–1.005**, confirming the shipped `deadcut=-2.5` trade.
- **The knob winners** come from `2026-09-26_coherent-knobs/` (scan + refine, 10 MLPs): `feedmult` has a flat
  optimum at 0.75–0.85 (0.8 → ×1.0058 alone), `k22c` improves monotonically down to 0.0 (×1.0032 alone),
  `sdmult 0.75` and `k4scale 0.85` are inside the noise and do not compose. **`feedmult 0.8` + `k22c 0.0` =
  ×1.0089 on 9/10 and costs zero FLOPs** — this is bankable today, independently of everything else.

## ⚠️ One arm in `ceiling_n3.json` must NOT be read as a measurement

`corr="full"` ("EXACT (2,1) correction") reported raw 2.5607e-8, i.e. *worse* than the rank-24 split, and
two different configurations (`EXACT+cutoff+knobs` and the same `+allages`) returned **identical** values in a
*shorter* runtime. `chain.py` takes a separate branch at `corr == "full"` in which our path/age switches are
inactive, so those arms measure a different algorithm, not an exact correction. `thin192` is the honest proxy,
and its saturation says the exact correction is worth ≈×1.02, not more.

## What is measured vs assumed

Measured: every number in the tables (torch float64, baked MLPs, both self-checks passing). Assumed: (a) the
0.920 flopscope/torch ratio transfers to these configurations — it is calibrated on k3l only; (b) that a merged
κ₃ representation can still host `allages` cheaply — **not** established, which is why the conservative row is
quoted alongside.
