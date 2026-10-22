# The strided / jumped old tier — closed for every age that matters (2026-09-29, Mac)

**The idea** (the ideation workflow's completeness critic, gap 1, the one class all seven lenses missed): legs are
moved one layer at a time only because every source is *read* at every layer. All W are known in advance, so the
old tier (age ≥ a0) can be read exactly only at checkpoints and **jumped** k layers through one shared operator
J = M_{c+k}⋯M_{c+1} (k−1 products per jump, shared by every old source). Between checkpoints the old tier's (t, D)
come from the exact repeated-index rule (state-closure §1, i.e. everything except the all-distinct term), and that
error is **reset** at each checkpoint — so it lives at most k−1 steps. Projected util: (4,2) ×0.79, (4,3) ×0.72,
(3,3) ×0.65. It also looked like the way to soften the cloud's MST tolerance 5–10×.

**Switch:** `chain.py stride=(a0, k, mode)` (default None). Modes: `mst` — old legs not materialised between
checkpoints, t_old and D_old from the rule; `Donly` — legs still move every layer (t_old exact, O(n²)), only the
old tier's D *pairings* skipped between checkpoints. Sources that just reached age a0 are always added exactly.
**Self-check:** `stride=(4, 1)` (every layer a checkpoint) reproduces the base to **2.2e-15** relative.

## MLP 0 (1 thread) — already decisive

| stride (a0, k, mode) | raw × base |
|---|---|
| (4, 2, mst) | **×3.51** |
| (4, 2, Donly) | ×2.49 |
| (6, 2, Donly) | ×1.49 |
| (8, 2, Donly) | ×1.18 |
| (8, 2, mst) | ×1.22 |
| **(10, 2, mst)** | **×0.9945** |

**Losing the old tier's all-distinct part for a single un-reset layer costs ×1.5–×3.5 for every a0 below ~10.**
Only content older than ~10 layers has become slice-concentrated enough for the rule to carry it — and that tier is
only ≈17.5% of source-layers (1+2+…+6 of 120), so striding it is worth a few percent of utilisation at most.
The reset does not rescue the idea because the loss per un-reset step is already ×1.5+ at a0 ≤ 6: the tolerance
problem is per step, not cumulative.

## What it says

1. **It sharpens §62 in the direction that matters for cost:** the all-distinct content is load-bearing not only in
   aggregate (28.8%, ×25.6 end-to-end) but **per source, for every source younger than ~10 layers, even for one
   step.** The "fresh content is less slice-concentrated" hint was right, but the crossover sits at age ≈10, far
   older than any useful a0.
2. **It predicts the cloud's Prompt A (MST, a0 = 2–5, no resets) will be far worse than ×1.05.** MST at a0 ≤ 5
   loses the same term at every layer without reset; `(4,2,mst)` losing it on every other layer is already ×3.5.
3. **For the leaders' puzzle:** whatever carries their old content for ~free must carry the **all-distinct** part of
   sources aged 4–9. It is not a slice closure, not a jump with a slice rule between checkpoints, and not a
   compression (§54–§59). The constraint keeps narrowing toward "a different method".

(A 10-MLP confirmation of the curve is was queued behind E1 and NOT run (the verdict is far outside the per-MLP spread); the verdict at a0 ≤ 6 cannot change sign — ×1.5–×3.5 is
far outside the ±15% per-MLP spread that bit us in §51.)
