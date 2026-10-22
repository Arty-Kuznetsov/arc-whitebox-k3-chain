# Age-truncated k = 4 in ARC's own representation (2026-10-13/14)

**Question.** Write-up §7 found that the gain of ARC's k = 4 over k = 3 lives in the κ₄ born in the last two layers
(substituted into our chain: young blocks exact + the old part on the closure's pattern recover the whole gain). Does
the same hold inside ARC's own algorithm, as an algorithm? ARC's SIMPLE k = 4 carries κ₄ as Sym(Σ_r A_r ⊗ B_r) and
appends 2n columns per ReLU, so its κ₄ work grows like L² over a network. Keeping only the blocks of age ≤ a makes it
linear in depth.

**Variants** (`k4age.py`; ARC's k = 4 stepped exactly as `mlp_kprop` does): full; age a (blocks older than a dropped
after every linear step); age a + closure (the dropped blocks replaced by one scale-mixture block v (CC + CC + CC), v
fitted on their diagonal before they are dropped, O(n r)). Truth: Monte-Carlo post-means at 10⁷ samples, MC noise
var/N subtracted; "gain kept" = (k3S − variant)/(k3S − full).

**Gates** (`gate.py`, n = 48, L = 6): the stepped full k = 4 equals `mlp_kprop`'s k4S to 1.6e-16; age ≥ depth equals
full exactly; the closure block's diagonal equals 3v·var² exactly. κ₄ factor columns per layer: full 2n·l; age 1 stays
2n (+1 effective column with the closure; the block is allocated n wide for bookkeeping).

**Prediction (written before the runs).** Age 2 keeps ≥ 90% of k = 4's gain over k = 3 at small width; the closure
block matters little ("the pattern, not the level", write-up §7.1); the κ₄ work drops from ∝ L² to ∝ L.

## Result (2026-10-13 16:34; `res/k4age_*.json`, `res/table.md`)
| shape | nets | variant | kept share of k = 4's gain, mean (min…max) | final MSE ÷ full k = 4 (median) | κ₄ columns ÷ full | wall time ÷ full | wall ÷ k = 3 |
|---|---|---|---|---|---|---|---|
| 128 × 8 | 8 | full k = 4 | 100% | 1 | 1 | 1 | 66.3 |
| 128 × 8 | 8 | age ≤ 1 | 28% (-44…67) | 5.67 | 0.25 | 0.70 | 46.8 |
| 128 × 8 | 8 | age ≤ 2 | 78% (49…100) | 2.45 | 0.46 | 0.79 | 52.7 |
| 128 × 8 | 8 | age ≤ 3 | 100% (94…111) | 1.13 | 0.64 | 0.85 | 56.4 |
| 128 × 8 | 8 | age ≤ 1 + closure | 85% (73…93) | 1.67 | 0.36 | 0.71 | 47.4 |
| 128 × 8 | 8 | age ≤ 2 + closure | 92% (86…95) | 1.49 | 0.55 | 0.82 | 54.4 |
| 128 × 8 | 8 | age ≤ 3 + closure | 97% (96…100) | 1.24 | 0.71 | 0.91 | 60.2 |
| 256 × 8 | 3 | full k = 4 | 100% | 1 | 1 | 1 | 207.2 |
| 256 × 8 | 3 | age ≤ 1 | -64% (-102…-13) | 28.99 | 0.25 | 0.66 | 136.6 |
| 256 × 8 | 3 | age ≤ 2 | 43% (24…64) | 10.47 | 0.46 | 0.78 | 160.5 |
| 256 × 8 | 3 | age ≤ 3 | 85% (78…92) | 3.40 | 0.64 | 0.87 | 180.6 |
| 256 × 8 | 3 | age ≤ 1 + closure | 89% (83…92) | 2.56 | 0.36 | 0.74 | 152.9 |
| 256 × 8 | 3 | age ≤ 2 + closure | 95% (94…97) | 1.68 | 0.55 | 0.83 | 171.8 |
| 256 × 8 | 3 | age ≤ 3 + closure | 98% (98…98) | 1.32 | 0.71 | 0.91 | 187.4 |
| 128 × 16 | 4 | full k = 4 | 100% | 1 | 1 | 1 | 83.9 |
| 128 × 16 | 4 | age ≤ 1 | 80% (-7…246) | 2.37 | 0.12 | 0.45 | 37.7 |
| 128 × 16 | 4 | age ≤ 2 | 121% (33…296) | 1.66 | 0.24 | 0.52 | 43.4 |
| 128 × 16 | 4 | age ≤ 3 | 156% (65…328) | 1.05 | 0.35 | 0.58 | 48.5 |
| 128 × 16 | 4 | age ≤ 1 + closure | 130% (93…221) | 0.88 | 0.18 | 0.49 | 40.9 |
| 128 × 16 | 4 | age ≤ 2 + closure | 125% (87…197) | 0.82 | 0.30 | 0.56 | 47.3 |
| 128 × 16 | 4 | age ≤ 3 + closure | 107% (83…150) | 1.04 | 0.40 | 0.64 | 53.4 |
| 256 × 16 | 3 | full k = 4 | 100% | 1 | 1 | 1 | 201.5 |
| 256 × 16 | 3 | age ≤ 1 | -45% (-127…30) | 8.90 | 0.12 | 0.42 | 85.4 |
| 256 × 16 | 3 | age ≤ 2 | 1% (-68…57) | 5.79 | 0.24 | 0.50 | 100.2 |
| 256 × 16 | 3 | age ≤ 3 | 50% (23…76) | 3.85 | 0.35 | 0.58 | 115.7 |
| 256 × 16 | 3 | age ≤ 1 + closure | 63% (44…83) | 3.08 | 0.18 | 0.47 | 93.9 |
| 256 × 16 | 3 | age ≤ 2 + closure | 78% (67…86) | 2.08 | 0.30 | 0.54 | 109.5 |
| 256 × 16 | 3 | age ≤ 3 + closure | 89% (84…92) | 1.63 | 0.40 | 0.61 | 122.4 |

"Kept share" = (k3S − variant)/(k3S − full), MC noise subtracted; at 128 × 16 the denominator is small and noisy (on one
network full k = 4 is barely better than k = 3), so the median MSE ratio to full k = 4 is the column to read there.

**Reading.**
1. **Dropping old κ₄ without replacement does not work** inside ARC's algorithm: age ≤ 1 is worse than k = 3 at n = 256
   (kept −45 to −64%), age ≤ 2 keeps 1–43%. Unlike our chain (where our own closure carried the old part's level, §7.1
   of the write-up), ARC's k = 4 has nothing else to carry it.
2. **One scale-mixture block for everything dropped restores it.** v (CC + CC + CC), v fitted on the dropped blocks'
   diagonal, O(n r): age ≤ 2 + closure keeps 92% / 95% of the gain at depth 8 (n = 128 / 256) and 78% at 256 × 16;
   age ≤ 3 + closure 97% / 98% at depth 8 and 89% at 256 × 16; at 128 × 16 the truncated chains are as good as or better
   than full k = 4 (median MSE ratio 0.82–1.04).
3. **The kept share falls with depth at fixed age** (age ≤ 2 + closure: 95% at 256 × 8, 78% at 256 × 16): part of k = 4's
   gain is born more than two layers back at depth 16, so the age that keeps ~90% grows slowly with depth.
4. **Cost.** The κ₄ factor columns fall to 0.30–0.40 of full k = 4 at depth 16 (constant per layer instead of growing),
   but the wall time only to 0.54–0.61: in ARC's implementation k = 4 is 66–207× the wall time of k = 3, and most of
   it is not the κ₄ factor columns. The saving grows with depth (the columns no longer grow with the layer index).
**Verdict on the prediction.** "Age 2 keeps ≥ 90%": true with the closure at depth 8, false at depth 16 (78%; age 3 gives
89%). "The closure matters little": false inside ARC's algorithm; it is what makes truncation work. "κ₄ work ∝ L instead of
L²": true for the factor columns; the wall time is dominated by other parts of the step.
**What it is for ARC:** a knob for their factored k = 4: keep three layers of κ₄ blocks and one scale-mixture block for the
rest, ≈ 90% of k = 4's accuracy gain over k = 3 at depth 16 for 0.4 of the κ₄ columns. Not a cheap k = 4.
