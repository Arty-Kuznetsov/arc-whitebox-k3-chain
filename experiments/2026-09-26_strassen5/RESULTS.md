# Strassen level 5 — closed by wall time and memory, not by the meter

**Date:** 2026-09-26, Mac. **Script:** `probe.py` (monkeypatches only the recursion floor in `_Strassen`,
`src/` untouched). One synthetic 1024×16 He-init MLP, `flopscope` budget 2**50.

The cost-frontier atlas named L5 the last modelled cost lever with **zero accuracy cost**
(billing 0.636 → ≈0.560, ×1.14). It was parked as "a residual-time decision, only on the Mac". After
332403 the residual headroom looked ample (grader 0.160 s of a 0.400 cap), so it was measured.

| config | util | residual | wall | peak RSS | vs L4 | final rel diff |
|---|---|---|---|---|---|---|
| L4, chunk 16 (production, = k3m) | **25.169%** | 0.271 s | **19.7 s** | 5.46 GB | — | — |
| L5, chunk 16 | **23.067%** | 0.336 s | **38.4 s** | 7.81 GB | **×0.9165** | 1.45e-6 |
| L5, chunk 8 | 23.066% | 0.364 s | 38.2 s | 7.81 GB | ×0.9165 | 2.08e-6 |

(Residual here reads ≈1.35× the 100-MLP local runner — this probe runs under desktop load on a synthetic
MLP; peak RSS is `ru_maxrss` of the whole process, so the L5 rows include the L4 run's peak.)

## Verdict: not shippable

- **The meter pays less than modelled: ×0.9165, not ×0.88.** Still a real 8.4% cost cut (25.17% → 23.07%),
  worth ≈×0.92 on the adjusted score if it were free.
- **Wall time doubles: 19.7 s → 38.4 s (×1.95).** 332403 measured grader wall **60.9 s mean / 75.3 s max**
  at 19.4 s local (so ≈3.1× mean, **3.9× on the worst MLP**). ×1.95 puts the grader's mean at ≈119 s and its
  worst MLP at ≈147 s against a **120 s cap** — a breach zeroes that MLP's prediction. Dead.
- **Peak RSS reaches the 8 GB cap.** The level-5 region alone is 2 × 7⁵ × chunk × 32² floats = 2.2 GB at
  chunk 16, on top of L4's 2.6 GB of regions. `chunk 8` does not help: it halves that region but leaves the
  wall unchanged (38.2 s) and raises the residual (0.364 s), and §49 already measured chunk 8 as a residual loss.
- **Why the wall doubles while the FLOPs fall.** Level 5 turns one 64×64-leaf product into 7 32×32-leaf
  products: 268 912 matmuls of 32³ instead of 38 416 of 64³. The *combine/recombine* traffic grows with it —
  20 element-wise passes over a (7⁵, chunk, 32, 32) = 275 M-float buffer per call, ≈1.1 GB touched per pass.
  The level is memory-bandwidth bound, and that is not something a different layout fixes (cf. §49, k3g).

**Consequence: the cost axis is now exhausted on every known switch.** deadcut −2.5 shipped (k3m, 24.84%),
pathrank withdrawn (§55), shared subspace parked (k3k, §54), window dead (§16), srank neutral, L5 closed here.
Anything further on cost requires the new representation of the old tail (FRONTIER (i), symmetric-CP cubes).
