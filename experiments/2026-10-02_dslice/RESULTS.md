# The step's inputs against Monte-Carlo oracles: D (κ₃ (2,1) slice) and K22 (κ₄ (2,2) slice) — 2026-10-02, Mac

Why: cloud #11 found that the step, fed the exact input covariance, still gives birth to 22–30% of the covariance
error at depth, while with ALL inputs exact it is accurate to ~0.1%. So the born error comes from the step's other
inputs. Two n×n inputs had never been checked against an oracle: **D = κ₃[i,i,j]** (carried by the source
representation) and **K22 = κ₄[i,i,j,j]** (rebuilt every layer by the closure's ansatz v(var_i var_j + 2C0²)).

**Oracles** (`d_oracle.py`, 1e7 samples, seeds 3101/3102, MLPs 0 and 1; ≈6 min each on the Mac): D and K22 of the
pre-activations at layers 9 and 12, post covariances at 8, 9, 11, 12, post means. Noise is measured from the seed
difference and subtracted everywhere. `chain.py` gained an oracle key `K22_layers` (off = bitwise: k4memo 3 = 3.7533e-08).
Base: B9 (torch), gate MLP 0 2.278922e-08, MLP 1 2.205732e-08.

## 1. D (`dslice.py`, `dslice2.py`, `dvar.py`)
| | MLP 0 | MLP 1 |
|---|---|---|
| D's error on the block the chain computes (alive, unsaturated rows × alive columns), noise-subtracted | 6.6% (l9) / 7.8% (l12), corr 0.997, slope 1.00 | 6.9% / 8.4%, corr 0.997, slope 1.01 |
| BORN covariance error at the step, with the exact D instead of the chain's | energy ×0.27 (l9) / ×0.22 (l12): **D makes 73–78% of the born error** | — |
| exact D at layers 9+12 — **all** of it (upper bound, Richardson) | **×0.885** | **×0.931** |
| … only on the block | ×0.900 | **×1.027 (worse)** |
| … only on the dead rows/columns (zeroed by the matched cut) | ×0.993 | **×1.094 (worse)** |
| … only on the saturated rows (zeroed by E2; acts through the κ₄ level restoration) | ×0.999 | ×1.001 |

**D's accuracy matters a lot — two layers of exact D are worth ×0.88–0.93 — but only the complete exact D helps on
both networks; any part alone can hurt (MLP 1).** The consistency principle again: the chain's errors in D's parts
compensate each other and the rest of the state. Where the 7% comes from (`dvar.py`, MLP 0, D error on the block):
the thin-leg rank, the band, the cuts and the κ₄ birth depth change nothing; path births at every age 6.6 → 5.8%,
`pathD="full"` 6.6 → 6.2%. **≈85% of D's error is not in any tunable approximation** — it is the truncation of the
births and errors in their inputs, i.e. the k = 3 truncation itself.

## 2. A small structural lever found on the way: `pathD="full"` on the shippable base (`pathd10.py`)
With path births at the first readout only (production), `pathD="full"` adds ONE product per layer (the newest
source's path leg against its tree leg) — the 09-28 note priced it at one product per source because it was measured
with path births at all ages. 10 baked MLPs: **raw ×0.9829, better on 8/10** (0.955 … 1.007).
**Priced on the production shapes it does not pay:** the product is (n_r × w) @ (w × npc) with w the newest source's
≈850 hubs, ≈1.0e9 FLOPs per layer, ≈1.55e10 per network = **+3.8% of k3y's FLOPs plain, +2.2% with Strassen L4** —
net ×1.005 at best. (A first estimate of "+0.7%" divided by the budget instead of by our own FLOPs.) Closed.

## 3. K22 (`k22.py`, `k22feat.py`) — MLP 1 so far
| | layer 9 | layer 12 |
|---|---|---|
| the closure's K22 vs the oracle | rel error 20%, **corr 0.62**, slope 0.93 | 25%, **corr 0.64**, slope 0.88 |
| the ansatz's two terms (var·var, C0²) fitted to the oracle — R² | 0.36 | 0.41 |
| 23 pair features of the same layer (ridge, in sample) — R² | 0.49 | 0.56 |
| split-half noise ceiling | 0.98 | 0.99 |

Payoff at layers 9+12 (MLP 1, upper bounds): exact K22 ×0.970; exact pattern at the chain's level ×0.975; the chain's
pattern at the exact level ×0.979. The closure's pattern is poor, but half of the true K22 is a random projection
(through W_l) of the previous layer's κ₄ tensor, which no pair feature sees; extra features add 0.10–0.15 of R².
**Out of sample** (`k22feat.py`, fit on one network, test on the other, same layer): the closure's two terms R² 0.35–0.45,
all 23 pair features **0.45–0.61** — the extra features DO generalise across networks (+0.10–0.17).
**But the payoff does not:** MLP 0, layers 9+12 — **exact K22 ×1.102 (worse)**, exact pattern ×1.040 (worse), exact level
×1.043 (worse), against ×0.970 / ×0.975 / ×0.979 on MLP 1. K22's error is part of the chain's compensating set of errors:
correcting it alone helps one network and hurts the other by 10%. **A learned K22 closure is closed** (it would aim, at
best, at a target that is not reliably worth having), and so is any partial fix of D (§1).

## Verdict
The step's two pair inputs are measurably wrong — D by 7% (85% of it the k = 3 truncation), K22 by 20% (half of it a
random projection no pair feature sees) — and both are worth something only when corrected **together with everything
else**: exact D at two layers ×0.885 / ×0.931 complete, but its parts and the exact K22 alone are network-dependent
(×0.90 vs ×1.03; ×0.97 vs ×1.10). `pathD="full"` on the shippable base is ×0.983 raw (8/10) but costs +2.2–3.8% of
the FLOPs — net zero. NEGATIVE §73.
