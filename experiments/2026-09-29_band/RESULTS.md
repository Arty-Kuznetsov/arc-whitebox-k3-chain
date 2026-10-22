# E1 — the transported band basis survives on accuracy; its cost depends entirely on the coefficient rotation

**Date:** 2026-09-29, Mac. **Script:** `e1.py` (10 baked MLPs, k3m config, base 2.3462e-08). No new code:
`chain.py ssp=(a0, r, 0, 0)` — the band basis is only *transported* (ov = 0, q = 0: the Rayleigh-Ritz step then
rotates within the transported span, which leaves the projector unchanged). `shared_sub=(a0, r)` = ideal refit.

| arm | raw | × base | better |
|---|---|---|---|
| k3m base | 2.3462e-8 | 1.0000 | — |
| ideal refitted band 6:320 | 2.3769e-8 | 1.0131 | 3/10 |
| transported 6:320 (control) | 2.6010e-8 | **1.1086** | 0/10 |
| transported 6:384 | 2.4040e-8 | **1.0246** | 0/10 |
| **transported 6:448** | **2.3624e-8** | **1.0069** | 1/10 |
| transported 5:448 | 2.3966e-8 | 1.0215 | 1/10 |

**Verdict on accuracy: survives the synthesis's kill rule** (384 ≤ +3.0%, 448 ≤ +2.0%). The control confirms the
skeptics: a transported basis at r = 320 is +10.9% (they predicted +7–8%) — no spectral gap near 320; a transported
rank-r basis captures what an ideal ~0.8r one does, so the working point is r = 384–448.

## Cost — priced with `2026-09-26_cost-frontier/cost.py`

| band | coefficient rotation | entry | util | × util | projected |
|---|---|---|---|---|---|
| (6,448) | every layer (as k3k) | 0.9 | 24.48% | 1.014 | ×1.02 |
| (6,448) | **none** | 0.9 | 21.40% | 0.886 | **×0.89** |
| (6,384) | every layer | 0.9 | 22.72% | 0.941 | ×0.96 |
| **(6,384)** | **none** | 0.9 | **20.45%** | 0.847 | **×0.87** |
| (5,448) | none | 0.9 | 20.55% | 0.851 | ×0.87 |

The rotation (r²·n per band leg per layer, plain billing: ≈0.57 units per band source-layer at r = 448) is the whole
difference between a dead lever and a live one. It exists in k3k because that basis was re-fitted every layer. A
transported basis without a per-layer QR does not need it: Wᵀ(P·Z) = (WᵀP)·Z, so Z never changes. **Open question,
decisive:** without re-orthonormalisation P loses conditioning over ~10 random transports, and production is
float32. A lazy QR every k layers puts the cost between the two rows (≈×0.90 at k = 3). Tasked to the cloud
(`notes/CLOUD-BRIEF.md`, port battery, step 1).
