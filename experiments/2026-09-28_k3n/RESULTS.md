# k3n — k3m + two re-tuned scalar levels, at zero FLOP cost

**Date:** 2026-09-28, Mac. **Package:** `submission-20260928-161016.tar.gz`
(`src/estimator_k3n.py`). **Change:** `feedmult = 0.8` (was 1.0) and `k22c = 0.0` (was 1.0), i.e. the level of
the κ₄→κ₃ feed and the coefficient of the `2 C0²` part of the (2,2) closure pattern.
Found by `experiments/2026-09-26_coherent-knobs/` (scan + refine, 10 torch MLPs).

## Self-checks (`k3n_check.py`)

| claim | result |
|---|---|
| `k3n(feedmult=1.0, k22c=1.0)` vs `k3m()` | **bitwise equal**, FLOPs equal, 25.169% of B |
| `k3n()` defaults | 25.167% of B — **0.002 points cheaper** (k22c=0 skips one n² product); final-layer difference 1.4e-5 relative |

## 100 MLPs, flopscope (`runs/k3n_n100.json`)

| | k3m | **k3n** |
|---|---|---|
| adjusted | 5.5209e-9 | **5.5045e-9 (×1.0030)** |
| final-layer MSE | 2.2172e-8 | 2.2108e-8 |
| utilisation | 24.894% | 24.892% |
| failures | 0 | **0** |
| residual mean / max | 0.197 / 0.227 s | 0.196 / 0.226 s |
| better on | — | **66 / 100 MLPs** |

Expected on the grader: **≈5.372e-9** (k3m graded 5.3906e-9).

## The finding that matters more than the 0.3%

**The gain shrank by 3× between the tuning set and production: torch ×1.0089 on 10 MLPs → flopscope ×1.0030
on 100, and the breadth fell from 9/10 to 66/100.** 66/100 is still significant (p ≈ 0.001 against a coin),
so the lever is real — but this is the measured transfer factor for a **scalar level fitted on 10 MLPs**, and
it should be applied to every future number of that kind. In particular the cost-free accuracy stack of
`2026-09-28_ceiling/` mixes *structural* levers (`allages`, `thin192`, `pathD`, `k4b` — terms added or rank
raised, no fitted constant) with *fitted* ones (`feedmult`, `k22c`, `k4k3`, `k31mult`). Only the fitted half
should be discounted by ≈3; quoting a projection that discounts neither, or both, would be wrong.

Two independent reasons the fitted half is the risky half: it is fitted on 10 of **our** 100 baked MLPs, and
under grader schema 2.0 the ranked number is a mean over **50 grader MLPs** we never see.
