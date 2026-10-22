# Selective Strassen L5 (on the transport only) — priced by MEASUREMENT, not by the model (2026-10-01)

Full L5 died on wall time (×2.2) and RSS (7.21 GB) — NEGATIVE §69. The question was whether L5 on the **transport
alone** pays. Instead of building a two-depth Strassen first, both configurations were run once through the
namespaced profiler (`estimator_prof.py`, `runs/v_prof.py` vs `runs/v_prof_L5.py`, one 1024×16 MLP) and compared per
call site:

| namespace | L4 FLOPs | L5 FLOPs | **ΔFLOPs (of the total)** | L4 backend | L5 backend | **Δ backend** |
|---|---|---|---|---|---|---|
| transport | 1.058e11 | 9.718e10 | **−1.96%** | 3.20 s | 7.23 s | **+4.03 s** |
| dpair | 8.694e10 | 8.199e10 | −1.12% | 3.49 s | 7.69 s | +4.20 s |
| band_members | 7.366e10 | 6.802e10 | −1.28% | 3.81 s | 8.66 s | +4.85 s |
| estimator-client (incl. the members' compressed pairing, also Strassen) | 6.631e10 | 6.518e10 | −0.26% | 5.21 s | 13.20 s | +7.99 s |
| ctransport | 3.750e10 | 3.427e10 | −0.73% | 1.13 s | 2.73 s | +1.60 s |
| band_entry | 3.001e10 | 2.920e10 | −0.18% | 1.05 s | 2.88 s | +1.83 s |
| cpre | 1.839e10 | 1.684e10 | −0.35% | 0.61 s | 1.50 s | +0.89 s |
| total | 4.407e11 (20.03%) | 4.148e11 (18.86%) | −5.88% | 19.6 s | 45.0 s | +25.4 s |

**L5 on the transport only: util ×0.980 (−1.96% of the total), backend +4.03 s = +20.6%** → on the grader (wall ≈ 3×
local; k3r's graded max 69.7 s) ≈ **84 s against the 120 s cap**. Strassen is exact up to float32 rounding (full L5:
raw ×0.9991), so the accuracy is unchanged. **The catch is memory:** the level-5 buffer (≈2.2 GB at chunk 16) is
needed by any L5 call, so peak RSS is ≈7.2 GB against the 8 GB cap, measured on macOS — the grader's Linux accounting
is unmeasured. Halving the chunk at level 5 would halve that buffer at the price of more flopscope calls (residual).

**Verdict: a real ≈2% lever with a known memory risk.** Shippable only as a probe submission first (the leaderboard
keeps the best entry; an OOM would show up as failed MLPs in the telemetry), never as a designated one unverified.
