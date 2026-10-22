# k4a — the port of the young-κ₄ reader without the last-layer pairing (R6, 2026-10-06)

**Target.** k3z + (a) the restructured D pairing: pair the current partners L1/L2 (and the members' compressed partners)
with the legs **as they were two layers back** (ungated snapshots), recover Dx = Dx2 · V(l−1) and D = Dx · V(l); (b) the
production reader (`young4.Young4Chain` semantics): per generation born at m ∈ {l−1, l−2}, K4d += 12·Σ_j x_aj d2_m(j)
Dx_m(a, j) (3+2, not at the last layer) + the path's coinciding-hub piece 12·d2² var·x² + the star 4·d3·x·U², then the
projection of the sum off the closure's pattern 3·var² ("resid"). Metered projection: raw ×0.879 (10/10 on the torch
chain with the band) at +6.25% FLOPs → 3.49e-9 (`../2026-10-04_r3/RESULTS.md` step 3).

**Algebra (transposed layout; rows = hub rows, columns = neurons).** Legs at l: Xᵀ(l) = (d1(l−1) ∘ Xᵀ(l−1))[:, idp_l] · W_l[idp_l, :].
D(l) = LTᵀ(l) · Xᵀ(l) = LTᵀ(l) · Xᵀ(l−1)[:, idp_l] · V_l = Dx · V_l with V_l = d1(l−1)[idp_l] ∘ W_l[idp_l, cols], and
Dx = Dx2 · V_{l−1} with Dx2 = LTᵀ(l) · Xᵀ(l−2)[:, idp_{l−1}] over the sources that existed at l−2. Sources born at l−1
(legs at l−1 = their birth legs: Bn and the identity on their hubs) enter Dx directly; the member sources enter Dx2
through P(l−2) (compressed) or through their dense snapshot rows if they entered the band after l−2.

**Gates, in order.** (G0) `young=None, restructure=False` → bitwise k3z, equal FLOPs (1024×16 synthetic and 128×8).
(G1) `restructure=True, young=None` → outputs equal k3z to float rounding (≤ 1e-5 rel on the final row), FLOPs ≈ +6%;
local wall and residual on 3 MLPs. (G2) reader on → per-MLP raw ratios against the torch chain's bCy2K4pdnl on MLPs 0–2
(0.935, 0.900, 0.861), through the production/torch transfer. (G3) 100 MLPs locally (`--no-residual-wall-time-limit`),
projection = local score × 0.976 (grader/local). (G4) pre-package checklist, submit as a PROBE (not designated).

**Kill conditions.** G1 wall increment > 10 s local (grader ×1.7–3.5) or residual increment that puts the quiet local max
above 0.45 s (grader ≈ 0.73×) → move the recovery products to level 4, then to a lower level; if the projection then
rises above 3.55e-9, stop and record. Time box: 10-09.

**Memory.** Two ungated snapshots of the non-member legs (≤ 2·Wcap × n each), two of Pᵀ (rb × n), per-layer vectors
(d1, d2, d3, var, idp) and two Dx buffers: ≈ +0.3 GB on k3z's 6.85 GB peak.
