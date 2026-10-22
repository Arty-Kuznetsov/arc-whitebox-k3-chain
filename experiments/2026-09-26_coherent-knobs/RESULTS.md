# Re-scanning the soft "coherent" scalar levels of the closure on the current base

**Dates:** scan 2026-09-26, refine 2026-09-27, Mac. **Scripts:** `scan.py` (5 knobs × 5 values), `refine.py`
(finer grid + joint settings). 10 baked MLPs, k3l torch config, base **2.3135e-08** (self-check printed by both).

**Why.** NEGATIVE §57 found the variance *level* is stiff (a parabola ~1e-5 wide), so coherent corrections there are
worthless. But the κ₄ / feed levels are **soft** — `k31mult` 1.0 → 0.65 had given ×1.058 — so their optima may have
moved after k3i/k3j/k3l landed. Every knob is a scalar multiplier on a term the chain already forms: **zero FLOPs**.

## scan.py (one knob at a time)
| knob (current) | values | best | × | better |
|---|---|---|---|---|
| `k31mult` (0.65) | 0.45 … 0.85 | 0.65 (current) | 1.0000 | — |
| `k22c` (1.0) — the 2·C0² part of the (2,2) closure pattern | 0.6 … 1.4 | 0.6 | 1.0014 | 8/10 |
| `sdmult` (1.0) — star-diagonal κ₄ birth | 0.5 … 1.5 | 0.75 | 1.0011 | 6/10 |
| `feedmult` (1.0) — κ₄→κ₃ P2 feed level | 0.6 … 1.4 | **0.8** | **1.0058** | 7/10 |
| `k4scale` (1.0) — κ₄ births | 0.85 … 1.15 | 0.85 | 1.0010 | 6/10 |

## refine.py
- `feedmult`: flat optimum 0.75–0.85 (0.8 → ×1.0058, 0.85 → ×1.0055 on 8/10).
- `k22c`: improves monotonically down to **0.0** (×1.0032, 7/10).
- Joint `feedmult 0.8 + k22c 0.6` → **×1.0072, better on 9/10**; `+ sdmult 0.75` → ×1.0078 (7/10); adding
  `k4scale 0.85` does not compose (×1.0073, 6/10).
- Later (`2026-09-28_ceiling/combo.py`): `feedmult 0.8 + k22c 0.0` = **×1.0089, 9/10**.

## What happened next — the result that matters
Shipped as **k3n** (`src/estimator_k3n.py`, `2026-09-28_k3n/`): flopscope 100 MLPs ×1.0030 (66/100), and on the
grader **5.3977e-9 vs k3m's 5.3906e-9 — 0.13% worse** (submission 332872). The transfer chain torch-10 ×1.0089 →
flopscope-100 ×1.0030 → grader ×0.9987 is NEGATIVE §60: **scalar tuning is below the grader's ~0.4–0.5% resolution.**
Do not re-tune these levels for the score. (`k31mult` and `k4k3` did move again inside the cost-free stack of
`2026-09-28_accuracy-free/` — k31mult 0.55, k4k3 0.5 — but only together with terms we cannot afford.)
