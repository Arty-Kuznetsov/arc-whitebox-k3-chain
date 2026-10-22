# Port battery + stack frontier — cloud session #8 (2026-09-29, branch `cloud/port-battery`)

**Question (the user's, stated as the score):** which EXACT-engineering levers can the Mac port by ~10 Oct, in which
combination, and how far does the full stack get against the top-5 cut of 2.1e-9? Base = k3m (332403 = 5.3906e-9 =
raw 2.1652e-8 × util 24.835%). All accuracy numbers: torch chain (`chain.py`), 10 baked MLPs, float64 except where a
switch says otherwise. All utilisations: **modelled** (`cost_pb.py`, below), reported as a ratio to k3m in the same
model and scaled to the graded 24.835%. Projected adjusted = 5.3906e-9 × raw ratio × util ratio (both vs k3m), exactly
as the brief prescribes. Wall and residual time measured here are not evidence (CLOUD.md §2) and are not reported.

**TL;DR for the Mac** (verdict §5, ranked port list with exact switch settings §6, frontier table §4):
1. **Float32 band is safe with NO re-orthonormalisation and NO coefficient rotation** when the entrant is projected
   through a QR of P (Z = R⁻¹QᵀX); the normal equations need a QR every ≤ 5 layers (k = ∞: ×1.12–1.52). §1.
2. **Priced honestly, the band's construction eats 71% of its gross saving**; net ×0.886 at its best point ((5,448),
   entry at the end of the previous layer, range-finder init) on the matched base. §2.
3. **New and the largest lever: E2, the saturated-row cut of the D pairing with the κ₄ level corrected in O(S n²)** —
   raw ×0.9405 (9/10) AND util ×0.932 on plain k3m → **k3m + E2 projects to 4.72e-9**; small port on the current
   layout — **ship it first**. §3.
4. E3 (receiver rows) survives only with the repeated-index fill-in (×0.953); E4 (age-dependent cut) is dead. §3.
5. **The full exact stack (matched −2.25 + E2 + band + E3) = 3.79e-9, 3.47e-9 with Strassen L5 (×0.70 / ×0.64 of
   k3m): rank ≈22–27. It cannot reach 3.0e-9; the gap to the top-5 cut 2.1e-9 is ×1.80 (×1.65 with L5).** §4–§5.

## 0. Anchors and gates (all exact)

| gate | required | measured here |
|---|---|---|
| `ablate.py k4memo 3` (every new switch off) | 3.7533e-08 | **3.7533e-08** (`logs/gate_k4memo3.txt`) |
| `ablate.py k3l_t 10` | 2.3135e-08 | **2.3135e-08**, per MLP 0/1/2/7/9 = 2.4074/2.1572/2.2646/2.3867/2.8053e-8 = the Mac (`logs/gate_k3l_t10.txt`) |
| k3m BASE (`pairs.py` BASE + deadcut −2.5) | 2.3462e-08 | **2.3462e-08**, per MLP within 5.4e-12 of the Mac's E1 base (`t3_base_k3m_n10.json`) |
| matched cut −2.25 (colcut `pairs.py`) | 2.4151e-08 | **2.4151e-08**, per MLP within 5.6e-12 (`t3_base_m225_n10.json`) |
| task 1 anchor: `bandz` float64, QR every layer, (6,448) | E1 = 2.3624e-08 to 1e-4 | **2.3624e-08**, max per-MLP difference 3.6e-12 |
| re-run at the end with the final `chain.py` (all switches in): `k4memo 3` / `k3l_t 10` | 3.7533e-08 / 2.3135e-08 | **3.7533e-08** / **2.3135e-08**, every per-MLP value identical to the first gates (`logs/gate_final_*.txt`) |

New `chain.py` switches, all default off: `bandz` (the band in coefficient form), `satrow_k4` (E2 κ₄-level
restoration: `"restore"` exact, `"sep"` production form), `satrow_ch` (E2 channel test), `recv` (E3 receiver rows),
`agecut` (E4 age-dependent cut), `margtail` (side probe, §8). Identities checked before any 10-MLP run (random
n = 128/256 MLPs): `satrow=1e9` with restore / sep / either channel, `recv` θ = −99, `agecut=(99, ·)` and
`agecut=(4, −2.5)` on matched −2.5, `margtail=1e9` reproduce their bases to ≤ 3e-16 (exactly 0 for most);
`bandz` float64 k=1 == `ssp=(a0, r, 0, 0)` to 5e-16.

## 1. Task 1 — the transported band in float32 with a lazy QR (`t1_band.py`, `chain.py bandz`)

**The mirror** (`chain.py _band_step / _band_enter`, `bandz=dict(a0, r, k, dtype, proj, entry)`): sources of age ≥ a0
live as legs = P·Z with ONE shared basis P (n × r) transported like a leg (gated by d1 at the end of every layer,
Wᵀ·P at the next) and FIXED per-leg coefficients Z (r × n), because Wᵀ diag(d1)(P Z) = (Wᵀ diag(d1) P) Z. P is
re-orthonormalised only every k layers (then P = QR, Z ← R Z for every member: the rotation), the entrant — exactly
one per layer ≥ a0 — is projected once by least squares on the current P, either by the normal equations
(`proj="ne"`: Z = (PᵀP)⁻¹PᵀX, one r × r LU solve, what the brief specified) or through a QR of P taken for the
projection only (`proj="qr"`: Z = R⁻¹QᵀX; the members are NOT rotated). All band arithmetic (transport of P, QR,
rotation, projection, materialisation P·Z) in float32; a float64 shadow of the same band (`shadow=True`) gives the
per-layer relative leg error ‖legs₃₂ − legs₆₄‖/‖legs₆₄‖ (diagnostic only, it does not feed the chain).
**Anchor:** float64, k = 1, (6,448) = **2.3624e-08 = E1**, max per-MLP relative difference **3.6e-12** (1e-4 required).

<!-- TABLE_T1 -->
| band (a0, r) | projection | QR every k | raw | x k3m | vs E1 float64 | worst MLP vs E1 | max cond(P) | max leg error |
|---|---|---|---|---|---|---|---|---|
| (6,448) | ne | 1 | 2.3624e-08 | 1.0069 | -0.00% | x1.000 | 17 | 1.6e-06 |
| (6,448) | ne | 2 | 2.3624e-08 | 1.0069 | +0.00% | x1.000 | 61 | 1.7e-06 |
| (6,448) | ne | 3 | 2.3624e-08 | 1.0069 | -0.00% | x1.000 | 201 | 3.5e-06 |
| (6,448) | ne | 5 | 2.3624e-08 | 1.0069 | -0.00% | x1.000 | 884 | 2.4e-05 |
| (6,448) | ne | inf | 2.6346e-08 | 1.1229 | +11.52% | x2.092 | 47044 | 1.0e+00 |
| (6,448) | qr | inf | 2.3624e-08 | 1.0069 | -0.00% | x1.000 | 47036 | 1.9e-05 |
| (6,384) | ne | 1 | 2.4040e-08 | 1.0246 | +0.00% | x1.000 | 11 | 1.9e-06 |
| (6,384) | ne | 2 | 2.4040e-08 | 1.0246 | -0.00% | x1.000 | 33 | 1.7e-06 |
| (6,384) | ne | 3 | 2.4040e-08 | 1.0246 | -0.00% | x1.000 | 81 | 2.3e-06 |
| (6,384) | ne | 5 | 2.4040e-08 | 1.0246 | +0.00% | x1.000 | 299 | 1.1e-05 |
| (6,384) | ne | inf | 2.4039e-08 | 1.0246 | -0.00% | x1.000 | 6131 | 2.9e-03 |
| (6,384) | qr | inf | 2.4040e-08 | 1.0246 | +0.00% | x1.000 | 6131 | 9.6e-06 |
| (5,448) | ne | 1 | 2.3966e-08 | 1.0215 | +0.00% | x1.000 | 17 | 1.7e-06 |
| (5,448) | ne | 5 | 2.3966e-08 | 1.0215 | -0.00% | x1.000 | 1386 | 3.7e-05 |
| (5,448) | ne | inf | 3.5582e-08 | 1.5166 | +48.47% | x3.438 | 98400 | 1.7e+00 |
| (5,448) | qr | inf | 2.3967e-08 | 1.0215 | +0.00% | x1.000 | 98367 | 7.8e-05 |

anchor float64 k=1 (6,448): raw 2.362390e-08 vs E1 2.362390e-08; max per-MLP relative difference 3.64e-12
<!-- /TABLE_T1 -->

**What limits it — and what does not.**
- **cond(P) grows ×2.6 per transport** (median over 10 MLPs: 12 after one transport → 40 → 112 → 295 → 759 → 1990 →
  4600 → 1.2e4 → **3.0e4 after nine**; max 4.7e4 at (6,448), 6.9e4 with the pre entry, 1.4e5 at a0 = 5, and faster
  for larger r: 1.7e6 at (5,512), 3.6e6 at (4,512)).
- **P itself stays accurate in float32 at these conditions:** with the QR projection the members' legs are within
  **≤ 3e-5** (a0 = 6, r = 448) … 5e-4 (a0 = 5) … 2.4e-3 (a0 = 4, r = 512) of the float64 band after the full depth —
  the leg error tracks ≈ 6e-10 × cond(P) — and at (6,448) the raw is E1's to the last digit on every MLP. The tail is
  a soft channel (NEGATIVE §58: 1% relative error costs 0.1–1.1% raw), so even the (4,512) error is negligible; no
  rank or span is lost at 16 layers (1/ε₃₂ ≈ 1.7e7).
- **The only fragile piece is the r × r solve of the normal equations** — κ(PᵀP) = κ(P)² reaches 2e9 at layer 15:
  with k = ∞ at (6,448) the leg error is 15% at layer 14 and **100% at layer 15** on MLP 3 (×2.09 there, ×1.12 on the
  mean; at (5,448) 170%, ×1.52).
  With a re-orthonormalisation every **k ≤ 5** layers κ(P) stays ≤ 1.4e3 and the result is exact.
- **Answer to the brief's question:** with the normal equations the largest k within +1% of E1 is **k = 5** for
  (6,448) and (5,448) (all of k = 1, 2, 3, 5 reproduce E1 to 5 digits; k = ∞ fails: ×1.12 and ×1.52, one MLP ×2.1 /
  ×3.4); at (6,384) even k = ∞ holds, because cond(P) grows slower at smaller r (6e3 at layer 15). **With the QR projection, k = ∞: no re-orthonormalisation
  and NO coefficient rotation at all** — the rotation cost (the whole difference between a dead and a live band in
  E1's pricing) disappears; what it costs instead is one n × r QR per layer (0.33 units, plain), priced in §2.

## 2. Task 2 — the band priced honestly (`cost_pb.py`, `t2_price.py`)

**The model.** `cost_pb.py` re-implements `2026-09-26_cost-frontier/cost.py` per source (same two billing factors:
0.636 for everything riding the Strassen stacks, 1.0 for plain products and linalg calls; same base products and
path leg) so that the column cut, the band and the row compactions are priced in one pass, from a 10-MLP α table
of the k3m chain (`alpha_dump.py` → `alpha_fractions_k3m.json`). Its k3m row-cut curve agrees with `cost.py` to
0.3% (×0.9660 vs ×0.9630 at −2.25); the model's k3m is 24.11% (cost.py 24.14%) and everything is reported as a ratio
to it, scaled to the graded 24.835%. The flopscope billing of every construction op was checked on the real meter
(`fnp`, float32, n = 1024, r = 448): PᵀX 0.4373 units (model q = 0.4375), `qr` reduced 0.3270 (model 0.3270),
`solve` r × 2n 0.4107 (model 0.4107), `inv` r × r 0.0837.
**What the band costs, item by item** — (6,448), QR projection, no rotation, pre entry, billed units of the 1024 in B:

| item | k3m | band | billed Δ |
|---|---|---|---|
| dense leg transports (members' removed) | 191.5 S | 100.9 S | −57.6 |
| D pairings (members' removed) | 151.7 S | 91.4 S | −38.4 |
| **materialise A, B = P Z for every member, every layer** | — | 48.1 S | **+30.6** |
| **compressed pairing Σ(L1 Z_Bᵀ + L2 Z_Aᵀ), then one ·Pᵀ** | — | 35.0 S | **+22.2** |
| transport of P | — | 3.7 S | +2.4 |
| entrant projection QᵀX (Strassen) + QR of P and the solve with R (plain) | — | 8.8 S + 7.4 P | +13.0 |
| entrant's own dense transport at age a0 (only with entry="post", E1's semantics) | — | (17.0 S) | (+10.8) |
| **total** | 246.9 | 219.1 | **−27.8 (×0.887)** |

**The construction eats 71% of the gross saving** (68.2 of 96.0 billed units) — the §54–§59 pattern again, but here
the remainder is positive. The two big construction items are inherent (the readout and L1, L2 are element-wise in
the materialised legs; the dense pairing of a member would cost 2·fo²·c instead of 2·q·fo·c), so this is close to the
band's real price. **E1's pricing (×0.886 at (6,448)) matches this total, but it priced one design and measured
another:** its cost had no entrant transport and a flat entry charge 0.9·r/320 ≈ the real projection (13.0) — i.e. it
priced the *pre* entry — while its accuracy (+0.69%) was measured with the *post* entry, which needs the entrant's
dense transport (+10.8 → ×0.930). Measured here: pre costs +0.37% raw at (6,448), so both corners are now on the
table (§2 table).

**Production design this points to (the Mac should mirror exactly this):** `bandz=dict(a0=5, r=448, k=None,
proj="qr", entry="pre", init="rr", dtype=float32)` (or the tied `a0=6, r=384`, init "cols") — P in the leg stack (r extra columns, gated and transported with the legs), at the end of
the layer the source that reaches age a0 is projected onto the gated P through `qr(P)` (Z = solve(R, QᵀX) for A, B and
the thin H); no re-orthonormalisation and no coefficient rotation ever; members materialised by one shared-left
product P @ [Z…] into their leg slots every layer; the members' pairing compressed. A cheaper equivalent of the solve
(inv(R) once, 0.084 plain, then (R⁻¹Qᵀ)·X on the Strassen stack) saves another ≈0.2 units/layer (not in the numbers).
**Kill rule (projected worse than ×0.97 of k3m):** every QR-projection arm and ne with 2 ≤ k ≤ 5 survive; ne with
k = 1 (rotation every layer: ×1.013 at (6,448), ×1.014 at (5,448)), ne with k = ∞ (×1.038) and the (4,448) band
(×1.003) are killed.

**The band frontier (tasks 1b/1c, all float32 + QR projection + no rotation; `t1b_band.py`, `t1c_band.py`).** Three
things move the band beyond E1's grid:
- **entry = "pre"** (project the source at the end of the layer before it turns a0, onto the gated P) saves the
  entrant's dense transport (util ×0.930 → ×0.887 at (6,448)) for +0.37% raw at a0 = 6 (+1.7% at a0 = 5): projected
  ×0.937 → ×0.897;
- **init = "rr"** (the first basis = the dominant r-subspace of the first entrant's [A B], one power step, instead of
  its first r columns; +0.8% util once) halves the loss at a0 = 5: (5,448) ×1.0387 → **×1.0196** (projected ×0.886 →
  **×0.878**); at a0 = 6 it barely matters (×1.0106 → ×1.0072);
- **there is a cliff below r ≈ 448 and a0 ≈ 5**: (5,384) ×1.13–1.15, (4,448) ×1.18–1.23, (4,512) ×1.05–1.07, while
  r = 512 is almost lossless at a0 = 5–6 (×1.0025, 5/10 better) but saves less. The optimum is flat: (5,448) rr
  ×0.878, (6,384) pre ×0.880, (5,448) pre ×0.886, (6,448) pre ×0.897, (5,512) pre ×0.907 — and on top of the rest of
  the stack (5,448) rr and (6,384) pre are tied (§4).

<!-- TABLE_T2 -->
| arm | raw × k3m | util (model) | util × | **projected ×** | max leg error | max cond(P) | kill ×0.97 |
|---|---|---|---|---|---|---|---|
| f32 (5,448) k=inf qr pre init=rr | 1.0196 | 21.38% | 0.8608 | **0.8777** | 3.18e-04 | 120141 | survives |
| f32 (6,384) k=inf qr pre | 1.0417 | 20.97% | 0.8444 | **0.8797** | 1.36e-05 | 8140 | survives |
| f32 (5,448) k=inf qr pre | 1.0387 | 21.18% | 0.8527 | **0.8857** | 1.13e-04 | 136017 | survives |
| f32 (6,448) k=inf qr pre | 1.0106 | 22.04% | 0.8874 | **0.8969** | 2.91e-05 | 68676 | survives |
| f32 (5,480) k=inf qr pre init=rr | 1.0130 | 22.03% | 0.8872 | **0.8988** | 5.58e-04 | 489438 | survives |
| f32 (6,448) k=inf qr pre init=rr | 1.0072 | 22.24% | 0.8955 | **0.9020** | 1.62e-03 | 59997 | survives |
| f32 (6,384) k=inf ne | 1.0246 | 21.93% | 0.8830 | **0.9047** | 2.92e-03 | 6131 | survives |
| f32 (5,512) k=inf qr pre | 1.0025 | 22.47% | 0.9046 | **0.9068** | 4.48e-04 | 1714146 | survives |
| f32 (6,384) k=inf qr | 1.0246 | 22.04% | 0.8873 | **0.9091** | 9.64e-06 | 6131 | survives |
| f32 (6,384) k=5 ne | 1.0246 | 22.10% | 0.8897 | **0.9116** | 1.06e-05 | 299 | survives |
| f32 (5,384) k=inf qr pre init=rr | 1.1340 | 20.09% | 0.8088 | **0.9172** | 1.09e-04 | 16845 | survives |
| f32 (5,448) k=inf qr | 1.0215 | 22.36% | 0.9003 | **0.9197** | 7.80e-05 | 98367 | survives |
| f32 (4,512) k=inf qr pre init=rr | 1.0474 | 21.91% | 0.8824 | **0.9242** | 2.36e-03 | 3500384 | survives |
| f32 (5,384) k=inf qr pre | 1.1540 | 19.92% | 0.8020 | **0.9255** | 3.80e-05 | 15540 | survives |
| f32 (6,384) k=3 ne | 1.0246 | 22.51% | 0.9065 | **0.9288** | 2.32e-06 | 81 | survives |
| f32 (6,384) k=2 ne | 1.0246 | 22.59% | 0.9098 | **0.9322** | 1.73e-06 | 33 | survives |
| f32 (4,512) k=inf qr pre | 1.0697 | 21.68% | 0.8730 | **0.9339** | 2.31e-03 | 3637214 | survives |
| f32 (6,512) k=inf qr pre | 1.0025 | 23.13% | 0.9315 | **0.9339** | 7.41e-05 | 687354 | survives |
| f32 (6,448) k=inf qr | 1.0069 | 23.10% | 0.9301 | **0.9365** | 1.95e-05 | 47036 | survives |
| f32 (5,448) k=5 ne | 1.0215 | 22.85% | 0.9202 | **0.9400** | 3.69e-05 | 1386 | survives |
| f32 (6,448) k=5 ne | 1.0069 | 23.19% | 0.9337 | **0.9402** | 2.38e-05 | 884 | survives |
| f32 (6,448) k=3 ne | 1.0069 | 23.75% | 0.9565 | **0.9631** | 3.47e-06 | 201 | survives |
| f32 (6,384) k=1 ne | 1.0246 | 23.43% | 0.9432 | **0.9665** | 1.91e-06 | 11 | survives |
| f32 (6,448) k=2 ne | 1.0069 | 23.86% | 0.9609 | **0.9676** | 1.67e-06 | 61 | survives |
| f32 (4,448) k=inf qr pre init=rr | 1.1792 | 20.38% | 0.8208 | **0.9679** | 4.16e-04 | 267784 | survives |
| f32 (4,448) k=inf qr pre | 1.2343 | 20.18% | 0.8127 | **1.0031** | 3.84e-04 | 275389 | KILL |
| f32 (6,448) k=1 ne | 1.0069 | 24.99% | 1.0063 | **1.0133** | 1.64e-06 | 17 | KILL |
| f32 (5,448) k=1 ne | 1.0215 | 24.66% | 0.9928 | **1.0141** | 1.72e-06 | 17 | KILL |
| f32 (6,448) k=inf ne | 1.1229 | 22.96% | 0.9246 | **1.0383** | 1.03e+00 | 47044 | KILL |
| f32 (5,448) k=inf ne | 1.5166 | 22.21% | 0.8943 | **1.3563** | 1.66e+00 | 98400 | KILL |
<!-- /TABLE_T2 -->

## 3. Task 3 — the cheap battery (`t3_battery.py`; bases `t3_base_{k3m,m225}_n10.json`)

Every arm alone on the matched base (k3m + matched cut −2.25, raw 2.4151e-08) and on k3m (2.3462e-08).
"projected x base" = raw ratio × modelled util ratio vs that base; "projected-better" = MLPs with raw_i/base_i × util
ratio < 1.

<!-- TABLE_T3 -->
base k3m: raw 2.3462e-08
| arm | raw | raw x base | better | util (model) | util x base | projected x base | projected-better |
|---|---|---|---|---|---|---|---|
| E2 satrow 3.0 restore | 2.3258e-08 | 0.9913 | 10/10 | 23.62% | 0.9509 | **0.9426** | 10/10 |
| E2 satrow 3.0 sep | 2.3093e-08 | 0.9843 | 10/10 | 23.62% | 0.9509 | **0.9359** | 10/10 |
| E2 satrow 2.5 sep | 2.2064e-08 | 0.9405 | 9/10 | 23.14% | 0.9318 | **0.8763** | 10/10 |
| E3 recv -3.0 a | 2.3746e-08 | 1.0121 | 1/10 | 23.45% | 0.9442 | **0.9556** | 10/10 |
| E3 recv -3.0 b | 2.3502e-08 | 1.0017 | 4/10 | 23.58% | 0.9494 | **0.9510** | 10/10 |
| E3 recv -3.25 a | 2.3512e-08 | 1.0022 | 3/10 | 23.65% | 0.9522 | **0.9543** | 10/10 |
| E3 recv -3.5 a | 2.3510e-08 | 1.0021 | 4/10 | 23.83% | 0.9596 | **0.9615** | 10/10 |
| E3 recv -3.0 zero | 2.3864e-08 | 1.0172 | 1/10 | 23.45% | 0.9442 | **0.9604** | 10/10 |
| E4 agecut (4, -2.0) | 2.5233e-08 | 1.0755 | 0/10 | 22.73% | 0.9150 | **0.9841** | 7/10 |
| E4 agecut (4, -1.5) | 3.4586e-08 | 1.4741 | 0/10 | 20.93% | 0.8429 | **1.2425** | 0/10 |

base m225: raw 2.4151e-08
| arm | raw | raw x base | better | util (model) | util x base | projected x base | projected-better |
|---|---|---|---|---|---|---|---|
| E2 satrow 3.0 restore | 2.3950e-08 | 0.9917 | 10/10 | 21.43% | 0.9520 | **0.9440** | 10/10 |
| E2 satrow 3.0 sep | 2.3778e-08 | 0.9845 | 10/10 | 21.43% | 0.9520 | **0.9372** | 10/10 |
| E2 satrow 2.5 restore | 2.3026e-08 | 0.9534 | 9/10 | 21.01% | 0.9331 | **0.8896** | 10/10 |
| E2 satrow 2.5 sep | 2.2707e-08 | 0.9402 | 9/10 | 21.01% | 0.9331 | **0.8773** | 10/10 |
| E2 satrow 2.75 restore | 2.3613e-08 | 0.9777 | 9/10 | 21.24% | 0.9432 | **0.9222** | 10/10 |
| E2 satrow 2.75 sep | 2.3323e-08 | 0.9657 | 9/10 | 21.24% | 0.9432 | **0.9109** | 10/10 |
| E2 satrow 3.25 restore | 2.4077e-08 | 0.9969 | 10/10 | 21.60% | 0.9595 | **0.9566** | 10/10 |
| E2 satrow 3.5 restore | 2.4134e-08 | 0.9993 | 7/10 | 21.75% | 0.9661 | **0.9655** | 10/10 |
| E2 satrow 2.25 restore | 2.3057e-08 | 0.9547 | 7/10 | 20.76% | 0.9221 | **0.8803** | 9/10 |
| E2 satrow 2.25 sep | 2.3177e-08 | 0.9596 | 7/10 | 20.76% | 0.9221 | **0.8849** | 9/10 |
| E2 satrow 2.0 restore | 2.6214e-08 | 1.0854 | 3/10 | 20.48% | 0.9098 | **0.9876** | 6/10 |
| E2 satrow 2.0 sep | 2.8053e-08 | 1.1616 | 1/10 | 20.48% | 0.9098 | **1.0568** | 5/10 |
| E2 satrow 2.5 restore ch=cov | 2.3499e-08 | 0.9730 | 9/10 | 21.01% | 0.9331 | **0.9079** | 10/10 |
| E2 satrow 2.5 restore ch=leg | 2.3811e-08 | 0.9859 | 8/10 | 21.01% | 0.9331 | **0.9200** | 10/10 |
| E3 recv -3.0 a | 2.4399e-08 | 1.0102 | 2/10 | 21.28% | 0.9453 | **0.9550** | 10/10 |
| E3 recv -3.0 b | 2.4199e-08 | 1.0020 | 4/10 | 21.41% | 0.9511 | **0.9530** | 10/10 |
| E3 recv -3.25 a | 2.4193e-08 | 1.0017 | 3/10 | 21.46% | 0.9533 | **0.9549** | 10/10 |
| E3 recv -3.5 a | 2.4195e-08 | 1.0018 | 5/10 | 21.62% | 0.9605 | **0.9622** | 10/10 |
| E3 recv -3.0 zero | 2.4523e-08 | 1.0154 | 3/10 | 21.28% | 0.9453 | **0.9599** | 10/10 |
| E4 agecut (4, -2.0) | 2.5497e-08 | 1.0557 | 0/10 | 21.70% | 0.9640 | **1.0177** | 2/10 |
| E4 agecut (4, -1.5) | 3.4795e-08 | 1.4407 | 0/10 | 19.91% | 0.8844 | **1.2741** | 0/10 |
| margtail 2.5 | 2.4149e-08 | 0.9999 | 5/10 | 22.51% | 1.0000 | **0.9999** | 5/10 |
| margtail 3.0 | 2.4112e-08 | 0.9984 | 7/10 | 22.51% | 1.0000 | **0.9984** | 7/10 |
<!-- /TABLE_T3 -->

### E2 — the saturated-row cut of the D pairing, with the κ₄ level restored: **a lever, and an accuracy gain**
`chain.py satrow=θ` zeroes the rows i of D with α_i > θ at the current layer (production: those rows of
Σ_s L1_s B_sᵀ + L2_s A_sᵀ are not computed); `satrow_k4` restores the κ₄ level the next layer reads:
`"restore"` = a second step on the full D, only its K22 kept (exact, the upper bound — not buildable cheaply);
`"sep"` = the production form below.
- **The naive cut's damage (Mac: ×1.0094 at 3.0, ×1.34 at 2.5, ×3.07 at 2.0) was entirely the κ₄ level.** With the
  level held exactly, the cut IMPROVES raw at every threshold, and the gain grows with the cut fraction until ≈2.5:
  3.5 ×0.9993, 3.25 ×0.9969 (10/10), 3.0 ×0.9917 (10/10), 2.75 ×0.9777 (9/10), 2.5 ×0.9534 (9/10),
  2.25 ×0.9547 (7/10), and it turns at 2.0 (×1.0854). On plain k3m (no matched cut) the same: 3.0 restore ×0.9913
  (10/10), sep 3.0 ×0.9843 (10/10), **sep 2.5 ×0.9405 (9/10)** against ×0.9402 on the matched base — E2 does not
  depend on the matched cut and can ship on the current k3m layout.
- **Mechanism — measured by channel** (`satrow_ch`, θ = 2.5, exact level in every arm; the cut D reaches ONE channel,
  everything else sees the full D): **covariance only (K11 → the next C) ×0.9730 (9/10); correction + path leg births
  only (K21 and the births' D) ×0.9859 (8/10); both (the full cut) ×0.9534** — the product of the two channels, ×0.9593,
  is within 0.6% of the full cut, so ~60% of the gain is in the covariance and ~30% in the births.
  **What it is not:** the marginal Edgeworth corrections of the tail coefficients are large — measured on MLP 0
  (matched base) for α > 2.5 the correction of coef(2,1) = d2 + d5 t/6 + d6 K4d/24 is −28% (layer 8), −40% (11),
  −79% (14) of d2 itself (median; 90th percentile 1.1–3.3×), and in the dead tail it is 1–2× the Gaussian mean (E3
  below) — **but switching those corrections off for |α| ≥ 2.5 in every channel (side probe `margtail=2.5`) is neutral,
  ×0.9999 (5/10).** So the gain comes from removing the saturated rows of D themselves from the step's pair terms, not
  from their weights. [U] Our reading: for a near-identity neuron i the exact pair terms involve κ₃ through the OTHER
  index (d2_j D_ji); the Wick expansion's i-curvature terms (d2_i D_ij, the pass-through in K21) are small but coherent
  over ~20% of the rows, and the chain is hypersensitive to coherent covariance changes (NEGATIVE §57: ×2.7 per
  coherent 1e-4) — the same *consistency* reading as the matched dead cut (METHOD §22): a neuron in a tail regime is
  better treated consistently than through small, systematically biased terms. The mechanism is not proven; the
  measured facts are the monotone curve, the channel split and the margtail neutrality.
- **Production form (`"sep"`), O(S n²):** the next layer reads K22post only through the level
  Ssum = Σ u_i u_j K22post_ij (u = row sums of W_{l+1}∘W_{l+1}). Its linear response to the cut rows, dropping the
  d2- and d3-suppressed terms, is separable:
  **dSsum = Σ_{i∈sat} u_i coef(2,2)_i (D (u∘β))_i, β = coef(1,2) − 2 P1 coef(1,1)** (β is the correction leg's α,
  marginally corrected), and D_sat·v = Σ_s L1_s[sat](B_sᵀv) + L2_s[sat](A_sᵀv) + the thin and feed terms needs only
  one n-vector per leg. The chain then adds dSsum to the next layer's Ssum (in `chain.py`: a constant off-diagonal
  shift of K22post, which only the level reads). The exact linear response also has C0∘D and Q11∘D terms (autograd on a
  random n = 256 chain: the separable part is 50–87% of it); computing those would cost as much as the rows it saves.
  **On the real MLP 0 (θ = 2.5, matched base) the saturated rows carry +3.4% (layer 6), +5.4% (9), +7.2% (12), +5.7%
  (14) of the level Ssum** — which is why the naive cut was catastrophic — **and sep recovers 96%, 93%, 92%, 88% of
  it.**
- **sep beats restore:** 3.0 ×0.9845 vs ×0.9917, 2.5 ×0.9402 vs ×0.9534 (9/10); on plain k3m 3.0: sep ×0.9843 vs
  restore ×0.9913 (10/10). sep leaves the κ₄ level 0.2–0.8% of Ssum below the exact one (the numbers just above); **that extra ×0.986 is a level effect — the class that arrived at ⅓ strength
  on the grader (NEGATIVE §60, k3n)**. The projection below uses sep's measured number (it is what production
  computes), and §5 states what is left if the level part does not transfer.

### E3 — receiver rows (`recv`): survives only with the repeated-index fill-in, or at −3.25
Rows of neurons with α_l < θ_out at the CURRENT layer are not computed in the leg transport (they are gated to 0 at
the end of the layer anyway, since θ_out < deadcut, so only this layer's readout of them changes).
- arm a (their t_i from the age-1 source only): −3.0 **×1.0102 — killed** (> +0.5%); −3.25 ×1.0017, −3.5 ×1.0018.
- arm b (plus the exact repeated-index rule on the previous layer's (t, D) for all older sources): −3.0 **×1.0020**.
- on plain k3m the same picture: arm a −3.0 ×1.0121 (killed), arm b −3.0 ×1.0017 → projected ×0.951; arm a −3.25
  ×1.0022 → ×0.954, −3.5 ×1.0021 → ×0.962; zero −3.0 ×1.0172.
- "zero" (t_i and the star κ₄ row dropped): −3.0 ×1.0154.
- **Mechanism (measured on MLP 0, matched base):** for the neurons with α < −3 the first-order Edgeworth correction
  of the post mean, d3·t/6, is **−1.0× (layer 8), −1.2× (11), −2.1× (14) the Gaussian mean itself** (median;
  90th percentile 2.4–7.7×; |t|/σ³ ≈ 0.10–0.15). So the dead neurons' own readout is not negligible — it is the
  largest term of their mean — and it reaches the next layer through their mean and variance (never through κ₃: they
  are gated). The repeated-index rule recovers 80% of the loss. (The marginal Edgeworth series is at the edge of validity
  in both tails; for E2 that turned out not to be the mechanism — see the margtail probe above.)
- **Production:** α_l must be known before the leg transport, so the fused Wt·C product runs first (same FLOPs);
  every dense leg except the age-1 source's B (and C, the path leg, the thin H) is transported only onto the rows
  α_l ≥ θ_out (2-D compaction, both Wᵀ index sets); the rows not computed are zero, so the standard readout yields
  the age-1-only t_i by itself; arm b adds t_rep = (M∘M∘M)t′ + 3((M∘M)[rows]·D′ ∘ M[rows])1 from the previous layer's
  gated slices ((1 − g) of a product, plain).

### E4 — the age-dependent matched cut (`agecut`): dead
Legs of age ≥ 4 are cut harder than the young ones (rows at the gating, their D contribution over alive_old², their
birth columns from age 4 on); young legs keep −2.25. On the matched base: **(4, −2.0) raw ×1.0557 (0/10) for util
×0.964 → projected ×1.018** — worse than doing nothing, and (4, −1.5) ×1.4407 → ×1.274. On plain k3m (where the old
legs' harsher cut also brings a column cut the base does not have): (4, −2.0) ×1.0755 → ×0.984, (4, −1.5) ×1.4741 →
×1.243 — killed on both bases (×0.97 rule). [U] The mechanism is plausibly the matched cut's own
one read backwards: the consistency gain needs row and column cut *at the same threshold on the same neuron*; a second, harsher
threshold for old legs makes a neuron dead for some sources and alive for others — the inconsistency the matched cut
removed (METHOD §22) — and sources older than 4 carry 69% of ‖t‖ at layer 15 (cost-frontier atlas, NEGATIVE §16),
so the loss is large.

## 4. Task 4 — the greedy stack on k3m + matched cut (`t4_stack.py`, every step re-measured on 10 MLPs)

Start = the matched cut at −2.25 (treated as landed, as the brief says). At every step each surviving lever is added
to the current stack and measured; the best projected adjusted is kept. Levers do not multiply (the brief's warning),
so no step's number is extrapolated from another's.

**The greedy frontier** (each row re-measured on 10 MLPs with everything above it switched on; projected adjusted =
5.3906e-9 × raw/2.3462e-8 × util/24.835%; util modelled by `cost_pb.py`):

| step | lever added — the exact switch settings the Mac mirrors | raw | raw × k3m | better vs k3m | util | util × k3m | **projected** | × k3m | + L5 ×0.9165 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | k3m (graded 332403) | 2.3462e-8 | 1.0000 | — | 24.84% | 1.0000 | 5.3906e-9 | 1.000 | — |
| 1 | matched dead cut: `deadcut=-2.25, colcut=(-2.25, inf)` | 2.4151e-8 | 1.0294 | 1/10 | 22.51% | 0.9065 | **5.030e-9** | 0.933 | 4.610e-9 |
| 2 | + E2: `satrow=2.5, satrow_k4="sep"` | 2.2707e-8 | 0.9678 | 7/10 | 21.01% | 0.8460 | **4.413e-9** | 0.819 | 4.045e-9 |
| 3 | + band: `bandz=dict(a0=5, r=448, k=None, proj="qr", entry="pre", init="rr", dtype=float32)` | 2.3084e-8 | 0.9839 | 6/10 | 18.47% | 0.7436 | **3.944e-9** | 0.732 | 3.614e-9 |
| 4 | + E3: `recv=(-3.0, "b")` | 2.3092e-8 | 0.9842 | 6/10 | 17.73% | 0.7141 | **3.789e-9** | 0.703 | **3.472e-9** |
| — | + E4 | killed alone (§3: ×1.018, ×1.274) | | | | | | | |

**What each step chose against (all measured, same base as the step):**
- step 2 (on the matched cut): E2 sep 2.5 **4.413e-9** · band (5,448) rr 4.457e-9 · band (6,384) pre 4.455e-9 ·
  band (6,448) pre 4.546e-9 · band (5,512) pre 4.615e-9 · E3 b −3.0 4.794e-9 · E4 > base.
- step 3 (on matched + E2): band (5,448) rr **3.944e-9** · band (6,384) pre 3.952e-9 · band (6,448) pre 4.023e-9 ·
  E3 b −3.0 4.183e-9.
- step 4 (on matched + E2 + band): E3 b −3.0 **3.789e-9**; with the (6,384) pre band instead 3.798e-9 (raw ×1.0129,
  util 17.28%) — the two bands are tied at the end of the stack.
- **The levers nearly multiply here after all:** E2 alone ×0.877 on the matched base, band alone ×0.886, E3 alone
  ×0.953; product ×0.741 vs the measured ×0.753 (step 1 → 4). E3's accuracy cost vanishes on top of E2 (×1.0003).

**The quick-ship stack — only levers that work on the CURRENT k3m layout** (no wide layout, no band;
`t4_k3m_quick.py`): k3m + E2 (`satrow=2.5, satrow_k4="sep"`) = raw ×0.9405 (9/10), util 23.14% → **4.724e-9**;
k3m + E2 + E3 (`recv=(-3.0, "b")`) = raw **×0.9414 (9/10)**, util 21.89% → **4.472e-9 (×0.830)**, L5 4.099e-9.

**The final stack with the matched cut at −2.5** (the brief's repeat): raw **2.2392e-8 = ×0.9544 (8/10 better than
k3m)**, util 18.34% → **3.800e-9** (L5 3.482e-9) — the same projected score as at −2.25 (3.789e-9) with a 3% better
raw. The choice of threshold no longer matters for the score; −2.5 is the more robust one (less of the projected gain
rests on the util model, more on measured accuracy). With the (6,384) band at −2.5: raw ×0.9797, util 17.93% →
3.813e-9 (L5 3.495e-9). All four end points (two thresholds × two bands) lie within 3.79–3.81e-9.

**Strassen level 5.** The final stack's modelled util is 17.73% (16.25% with L5). The brief's admissibility line is
~17%: NEGATIVE §56's wall veto was ≈119 s mean / ≈147 s worst at 24.8% util against the 120 s cap; if the grader
wall scales with the meter, 17.7% gives ≈85 s mean / ≈105 s worst at L5 — inside the cap but with ~13% margin on the
worst MLP, and the band/E3 add non-FLOP work (QR, gathers) that the meter does not see. **L5 is a Mac decision on
measured wall time**; the projection is given both ways. (The ×0.9165 applies to everything on the Strassen stacks;
the stack's plain part — the band's QR/solve and E3's t_rep, 9.4 of 176.3 billed model units — does not shrink, so the
true factor is ≈×0.921.)

<!-- TABLE_T4 -->
| file | stack (on the matched cut) | raw | raw x k3m | better vs k3m | util (model) | util x k3m | projected adjusted | x k3m | + L5 x0.9165 |
|---|---|---|---|---|---|---|---|---|---|
| — | k3m (graded 332403) | 2.3462e-08 | 1.0000 | — | 24.84% | 1.0000 | 5.3906e-09 | 1.000 | — |
| t3_base_m225 | matched −2.25 | 2.4151e-08 | 1.0294 | — | 22.51% | 0.9065 | 5.0304e-09 | 0.933 | 4.6104e-09 |
| t4_step1d_n10 | t=-2.25 band_5_512_pre | 2.4216e-08 | 1.0321 | 0/10 | 20.60% | 0.8295 | **4.6151e-09** | 0.856 | 4.2297e-09 |
| t4_step1_n10 | t=-2.25 band_6_448_pre | 2.4416e-08 | 1.0407 | 0/10 | 20.13% | 0.8104 | **4.5461e-09** | 0.843 | 4.1665e-09 |
| t4_k3m_quick_n10 | k3m + E2 sep 2.5 + E3 b -3.0 | 2.2087e-08 | 0.9414 | 9/10 | 21.89% | 0.8812 | **4.4720e-09** | 0.830 | 4.0986e-09 |
| t4_step1b_n10 | t=-2.25 band_5_448_pre_rr | 2.4593e-08 | 1.0482 | 0/10 | 19.59% | 0.7887 | **4.4566e-09** | 0.827 | 4.0844e-09 |
| t4_step1c_n10 | t=-2.25 band_6_384_pre | 2.5195e-08 | 1.0739 | 0/10 | 19.11% | 0.7696 | **4.4552e-09** | 0.826 | 4.0832e-09 |
| t4_step2d_n10 | t=-2.25 e2_2.5_sep + e3_-3.0_b | 2.2714e-08 | 0.9681 | 7/10 | 19.91% | 0.8016 | **4.1831e-09** | 0.776 | 3.8338e-09 |
| t4_step2c_n10 | t=-2.25 e2_2.5_sep + band_6_448_pre | 2.2958e-08 | 0.9785 | 6/10 | 18.94% | 0.7627 | **4.0230e-09** | 0.746 | 3.6871e-09 |
| t4_step2a_n10 | t=-2.25 e2_2.5_sep + band_6_384_pre | 2.3734e-08 | 1.0116 | 4/10 | 18.00% | 0.7247 | **3.9516e-09** | 0.733 | 3.6217e-09 |
| t4_step2b_n10 | t=-2.25 e2_2.5_sep + band_5_448_pre_rr | 2.3084e-08 | 0.9839 | 6/10 | 18.47% | 0.7436 | **3.9437e-09** | 0.732 | 3.6144e-09 |
| t4_final25b_n10 | t=-2.5 e2_2.5_sep + band_6_384_pre + e3_-3.0_b | 2.2985e-08 | 0.9797 | 7/10 | 17.93% | 0.7220 | **3.8128e-09** | 0.707 | 3.4945e-09 |
| t4_final25a_n10 | t=-2.5 e2_2.5_sep + band_5_448_pre_rr + e3_-3.0_b | 2.2392e-08 | 0.9544 | 8/10 | 18.34% | 0.7385 | **3.7995e-09** | 0.705 | 3.4822e-09 |
| t4_step3b_n10 | t=-2.25 e2_2.5_sep + band_6_384_pre + e3_-3.0_b | 2.3764e-08 | 1.0129 | 4/10 | 17.28% | 0.6957 | **3.7984e-09** | 0.705 | 3.4812e-09 |
| t4_step3a_n10 | t=-2.25 e2_2.5_sep + band_5_448_pre_rr + e3_-3.0_b | 2.3092e-08 | 0.9842 | 6/10 | 17.73% | 0.7141 | **3.7888e-09** | 0.703 | 3.4724e-09 |
<!-- /TABLE_T4 -->

## 5. Verdict

**The full exact stack — matched cut + E2 + band + E3 — projects to 3.79e-9 (3.47e-9 with Strassen L5), ×0.703
(×0.644) of k3m's graded 5.3906e-9: from rank ≈69 to ≈ rank 22–27 on the 09-29 board (#16 = 3.0e-9, #29 = 4.0e-9).
It cannot reach 3.0e-9** — that needs ×0.556 of k3m, the stack delivers ×0.64–0.70, and L5 is the last lever on the
table. **The gap to the top-5 cut of 2.1e-9 is ×1.80 (×1.65 with L5)**, and the cut is moving to ≈1.9–2.0e-9 by
17 Oct. It is not an engineering distance: the stack's util (17.7–18.3%) is still ×1.8 above the 10% floor, *and* its
raw (2.24–2.31e-8 torch ≈ 2.07–2.13e-8 graded at k3m's ratio 0.923) is ×1.2–1.7 worse than the leaders' 1.26–1.70e-8 —
even this chain at the floor would only reach today's #5 cut (≈2.07–2.13e-9), not the one at the close. **The ports fit by ~10 Oct** if k3o lands in the next days: E2 is small and does not
need the wide layout (a second row index set for the D pairing plus one O(S n²) scalar per layer), E3 is about a day,
the band is the only multi-day port (2–3 days on k3o's wide layout; float32 is safe as measured). **The first thing to ship is E2 on the current k3m** (k3m + `satrow=2.5, satrow_k4="sep"`: raw ×0.9405 on 9/10,
projected **4.72e-9**, a small port that needs neither k3o nor the band; with E3 too, 4.47e-9) — it is the largest single lever of this
battery and can be validated on the grader while the rest is built. **Recommendation: the
last week goes to hardening and shipping this stack** — it is worth ×0.70, a ~40-rank move, and nothing measured here
or in NEGATIVE §54–§66 points to a new method that could be found, ported and validated in one week. The transfer
risk to watch is E2's accuracy gain: its level part is the class that arrived at ⅓ strength on the grader (§60) —
with that discount the stack is ≈3.82e-9, and with no E2 accuracy gain at all ≈4.03e-9 (still ×0.75).

What would change this verdict (none of it is engineering): a raw of ≈1.6e-8 at the same cost — i.e. the accuracy
the leaders have and our chain does not have at any cost (NEGATIVE §62's puzzle) — would put this stack at ≈2.8e-9.

## 6. Ranked port list — what the Mac mirrors, exactly

Ranked by projected gain on the matched base ÷ porting effort. Every lever is exact in the sense of the brief (no
fitted constant; thresholds from these 10 MLPs), and every one has a torch per-MLP reference to port against
(`t3_*_n10.json`, `t4_*_n10.json`).

| # | lever | torch switches (all other switches = the k3m BASE) | raw × (its base) | util × | projected × | construction | effort |
|---|---|---|---|---|---|---|---|
| 1 | **E2** saturated-row cut + level correction | `satrow=2.5, satrow_k4="sep"` | ×0.9402 (9/10) on matched; **×0.9405 (9/10) on plain k3m** | ×0.933 | **×0.877 / ×0.876** | none (rows not computed) + O(S n²) scalar | small — works on the k3m layout: **k3m + E2 alone projects to 4.72e-9** |
| 2 | **matched dead cut** (k3o, in progress) | `deadcut=t, colcut=(t, inf)`, t ∈ {−2.25, −2.5} | ×1.0294 / ×0.9928 | ×0.9065 / ×0.9502 | ×0.933 / ×0.943 | wide (n × Σw) leg layout | in progress |
| 3 | **band** | `bandz=dict(a0=5, r=448, k=None, proj="qr", entry="pre", init="rr", dtype=torch.float32)` — tied alternative `a0=6, r=384, init="cols"` | ×1.018 (matched), ×1.0196 (k3m) | ×0.87 | ×0.886 | QR of P + one r × 2n solve per layer, materialise P·Z, compressed pairing: 71% of its gross saving | 2–3 days on k3o |
| 4 | **E3** receiver rows | `recv=(-3.0, "b")` (or `recv=(-3.25, "a")`: ×0.955, no t_rep) | ×1.0020 alone, ×1.0003 on E2 | ×0.951 | ×0.953 | unfused Wt·C first; t_rep (1 − g of a plain product) | ~1 day |
| 5 | **Strassen L5** | (production parameter) | 1.0 | ×0.921 | ×0.921 | none | wall-time decision on the Mac at util ≈17.7% |
| — | dead | E4 (`agecut`) both thresholds; band with the normal equations and no re-orthonormalisation; any band with a0 = 4; E2 below 2.5 (2.0: ×1.09 restore, ×1.16 sep) | | | | | |

**Recipes the Mac needs beyond the switch names:**
- **E2 in production.** At layer l (not the last): sat = {i: α_l,i > 2.5}; the D pairing's LEFT-factor rows are
  alive ∧ ¬sat, its RIGHT-factor rows alive (two index sets for the rectangular Strassen, instead of one); D's sat rows
  are zero. Level correction for the next layer: with u = row sums of W_{l+1}∘W_{l+1},
  β = coef(1,2) − 2·P1·coef(1,1) and γ = coef(2,2) (marginally corrected pair coefficients: w(k,p) + w(k+3,p)·t/6 +
  w(k+4,p)·K4d/24, P1 the corrected post mean), v = u∘β∘alive: **dS = Σ_{i∈sat} u_i γ_i (D_full v)_i** with
  (D_full v)_sat = Σ_s L1_s[sat]·(B_sᵀv) + L2_s[sat]·(A_sᵀv) + the thin term (A²Utᵀ)[sat]·(Hᵀv) + the feed terms
  (q mᵀ, 2m B d1b Aᵀ) and the path-merge terms that ride L1/L2, minus the diagonal term v_i·D_ii (D is zero-diagonal;
  D_ii = rowsum(L1∘B + L2∘A + …)) — O(S n²). Add dS to the next layer's Ssum (the k4lvl level; trM follows through
  Ssum). The torch mirror drops coef(2,2)'s −2·P1·coef(2,1) partner and the C0∘D, Q11∘D terms of the exact response
  (they cost as much as the rows they would restore); `satrow=1e9` must reproduce k3m bitwise.
- **Band in production.** The first entrant (born at layer 0) reaches age 4 at the end of layer 4: P = qr([A B]·Ω,
  one power step).Q (r = 448 columns; `chain.py _band_enter`, seed irrelevant); from then on P rides the leg stack
  (gated by d1, transported by Wᵀ, dead rows cut like a leg). At the end of every layer ≥ 4 the source turning 5 is
  projected onto the gated P: Q, R = qr(P) (float32), Z = solve(R, Qᵀ[A | B | H]) — no re-orthonormalisation of P and
  no coefficient rotation, ever (cond(P) reaches 1.2e5 at layer 15 and the float32 legs stay within 3e-4 of float64).
  Members are materialised each layer by one shared-left product P @ [Z_A Z_B …] into their leg slots (the readout and
  L1, L2 need them element-wise); their pairing is compressed: (Σ_members L1 Z_Bᵀ + L2 Z_Aᵀ) then · Pᵀ.
- **E3 in production.** α_l is needed before the leg transport: run the fused Wt·C product first (same FLOPs), take
  rows R = {α_l ≥ −3.0}; every dense leg except the age-1 source's B, C and the path leg is transported only onto R
  (2-D compaction with the dead-row cut's column set); rows outside R are zero, so the standard readout gives their
  age-1-only t_i automatically; add t_rep = (M∘M∘M)·t′ + 3·rowsum(((M∘M)[¬R]·D′) ∘ M[¬R]) with M = Wᵀ and the previous
  layer's gated t′ = d1³t, D′ = d1² D d1 (zero diagonal).

## 7. Files, switches, how to reproduce

| file | what |
|---|---|
| `../2026-09-24_k3-chain/chain.py` | new switches, all default off (gates exact): `bandz` (`_band_step`, `_band_enter`, `_band_enter_pre`), `satrow_k4` ("restore" / "sep"), `satrow_ch` (E2 channel test), `recv` (E3), `agecut` (E4), `margtail` (side probe, §8) |
| `pb.py` | harness: the k3m BASE, the matched cut, data, per-MLP arm runner with JSON caching |
| `cost_pb.py` | the honest per-source cost model (§2), α tables from `alpha_dump.py` → `alpha_fractions_k3m.json` |
| `t1_band.py`, `t1_rest.py` | task 1 grid (float32, lazy QR, ne/qr) → `t1_band_n10_*.json` |
| `t1b_band.py`, `t1c_band.py` | the band frontier: pre entry, r = 384–512, a0 = 4–6, range-finder init → `t1b_*`, `t1c_*` |
| `t2_price.py` | prices every band arm and applies the ×0.97 kill → `t2_price.json` |
| `t3_bases.py`, `t3_battery.py` | E2/E3/E4 (+ E2 channels, margtail) on both bases → `t3_*_n10.json` |
| `t4_stack.py` | the greedy stack (candidates = switch sets + cost kwargs) → `t4_*_n10.json` |
| `make_tables.py` | renders every table of this file from the JSON |
| `queue.sh`, `watch*.sh`, `jobs1.txt*` | the 4-worker job queue used here (1 thread per job) |
| `logs/` | gates, every job's stdout |

```bash
cd experiments/2026-09-29_port-battery
PB_THREADS=1 ../../.venv-torch/bin/python t4_stack.py t4_step3a_n10.json -2.25 e2_2.5_sep band_5_448_pre_rr e3_-3.0_b
../../.venv-torch/bin/python make_tables.py all
```

## 8. Side probe (not in the brief): the tail marginal-Edgeworth corrections

Suggested by the E2/E3 measurements (the first-order Edgeworth corrections of the pair coefficients are 30–80% of the
leading term at α > 2.5, and 1–2× the Gaussian mean at α < −3). `chain.py margtail=θ` drops the marginal correction of
the pair Wick coefficients for |α| ≥ θ (zero FLOPs). On the matched base: **θ = 2.5 ×0.9999 (5/10) — neutral**,
θ = 3.0 ×0.9984 (7/10) — below any resolution. Not a lever; its value is the negative control for the E2 mechanism (§3).
