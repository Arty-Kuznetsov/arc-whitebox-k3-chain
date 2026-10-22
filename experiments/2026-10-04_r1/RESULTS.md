# R1 — the minimal consistent set of the next order, at small width (2026-10-04)

Plan: `notes/PLAN-2026-10-04.md` (R1, kill rule: some PROPER subset must reach within ×1.2 of the full joint gain).
Instrument: ARC's k_max = 4 SIMPLE (full κ₄, factored) stepped layer by layer (`r1lib.k4_stats`), its pre-activation
statistics substituted into our float64 chain (research base B9 + marg2) at layers 1 … L−2 (the last step is the
chain's own), final-layer MSE against Monte-Carlo (1e7, `../2026-10-03_korder/res/`), MC noise var/N subtracted.
Gates: the stepped k = 4 reproduces `mlp_kprop`'s k4S to the last digit (5.542e-08, n=256 net 0); the base arm = korder's
B9m2 (3.1499e-07); V+t+K4 substituted at EVERY layer reproduces k4S exactly (the mean depends only on var, t, K4d);
`chain.py` gate k4memo 3 = 3.7533e-08 after every edit.

## 1. The full factorial (`r1fact.py`, `r1analyze.py`): 8 channels × 256 subsets
Channels: V (variance), C0 (off-diagonal covariance), t (κ₃ diag), D (κ₃ (2,1)), K4 (κ₄ diag), K22 (κ₄ (2,2)),
K31 (κ₄ (3,1), new oracle key `K31_layers`), S (κ₄ (2,1,1) feeding the κ₄→κ₃ births, `dense3`).

| | n = 256, L = 8 (8 nets) | n = 128, L = 8 (8 nets: 6 complete in the table run) |
|---|---|---|
| k = 4 itself | ×0.161 | ×0.360 |
| all eight upstream | ×0.313 | ×0.485 |
| pass line (×1.2) | ×0.375 | ×0.582 |
| best single | K4 ×0.824 | K31 ×0.851 |
| **K4 + K31** | **×0.639** | **×0.595** |
| K4 + K31 + S ("the κ₄ world") | ×0.479 (per net 0.35–0.62) | — |
| leave-one-out from all eight | −K4 ×1.605, −t ×1.185, −K31 ×1.125, −S ×1.122, −D ×1.064; −V, −C0, −K22 ×1.00 | −K4 ×1.595, −t ×1.130, −K31 ×1.114; −S ×0.976, −K22 ×0.956 |
| minimal passing subsets | 9, all of 5 channels (e.g. t+D+K4+K22+K31 ×0.347, t+D+K4+K31+S ×0.361) | 6, from 3 channels (C0+K4+K31 ×0.531, D+K4+K31 ×0.532, V+K4+K31 ×0.551) |
| log-linear fit (main + pairwise, R² 0.99) | strongest synergy K4×K31 ×0.912 | K4×K31 ×0.871 |

- **The covariance and K22 follow by themselves** when the higher readouts are right (removing V, C0 or K22 from the
  full set costs nothing); V and C0 are antagonistic with K31 and D (they fix the same error).
- **The core everywhere is K4 + K31**: the κ₄ diagonal and its (3,1) slice, the strongest synergy of all pairs.
- Kill rule R1: **passes** (proper subsets within ×1.2 at both widths).

## 2. How far back the κ₄ reaches (`r1age.py`): ARC's κ₄ by birth block
ARC appends per ReLU a TREE block (paths, stars, 3+2) and a CORRECTION block (exact repeated slices); columns keep their
place, so a block's age is known. Substituted: K4+K31+S = young blocks (age ≤ a) exact + the older blocks' least-squares
fit on the scale-mixture pattern (the closure carrying the old part at its level).

| n = 256 | net 0 | net 1 | net 2 |
|---|---|---|---|
| κ₄ world (exact) | ×0.598 | ×0.427 | ×0.535 |
| closure fit only (right level, closure pattern) | ×1.176 | ×1.067 | ×0.984 |
| young ≤ 1 + closure | ×0.833 | ×0.548 | ×0.564 |
| **young ≤ 2 + closure** | **×0.609** | **×0.471** | **×0.602** |
| young ≤ 3 + closure | ×0.615 | ×0.435 | ×0.551 |

The closure's residual (R² of the closure fit: K4d 0.48–0.73, K31 0.47–0.61, S 0.27–0.43) sits in the young blocks:
age ≤ 2 holds 68–100% of it, age ≤ 3 85–100%. **The pattern, not the level** (the closure at k = 4's level is worse).

## 3. Which births (`r1types.py`, `r1k4split.py`: ARC's tree block split into path | star | 3+2 by exact source edits)
Young ≤ 2 + closure, by the types carried (K4+K31+S; n = 128 nets 0/1/2 | n = 256 nets 0/1):
| carried | n=128 | n=256 |
|---|---|---|
| all | 0.447 / 0.626 / 0.772 | 0.609 / 0.471 |
| without 3+2 | **1.063 / 1.762 / 1.827** | **1.254 / 0.909** |
| without path | 0.666 / 0.902 / 0.746 | 0.763 / 0.741 |
| without star | 0.527 / 0.706 / 0.893 | 0.685 / 0.579 |
| without correction (trees only) | 0.430 / 0.673 / 0.900 | 0.758 / 0.537 |
**The 3+2 tree — κ₃ handed on through an edge, 12 d1 C0 d2 κ₃(j,k,l) d1 d1 — is indispensable**; paths matter, stars
less; the correction block (where the κ₃→κ₄ transfer T3 of 10-03 lives) is secondary. This is why `t3k4` / `zlegs`
(NEGATIVE §74) did not pay: they carried the self-variance transfer, not the edge transfer.

## 4. The bridge to a carrier inside the chain (`r1res.py`)
The chain keeps its OWN closure (level and pattern) and receives only what the young objects hold beyond the closure's
pattern: X = X_chain + (Y − lsfit(Y, P) P) (new additive oracle keys `k4add`, `K31add`; S via `dense3` "add").
n = 128 nets 0/1/2, young ≤ 2 trees: K4+K31+S ×0.502 / 0.711 / 0.908; **K4+K31 only ×0.553 / 0.767 / 0.869** (exact K4+K31
×0.464 / 0.650 / 0.691). → R2 builds exactly this from the chain's own state.

## Verdict
R1 passes. The next order is reducible: a carrier of the **young (≤ 2 layers) κ₄ trees, the 3+2 above all**, read
into **K4d and K31** on top of the chain's closure, is the target. Continued in `../2026-10-04_r2/`.
