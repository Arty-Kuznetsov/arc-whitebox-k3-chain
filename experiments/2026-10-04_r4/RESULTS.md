# R4 — can the old κ₃ tier be made cheap? Three direct tests, all closed (2026-10-04, evening)

**Why.** On the live board (2026-10-04) many teams sit at OUR raw (≈2.0–2.3e-8) at 10–13.5% of the budget (lode_dockx
raw 2.34e-8 at ≈10.3%, a_s6 2.0e-8 at 12.5%, Puffi 2.0e-8 at 13.5%), against our 18.3%; and 504aldo's ledger (forum
18218, §6) puts the leaders' whole bill at about the price of a young tier alone. Our old tier (the band members:
materialisation 16.8% + compressed pairing 11.7% + entry 8.2%) is ≈37% of our FLOPs; our young tier alone would be
≈2.5e11 FLOPs ≈ J2W's 2.4e11. So: is our old tier cheaper than we carry it?
Harness `r4window.py`: production accuracy configuration (research base B9 + marg2 + band 5/448 rr), 10 baked MLPs at
1024 × 16, × of the band base's final-layer MSE (one file per (MLP, arm)).

## 1. Drop the oldest sources, with or without the amplitude-mode tail (`window`, `tail="amp"`)
| sources kept (W) | dropped member-layers | with amplitude tail | without |
|---|---|---|---|
| 12 (the 3 oldest gone) | 6/55 | ×1.169 (10/10 worse) | ×1.267 |
| 10 | 15/55 | ×1.849 | ×2.473 |
| 8 | 28/55 | ×3.784 | ×6.250 |
Even the three OLDEST sources (born at layers 0–2, read at 12–15) cost ×1.17. The amplitude closure (positive
homogeneity) removes about half of the damage at every W — the right shape, an order too weak (cf. NEGATIVE §39, §46).

## 2. The old tier only in the readout t, the (2,1) slice D from the y newest sources (`d21_ages`)
| y | ×  |
|---|---|
| 11 | ×1.066 |
| 9 | ×1.514 |
| 7 | ×3.254 |
| 5 (= the members' pairing dropped, −11.7% FLOPs) | ×9.251 |
The old tier is load-bearing in D as much as in t (cf. NEGATIVE §19 on the old chain).

## 3. Read the members through a smaller, optimal basis (`bandz readrank`: per layer, the members' materialised legs
projected on their own top-r' SVD subspace — the best rank-r' basis, re-fitted every layer; Z keeps the full 448)
| r' | × (10 MLPs) |
|---|---|
| 384 | ×1.00–1.02 |
| 320 | ×1.03–1.04 |
| 256 | ×1.11–1.15 |
| 192 | ×1.46–1.60 |
| 128 | ×2.7–3.1 |
| 64 | ×6.4–7.5 |
Even with an optimal basis the old tier needs ≈320–384 dimensions: the read cost can come down only ≈4–8% of the FLOPs
(r' 384/320) — a crumb, not the leaders' "old tier for nothing".

## Verdict
In our factorised representation the old tier is necessary in t and in D, at every age, and in ≈350–450 dimensions:
it costs what it costs (NEGATIVE §76). Same wall as 504aldo's F72–F76 (row-space law r/n ≈ 3/8 at age 4, flat
column-importance spectrum, capped atoms dead): **the cheap teams run a different expansion** — not a compression of
this one. Next: external evidence on what that expansion is (the organizers' IPAM talks, forum 18206; Puffi's SSC).
