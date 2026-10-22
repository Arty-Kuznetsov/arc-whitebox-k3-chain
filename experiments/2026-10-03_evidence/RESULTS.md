# Cloud #12 — evidence for the write-up on the FINAL stack (k3y) — RESULTS

Branch `claude/laughing-hopper-61l0bn` (continuation of #11), from main 6d3ecf4. RESULTS.md is regenerated from the JSONs
by `make_results.sh` (tables by `tables.py`) after every finished arm.

## 0. Gate (`runs/base_n3.json` vs `../2026-10-02_csym/runs/k3y_n100.json`)
k3y with its defaults, mini MLPs 0-2, flopscope, `--no-residual-wall-time-limit --wall-time-limit 600`:
**FLOPs per MLP equal the Mac's exactly** (4.13936942348e11, 4.07452290989e11, 4.00679637716e11); adjusted relative to the
Mac: −0.03%, −0.82%, −0.45% (x86 + MKL vs arm64 + Accelerate, float32; larger than CLOUD.md's ~1e-3 on MLP 1 — every
ratio below is cloud-vs-cloud, so it cancels).

## A. Production ablation of k3y (flopscope, exact FLOPs; mini MLPs 0-31; `runs/<arm>_n32.json`, wrappers `arms/`)
One constructor switch per arm vs k3y defaults. adjusted = graded metric (mean over MLPs of raw × max(0.1, util)),
raw = final-layer MSE, util = FLOPs / 2^41. "10-31" = the 22 networks outside every fitted choice (the κ4 corrector
was trained, and the thresholds tuned, on MLPs 0-9). Arms run one at a time (two in parallel OOM: 8.3 GB RSS each).
"not run" = not reached yet / out of time.

| arm | set | adjusted | raw | util | adj × | raw × | util × | MLPs worse (adj) |
|---|---|---|---|---|---|---|---|---|
| base | 0-31 | 3.8370e-09 | 2.0802e-08 | 18.44% | 1.0000 | 1.0000 | 1.0000 | 0/32 |
| base | 10-31 | 3.7621e-09 | 2.0405e-08 | 18.43% | 1.0000 | 1.0000 | 1.0000 | 0/22 |
| k4corr_off | 0-31 | 3.8876e-09 | 2.1090e-08 | 18.43% | 1.0132 | 1.0138 | 0.9994 | 23/32 |
| k4corr_off | 10-31 | 3.7984e-09 | 2.0616e-08 | 18.42% | 1.0097 | 1.0103 | 0.9994 | 15/22 |
| marg2_off | — | not run | | | | | | |
| satrow_off | — | not run | | | | | | |
| band_off | 0-31 | 4.2126e-09 | 2.0250e-08 | 20.80% | 1.0979 | 0.9735 | 1.1277 | 32/32 |
| band_off | 10-31 | 4.1314e-09 | 1.9874e-08 | 20.78% | 1.0982 | 0.9739 | 1.1274 | 22/22 |
| cuts_off | 0-31 | 5.4979e-09 | 2.5731e-08 | 21.37% | 1.4329 | 1.2370 | 1.1587 | 32/32 |
| cuts_off | 10-31 | 5.3430e-09 | 2.5007e-08 | 21.37% | 1.4202 | 1.2255 | 1.1592 | 22/22 |
| colcut_off | — | not run | | | | | | |
| l5_off | — | not run | | | | | | |
| csym_binv_off | — | not run | | | | | | |
| k31scale_1 | — | not run | | | | | | |
| feed_off | — | not run | | | | | | |
| paths_off | — | not run | | | | | | |
| c0d_off | — | not run | | | | | | |
| marg_off | — | not run | | | | | | |
| k31_off | — | not run | | | | | | |
| star_off | — | not run | | | | | | |
| c0k22_off | — | not run | | | | | | |

## B. Today's cost-free ceiling (torch chain, float64, the 10 baked MLPs, COST IGNORED; `b_ceiling.py`, `resB/`)
Base = B9 + marg2 (#9A arm `marg2_sat_m2.0`, reproduced bitwise per MLP: 2.15516e-08). Levers refused on cost only.

| arm (torch, 10 baked MLPs, cost ignored) | raw | raw × vs base | better | per-MLP × |
|---|---|---|---|---|
| base | 2.1552e-08 | 1.0000 | 0/10 | 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 1.000 |
| allages | 2.0664e-08 | 0.9588 | 8/10 | 1.003 1.060 0.942 0.989 0.903 0.961 0.940 0.950 0.957 0.890 |
| thin384 | 2.1131e-08 | 0.9805 | 10/10 | 0.956 0.988 0.977 0.989 0.975 0.960 0.997 0.997 0.980 0.986 |
| k4k3_0.5 | 2.0903e-08 | 0.9699 | 10/10 | 0.977 0.973 0.973 0.965 0.971 0.972 0.990 0.974 0.966 0.945 |
| stack | 1.9851e-08 | 0.9211 | 9/10 | 0.946 1.007 0.907 0.949 0.875 0.918 0.919 0.940 0.913 0.847 |

## C. The κ3 projection test — SKIPPED for budget (the sequential A run takes ~11 h of the remaining credit).

## Reading
- **Stack today, cost ignored (B): torch raw 1.985e-8 = ×0.921 of B9+marg2 (9/10)** — replaces the 09-28 figure 1.72e-8
  (which was measured on an older base with since-closed terms); the three refused levers roughly multiply
  (0.959 × 0.980 × 0.970 = 0.912 vs 0.921 measured). Note: k3y itself is graded at raw 2.03e-8.
- A: the band and the matched cuts are the big engineering levers (band off adjusted ×1.10, util ×1.13; cuts off ×1.43,
  and raw ×1.24 — the cuts are an ACCURACY gain, the consistency mechanism); the κ4 corrector is worth ×1.013 on all 32
  and ×1.010 out of its training set (MLPs 10-31) — it transfers. Remaining arms: see the table as it fills.


## A, completed on the Mac (2026-10-03; the cloud's container stopped after base + 3 arms)
`mac_runs.sh` (chunks of ≤4 arms), results in `runs_mac/`, table by `tables_mac.py` — every ratio Mac-vs-Mac (the
Mac base is bitwise the 10-02 k3y 100-MLP run on MLPs 0–31; Mac vs cloud per-network raw differs by up to ±4%,
FLOPs by ≤3e-6, so the two machines' arms are never compared with each other's base). The cloud's three arms
(k4corr_off, band_off, cuts_off) above stand cloud-vs-cloud. Write-up §5.2 merges both (`writeup/make_s52.py`).

| arm | set | adjusted | raw | util | adj × | raw × | util × | MLPs worse (adj) |
|---|---|---|---|---|---|---|---|---|
| base | 0-31 | 3.8372e-09 | 2.0804e-08 | 18.44% | 1.0000 | 1.0000 | 1.0000 | 0/32 |
| base | 10-31 | 3.7669e-09 | 2.0433e-08 | 18.43% | 1.0000 | 1.0000 | 1.0000 | 0/22 |
| k4corr_off | — | not run on the Mac | | | | | | |
| marg2_off | 0-31 | 3.9155e-09 | 2.1227e-08 | 18.44% | 1.0204 | 1.0203 | 1.0000 | 30/32 |
| marg2_off | 10-31 | 3.8433e-09 | 2.0845e-08 | 18.43% | 1.0203 | 1.0202 | 1.0000 | 21/22 |
| satrow_off | 0-31 | 4.3614e-09 | 2.2163e-08 | 19.67% | 1.1366 | 1.0653 | 1.0668 | 32/32 |
| satrow_off | 10-31 | 4.2863e-09 | 2.1792e-08 | 19.66% | 1.1379 | 1.0665 | 1.0669 | 22/22 |
| band_off | — | not run on the Mac | | | | | | |
| cuts_off | — | not run on the Mac | | | | | | |
| colcut_off | 0-31 | 4.0347e-09 | 2.0879e-08 | 19.32% | 1.0515 | 1.0036 | 1.0477 | 31/32 |
| colcut_off | 10-31 | 3.9475e-09 | 2.0433e-08 | 19.31% | 1.0479 | 1.0000 | 1.0479 | 22/22 |
| l5_off | 0-31 | 4.0351e-09 | 2.0797e-08 | 19.40% | 1.0516 | 0.9997 | 1.0519 | 32/32 |
| l5_off | 10-31 | 3.9613e-09 | 2.0428e-08 | 19.39% | 1.0516 | 0.9997 | 1.0518 | 22/22 |
| csym_binv_off | 0-31 | 3.8863e-09 | 2.0805e-08 | 18.68% | 1.0128 | 1.0000 | 1.0128 | 32/32 |
| csym_binv_off | 10-31 | 3.8155e-09 | 2.0436e-08 | 18.67% | 1.0129 | 1.0001 | 1.0128 | 22/22 |
| k31scale_1 | 0-31 | 4.1112e-09 | 2.2296e-08 | 18.44% | 1.0714 | 1.0717 | 1.0000 | 32/32 |
| k31scale_1 | 10-31 | 4.0367e-09 | 2.1902e-08 | 18.43% | 1.0716 | 1.0719 | 1.0000 | 22/22 |
| feed_off | 0-31 | 4.1932e-09 | 2.2801e-08 | 18.39% | 1.0928 | 1.0960 | 0.9970 | 30/32 |
| feed_off | 10-31 | 4.1463e-09 | 2.2555e-08 | 18.38% | 1.1007 | 1.1039 | 0.9970 | 21/22 |
| paths_off | 0-31 | 4.2162e-09 | 2.3821e-08 | 17.69% | 1.0988 | 1.1450 | 0.9594 | 32/32 |
| paths_off | 10-31 | 4.1142e-09 | 2.3258e-08 | 17.68% | 1.0922 | 1.1383 | 0.9594 | 22/22 |
| c0d_off | 0-31 | 4.9989e-09 | 2.7131e-08 | 18.43% | 1.3027 | 1.3041 | 0.9992 | 32/32 |
| c0d_off | 10-31 | 4.8918e-09 | 2.6564e-08 | 18.42% | 1.2986 | 1.3000 | 0.9992 | 22/22 |
| marg_off | 0-31 | 5.9772e-09 | 3.2420e-08 | 18.44% | 1.5577 | 1.5584 | 1.0000 | 32/32 |
| marg_off | 10-31 | 5.8289e-09 | 3.1630e-08 | 18.43% | 1.5474 | 1.5480 | 1.0000 | 22/22 |
| k31_off | 0-31 | 4.4759e-09 | 2.4269e-08 | 18.43% | 1.1664 | 1.1665 | 0.9993 | 32/32 |
| k31_off | 10-31 | 4.3821e-09 | 2.3773e-08 | 18.42% | 1.1633 | 1.1635 | 0.9993 | 22/22 |
| star_off | 0-31 | 3.8834e-09 | 2.1059e-08 | 18.44% | 1.0120 | 1.0123 | 0.9998 | 24/32 |
| star_off | 10-31 | 3.8108e-09 | 2.0675e-08 | 18.43% | 1.0116 | 1.0118 | 0.9998 | 16/22 |
| c0k22_off | 0-31 | 3.8638e-09 | 2.0963e-08 | 18.43% | 1.0069 | 1.0077 | 0.9993 | 24/32 |
| c0k22_off | 10-31 | 3.7980e-09 | 2.0616e-08 | 18.42% | 1.0082 | 1.0090 | 0.9993 | 19/22 |
