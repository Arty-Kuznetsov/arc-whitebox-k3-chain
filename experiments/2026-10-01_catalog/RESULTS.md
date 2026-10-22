# The catalog re-screen on the new base (cloud, Prompt B of `notes/CLOUD-BRIEF.md`) — RESULTS

Branch `cloud/catalog` = `cloud/tail-accuracy` (Prompt A, finished: `experiments/2026-09-30_tail-accuracy/RESULTS.md`)
+ `origin/main` (bc6c800), so that the tail switches of A (`tailr`, the confirmed `marg2`) are available to the
stacks; `chain.py` is unchanged otherwise. Torch chain, float64; FLOPs and MSE only (CLOUD.md §2). Every table is
rendered by `report.py` from `res/` (fill: `python report.py t1|grid3|stack`).

## 0. Setup, gates
- Harness: the tail-accuracy one (`ta.py`: per-(arm, MLP) result files, the lossless early stop at 4 worse of the
  10 screening MLPs, MLPs 10-41 = the 32 fresh MLPs never used for selection); arms in `cat_grid.py`, jobs `run.py`.
- **Gates on this branch, every switch off:** `ablate.py k4memo 3` = **3.7533e-08**, `ablate.py k3l_t 10` =
  **2.3135e-08**; k3m BASE 2.3462e-08 and #8's `t4_final25a` (FULL) 2.23915e-08 are the per-MLP files of the
  tail-accuracy session (same `chain.py` code path), copied into `res/`.
- **Base B9** = k3m BASE + matched cut -2.5 + E2 (satrow 2.5, sep) = **2.1967e-08** (x0.9363 of k3m).
- Smoke test (`smoke.py`): all 53 arms run on a random n=128 MLP and every one moves the output.
- **Pricing** (`cost_cat.py`): the port-battery model (`cost_pb.py`) with each arm's own cut thresholds, plus the
  products it does not know — path legs of ages 2..a (`e3age`, `allages`), the `pathD="full"` pairing, `k4k3`
  (~4 products per source per layer, NEGATIVE 61), the `pd`/`p22` kappa4 births, and the thin rank (NEGATIVE 61's
  +5 points at rank 384, taken linear: an estimate, not a meter reading); anchored to B9's meter-anchored 22.69%.
  Every other switch is O(n^2) or a coefficient change and is priced at zero.

## 1. Every structural switch alone on B9 (task 1)
The fitted levels (k31mult, feedmult, k22c, sdmult, k4scale/k4mult, intermediate k4k3) and the pure cost
approximations (window, srank, dlr, d21_ages, ssp/shared_sub, tailx, stride, bandz, agecut) are not in the list; the
tail restrictions are Prompt A's. Sorted by raw x B9:

<!-- T1 -->
| arm | raw x B9 (10) | better | status | util (est.) | **projected x B9** | raw x B9, 32 fresh |
|---|---|---|---|---|---|---|
| allages | 0.9461 | 9/10 | survives (raw) | 28.56% | **1.1908** | 0.9462 (27/32) |
| k4k3_1 | 0.9607 | 10/10 | survives (raw) | 54.33% | **2.3004** | 0.9736 (26/32) |
| e3age3 | 0.9704 | 9/10 | survives (raw) | 24.26% | **1.0376** | 0.9517 (31/32) |
| thin384 | 0.9799 | 10/10 | survives (raw) | 27.69% | **1.1959** | 0.9795 (28/32) |
| pathD_full | 0.9829 | 8/10 | survives (raw) | 23.42% | **1.0146** | 0.9861 (26/32) |
| e3age2 | 0.9846 | 7/10 | survives (raw) | 23.51% | **1.0202** | 0.9703 (29/32) |
| thin192 | 0.9849 | 8/10 | survives (raw) | 25.02% | **1.0862** | 0.9858 (27/32) |
| thin96 | 0.9892 | 8/10 | survives (raw) | 23.69% | **1.0328** | 0.9880 (26/32) |
| thin48 | 0.9909 | 9/10 | survives (raw) | 23.02% | **1.0055** | 0.9949 (23/32) |
| k4b_pd | 0.9941 | 7/10 | survives (raw) | 23.68% | **1.0374** | 0.9920 (26/32) |
| cuts_off | 0.9944 | 7/10 | survives (raw) | 29.52% | **1.2939** | — |
| q22x2 | 0.9992 | 8/10 | survives (raw) | 22.69% | **0.9992** | 0.9994 (24/32) |
| k4lvl_off | 1.0001 | 4/8 | killed | 22.69% | **1.0001** | — |
| q22x1 | 1.0001 | 1/5 | killed | 22.69% | **1.0001** | — |
| tet4 | 1.0009 | 0/4 | killed | 22.69% | **1.0009** | — |
| pairx_off | 1.0015 | 3/7 | killed | 22.69% | **1.0015** | — |
| tet2 | 1.0021 | 2/6 | killed | 22.69% | **1.0021** | — |
| marg_11_21 | 1.0021 | 1/5 | killed | 22.69% | **1.0021** | — |
| colcut_off | 1.0035 | 3/7 | killed | 23.84% | **1.0546** | — |
| paths_no_k22 | 1.0052 | 0/4 | killed | 22.69% | **1.0052** | — |
| paths_no_e3t | 1.0070 | 1/5 | killed | 22.69% | **1.0070** | — |
| k4b2 | 1.0077 | 0/4 | killed | 22.69% | **1.0077** | — |
| k4b_full | 1.0080 | 2/6 | killed | 24.67% | **1.0959** | — |
| paths_no_cc | 1.0104 | 0/4 | killed | 22.69% | **1.0104** | — |
| k4b_s22 | 1.0123 | 1/5 | killed | 22.69% | **1.0123** | — |
| feed_P1P2 | 1.0133 | 0/4 | killed | 22.69% | **1.0133** | — |
| e2_restore | 1.0137 | 1/5 | killed | 22.69% | **1.0137** | — |
| margleg | 1.0142 | 2/6 | killed | 22.69% | **1.0142** | — |
| k4b3 | 1.0152 | 0/4 | killed | 22.69% | **1.0152** | — |
| k4b0 | 1.0313 | 0/4 | killed | 22.69% | **1.0313** | — |
| k4star_off | 1.0313 | 0/4 | killed | 22.69% | **1.0313** | — |
| edge2 | 1.0417 | 0/4 | killed | 22.69% | **1.0417** | — |
| c0d_21_22 | 1.0562 | 0/4 | killed | 22.69% | **1.0562** | — |
| pathsub_off | 1.0630 | 0/4 | killed | 22.69% | **1.0630** | — |
| paths_no_e3 | 1.1134 | 0/4 | killed | 22.69% | **1.1134** | — |
| paths_off | 1.1210 | 0/4 | killed | 22.69% | **1.1210** | — |
| e3 | 1.1261 | 0/4 | killed | 22.69% | **1.1261** | — |
| feed_off | 1.1422 | 0/4 | killed | 22.69% | **1.1422** | — |
| k31pat_off | 1.1814 | 0/4 | killed | 22.69% | **1.1814** | — |
| k22pat_off | 1.1965 | 0/4 | killed | 22.69% | **1.1965** | — |
| deadC | 1.1974 | 0/4 | killed | 22.69% | **1.1974** | — |
| c0d_off | 1.3841 | 0/4 | killed | 22.69% | **1.3841** | — |
| c0d_22 | 1.3861 | 0/4 | killed | 22.69% | **1.3861** | — |
| c0d_11_22 | 1.4079 | 0/4 | killed | 22.69% | **1.4079** | — |
| marg_22 | 1.6495 | 0/4 | killed | 22.69% | **1.6495** | — |
| marg_off | 1.6514 | 0/4 | killed | 22.69% | **1.6514** | — |
| marg_11_22 | 1.8722 | 0/4 | killed | 22.69% | **1.8722** | — |
| marg_11 | 1.8732 | 0/4 | killed | 22.69% | **1.8732** | — |
| k4pat_off | 1.9468 | 0/4 | killed | 22.69% | **1.9468** | — |
| aug | 1.9705 | 0/4 | killed | 22.69% | **1.9705** | — |
| marg_21 | 2.8786 | 0/4 | killed | 22.69% | **2.8786** | — |
| marg_21_22 | 2.8818 | 0/4 | killed | 22.69% | **2.8818** | — |
| pathfold | 4.9810 | 0/4 | killed | 22.69% | **4.9810** | — |
<!-- /T1 -->

**12 of 53 switches survive the raw kill rule, and every one of them is a cost-for-accuracy trade** (more path legs,
a richer correction leg, extra kappa4 products) or the removal of a cost cut. Confirmed on the 32 fresh MLPs (raw x
B9 / better): allages 0.9462 (27/32), e3age3 0.9517 (31/32), e3age2 0.9703 (29/32), k4k3 = 1.0 0.9736 (26/32),
thin384 0.9795 (28/32), thin192 0.9858, pathD full 0.9861, thin96 0.9880, k4b pd 0.9920, thin48 0.9949 (23/32),
q22x2 0.9994 (24/32) — the accuracy gains are real and mostly hold out of sample (e3age2/3 are even stronger on
fresh MLPs). **None pays at today's utilisation**: projected on the fresh ratios, allages x1.19, k4k3 x2.30, thin384
x1.20, e3age3 x1.018, pathD full x1.018, thin48 x1.010, k4b pd x1.035; the closest is **e3age2 (path legs live two
layers): x1.005** — break-even within the util model's own 1-3% error, i.e. not decidable here; it is the one the
Mac could meter. `cuts_off` (x0.9944 raw) is the cost cut undone: x1.29 projected, not confirmed (it cannot enter the
frontier). The only free survivor, **q22x2** (second-order kappa3 x kappa3 diagrams of the (2,2) power cumulant,
O(n^2)), is worth 0.06% (fresh x0.9994, 24/32) — noise-level.

Killed on accuracy (selection; full table above): every marg-per-channel arm (the existing `marg` string selects power
cumulants for BOTH coefficient orders, and every partial choice is worse — the k-selective `marg2` of Prompt A is the
form that works), c0d off/partial, pairx off, k31pat off, the path subsets, pathsub off, pathfold, aug, edge2, e3,
tet 2/4, the kappa4 birth variants other than pd, feed off/P1P2, k22pat/k4pat off, k4lvl off (x1.0001: the O(n^2)
levels are as good as the full memo22 transport), colcut off, deadC, E2 restore instead of sep (x1.014).

## 2. The threshold grid on the full stack (task 3)
FULL = matched cut -2.5 + E2 2.5 sep + band (5,448,rr,pre) + E3 recv (-3.0, b). Grid: cut {-2.25, -2.5, -2.75} x E2
theta {2.25, 2.5, 2.75} x recv {-3.0, -3.25}, and band (6,384) instead of (5,448); kill rule on raw vs FULL, util from
the port-battery model at each setting, anchored to FULL's meter-anchored 18.95%:

<!-- GRID3 -->
| on FULL | raw x FULL (10) | better | status | util | **projected x FULL** |
|---|---|---|---|---|---|
| FULL (cut -2.5, E2 2.5, recv -3.0; band (5,448)) | 1.0000 | — | base | 18.95% | **1.0000** |
| cut -2.25, E2 2.5, recv -3.0 | 1.0305 | 1/5 | KILLED | 18.32% | **0.9964** |
| band (6,384) | 1.0146 | 1/5 | KILLED | 18.70% | **1.0012** |
| cut -2.25, E2 2.5, recv -3.25 | 1.0300 | 0/4 | KILLED | 18.43% | **1.0020** |
| cut -2.5, E2 2.5, recv -3.25 | 1.0004 | 1/5 | KILLED | 19.06% | **1.0065** |
| cut -2.25, E2 2.25, recv -3.0 | 1.0554 | 2/6 | KILLED | 18.12% | **1.0095** |
| cut -2.25, E2 2.25, recv -3.25 | 1.0592 | 0/4 | KILLED | 18.23% | **1.0193** |
| cut -2.5, E2 2.25, recv -3.0 | 1.0305 | 2/6 | KILLED | 18.74% | **1.0194** |
| cut -2.5, E2 2.25, recv -3.25 | 1.0294 | 2/6 | KILLED | 18.86% | **1.0245** |
| cut -2.75, E2 2.5, recv -3.0 | 0.9943 | 8/10 | done | 19.53% | **1.0250** |
| cut -2.25, E2 2.75, recv -3.25 | 1.0494 | 0/4 | KILLED | 18.61% | **1.0308** |
| cut -2.75, E2 2.5, recv -3.25 | 0.9943 | 8/10 | done | 19.65% | **1.0313** |
| cut -2.25, E2 2.75, recv -3.0 | 1.0586 | 1/5 | KILLED | 18.50% | **1.0336** |
| cut -2.5, E2 2.75, recv -3.0 | 1.0237 | 1/5 | KILLED | 19.13% | **1.0338** |
| cut -2.5, E2 2.75, recv -3.25 | 1.0239 | 1/5 | KILLED | 19.25% | **1.0401** |
| cut -2.75, E2 2.25, recv -3.0 | 1.0316 | 3/7 | KILLED | 19.32% | **1.0521** |
| cut -2.75, E2 2.25, recv -3.25 | 1.0306 | 3/7 | KILLED | 19.44% | **1.0574** |
| cut -2.75, E2 2.75, recv -3.0 | 1.0160 | 1/5 | KILLED | 19.72% | **1.0577** |
| cut -2.75, E2 2.75, recv -3.25 | 1.0163 | 1/5 | KILLED | 19.84% | **1.0644** |
<!-- /GRID3 -->

**FULL's thresholds are optimal within the resolution of the cost model.** Every threshold change is worse on raw
except the softer cut -2.75 (x0.9943, 8/10), which is dearer (19.5-19.7% util) and loses on projected (x1.025-1.031).
The harsher cut -2.25 with E2 2.5 / recv -3.0 projects to x0.9964, but it is 3% worse on raw and its 0.4% projected gain
is below the util model's error (it was optimistic by 1-3% against the Mac's meter in every port so far): not a
finding. Band (6,384) instead of (5,448): x1.0146 raw, x1.0012 projected — #8's choice stands.

## 3. Greedy stack, confirmation on 32 fresh MLPs, frontier (tasks 2 and 4)

<!-- STACK -->
| stack | x base (10) | x base (32 fresh) | util | **projected (fresh)** | + L5 |
|---|---|---|---|---|---|
| B9 | 1.0000 (0/10) | 1.0000 (0/32) | 22.69% | **4.611e-09** | 4.226e-09 |
| B9 + marg2 | 0.9811 (8/10) | 0.9838 (29/32) | 22.69% | **4.536e-09** | 4.157e-09 |
| B9 + marg2 + q22x2 | 0.9810 (8/10) | 0.9836 (29/32) | 22.69% | **4.535e-09** | 4.156e-09 |
| B9 + marg2 + q22x2 + thin48 (greedy step, rejected) | 0.9721 (9/10) | not run | 23.02% | **pending** | — |
| FULL | 1.0000 (0/10) | 1.0000 (0/32) | 18.95% | **3.972e-09** | 3.640e-09 |
| FULL + marg2 | 0.9802 (9/10) | 0.9842 (28/32) | 18.95% | **3.909e-09** | 3.582e-09 |
| FULL + marg2 + q22x2 | 0.9800 (9/10) | 0.9840 (28/32) | 18.95% | **3.908e-09** | 3.582e-09 |
<!-- /STACK -->

Greedy stack by projected adjusted (re-measured at each step): **marg2** (Prompt A's confirmed lever) first, then
**q22x2** (fresh x0.9836 vs x0.9838 without it: nothing), then the cheapest cost-for-accuracy step **thin48**, which is
rejected (screen projected x0.9863 vs x0.9810 for marg2 + q22x2). No other survivor has a positive projected effect,
so the greedy stops. On FULL: FULL + marg2 x0.9842 fresh (28/32) -> **3.909e-9**; FULL + marg2 + q22x2 x0.9840
(28/32) -> 3.908e-9; with Strassen L5 3.58e-9.

## 4. Verdict (task 5)

- **Best projected adjusted: FULL + marg2 = 3.91e-9 (3.58e-9 with L5)** — the same as Prompt A's verdict; the catalog
  adds nothing measurable to it. Exact settings: `pb.matched(-2.5, satrow=2.5, satrow_k4="sep")` + `bandz=(5, 448, rr,
  pre)` + `recv=(-3.0, "b")` + `tailr=(("marg2", "sat", None, -2.0, "gauss"),)`; optional `q22x=2` (O(n^2), ~0.02%).
- **Production cost of the levers that were measured to help raw** (per source and layer, n x n x n products): path
  legs beyond age 1 (`e3age`, `allages`): +1 transport per extra age; `pathD="full"`: +1 pairing; `k4k3`: ~4; the
  thin rank: (k - 24)/n of a product per source in the leg stack (NEGATIVE 61: +5 points at 384); `k4b` with `pd`:
  +1 (X C0) per k4b source. `q22x2`, `marg2`: O(n^2) / O(n), zero products.
- **What the catalog says about the goal.** On the new base every structural accuracy gain the chain still has is a
  cost-for-accuracy trade, and at ~19-23% utilisation each costs more than it buys — the same wall as NEGATIVE 61, now
  confirmed out of sample. They would pay only near the 10% multiplier floor (where e3age3 x0.952, allages x0.946,
  k4k3 x0.974 and thin384 x0.980 would stack towards the 09-28 cost-free ceiling), so the cost axis remains the gate
  for them; the free axis is exhausted except for marg2. The open accuracy question is Prompt A's: the kink-core
  moments, i.e. Prompt C's order scan.
