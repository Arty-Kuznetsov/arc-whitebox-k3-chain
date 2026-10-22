# Ideation workflow — synthesis (research lead agent, verbatim; translated from Russian on 2026-10-13)

# Decision on cost: what to do from 29 September to 17 October

Read-only mode: no file was changed, nothing was run except reads. Checked: `NEGATIVE-RESULTS.md` §54–§62, `CLOUD-BRIEF.md`, `2026-09-29_colcut/RESULTS.md`, `2026-09-29_satrow/scan_n10.log` and the `ssp` and `satrow` blocks in `chain.py`.

## 0. Two corrections to the brief

1. **First place is a problem of both cost and accuracy.** J2W has raw 1.62e-8. Our accuracy stack, even with no cost limit, gives 1.724e-8 (1.763e-8 without the two fitted scalars, and those do not transfer to the grader, §60). So even at the 0.1 multiplier floor we would get about 1.72–1.76e-9 today against J2W's 1.70e-9, i.e. #2 at best. The leaders are more accurate than us on every one of the 50 public MLPs (telemetry, intel lens).
2. **The cloud test described in the brief is already closed.** The question of whether (t, D) is sufficient was settled in §62 (28 September): the all-distinct part is 28.8% per layer, ×25.6 in total, and the cheap basis catches 19.5%. Only **Prompt A of `CLOUD-BRIEF.md` (MST)** remains open: what the **old tail's** own all-distinct contribution is. Nobody has measured it, and in what follows it is the only ticket to the floor.

## 1. Deduplication: 17 ideas reduce to 6 mechanisms

| # | Mechanism | Lenses that converged | Verdict |
|---|---|---|---|
| M1 | Resume the parked band branch of §54 with a cheap update at entry. Two variants: (a) the basis is only transported, no Gram, RR or eigh, this is `ssp q=0`; (b) exact top-r through `svd(F,k=)` at the 4mnk rate | **6** (billing, rmt, restructure, learned, intel, tensor-formats). The same idea sits in CLOUD-BRIEF as Prompt B | **alive**, but works only at r≥384 |
| M2 | Cut the rows of saturated neurons (α>θ) in the D contraction and restore the κ4 level with two matvecs at O(S·n²) | **4** (tensor-formats, stat-sketch, intel, restructure: the second half of idea 2) | **alive** |
| M3 | Skip the receiver rows of dead neurons (α_l<θ_out) in the leg transport; for t either 0 or t_rep over repeated indices | **2** (intel, restructure) | **alive** at θ_out≤−3.0 |
| M4 | Closure of the old tail through slices (MST, slice-exact sliding drop, feed-form carrier, chaos ceiling) | **3 lenses, 4 ideas** | The skeptics closed it via §62. Honest gap: f for the old tail alone is not measured. That is exactly Prompt A, **do not duplicate** |
| M5 | Symmetric C_pre (upper block triangle) | 1 | marginal, only as part of a package |
| — | Closed or marginal: one-sided projection of D_old (×0.99–1.0); half-column cut of saturated hubs (×0.99–1.03, useful only for the write-up); L5 on the contractions only, and band 5:384 + full L5 (closed by §56 and §54, catastrophic wall risk); sampling in the readout norm (§59); Oseledets band with slice-fix at r=256 (closed by Eckart–Young; its r≥384 variant is M1) | | |

**On lens convergence.** 6 lenses on M1 confirm that it is the only live lever that changes the representation itself. But the mechanism they attributed to it is refuted: three independent skeptic tests (two synthetic at n=1024 and one on real weights from the baked MLPs) show that a transported basis of rank r catches about as much as an ideal basis of rank 0.8r. The convergence of 4 lenses on M2, plus the naive satrow scan that has already been run, is the most reliable evidence in this package.

## 2. Ranking: gain × probability

The gain g is the fraction by which the adjusted score drops. P is the probability that the lever works and makes it into a submission before 17 October.

| # | Lever | Central multiplier | g | P | g×P | Where P comes from |
|---|---|---|---|---|---|---|
| 1 | **M1 band, transport at r=384–448** (fallback: svd entry, ×0.945, P≈0.45) | ×0.92 | 8% | 0.35 | **2.8%** | 0.6 on accuracy: ideal 6:320 was +1.3% on the k3j base, not measured on k3m. 0.7 on residual: k3k added +0.137 s, without Gram/eigh it comes to about +0.05–0.10. 0.8 on the port on top of the wide colcut layout |
| 2 | **M2 satrow + level restoration, θ=3.0–3.25** | ×0.955 | 4.5% | 0.55 | **2.5%** | The naive scan without restoration already gave ×1.0094 at θ=3.0, but only 6/10 and a spread of ×0.84…×1.35, which looks like a hard κ4 level. The risk is that with marg=True the cancellation in K21 is inexact, leaving a remainder of 3–5% of the kink weight |
| 3 | **M3 receiver rows, −3.5 with t=0 or −3.0 with t_rep** | ×0.955 | 4.5% | 0.55 | **2.5%** | t_i is also read through K2 and marg. So −2.5 is dead, and −3.0 needs t_rep. WORKLOG says "~4e-9", 30 times the skeptic's estimate, and the test will settle it |
| 4 | **M4 MST (cloud, Prompt A)** | ×0.45 (route to the floor) | 55% | 0.04 | 2.2% | For the whole tensor §62 gives 28.8%, and the MST threshold is ≈8%. The tail's share would have to be 3–4 times lower than the whole tensor's. There is a hint of that (35–61% in the young layers 1–3), but the star capture only grows to 26.7% |
| 5 | M5 symmetric C_pre | ×0.983 | 1.7% | 0.4 | 0.7% | The FLOP gain is deterministic, but costs about +80 ms of residual |

**The order of work does not match the g×P order.** Dividing g×P by days of engineering work gives: M2 ≈5%/day (only the index array in the existing `take` changes), M3 ≈2.5%/day, M1 ≈0.6%/day (4–5 days of porting). So M2 and M3 go together with the colcut port, and M1 after them.

**Residual is a shared scarce resource.** k3m has 0.197/0.227 s locally, and the rule requires ≤0.2 s. Per millisecond of residual: M2 gives about 4.5% for ~5 ms, M3 4.5% for 10–40 ms, M1 8% for 50–100 ms, M5 1.7% for ~80 ms. M5 goes last.

**The levers do not stack multiplicatively.** After M1 the gain of M3 is roughly halved, because only the young dense legs remain.

## 3. First experiments and the numbers that kill them

Everywhere `BASE` from `2026-09-29_colcut/pairs.py`, `deadcut=-2.5`, 10 baked MLPs, about 65 s per arm (as in `satrow/scan_n10.log`). Control: switch off → 2.3462e-08 bit for bit. Each arm is also run on the matched colcut −2.25 base, because that is the one going into the submission.

**E1: M1 band. About 15 minutes, no new code needed.** I checked `chain.py`: `ssp=(a0,r,0,0)` really works as "projection only". `ss_state` is gated like a leg (lines 561–562), `X0 = Wt@P` holds the old legs exactly, RR rotates the basis within its span.
- Arms: `ssp=(6,384,0,0)`, `(6,448,0,0)`, `(5,448,0,0)`. Ideal reference: `shared_sub=(6,320)` on the k3m base. Toy check: `(6,320,0,0)`, I expect +7–8%.
- In parallel, 1 minute: in `BudgetContext` call `fnp.linalg.svd(F,k=320)` on F of size 1024×2368. The count should be 3.10e9.
- **Kill:** transport at 384 gives more than +3.0% and at 448 more than +2.0%: then variant (a) is dead. The svd variant survives only if ideal 6:320 on k3m is ≤ +2.5% and the count = 3.10e9. Then the price through `cost.py` with every part of the construction (Z·Pᵀ, QR-solve in float32). Kill if util×raw > 0.95.
- Before the port a residual gate is needed. I propose: predicted maximum on the grader ≤0.24 s at a grader/local ratio of 0.75. This is the same 1.7× margin to 0.4 s, but formally a relaxation of the "≤0.2 s locally" rule, and that is the user's decision.

**E2: M2 satrow with restoration. About 12 minutes and ~20 lines of code.** A second call of `nonlin_slices` on the full D, taking only the κ4 level from it. A separate arm exactly as in production: the level from the trimmed D plus 2Σ_{i∈sat} u_i (D w)_i, where w_j = 2u_j·P1_j·(1−d1_j), and the c0d and Q11² terms weighted by d2 are dropped.
- Grid θ ∈ {3.5, 3.25, 3.0, 2.75, 2.5}.
- **Kill:** at θ=3.0 raw worse than +1.0% or adjusted better on fewer than 9/10 MLPs, and also if the per-MLP spread has not collapsed. If at 2.5 with restoration ×1.10 or worse remains, then the κ4 level is not the main mechanism, and we ship 3.5 at most.

**E3: M3 receiver rows. About 10 minutes and ~20 lines.** On rows with α_l<θ_out: arm a replaces t_i with the t_i of the source of age 1 alone; arm b adds t_rep for sources of age ≥2 (from the post slices of layer l−1, which have already gone through the d1 gating).
- θ_out ∈ {−3.5, −3.25, −3.0}.
- **Kill:** arm a at −3.5 worse than +0.5%: then only b remains. Arm b at −3.0 worse than +1.0%: then the limit is −3.5. If both arms at −3.5 are worse than +0.5%, the mechanism is closed.

**E4: M4. This is the cloud's task (Prompt A); do not duplicate it on the Mac.** The deciding number is f(a0, layer), the all-distinct share of the old tail itself. **Kill:** f > 0.08 at any layer ≥9 at a0=3, or end-to-end worse than ×1.05 at a0≤3.

**E5: M5.** The count of one layer in `BudgetContext` and a timing of 1–2 levels. Kill if the residual increase is more than 30 ms.

Everything that passes on 10 MLPs is checked on 100 MLPs through flopscope before the submission (about 11 minutes). The local→grader discrepancy is 0.4–0.5%, so nothing below 3% ships.

## 4. Where the skeptic and the author disagree, and whom I believe

- **Transported basis at r=320 ("Oseledets aligns").** I believe the skeptics. Three independent tests (one on real weights) showed no gap in the spectrum around 320. In theory this is also clear: the newcomer's subspace is set by the last a0 matrices, and those are independent of the basis history, so the transported basis works as a range finder with q=0. Hence the working point 6:384–448, not 6:320. The arms in Prompt B ((3,256), (4,256), (4,384), (5,320)) all fail on the toys. If Prompt B has already been sent, it needs corrected arms. If not, do not send it: E1, 15 minutes on the Mac, replaces it.
- **Price of the svd entry.** I believe the skeptic, who counted ~−3.75u per entry from lines 540–583 of `estimator_k3k.py`; that comes to ×0.945. About half of the gain is purely the difference in how svd(k=) is priced. This is allowed by the documented API, but it has to be disclosed in the write-up.
- **Satrow "low risk at 3.5".** I believe the skeptic: with marg=True the cancellation in K21 is inexact. The test does not change, but the breadth requirement becomes 9/10.
- **Receiver rows "are read once".** I believe the skeptic: the K2 and marg channels remain. So −2.5 is dead, and t_rep is mandatory at −3.0.
- **Closing M4 via §62.** I agree with the skeptic's conclusion for the whole tensor. But the skeptics themselves write that the tail-only share was not measured, so I keep M4 open with P≈4%.
- **The intel lens's inference from telemetry** ("the leaders have 0.6–0.8 of our deep work; the road to success is a stack of levers at ×0.8 each"). This is a model with several free parameters. Against it stands a direct measurement: J2W's raw is better than our ceiling with no cost limit. On the first-place question I believe the second: #1 has a different method.

## 5. Bottom line

- **Realistic finish.** The colcut port gives ≈4.95e-9 (util modelled so far). With M2 and M3 added, about 4.6–4.75e-9. With M1 as well, about 4.2–4.4e-9. The probability-weighted median is ≈4.55e-9, roughly rank 35–50 on today's board, and the board will move down over the 18 days.
- **P(first place) < 0.5%.** It needs both the route to the floor (≈4%) and a raw better than J2W's, and our accuracy stack does not reach that even with no cost limit.
- **P(top-5) ≈ 1–2%.** Only if Prompt A gives f ≤ 0.08 at a0≤3, the MST port fits in about 10 days, and #5 does not drop below ~2.1e-9.
- **The real payoff is the $20k Algorithmic Contribution prize.** The write-up draft has to start no later than ~10 October (due 24 October). Freeze all cost engineering after 11–12 October.
- **The most valuable next action.** Today, before the layout of the colcut port is fixed, run E1: the band arms (6,384,0,0), (6,448,0,0) and ideal 6:320 on the k3m base. That is 15 minutes with no new code. It is the largest lever, and it alone decides the port's architecture: legs in coefficient form or a wide dense layout. A mistake here costs days. Right after it, in the same session, E2 and E3 (~25 minutes): they go into the same port as colcut.

Files:
- `<repo>/experiments/2026-09-24_k3-chain/chain.py` (ssp: lines 309–340, satrow: 512–516)
- `<repo>/experiments/2026-09-29_satrow/scan.py`
- `<repo>/experiments/2026-09-29_colcut/pairs.py` (BASE)
- `<repo>/src/estimator_k3k.py` (entry block, lines 540–583)
- `<repo>/notes/CLOUD-BRIEF.md` (Prompt A and Prompt B; the arms in B need fixing)