# SPEC — ARC WhiteBox Estimation Challenge 2026 (Phase 2)

> Distilled from the official starter kit (`external/whest-starterkit/docs/`).
> This is a cheat-sheet; the kit docs are the source of truth. Re-verify numbers
> against AIcrowd before relying on prizes/deadlines (see the case brief).

## The task in one sentence
Given a randomly-initialised ReLU MLP and a FLOP budget, predict the **per-neuron
mean post-ReLU activation** under standard-normal `N(0,1)` input — using the
network's *structure*, not brute-force forward passes.

## Fixed setup (Phase 2)
- **Shape:** width `n = 1024`, depth `d = 16`. `predict()` returns a `(16, 1024)` array.
  - Row `i` = expected neuron values **after** layer `i` (post-ReLU).
- **Layer:** `y = ReLU(W.T @ x)`, `W` is `(n, n)`, He init `N(0, 2/n)`, stored `(input, output)`.
- **Input:** each of `n` neurons i.i.d. `N(0,1)`.
- **FLOP budget per MLP:** `B = 2**41 = 2,199,023,255,552`.

## Scoring (the only ranked metric)
```
adjusted_final_layer_score = mean_m( final_layer_mse_m × max(0.1, C_m / B) )
```
- `final_layer_mse_m` = MSE on the **final layer only** vs N=1e9 Monte-Carlo ground truth.
- `C_m = F_m` = analytical FLOPs counted by flopscope. **Nothing else** (wall-time is NOT priced).
- **Multiplier floors at 0.1** once you are ≤10% of budget. Below 10%, spending less does
  NOT help — **accuracy is the only lever.**
- **Lower is better.** `all_layers_mse` is a diagnostic only (no multiplier).

### ⚠️ Grader schema 2.0 — the test set is 50 public + 50 private (first seen 2026-09-26, submission 332253)
The submission page now states: **"TEST SET PARTITION · 50 public / 50 gated"**, the headline score is
*"the mean over the 50 public MLPs"*, and **"FINAL SCORE · FULL TEST SET — graded on all 100 MLPs — the private
split is sealed until results release"**. Submissions 332128 / 332170 / 332247 were graded and displayed on 100.
- What we still get on every submission: per-MLP **telemetry for all 100** (FLOPs, residual and predict wall time,
  exhaustion flags) — that is where the residual/wall numbers in `CLAUDE.md` come from. Only the private half's
  **scores** are hidden.
- What it changes: the number we iterate against is a 50-MLP mean, so its sampling noise is ≈√2 larger than before,
  and the final rank is decided on a half we never see. Anything hand-tuned on the grader's public MLPs would be
  overfitting; our constants (`k31_scale`, `deadcut`) are tuned on the 100 baked public-dataset MLPs, a disjoint set.
- Practical rule: prefer a lever that improves **99–100 of our own 100 MLPs** over one that wins on the mean through
  a few MLPs. `deadcut=-2.5` passes that test (99/100); `k31_scale=0.65` passes it (98/100).

## The numbers that matter (Phase 2, `mini` split = 100 MLPs, baked 1e9 GT)
| Estimator | final_layer_mse | FLOPs | % of B | adjusted score (= mse × 0.1) |
|---|---|---|---|---|
| zeros baseline | 0.9095 | ~0 | 0.000% | 0.09095 |
| random | 0.6856 | 131,072 | 0.000% | 0.06856 |
| **mean propagation** (diag) | 2.216e-4 | 86,639,616 | 0.004% | 2.216e-5 |
| **plain sampling (floor)** | ~1.19e-5 | ~2.2e11 (6.5k passes) | 9.9% | **≈1.19e-6** |
| **covariance propagation** (full) | **4.050e-6** | 51,709,240,799 | 2.351% | **≈4.050e-7** |

**Milestone framing:** "beat sampling." Sampling bottoms out at ≈1.19e-6 adjusted.
Covariance propagation (≈4.05e-7) is the *only* bundled example that beats it — **2.9× better**.
**→ The number to beat is covariance propagation's `final_layer_mse = 4.05e-6.**

## Free compute headroom (key strategic fact)
- Multiplier is floored at 0.1 for anything ≤10% of `B` ≈ **2.199e11 FLOPs/MLP**.
- Covariance propagation uses only **2.351%**. So there is **~4.25× more compute to spend
  with ZERO score penalty.** 10% of budget ≈ **6,500 forward passes** worth of FLOPs.
- Implication: an accuracy improvement costing up to ~1.6e11 extra FLOPs is *free* on the metric.

## Allowed code (hard rules — violation = disqualification)
- **Only:** the grader's Python interpreter + the **flopscope client API**
  (`import flopscope as flops`, `import flopscope.numpy as fnp`) + `whestbench` types +
  the **pure-Python standard library**.
- **Shipping data files is allowed** (weights, lookup tables, precomputed artifacts) — load in `setup()`.
- **Prohibited:** vendored numpy/scipy/BLAS, compiled kernels, ctypes/cffi/FFI,
  threads/subprocess/async/multiprocessing, any numerical work outside flopscope's accounting.
- All numerics must go through `fnp.*` (flopscope numpy) so they are FLOP-counted.

## Estimator contract
- Interface: `predict(self, mlp: MLP, budget: int) -> fnp.ndarray` → shape `(mlp.depth, mlp.width)`, all finite.
  - Read shape off `mlp` (validate probes a tiny width=4, depth=2 MLP — don't hard-code 16×1024).
- Optional: `setup(self, ctx: SetupContext) -> None` (load, don't compute) and `teardown(self)`.
  - `setup()` runs once per worker (several times/submission), **5 s** hard cap → overrun fails whole submission.
  - `setup()` is OFF the FLOP budget and OFF residual time.
- **Seeding (prize eligibility):** use `fnp.random.default_rng(mlp.seed)` in `predict()`,
  `fnp.random.default_rng(ctx.seed)` in `setup()`. Never `fnp.random.seed(...)`; never a constant like `42`.

## Per-MLP caps (crossing any → that MLP's preds zeroed + multiplier forced to 1.0)
| Cap | Value |
|---|---|
| FLOP budget | `2**41` |
| Wall-clock | 120 s |
| Residual wall-time (plumbing only) | **400 ms** |
| Process memory | 8 GB |

A failed MLP scores **10× worse** than the cheapest valid submission (which earns the 0.1 floor).

## flopscope billing gotchas (cheap accuracy-neutral savings)
- **float64 bills 2× float32.** `fnp.zeros/ones/eye` default to **float64** → seed state at `float32`.
  (Worth ~44% on mean prop, ~50% on covariance, MSE unchanged.)
- `flops.stats.norm.pdf/cdf` promote float32→float64 (to match scipy) → `.astype(fnp.float32)` right after.
- **Symmetric covariance:** use one `fnp.einsum("ij,ia,jb->ab", cov, w, w)` (NOT chained `w.T@cov@w`),
  and **re-tag after every write** into `cov` with `flops.as_symmetric(cov, symmetry=(0,1))` — a write
  silently voids the tag. Symmetric rate saves ~25%/layer on the big contraction.
- `x * x` not `x ** 2` (power is 16× tier). `exp`/`log` are 16×/element. `where`/gathers 4×/element.
- matmul `(M,K)@(K,N)` costs `M·N·(2K−1)`. One `(1024,1024)@(1024,1024)` ≈ 2.146e9 FLOPs.
- **When the symmetric-output discount applies (measured, k=3 chain phase):** only when the two operands **alias**
  (`A @ A.T`, an inner product of an array with itself) or when the output is explicitly tagged `as_symmetric`.
  `(A*c) @ A.T` — a scaled copy — bills **in full**, even though the result is symmetric. Contraction family:
  `(2K−1)·M` per output element.
- `fnp.einsum` bills like matmul + sum but runs **~70× slower** in wall time — never in a hot path (the 120 s wall cap).
- **Strassen-Winograd built from flopscope ops is permitted with no recursion-level limit** (sponsor, 2026-09-11, forum;
  see `notes/INTEL-forum-2026-09-29.md`) and flopscope bills the op tree honestly. Our level is limited by **wall time and
  residual time**, not by the meter: L4 ships (billing ×0.636 of plain), L5 pays on the meter (×0.9165) but doubles wall
  at 24.8% util (NEGATIVE §56).

## Grader vs local — measured transfer ratios (use them to predict a grade)
| quantity | ratio | source |
|---|---|---|
| adjusted score, grader / local 100-MLP `whest run` | **0.976** (0.9756 k3l, 0.9764 k3m) | 332253, 332403 |
| raw final MSE, flopscope / torch chain | ≈0.920 | ports k3i … k3m |
| residual time, grader / local runner | 0.753 (subprocess runner reads ≈2× the grader) | 332253 |
| predict wall, grader / local | ≈3.1× on the mean, ≈3.9× on the worst MLP; ~90% of it is flopscope kernel time | 332253, 332403 telemetry |
| grader resolution on a 50-MLP mean | ≈0.4–0.5%: a lever < 2–3% cannot be validated by a submission | 332872 vs 332403 (NEGATIVE §60) |
| a scalar fitted on 10 torch MLPs | arrives at ≈⅓ strength on 100 flopscope MLPs, may change sign on the grader | k3n |
Telemetry for all 100 MLPs (public + private half) is readable from the public submission page with
`experiments/2026-09-24_k3-chain/prof/grader_telemetry.js`.

## Workflow (5-stage ladder)
1. `python estimator.py` — local MC-convergence table (vs fresh MC, NOT baked GT). Fast iteration.
2. `whest validate --estimator <file>` — contract check (shapes/finite) on a tiny MLP.
3. `whest run --estimator <file> --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner local` — **authoritative scoring** vs baked 1e9 GT (downloads public dataset).
4. `... --runner subprocess` — grader-like isolation.
5. `whest package` / `whest submit` (run `whest login` once). Packages only `estimator.py` unless you pass a folder.

## OFFICIAL RULES — full read, v14, verified 2026-09-22 (aicrowd.com/.../challenge_rules)
The rules page is the binding document (it overrides the starter kit). Read in full on this date; key points:

### Timeline (all UTC)
| Milestone | Date |
|---|---|
| Competition Period | 28 May 2026 00:00 → 31 Oct 2026 23:59 |
| Phase 2 Start | 22 Aug 2026 00:00 |
| **Registration & Team Freeze Deadline** | **2 Oct 2026 23:59 — HARD GATE: no new individual or Team registrations accepted after this, even though the competition runs to 31 Oct** |
| Phase 2 End (submissions close) | 17 Oct 2026 23:59 |
| Phase 2 algorithmic-contribution write-up deadline | 24 Oct 2026 23:59 (write-up only; the evaluator is already closed) |
| Phase 2 Private Re-evaluation | within 3 weeks of Phase 2 End |
| Tentative winner announcement | on or about 15 Nov 2026 |

### Eligibility (§3, §4)
- 18+ and age of majority in own jurisdiction; valid email; AIcrowd account; **registration = accepting the rules on the
  Competition Site** (current version **v14**; Sponsor may require re-acceptance after a material amendment).
- **Phase 1 participation is NOT required.** Each Phase has its own prize pool, its own designation and its own Private
  Re-evaluation; "a Participant may receive prizes in both Phase 1 and Phase 2". Warm-up scores don't carry forward.
- **Geographic prize eligibility:** residents of comprehensively sanctioned jurisdictions (rules name Cuba, Iran,
  North Korea, certain sanctioned regions of Ukraine; non-exhaustive) **may participate and keep leaderboard rank but
  forfeit prizes** (awarded to the next eligible participant). Separately, "Sanctioned Persons" may not participate at all.
- Insiders (ARC/AIcrowd staff, contractors, judges, their households) ineligible, as is anyone materially involved in
  building the competition.
- **One registration per person**, individually OR in one Team (≤5 people, Team Representative acts for the team).
  Multiple accounts = disqualification. **Teams must be finalized by 2 Oct.**
- Payouts run through AIcrowd, subject to sanctions screening, AML and tax forms (W-9 for US persons, **W-8BEN for
  non-US persons**); withholding may apply; Prize Winner Documents due within 14 days of notification (government ID may
  be required); prizes distributed within 6 months.

### Prizes (§6, §7) — Phase 2 is $100k of the $150k total
| Prize | Amount | Decided by |
|---|---|---|
| Phase 2 1st / 2nd / 3rd | $50,000 / $20,000 / $10,000 | **Private Re-evaluation only** — the public leaderboard does not decide |
| Phase 2 Algorithmic Contribution | $20,000 (1 prize; Sponsor may add more) | ARC's discretion, from code + technical write-up |
| Community-contribution awards | typically **$500–$5,000**, discretionary, no application process | Sponsor reaches out; e.g. helpful bug reports, high-quality submissions, community engagement |
- One place prize per Phase **per category** per participant → the same person can take a score prize AND the
  algorithmic prize. Only a participant's best-scoring submission is ranked, so each occupies one ranking slot.
- Ties: extra MLPs drawn until separable, then shared rank with the prize amounts combined and split.

### ⚠️ Designation: up to **TWO** submissions (§6) — earlier notes said one; that was wrong
"Each Participant (individual or Team) will designate up to two (2) submissions from among their valid Phase 2
submissions to be carried forward into the Private Re-evaluation." They need not be the latest or the best public score.
The selection interface and deadline are published on the Competition Site near the end of Phase 2.
**Default if you don't designate:** your two best-scoring valid submissions on that Phase's public leaderboard.
→ Strategy: designate one safe generalizing entry AND one aggressive entry.

### Test suites (§5.4) — three distinct things
1. **Public verification examples** — openly released via the starter kit (this is our local `mini` split with baked GT).
2. **Per-Phase grader suite — 100 hidden MLPs**, split into **50 public** (scores shown on the live leaderboard during
   the phase) + **50 private** (graded in parallel, withheld until the phase closes). Phase-end standings use all 100.
3. **Private Re-evaluation suite** — separate, fresh, never-seen, from private seeds, same distribution.
   **Phase prize ranking comes exclusively from this.** Solutions overfitted to specific MLPs/seeds are penalized.

### Write-up rules (§6)
- **Required** for the algorithmic-contribution prize; encouraged for score prizes (and not needed for score eligibility).
- **Must correspond to exactly one submission from the same Participant that was successfully evaluated by the grader
  for that Phase**; that submission need not have placed in the score rankings.
- Due within one week of the phase close (→ 24 Oct). ARC may publish all or part of it.
- LLM assistance allowed for code and text, but the participant owns the accuracy, must be **fully transparent about the
  extent of LLM use**, and must hedge or label anything they cannot explain ("guesswork"); unhedged dubious claims make
  judges skeptical and may get the entry skipped entirely.
- ARC's stated bias (guidelines #18041): "mechanistic" understanding; **less interested** in heavy sampling, fitted
  constants, performance optimization and opaque LLM-optimized code. Negative results and ablations are welcome.

### ⚠️ Source authenticity (§4, §5.7, §6) — affects any plan to build on a rival's public code
- §5.7: "Used datasets, libraries, or pre-existing code must be publicly available and equally accessible to all
  Participants" (public MIT code is usable in principle), BUT
- §4: "Submissions authored substantially by other Participant(s), including but not limited to plagiarized
  submissions, will be considered a violation of this rule", and
- §6 prize verification: "**Source authenticity** — the submitted code is the Participant's own work (or appropriately
  attributed third-party code) and **is not derived from another Participant's submission**."
→ Basing our estimator on 504aldo's MIT repo (github.com/504aldo/whest-p2-cumulant-k3) would likely **fail prize
verification**, even though running it is legal. The ~19 identical 5.40e-9 leaderboard entries sit in exactly that spot.
**Ask the Sponsor in writing (arc-whestbench@aicrowd.com) before relying on it.** A written confirmation binds only for
the submissions it is addressed to — 504aldo's Strassen-Winograd clearance was addressed to *their* submissions, not ours.

### IP / publication (§8) — we keep the paper rights
- **The participant retains ownership and all IP in their Solution and Content.** Publishing our own paper (arXiv etc.)
  is unrestricted by these rules.
- We grant ARC + AIcrowd a perpetual, worldwide, non-exclusive, royalty-free, irrevocable copyright **and patent**
  licence to the Solution, plus broad rights to use our name/likeness and to publish our write-up.
- **Open-source release is a condition of receiving any place prize:** full source + artifacts under an OSI-approved
  licence, at a public URL, within **30 days** of winner notification, kept accessible for **3 years** (Phase 1 had 7 days).
  Failing to release counts as a verification failure and the prize goes to an alternate.

### Other hard rules worth keeping in view
- **Fair accounting (§5.5):** the benefit must come from the estimation method, not from how computation is declared or
  priced; bit-packing several logical values into one wider element is named as prohibited. Genuinely reducing
  computation — lower precision, symmetry, better algorithms — is explicitly encouraged. flopscope dtype rates:
  float16/int8/int16/int32/float32 = 1.0, float64/int64 = 2.0, complex64 = 1.0, complex128 = 2.0 → **float16 buys nothing.**
- **Zero-prediction fallback (§5.5):** over budget, exception, non-finite, OOM, wall-clock >120 s or residual >400 ms →
  that MLP scores as all-zeros. No retries, no post-deadline resubmission.
- **Reproducibility (§5.5):** the grader supplies a fixed per-MLP seed identical for everyone; prize-eligible submissions
  must reproduce under it. Grader env: 2 vCPU for our process, 8 GB, CPU-only, no network, **numpy not installed**.
- Sponsor may use LLM-assisted review and may invalidate a graded submission retrospectively.

**Algorithmic Contribution — how to enter and how it is judged** (ARC's guidelines thread #18041):
- Entry = **one PDF technical write-up + exactly one submission ID; both mandatory.** Email to
  `arc-whestbench@aicrowd.com` (private) or post on the challenge forum (public).
- Awarded "at ARC's discretion to the method we think most improves our understanding of white-box estimation for
  random MLPs" — **score alone does not decide**; mechanistic insight does. ARC is "less interested in methods that
  rely heavily on sampling, fine-tuned constants, careful performance optimization, and opaque LLM-optimized code".
- **Negative results, ablations and measurement studies are welcome** (a Phase-1 write-up characterizing a
  systematic bias is cited approvingly).
- **LLM use is encouraged but must be disclosed** (extent, for code and text); uncertain claims must be hedged or
  labeled as guesswork — "if we notice unhedged, dubious claims … we may skip over submissions entirely".
