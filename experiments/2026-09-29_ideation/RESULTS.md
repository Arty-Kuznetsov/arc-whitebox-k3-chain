# First-place ideation workflow (2026-09-29) — what it was, what it found, what happened to each lead

**Why it was run.** After the cost axis closed five times (§54–§59) the user asked for broader, more creative
thinking about first place. A structured multi-agent pass was used instead of one mind: seven independent research
"lenses", each vetted by an adversarial skeptic against all negative results and the rules, then deduplicated and
ranked, then attacked by a completeness critic.

**Files.** `workflow.js` (the exact script) · `ideas-and-verdicts.md` (all 20 ideas, proposer's pricing, skeptic's
re-pricing and reason, cheapest falsifying test) · `batches.json` (same, machine-readable) · `synthesis.md` (research
lead, verbatim) · `critique.md` (completeness critic, verbatim).
**Run.** Relaunched on 2026-09-29 14:58 after the first run (same script) died with every agent on a weekly usage
limit. 16 agents, all succeeded, ~4.5M sub-agent tokens, 455 tool calls, ~1 h 54 min. Read-only: no agent edited
files or ran long computations.

## The seven lenses
billing rules & fast algorithms · random-matrix / mean-field · algebraic restructuring of the chain · tensor formats
beyond CP/Tucker · statistics & sketching · learned closures · competitive intelligence (incl. the public forum).

## What the synthesis concluded (verbatim numbers in `synthesis.md`)
- **P(#1) < 0.5%, P(top-5) ≈ 1–2%.** Realistic finish ≈4.2–4.7e-9 if the live cost levers are ported; the board
  will move down in the remaining 18 days.
- Two corrections to our own briefs: (i) the (t, D) sufficiency question was already closed by the cloud (§62) —
  only the OLD tail's own all-distinct share was unmeasured (MST, Prompt A); (ii) **Prompt B's band arms were wrong**
  — three skeptic tests (one on real weights) show a transported rank-r basis captures what an ideal ~0.8r one does,
  no spectral gap near 320, so the working point is r = 384–448.
- 17 ideas deduplicated into 6 mechanisms. Ranked by gain × probability:
  **M1** band with a cheap entry (6 lenses converged) · **M2** saturated-row cut of the D pairing with the κ₄ level
  restored (4 lenses) · **M3** dead receiver rows in the leg transport (2 lenses) · **M4** MST (the cloud's
  Prompt A) · **M5** symmetric C_pre (×0.983, marginal, ~+80 ms residual).
- Proposed first experiments: E1 (band, no new code), E2 (satrow + κ₄ restoration), E3 (receiver rows).

## What the completeness critic added (verbatim in `critique.md`)
1. **Transport need not match readout — jump the old tier k layers through one shared operator J**, a slice rule
   between checkpoints, exact reset at checkpoints. The one class all seven lenses missed.
2. **Strassen L5's wall veto depends on the count.** Leaders' billed throughput on the grader is 3.0–4.2 GFLOP/s
   against our 8.97 — consistent with one recursion level deeper. L5 re-opens for us once util ≲ 17%. Build the
   port with the recursion depth as a parameter.
3. **Per-MLP telemetry as an instrument.** Our per-MLP error ≈ the leaders' common factor × an *independent* excess
   (residual sd 0.079, R² 0.67; the leaders' own 0.044–0.057). We carry an extra error channel with its own per-MLP
   signature; our six graded submissions can serve as a fingerprint basis to find it. (Not yet done.)
4. **Age-dependent cut thresholds** — the old tail is a soft channel, young legs are stiff.
5. **The "<3% do not ship" rule is wrong for cost levers**, since FLOPs are analytic. Checked here: the util ratio
   k3m/k3j is 0.9447 on the grader vs 0.9476 locally — a 0.3% gap (data-dependent through the dead cut). Cost ratios
   transfer ~1.5× better than raw ratios, not exactly.

## What happened to each lead (measured on the Mac the same evening)
| lead | result | where |
|---|---|---|
| critic #1, jump / stride | **closed**: one un-reset layer costs ×1.5–×3.5 for every a0 < 10 | `2026-09-29_stride/`, NEGATIVE §63 |
| M4 / MST (Prompt A) | **predicted dead** by the stride result; prompt withdrawn, but the cloud had already run it: closed, NEGATIVE §66 | `notes/CLOUD-SESSIONS.md` |
| M1 / E1 transported band | **survives on accuracy** (+0.69% at 6:448, +2.46% at 6:384); **cost hinges on the coefficient rotation** (×1.02 with it, ×0.87 without) | `2026-09-29_band/` |
| M2 satrow, naive (no κ₄ restoration) | ×1.0094 at 3.0 (6/10, spread ×0.84–1.35), ×1.34 at 2.5 — restoration untested | `2026-09-29_satrow/` |
| M3, critic #2, #4 | handed to the cloud port battery | `notes/CLOUD-BRIEF.md` |
| critic #3 fingerprinting | not done yet (Mac/browser task: the grader pages are not reachable from the cloud) | FRONTIER |
| critic #5 | checked: 0.3% util transfer gap | above |
