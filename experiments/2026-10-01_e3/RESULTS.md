# E3 (receiver columns of the leg transport) — measured and CLOSED, 2026-10-01

**What E3 is:** neurons that are receivers at this layer (α = μ/σ < θ_out) get no leg columns, so the transport's
output axis is compacted. θ_out ≤ deadcut, so every receiver is already dead: its D rows/columns are zeroed by the
matched cut and its leg columns are multiplied by d1 = 0 at the end of the layer. The only thing that changes is this
layer's readout of t for those neurons. Arm "zero" drops t entirely; arm "b" restores it from the newest source plus
the repeated-index rule on the previous layer's (t, D).

Cloud #8 projected **util ×0.953** for arm "b" from its calibrated cost model. The catalogue's pricing put FULL
(= k3q + E3) at util 18.95% against k3q's 19.735%, i.e. **×0.960**.

## 1. Price the construction first (project lesson #9) — and the saving turned out to be the problem

`price.py`: the receiver share of the transport's output axis at 1024×16, from the chain's own α per layer:

| layer | 0–2 | 3 | 5 | 8 | 10 | 12 | 15 | mean 1–15 | **cost-weighted** |
|---|---|---|---|---|---|---|---|---|---|
| α < −3.0 | 0% | 1.1% | 3.2% | 8.4% | 11.9% | 14.0% | 20.2% | 8.3% | **≈11.7%** |

(cost-weighted by the layer's source count, since late layers carry more sources). So E3 removes ≈11.7% of the
transport. **The transport, however, is only ≈10% of the budget** — which is the whole story.

## 2. Arm "zero" implemented and measured (`src/estimator_k3s.py`, switch `recv`)

Fifteen lines: take the kept output columns of W (plus a zeroed pad block, so the width stays a multiple of the
Strassen quantum), contract into a compact buffer, scatter back through a column map that sends every receiver to the
zero column. **t then comes out exactly 0 at the receivers for free**, because `region_readout` reduces over hub rows
of a zero column — no fill-in needed, no extra Strassen call, no layout change. (Arm "b" needs the newest source at
full width and its rows sit in the MIDDLE of the stack, so it costs two or three separate chunked Strassen calls per
layer, i.e. +25–50% flopscope calls = most of our residual headroom.)

| check | result |
|---|---|
| k3s(`recv=None`) vs k3r, 1024×16 and 128×8 | **bitwise, FLOPs equal** |
| `recv=-3.0`, 1024×16 | FLOPs **×0.98852** (= **1.15%** saved), residual 0.377 → 0.384 s, max rel move 3.3e-5 |

**So the measured saving is 1.15%, not the model's ≈4.7% — the cost model overestimated by 3.5× and my own estimate
by 4×.** Lesson #9 again, and for the first time on the *saving* itself rather than on the construction.

## 3. The 10-MLP screen on the shipping base (k3r)

| arm | raw × k3r | util × | **adjusted ×** | better |
|---|---|---|---|---|
| E3 "zero" θ = −3.0 | 1.0093 | 0.98693 | **0.9960** | 3/10 |
| E3 "zero" θ = −2.5 | **diverges** (one MLP at 2.5e-1) | 0.97932 | — | 0/10 |

**0.4% at θ = −3.0 — below the grader's resolution (§60: a lever under 2–3% cannot be validated by a submission).**

**And the divergence at θ = −2.5 is mechanistic, not a bug:** zeroing the transport's output columns also zeroes the
corresponding **rows of the band basis Pᵀ**, which rides the transport. At 25% of rows zeroed, P — whose condition
number already reaches 1.2e5 at depth (cloud #8) — loses rank, `qr(Pᵀ)` gives a singular R, and `solve(R, ·)` blows
up. **E3 and the transported band are structurally in conflict**, and even θ = −3.0 (8–20% of rows) is fragile.

Projecting arm "b" from the same numbers: the same ≈1.15% of util at raw ×1.002 → **≈×0.991, i.e. 0.9%** — also below
the grader's resolution, for several hours of surgery on the most intricate part of the code, with the same band
fragility. **Not built.**

## 4. Verdict

**E3 is closed on the meter.** With it, the engineering track of this project is finished: the shipping point stays
**k3r = 4.0579e-9 (333307)**. The earlier "≈3.90e-9 with E3" projection, which came from the cloud's cost model, was
wrong by a factor of ~9 in the gain; the honest endpoint of engineering is **≈4.04e-9**.

Side result worth more than E3 itself: the transport is ~10% of the budget and C_pre ≈ 8%, so **≈80% of the budget
sits in the pairing, the births and the band** — and nobody in this project has ever measured the split by call site;
every cost argument so far came from a model. `estimator_prof.py` + `prof_run.py` instrument the shipping estimator
with `flopscope.namespace()` for exactly that, and the breakdown is readable from the official runner's
`breakdowns.estimator.by_namespace`.

Files: `price.py`, `k3s_check.py`, `runs/` (`v_k3s_r30.py`, `v_k3s_r25.py`, screens), `PLAN.md` (the arm-"b" design,
kept for the record), `estimator_prof.py`, `prof_run.py`. `src/estimator_k3s.py` keeps the switch, default
`recv=-3.0` — **do not ship it without re-reading §4 above**; the shipped file is `src/estimator_k3r.py`.
