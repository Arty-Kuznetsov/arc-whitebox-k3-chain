# k4a — the port of the young-κ₄ reader without the last-layer pairing (R6, 2026-10-06)

Plan and kill rule: `PLAN.md`. Torch reference: `../2026-10-04_r3/RESULTS.md` step 3 (bCy2K4pdnl ×0.879 raw on 10/10 at
+6.25% metered FLOPs → 3.49e-9). Estimator: `src/estimator_k4a.py` (= k3z + `restructure` + `young`; both off is bitwise k3z).

## 1. Gates
| gate | result |
|---|---|
| G0 restructure=False, young=None | **bitwise k3z, equal FLOPs** at 1024×16 (synthetic) and 128×8 |
| G1 restructure=True, young=None | outputs equal k3z to float rounding (max rel move **5.0e-6**), **FLOPs +4.91%** (the meter's stub said +6.25%), wall +1–2 s local |
| G2 defaults | FLOPs **+5.05%**, finite |

Two bugs found by the gates and fixed on the way: the snapshot ring needed three slots (at layer l the snapshots of l−1
and l−2 are read while l's is written; two slots overwrote l−2); the right factor of the old sources' pairing takes
the gate d1(l−2) (the transport into l−1 ran on gated legs) while the reader's Dx(l, l−2) is the ungated pairing.

## 2. The reader against the torch reader, network 0, part by part (`prod_parts.py`, `torch_parts.py`)
pathdiag: corr 1.000, norm ratio 1.000 at every layer; star: the same; 3+2: corr 0.98–1.00 up to layer 12, 0.74–0.83
at layers 13–14; **the total diverged with depth (corr 0.82 at layer 4 → 0.01 at layer 14, norm ratio 0.97 → 0.65)**:
production computes the 3+2 only on the rows of D (alive and not saturated; the E2 and dead cuts), and the projection
coefficient off the closure's pattern, taken over the zero-filled vector, then shifts every neuron's K4d coherently.

## 3. Torch variants that explain it (10 networks, pooled raw × vs the band base; `../2026-10-04_r3/r3full.py`)
| torch reader variant | × | better |
|---|---|---|
| full (bCy2K4pdnl) | 0.879 | 10/10 |
| Dx as the production partners give it: no thin remainder at the snapshot, no q mᵀ (…P) | 0.886 | 10/10 |
| … + the 3+2 zeroed on production's cut rows, coefficient from the zero-filled vector (…Pc) = **what production did** | **1.023** | 4/10 |
| … + the coefficient from the uncut rows only (…Pc2) = **what production can do** | **0.899** | 10/10 |
| … rows zeroed AFTER a projection with the full coefficient (…PcA) | 0.886 | 10/10 |
So the missing thin/feed terms cost nothing (0.879 → 0.886), the cut rows' values cost little (0.886 → 0.899 via the
coefficient), and the coefficient computed on the wrong support cost everything (0.899 → 1.023).

## 4. Production with the corrected coefficient (`runs/k4a_n3_v4.json`, local runner, quiet-ish Mac)
| MLP | raw × vs k3z | torch Pc2 | FLOPs × | wall s (k3z) | residual s (k3z) |
|---|---|---|---|---|---|
| 0 | 0.973 | 0.985 | 1.0520 | 38.1 (37.5) | **0.490** (0.350) |
| 1 | 0.944 | 0.941 | 1.0513 | 37.4 (36.5) | **0.531** (0.344) |
| 2 | 0.880 | 0.859 | 1.0513 | 37.4 (35.8) | **0.561** (0.345) |
Accuracy reproduces torch (pooled 0.933 vs 0.930 on these three; ≈0.90 expected over ten). **Projection: 0.899 × 1.052 =
0.946 → ≈3.53e-9** (display 3.5e-9, shared rank 29) — below the 3.55e-9 bar on FLOPs and accuracy.

## 5. The residual cap, the binding constraint (`opcount.py`)
k3z makes 23,911 flopscope calls per network; k4a 29,492 (**+5,581, +23%**): multiply +1,747, subtract +1,447, add +1,414
(the Strassen combinations of the four small products per layer — two newborn contractions, Dx(l, l−1) = M·Wg(l−1),
D = Dc·Wg(l) — at depth 4, ≈120 calls each), reshape +350, take +319. k3z's grader residual is 0.31 s median / 0.331 s
max of the 0.400 s cap (local 0.35); k4a's local 0.49–0.56 s projects to ≈0.36–0.53 s on the grader: **networks would be
zeroed**. Cheaper depths cut the calls but raise the FLOPs: depth 2 → projection 3.65e-9, plain matmul → 3.66e-9, both
worse than k3z's display. There is no depth at which both caps hold. The 100-network local run (`runs/k4a_n100.json`)
gives the residual distribution; a grader probe (not designated) is the only true measurement of it.

## Verdict (FINAL 2026-10-07 — see §6 for the 100-network run and §7.4 for the grader probe 334486)
The reader is correct and worth ×0.90 raw at +5.2% FLOPs in production; it costs 23% more flopscope calls than the
residual budget has. The same mechanism as NEGATIVE §54–§59, one level down: here it is not the FLOPs of the
construction but its *call count* that costs what the accuracy buys.

## 6. The 100-network local run (`runs/k4a_n100.json`, 2026-10-07 00:xx, official runner, `--no-residual-wall-time-limit`)
| | k4a | k3z (same run type, 10-02) |
|---|---|---|
| adjusted (local) | **3.7187e-9** | 3.8289e-9 |
| util | 19.31% | 18.37% (FLOPs ×1.0512, min 1.0488, max 1.0539) |
| raw × vs k3z | **pooled ×0.924**, mean ×0.926, **better on 95/100** (0.815–1.039) | — |
| residual (local, shared Mac) | median **0.439**, max 0.499, min 0.412 s; 100/100 above 0.400 | median 0.361, max 0.476 (its grader: 0.310 / 0.331) |
| wall (local) | median 37.1, max 38.7 s | 36.4 / 40.6 |
**Projection on the grader: 3.7187e-9 × 0.976 = 3.63e-9** (display 3.6e-9, shared rank ≈ 31) — not the 3.53e-9 the
10-network torch number suggested: production's raw gain is ×0.924 over 100 networks, not ×0.899 (the torch Pc2 value
over 10). The residual projects to ≈0.38 s median on the grader (k3z's local-to-grader ratio 0.86–0.70), with a tail
near or above the 0.400 cap. **The user approved one non-designated grader probe** (10-06 night) to measure that tail → submitted as **334476**
(`submission-20261006-190144.tar.gz`, estimator byte-identical to `src/estimator_k4a.py`, sha256 28af7e96…).

## 7. The grader probe 334476 — FAILED at the smoke test (2026-10-07, 00:14 poll)
`status: FAILED`, `gradingMessage: "Error : Smoke test failed"`, `smoke_test_result: {passed: false, duration_ms: 6689.65,
worker_passes: 0, worker_failures: 1, error_code: ESTIMATOR_EXCEPTION}`, `mlps_failed: 100`, `per_mlp: []`, no grading log
on the page (`gradingLogUrl: None`, `error: {}`). The exception fired within ~6.7 s of the smoke network — before the
first layers finish (k3z needs ~37 s per network), so at import, construction, workspace allocation or the first
restructured layer. Locally `whest validate`, the 100-network run (local in-process runner) and `validate-package` had
all passed, so the cause is something the grader's environment does differently: the flopscope **client/server split**
(`RemoteArray` handles instead of in-process `FlopscopeArray`; `_server_only.py` lists the names missing on the client)
or the 8 GB memory cap (k4a's workspace adds ≈290 MB to k3z's ≈6.9 GB peak). Diagnosis: reproduce under
`whest run --runner subprocess` (the isolated, client/server backend), then isolate the operation. Nothing designated
was touched: k3z 333534 + k3w 333388 remain the designation pair.

### 7.1 Diagnosis, step 1 (10-07 00:15–00:35): what it is NOT
- **Not the isolated runner, not RSS.** `whest run --runner subprocess` (RLIMIT_AS 8 GiB on the worker; not enforced by
  macOS) passes k4a on one network (same FLOPs 432,867,955,432, residual 0.52 s). Peak RSS of the whole `whest run`
  process (dataset included): k4a 13.71 GB vs k3z 13.67 GB — **+37 MB** only; the +290 MB estimate was the *allocated*
  (mostly untouched) workspace.
- **Not the client/server semantics.** Built the grader's split locally: `external/flopscope/flopscope-server`
  (`python -m flopscope_server --url ipc:///tmp/fs_k4a.sock`) + `flopscope-client` (uv envs; the editable `.pth` files
  are not applied, so PYTHONPATH points at both `src` dirs) + a stub `whestbench` shim (`cs_stub/`) + harness
  `cs_run.py` (setup, `fnp.load` of the baked weights, predict — all inside one `BudgetContext`, because on the
  client every op outside a session fails with `NoBudgetContextError`). With the server's **default** caps both k3z and
  k4a fail at the first `fnp.empty` of the workspace: `result array too large: 131858432 bytes exceeds 104857600 byte
  limit` (`FLOPSCOPE_MAX_ARRAY_BYTES`, default 100 MiB, also in the PyPI 0.12.1 wheel). That is NOT the grader's
  setting: forum 18039 (mohanty, 2026-06-23) — the grading sandbox's per-array cap was **raised to 4 GiB and the live-array
  count to 10 million**. With `FLOPSCOPE_MAX_ARRAY_BYTES=4294967296 FLOPSCOPE_MAX_ARRAY_COUNT=10000000` **k4a runs clean
  through the client/server: 47.0 s, FLOPs identical to the in-process run, finite output.**
- **Timeline of the smoke test.** Passing smoke tests of our earlier submissions took 16–60 s (k3z 333534: 29.96 s,
  6 worker passes); the probe's lasted 6.7 s with 0 passes and 1 failure, and the evaluation record spans
  19:02:16Z–19:02:18Z. So the exception fired at the very start of `predict` (worker start + the 64 MB weight transfer
  as JSON take the first seconds), i.e. around the workspace allocation — not inside the restructured layers.
- **Remaining hypothesis: memory as the grader enforces it** (rules §5.6: 8 GB; forum 18238 asks whether it is
  RLIMIT_AS/VmPeak or resident — unanswered). k4a allocates ≈ +235 MB of `fnp.empty` buffers over k3z (SNX 3 × 56 MB,
  nine n×n float32 buffers, YA–YD, SNB); under an address-space limit an allocation fails immediately with
  `MemoryError`, which fits the timeline. Measuring now: the server's RSS peak and the full list of ≥ 1 MiB constructor
  allocations for k3z vs k4a (`cs/cs_run3.py`).

### 7.2 The cause: k4a breaks on shapes other than 1024 × 16 — and the grader's smoke test uses one (10-07 00:40)
- Memory is not it either: the ≥ 1 MiB constructor allocations (client-side log) total **6.101 GiB for k3z and 6.374 GiB for
  k4a** (+0.27 GiB: SNX 3 × 54 MB, nine n² buffers, YA–YD, SNB); 4.6 GiB of both are the Strassen regions.
- **The smoke test is not full-size.** Two Phase-2 write-ups say so (venture_agent_ai 18214: "the grader's smoke test runs a
  shape other than 16 x 1024, so anything indexed by layer must be guarded"; yujigon 18216: a submission "failed the
  grader's smoke test on a depth-indexed constant table"), and the timings agree: k3z's graded networks take 91 s each on
  the grader, its smoke test took 30 s. Our earlier note "the grader's smoke test is full-size" (WORKLOG 10-01, l5t
  RESULTS) was an inference from k3t passing, not a measurement — **wrong**.
- **A shape grid finds it** (`shape_grid.py`, in-process, widths 16–512 × depths 1–16, budgets 2^41 and
  scaled): k3z passes everything; **k4a fails at depth ≥ 12 for every width ≤ 512** — two errors:
  (1) widths 64–512: `operands could not be broadcast together with shapes (1362,64) () (1280,64)` — the snapshot copy
  `fnp.multiply(XT[:st_rP + 2·st_W], 1.0, out=SNX_[…])`: SNX was sized by the band bound `2(a0+1)n + n + r` from
  `self.band`, but predict disables the band on small widths (`band[1] > n//2 or band[1] % quantum`), so the stack grows to
  CAPR rows and no longer fits; (2) widths 16–32: `output array does not match result of ndarray.take` in the reader's
  gathers. Neither can fire at 1024 × 16 (the band is on, the gathers are n-sized), which is why validate, the 100-network
  run and the client/server reproduction all passed. Fix: size SNX by the *effective* band (same rule as predict) and
  guard the copy (fall back to the k3z pairing when the stack would not fit); make the reader's gathers shape-safe; then
  re-run the grid, the gates and a bitwise before/after comparison at 1024 × 16 before re-packaging.

### 7.3 Fix, verification, resubmission (10-07 00:43–00:50)
- **Fix** (`src/estimator_k4a.py`, header "10-07 shape fix"): (a) `restr` additionally requires the band *in use for this
  width* (the per-call `band`, after predict's small-MLP rule) — without the band the snapshot stack is unbounded, and k4a
  then runs exactly as k3z; (b) SNX is sized by the same rule in `_workspace` (one row when the band is off); (c) the
  snapshot copy checks that the stack fits, else the rest of the network takes the k3z pairing; (d) the restructured
  pairing checks the layout it assumes (non-negative row counts), else that layer takes the k3z pairing. None of (b)–(d)
  can fire at 1024 × 16 with the production configuration.
- **Verification:** 1024 × 16 MLP 0 in-process **bitwise identical** before/after (FLOPs 432,867,955,432, same output);
  shape grid widths 16–768 × depths 1–16 × budgets {2^41, scaled} (`shape_grid.py`, log `runs/shape_grid_k4a_fixed.log`):
  **152 OK, 8 FAIL — exactly k3z's own failure set** (widths 16–64 at depth ≥ 12: a pre-existing tiny-shape `take`
  mismatch in the shared pairing code, which the grader's smoke shape evidently avoids since k3z passed); the depth grid at
  width 1024 (depths 2–12, both budgets) passes for both estimators (`runs/depth_grid_1024.log`); through the local
  client/server pair with the grader's caps (`cs_run_shape.py`): 512 × 16, 768 × 12, 256 × 16 OK and finite; `whest
  validate` OK; static import scan clean.
- **Package** `submission-20261006-194746.tar.gz` (estimator byte-identical, sha256 `aca032ca…`; `validate-package` OK)
  → **submitted as 334486** (00:49 local) — the same user-approved, non-designated probe, resubmitted after the fix.
  Poll: `scripts/poll_sub.py 334486` (the smoke result shows within minutes, the grade in ≈ 35 min). Decision rule
  unchanged: designate k4a only if 0 failed networks AND graded better than k3z; the probe measures the residual tail
  (projected ≈ 0.38 s median against the 0.400 s cap).

### 7.4 Grader result of 334486 (10-07, in progress)
- **Smoke test PASSED** (30.3 s, 5 worker passes, 0 failures) — the shape fix was the whole story of the 334476 failure.
- Scoring: at 14% (00:54 local) **1 network already failed** out of 14 graded. Under the decision rule (0 failed AND better
  than k3z) k4a is therefore **not designated**; the probe now delivers what it was for — the grader's residual distribution
  and the failure mode per network (filled in below when the grade completes).

- **Final (01:23 local, `runs/grader_334486.json`): 84 completed, 16 failed — all `RESIDUAL_WALL_TIME_EXHAUSTED`**
  (indices 4, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 70, 72, 86, 92, 98; 0.401–0.479 s). Residual over the 100 networks:
  min 0.373, p10 0.387, **median 0.393**, p90 0.416, max 0.479 s (k3z 333534 on the same networks: 0.297 / 0.310 / 0.331);
  paired delta +0.084 s median (+0.055 … +0.169), ratio ×1.27 — in line with the +23% call count. Wall 96.8 s median /
  112.4 max (k3z 91.2 / 98.8; cap 120). FLOPs ×1.0509 (util 19.26% vs 18.33%), exactly the local figure. Server kernel /
  overhead / comms 82.8 / 13.6 / 1.4 s vs 80.0 / 10.8 / 1.1. The failures run every fifth network from 19 to 64 — the slow
  side of the grader's worker spread — so the margin to design for is the slowest worker's, not the median's. The score
  field stays empty on the page (zeroed networks dominate the mean), so the port's accuracy cannot be read off the grader;
  the local 100-network ×0.924 stands.
- **Verdict: k4a is NOT designated (rule: 0 failed networks); R6 closed; NEGATIVE §77 final.** The probe did what it was
  for: the residual tail is measured on the grader, and §7 of the write-up ("priced in calls as well as FLOPs") rests on
  a grader measurement, not a projection.
