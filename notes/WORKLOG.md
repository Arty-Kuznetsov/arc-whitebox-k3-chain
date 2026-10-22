# WORKLOG

Newest first. One dated entry per working session.

---


## 2026-10-22 (Mac, Opus 5.5) — after the close: release and dataset prepared
- Board at the close #77 of 309. Release dry run clean (6,189 files, 75 MB); README extended; forum drafts + the unapproved
  paper excluded from the release. Dataset zip built (94.6 MB) + Zenodo fields. Waiting for the repo name and the go-ahead.

## 2026-10-21 (Mac, Opus 5.5) — cited by 18254; plan and benefits for the user
- 18254 (Susupro) cites our write-up twice (table of the eight public write-ups; the 4% per-neuron bound, contrasted with their
  W-reading corrector ×0.887 — consistent with Prop. 1). Their exact K=4 at 1024 (11.8× below augmented K=3) continues our §7
  trend. Reply draft written. Board #74 of 306. Plan + honest benefit assessment given to the user.
- Nomination done by the user (333534 + 333388). Reply posted by the user in 18254; Susupro confirmed the K=4 trajectory
  (×3 / ×6–7 / ×11.8), the corrector hierarchy (≤4% / ×0.988 / ×0.887) and birth 89.8% vs transport 10.2% — to cite in the final PDF.

## 2026-10-17 (Mac, Opus 5.5) — deadlines extended; nomination opened; flopscope 0.12.3 leaves k3z unchanged
- Forum 18251: submissions to 21 Oct, nomination to 22 Oct (Submissions page → Your team → Select), write-ups to 28 Oct;
  grader on flopscope 0.12.3. k3z re-metered under 0.12.3 + whestbench 0.16.2 (scratch venv): identical FLOPs on MLPs 0–1,
  MSE ±0.1% → unaffected. Board #59. The user asked where the nomination form is: answered.

## 2026-10-16 (Mac, Opus 5.5) — live check before the close; concurrent work credited; the to-do list
- Board #55 (3.7e-9); Phase 2 closes 10-17 23:59 UTC; no selection form yet; write-up rules re-read (18041).
- Forum 10-15: il.miscusi 18248 (scale mode of a homogeneous network, removed in an estimator) and kai_leal_gonzalez 18246
  (conditional skew identity ≈ Prop. 5's decomposition) — independent, posted after our 10-14 commits but before our release;
  credited in paper §5, write-up §6.4 + credits, forum draft.
- The user asked for the ordered list of what to upload where; given in the chat (and in CLAUDE.md CURRENT STATE).
- Write-up v4 final for posting: board numbers of 10-16, §9 audit sentence, §10 placeholder replaced (release at the close,
  links in the thread). **Posted by the user as forum topic 18249** (https://discourse.aicrowd.com/t/phase-2-write-up-a-third-cumulant-chain-at-its-boundary-random-projections-a-skew-law-and-where-the-next-orders-accuracy-lives-submission-333534/18249); checked rendering and the attachment.

## 2026-10-14 (evening, Mac, Opus 5.5) — v ∥ m: Proposition 5, the radius as an exact scale mode, the amplitude mode
- The user: "derive v ∥ m, continue, do the science items, we need a contribution".
- `experiments/2026-10-14_vm`: theory + pre-registered P1–P4 written before the runs; `vm_mc.py` (two passes over the same
  stream → exact central third moments, no Gram), 16 runs ≈ 50 min. P1 (Prop. 5: slope = 3q/n ⟨v,m⟩/‖m‖²) holds on truth
  (median 0.6–1.6% at 1024); P2 (gauss − sphere slope = 3σ̄²_w; sphere c(1)·n = 9.82) holds to ≤ 0.024; P3 (alignment made
  inside the network) holds; P4: v in span{μΦ(α), σφ(α)}, R² 0.73 → 0.97, λ₂/λ₁ 1.3–1.8 (not a pure rescaling).
  Mechanism measured link by link (amplitude mode, Euler response, v ≈ κCm̂). Closed forms: εn(0) = π − ½ / π − 1;
  D ≈ t/3 from Isserlis. Open: growth/saturation of εn with depth, the 62–78% purity.
- Paper §5 + write-up §6.4 rewritten with it; second sub-agent fact-check: derivations all correct; fixes (12 runs, λ ratio to
  1.9, width scopes, "exactly in expectation", q_v/σ̄²_w defined, radius renamed r). Closure ladder: `ptrace` arm (κ₃ ≈
  Sym(I⊗v)/n) keeps 72–73% (law 63–65%); log gain 28% vs 23% at 1024. A --only-ptrace run had overwritten the full-arm JSONs;
  restored from git and re-run with all arms (bitwise equal); ptrace-only runs now write *_ptrace_* files.
- Prop. 3 on the true κ₃ tensor (MC n³, two seeds of 2e6; `prop3_truth.py`): 0.94 ± 0.10 (n = 64), 1.01 ± 0.13 (n = 128), the same
  as on the chain's tensor; Prop. 5's R² clause predicts the measured R² on average (0.763 vs 0.762 at n = 128). Paper + write-up
  updated. n = 256 started (one network).
- Activation test (pre-registered A1–A3): Prop. 5 activation-agnostic (A1 ✓); radius = scale mode only for homogeneous
  activations (A2 ✓); A3 ✗ as stated (GELU's Euler link not weaker) but SiLU breaks the alignment (cos(v,m) 0.75, law R² 0.47 at
  layer 15): the skew law at depth needs positive homogeneity. Radius formula refined: + c_sphere (O(1/n)). Paper/write-up updated.
- Prop. 3 on the true tensor at n = 256 (one net, 22 min): 0.97 ± 0.04; Prop. 5's R² clause layer by layer 0.80–0.83 vs
  0.80–0.85. All three widths now agree with the chain-based check. Forum draft 2b gained one Prop. 5 sentence.
- Mistake recorded: the first n = 256 launch (`&` inside a foreground call) looked dead after 1.5 min but ran to 20:52, so two
  heavy processes overlapped for ≈ 20 min against the machine policy. Lesson: launch long jobs only with run_in_background and
  check `pgrep -f "[p]rop3_truth"` before concluding a job died. Results unaffected (deterministic, identical).

## 2026-10-14 (Mac, Opus 5.5) — the skew law, derived; the k = 2.5 closure; plan of the short paper
- The user: "continue, and update the long-term context as each piece finishes".
- `experiments/2026-10-14_skew`: κ₃ᵢ ≈ c(l)μᵢ at every shape; c(1)·n = 12(κ₃(R) + B₂)/E[R] = 13.91 derived (Mehler, ρ²
  term with E ρ² = 1/n), measured 13.94 ± 0.14 / 13.98 / ≈14.2 at n = 1024 / 512 / 256. Closure t = a + cμ, D = t/3 in the
  chain via `mcorr` (new "D" key, gate 3.7533e-08): keeps 63–65% of the κ₃ gain at L/n ≤ 1/32 with offline constants
  (= own constants), its residual has ≈0% learnable (the κ₃-free arm: 60–65%) → the learnable part of a Gaussian
  closure's error is exactly the slaved skew; k = 3 still ×37 better; fails at L/n ≥ 1/16; D = 0 worse than nothing.
- `writeup/short/PLAN.md`: 8-page paper "Where the error of moment propagation lives" (sections, sources, credits).
- Write-up §6.4 + §0 + §10 updated with the skew law and the closure table (PDF 32 pp.); forum draft 2b extended.
- Short paper drafted: `writeup/short/paper.md` → `paper.pdf` (8 pp.), new Figure 1 (κ₃ vs μ; c(l)·n by layer for 5 shapes;
  at width 256 depths 8/16/32 give one curve). Proposition 4 with a proof sketch (App. A). Fact-check corrections applied
  before the build: ×108 (not ≈110), k = 4 vs SIMPLE k = 3 is ×8–17 (×3–7 was vs our chain), k = 3 learnable can exceed
  its coherent share (512 × 16: 4.1% vs 1.2%), dataset covariance = 4 layers of 2 networks.
- Independent fact-check of the paper (sub-agent, read-only; Prop. 4 re-derived, quadrature to 5e-7, exact population
  slope over w for 4 random W0 at n = 1024: 13.80–14.09, mean 13.96 → the linear-in-ρ term sets the spread). Fixed: R² at
  depth 0.71–0.90 (not 0.8–0.9; the RESULTS 256 × 32 row had been read off a truncated print — layer 19, not 31); slope
  spread a few % only at 1024, 13–77% at the last layer of smaller widths; κ₃ = 0 is 1–12% below ARC k = 2 (not 6–14%);
  public oracles 2–29 are single-seed; abstract item 4 now says young blocks + one scale-mixture block (78–98%); "layer 1 =
  first non-Gaussian layer" (layer 0 is the first hidden layer); k4age ≈ 12 min/network (write-up §10 too); Prop. 3's G
  defined as T∘(d₁⊗d₁⊗d₁); cos(v, m) at layers ≥ 8; §3 measured on our augmented chain; ±0.14 = standard error.

## 2026-10-13 (evening, Mac, Opus 5.5) — what is unique; three new results in the write-up
- The user asked what is original; answer: Propositions 1–2 + random directions, where k = 4's gain lives, (t, D)
  insufficiency (shared with 504aldo); weak spots: one shape, few networks, no cheap algorithm for 3+2. User chose
  "#1 and everything useful". Credit added: hyojun_kwon 18154 (consistency principle's precursor).
- #1 `experiments/2026-10-13_coherence-law`: prediction (learnable share grows with L/n) pre-registered and FAILED;
  the law found instead: the learnable share is set by the truncation order (ARC k = 2 ≈ 65% at 5 shapes, k = 3 ≤ 4%).
  Mechanism: true κ₃ ≈ function of the neuron's own mean (corr 0.87–0.93; 79–88% explained). mliston's Phase-1 GRU used
  MC sample statistics → no counterexample (our INTEL §5.6 guess corrected).
- #3 `experiments/2026-10-13_prop3`: Proposition 3 derived (Isserlis) and verified: measured/predicted 0.94/1.00/0.99 at
  n = 64/128/256; the share falls slowly with n (46 → 40%, 28.8% at 1024); partial trace v ∥ mean vector (cos 0.92–0.99).
- #4 `scripts/error_anatomy.py` (any estimator, outputs only): covariance propagation 26% coherent / 66% learnable / parity
  z² 0.98; k3z 0.6% / 0 / 1.11. Write-up §0, §6.2, §6.4, §10 updated; PDF 31 pp.
- #2 `experiments/2026-10-14_k4age` (gates: stepped = mlp_kprop 1.6e-16; age ≥ L exact): dropping old κ₄ in ARC's k = 4 fails;
  young blocks + one scale-mixture block keep 92–98% of k = 4's gain at depth 8, 78% (age 2) / 89% (age 3) at 256 × 16,
  for 0.3–0.4 of the κ₄ columns and 0.54–0.61 of the wall. Prediction half right (closure essential, not "matters little").
  Write-up §7.3 (new), §0 item 5, §10 row. Oracle dataset builder + card (94.6 MB). Forum draft: 2b + Prop. 3.

## 2026-10-13 (afternoon, Mac, Opus 5.5) — the write-up read as a judge would; the release made runnable
- Read all 28 pages against the rules' prize criteria. Fixes: §6.2 formula glyphs (ₖ/ₘ missing in Helvetica → <sub>);
  catalogue references §54…§76 → N54…N76 (they looked like the write-up's own sections); §0 "close to the leaders' raw
  error", rank moved out of §0; §1/§7 notation; A.3 designation text true with or without the selection form; dash chains.
- §10 "Check it yourself": 13 headline claims → folder → entry point → cost to rerun (criterion iii of the rules).
- Release: the builder renamed experiments/ → measurements/ and left out pb.py and fetch_baked.py, so half the scripts
  could not run from the public repo. Now: original layout, all 62 experiment folders (54.7 MB), scripts/setup.sh (=
  cloud_setup.sh, pointed at k3z), fetch_baked.py, poll_sub.py, shape_grid.sh, prepkg_check.sh, notes/SPEC.md; absolute
  paths, session paths and two private journal notes redacted; three k4a scripts made portable. Clean-copy test:
  `whest validate` k3z OK, `ablate.py k4memo 3` 3.7533e-08, `ablate.py full 3` 4.2509e-08, `parity.py 0` = committed JSON.
- Eight files were half Russian (cloud sessions' RESULTS: old-tail, tail-sampling, state-closure, mst, k3m-port,
  cost-frontier; ideation synthesis/critique) — translated by two agents with a numeric-token check; originals kept as *.ru.md.

## 2026-10-13 (Mac, Opus 5.5) — actualisation: #47 at the same score (the second wave); no organiser news; numbers updated
- Board (curl): #47 of 290 (10-05: #33 of 266) at 3.7357e-9; 32 of the top 50 improved since 10-07; top five 1.4–1.5e-9,
  raw 1.15–1.36e-8 at 10.3–12.5% (INTEL §5.8; snapshot `notes/board_2026-10-13.json`). Our 333534 page unchanged (10-02).
- Forum: only AndreasHad04 (18241: 14 billing-fix PRs; 18238: client/server answer). No flopscope release, no
  re-evaluation, no selection form. Rules: Phase 2 End Oct 17 23:59 UTC; default designation = the two best public
  (k3z 333534 + k3y 333402).
- Write-up: 'as we write' numbers → 10-13 in §0/§4/§6.2/§7.2/§7.3/§8; §9 dates Fable 5.1 to 10-05…10-07; PDF rebuilt.
  Articles, LinkedIn, Habr pack and the forum draft: rank 47/290, top three 1.4–1.5e-9 (brackets stay for the close).
- Forum 18238 reply rewritten (natural-writing): telemetry shows setup() once per worker (networks 0–4 and 44 on 333534;
  0–4 on 334486) = the smoke-test worker passes (6 and 5) → one process per worker, ~20 networks each; 6.1/6.4 GiB
  pass; the smoke test is not 1024 × 16; grader/local ratios. Flopscope issues A (einsum wall) and B (solve on a
  triangular matrix) checked against all 108 issues: not filed by anyone.

## 2026-10-07 (Mac, Fable 5.1) — probe 334476 FAILED the grader's smoke test; diagnosis
- 00:14 poll: `FAILED`, "Smoke test failed", `ESTIMATOR_EXCEPTION`, smoke 6.7 s, no traceback or log on the page
  (`scripts/poll_sub.py` itself crashes on the empty per_mlp — to fix). Recorded in RESULTS §7, NEGATIVE §77 (status), CLAUDE.md, memory.
- Hypotheses: (a) flopscope client/server semantics — the grader runs `flopscope-client` (`RemoteArray` over ZMQ + msgpack);
  every local runner uses the in-process library, so neither `validate` nor the 100-network run exercised that path. The one
  fnp name new in k4a is `fnp.concatenate` (registered client-side), so the suspects are `out=` into column-slice views
  (`Yb[:, :wbN]`), 1-D `take(..., out=)`, argsort results as indices. (b) the 8 GB cap (RLIMIT_AS?) — k4a ≈ +290 MB over k3z.
- Running: `--runner subprocess` n=1 (RLIMIT_AS 8 GB locally), peak RSS k4a vs k3z (`/usr/bin/time -l`), then a local
  client/server reproduction with `external/flopscope/flopscope-server` + `flopscope-client` to get the traceback.

- Results (RESULTS §7.1): subprocess runner passes; RSS +37 MB; the client/server pair reproduces the grader's split
  (stub `whestbench` shim + `cs_run.py`, kept in `experiments/2026-10-06_k4a/`; the server's default per-array cap 100 MiB
  fails BOTH estimators at the first `fnp.empty` — the grader's cap is 4 GiB, forum 18039) and **k4a runs clean through it**
  (47 s, identical FLOPs).
- **Cause (RESULTS §7.2):** the grader's smoke test is NOT 1024 × 16 (write-ups 18214/18216; k3z smoke 30 s vs 91 s per
  network), and a shape grid (`experiments/2026-10-06_k4a/shape_grid.py`) shows k4a crashing at every width ≤ 512 with
  depth ≥ 12: the snapshot stack SNX was sized by the band bound while predict turns the band off below width 896 →
  overflow at the snapshot copy. (k3z itself fails only at widths 16–64 with depth ≥ 12, a pre-existing tiny-shape `take`
  mismatch the grader's smoke shape evidently avoids.) Fix: restructure gated on the band in use + SNX by that rule +
  guards (k3z pairing as the fallback). Verified: 1024 × 16 MLP 0 bitwise identical before/after (same FLOPs); grid
  16–768 × 1–16: 152 OK, the 8 failures = k3z's own tiny-shape set; client/server OK at 512 × 16 / 768 × 12 / 256 × 16.
- Re-packaged (`submission-20261006-194746.tar.gz`, byte-identical, validate-package OK) and **resubmitted as 334486**
  (00:49 local). Poll `scripts/poll_sub.py 334486` → RESULTS §7.3 / NEGATIVE §77 final.
- **01:23 — 334486 graded: smoke passed (30 s); 84/100 completed, 16/100 zeroed by `RESIDUAL_WALL_TIME_EXHAUSTED`**
  (residual median 0.393 vs k3z 0.310, max 0.479; FLOPs ×1.0509; wall max 112 s). k4a NOT designated; R6 closed; §77 final.
  Write-up §7.3 + articles §6 updated with the grader numbers; PDF rebuilt. Board 10-07: #36 (INTEL §5.7).
- **Daytime — plan W:** §0 item 5 (grader price of the port), arXiv abstract + Zenodo text, release list + dry run, forum
  18238 reply draft, Habr author pack. **Task C run and positive** (`experiments/2026-10-07_randdir/`): the per-neuron
  κ₃/variance errors are distributed exactly like the chain's errors along random directions (κ₃ spread ratios
  0.96–1.12 over MLPs 0–1, equal means; variance: same coherent level, spreads 0.98–1.03 at layers 8–14) → §6.4 "Measured,
  random directions", §0 item 1, articles §4.2. MC 10⁷ × 2 seeds ≈ 10 min each per network; chain 5 × 1.5 min.
- Lesson: **run the shape grid (widths 16–1024 × depths 1–16, budget 2^41) before every package** — `whest validate`'s tiny
  MLP never reaches the restructured layers, and the 100-network run is one shape.

## 2026-10-07 (early, Mac, Fable 5.1) — k4a on 100 networks: ×0.924 raw (95/100), FLOPs ×1.051, projection 3.63e-9; residual 0.44/0.50 s local; the probe goes out
- `experiments/2026-10-06_k4a/runs/k4a_n100.json`: adjusted 3.7187e-9 local (k3z 3.8289e-9) → grader projection
  **3.63e-9** (display 3.6e-9), 0 failed locally (residual limit off), residual median 0.439 / max 0.499 s local
  (k3z's local run 0.361 / 0.476 → grader 0.310 / 0.331). Raw ×0.924 pooled, better on 95/100. FLOPs ×1.0512.
- The user approved the non-designated probe (10-06 night); packaged `submission-20261007-*.tar.gz` (byte-identical
  estimator, `validate-package` ok) and submitted as **334476** (10-07 ~00:05 local). Grade and telemetry → RESULTS §7, NEGATIVE §77 final.
- English article written (`writeup/article/article-en.md`) while the run was going.

## 2026-10-06 (night, Mac, Fable 5.1) — k4a ported and gated; the reader is right; the residual-call cap kills it (NEGATIVE §77, provisional)
- `src/estimator_k4a.py` = k3z + restructured D pairing (snapshots two layers back, 3-slot ring) + the Young4Chain reader.
  G0 bitwise k3z (1024×16, 128×8); G1 D to 5e-6 with the reader off, FLOPs +4.91%; G2 +5.05%; `whest validate` passes;
  import scan clean. Two bugs found by the gates (ring size; the d1(l−2) gate on the old legs' pairing).
- Reader mismatch diagnosed part by part against torch (pathdiag/star/3+2 each corr ≈1.0; the total diverged): the
  projection coefficient over the zero-filled cut rows. Torch variants: production-style ×1.023 (4/10), coefficient on
  the uncut rows ×0.899 (10/10), full coefficient ×0.886, full reader ×0.879. Production with the fix: ×0.973/0.944/0.880
  on MLPs 0–2 (torch 0.985/0.941/0.859) → projection ≈3.53e-9 (display 3.5e-9).
- Residual: +5,581 flopscope calls (+23%; Strassen combos of the four small products per layer) → 0.49–0.56 s local vs
  k3z 0.35 (grader cap 0.400, k3z max 0.331). No Strassen depth satisfies both caps. 100-MLP local run started
  (`experiments/2026-10-06_k4a/runs/k4a_n100.json`, ~70 min) for the residual distribution; a grader probe (not designated)
  is the only true measurement — the user decides.
- Torch tooling: `young4.py` (dx="prod", rowmask, rec), `chain.py` (rowcut, cproj, rowcut_after, rec), r3full arms
  bCy2K4pdnlP/Pc/Pc2/PcA; meter.py `nolast` mode.

## 2026-10-06 (late, Mac, Fable 5.1) — R6: the unpriced variant passes the kill rule; port of k4a starts
- The user chose "fundamental improvements first" (bounded, kill rule in PLAN-10-06 §5). Re-reading R3's raw data: the
  "no last-layer 3+2" variant (bCy2K4pdnl) is ×0.879 pooled on 10/10 networks and was never metered. Metered now:
  +6.25% FLOPs (ages 1–2 upstream D = Dx·V products only) → projection **3.49e-9 < 3.55e-9**: the kill rule passes.
  R3's "worse" verdict for this row compared raw, not the metric. RESULTS step 3 written; port target k4a defined.
- Board 10-06: #34 of 269; 3.49e-9 would display 3.5e-9 = shared rank 29.

## 2026-10-06 (evening, Mac, Fable 5.1) — v4.1 prose pass; release dry run; flopscope reports; forum post final candidate
- Write-up: natural-writing pass over §3/§5/§6 (56 whitespace-tolerant edits; em-dashes 22/14/44 → 3/2/4; the three
  staging phrases and two "not X, it is Y" constructions rewritten; §8 leaders' numbers 1.5–1.6e-9; §4 floor sentence
  "around seventh place"). PDF rebuilt (28 pp.), pages 9 and 20 spot-checked.
- Release dry run: `scripts/build_release.py <dir>` assembles the public folder per RELEASE-PLAN (3,886 files, 28.4 MB;
  `*.npz`/`*.npy` and files > 2 MB left out, regenerable); `writeup/release/README.md` + `LICENSE` (MIT) written;
  secrets grep clean (only the word "secret" in prose); no personal mentions in the journal. `cov_oracle.py` /
  `d_oracle.py` live in their measurement folders, not in the chain folder (plan corrected).
- flopscope reports (`experiments/2026-10-06_flopscope-reports/`): (1) `einsum` 12–62× the wall of matmul+sum at an
  identical bill (5 shapes); (2) `linalg.solve` bills LU + two substitutions on a triangular matrix (×2.3–2.7 of one
  triangular solve), no `solve_triangular`. Issue texts ready; FILING IS THE USER'S ACTION. The `fnp.zeros` symmetry
  tag is cleared correctly by an `out=` write (no under-billing) — nothing to report there.
- Forum post: final candidate in `writeup/forum-post-draft.md` (rank 34 of 269 as of 10-06; J2W back at #1 1.5e-9).
  Awaiting the user's explicit confirmation before posting.
- Board 10-06 [M]: #1 J2W 1.5e-9 · #2 marius_binner / Luna 1.6e-9 · #4 mliston 1.7e-9 · **#34 us** (269 ranked teams).

## 2026-10-06 (afternoon, Mac, Fable 5.1) — write-up v4.1 built: judged-first summary, §7 "One order up", boxed propositions, appendix
- The user: forum post before the close YES, but only after their explicit confirmation of the final text; "take write-up
  v4.1 and do it professionally".
- v4.1 (28 pp.): §0 rewritten around "what we would like judged" (6 statements); §6.6 promoted to §7 with the 3+2
  identity [D], two new figures; §6.4 Propositions 1–2 boxed with proofs; §6 count 76, fifth §6.1 instance (§76), 10-05
  board fact in §6.2; §3.8/§4.3 → Appendix A (A.3 = designation k3z + k3w, the 9e-3 cost of a zeroed network); §8–§10
  renumbered, §9 names Opus 5.5 + Fable 5.1; figures 1–7 in order; stale refs 0 (grep). Figure script
  `writeup/make_v4_1_figures.py`; archives `v4-s0-summary.v4.0-archive.md`, `v4-s6-boundary.v4.0-archive.md`.
- Fixed a CSS bug that made a mid-sentence italic a block (`p > em:only-child` → `p.note > em`).
- Not done yet: natural-writing pass over §3–§5 prose; the user's §9 list; final numbers 10-16; §10 URL.

## 2026-10-06 (Mac, Fable 5.1) — natural-writing skill; article v2; write-up critique; the 11-day plan
- The user asked for texts that do not read as AI-written, and for a fresh read of everything with 11 days left.
- Sources read: Wikipedia "Signs of AI writing" (full list), blader/humanizer (26 patterns + procedure), petekp/de-slop,
  Russian markers (iphones.ru), Habr sandbox rules (rejects auto-generated-looking or list-heavy texts; ≥75%
  originality). No writing skill exists in the user's enabled skills (ListSkills/SearchSkills: none) → wrote the project
  skill `Research work/.claude/skills/natural-writing/SKILL.md` (RU+EN tells, procedure, Habr facts, checklist).
- Applied it: `writeup/article/article-ru.md` v2 (v1 kept as `article-ru.v1-2026-10-05.md`): no bold-label paragraphs,
  no staged openers/closers, Russian dashes where grammar needs them, first-person dates added; `linkedin.md` v2
  (prose, no bullet wall, no hashtag wall); forum draft opener softened.
- Write-up re-read as a judge → `notes/PLAN-2026-10-06.md` §1 (eight weaknesses: §0 order, §6.6 buried, propositions as
  prose, engineering weight, stale counts incl. "73" → 76 and §8 model families, figures, prose, release dry run) and
  the day plan to 10-17 (§3). Designation: k3z + k3w (one zeroed network ≈ 9e-3 on the mean). Nothing published.

## 2026-10-05 (evening, Mac, Fable 5.1) — the user approved the maximum-extraction plan; article, arXiv plan, forum draft written
- Written, nothing published: `writeup/article/article-ru.md` (the universal long-read, ≈24k characters, Habr-ready,
  slots for the final board; LLM disclosure = §8 verbatim), `writeup/article/linkedin.md` (two EN posts + one RU),
  `writeup/arxiv/ARXIV-PLAN.md` (conditions from arXiv's own pages: endorsement is the one gate;
  PDF-only accepted because ours is not from TeX; licence irrevocable → CC BY 4.0 recommended; the 2025-10-31 CS rule
  hits reviews/position papers only; Zenodo as the no-endorsement fallback), `writeup/forum-post-draft.md` (findings
  only, no code; option A before the close / option B with the write-up).
- Pending the user: go/no-go for the forum post before 10-17; the §8 list of their ideas; arXiv endorsement candidates.
  Next block: re-weight the write-up toward mechanisms (§0 order, §3 ↔ appendix), then the flopscope cost-model reports.

## 2026-10-05 (Mac, Fable 5.1) — unbiased reassessment of the prize chances; live facts re-verified; no experiments
- The user asked for an honest, unbiased reading of the real chances and of where work could still move us toward a
  prize. Read: CLAUDE.md, PLAN-10-04, STRATEGY-10-01, INTEL §5, R3–R5, korder, SPEC, FRONTIER, NEGATIVE boundary.
- Live facts [M]: board via curl (we are #33, 3.7e-9; top-3 = 1.6e-9; unchanged since 10-04); forum JSON (three new
  items: flopscope re-billing + re-evaluation before designation — mohanty 18118; MIT-code ruling — mohanty 18240;
  Strassen L7 passes the grader — ngmitam 18218); flopscope tracker issues #264–#278 read (under-billing edge cases;
  k3z uses none: no symmetry tags, exact-shape `out=`, reduced QR).
- Later the same day the user re-set the goal to "extract the maximum" (secondary prizes + a LinkedIn/Habr article):
  plan in STRATEGY-10-05 §8 (channels, timeline, consistency and timing rules, odds). Board: 1,140 participants, 266
  ranked teams, we are #33 = top 12.4%. No Phase 1 algorithmic-prize announcement found on the forum.
- Verdict written to `notes/STRATEGY-2026-10-05.md`: every measured/estimated lever stacks to ≈ ×0.75 at best (rank ≈ 20,
  no prize); P(top-3) < 1%; the prize EV is in the write-up (P ≈ 5–15% for the $20k) and the release (community
  awards); recommendation = write-up re-weighted toward mechanisms, release on the user's word, score housekeeping
  only (designate after the re-billing), stop score research. Awaiting the
  user's decision. INTEL §5.5 added (snapshot + news). No code, no runs, nothing sent or posted.

## 2026-10-04 (night) — R4 closures written; R5 = a duplicate of §58; external sources exhausted
- R4 (window / D from young sources / read rank): the old tier cannot be truncated, dropped from D or read in fewer
  than ≈350–450 dims (NEGATIVE §76, commit a2b1479). R5 (`experiments/2026-10-04_r5/`): CP-ALS of the old κ₃ at
  n = 128, rank n → 1.5–2% on t/D = NEGATIVE §58's finding; stopped. ARC reference repo unchanged since 08-04; no
  public cheaper method found. Leads left (estimates): age-dependent nested band ranks, breadth-first Strassen L5–L6.

## 2026-10-04 (Mac) — R1, R2 passed; the missing mechanism found (young κ₄ trees, ×0.70 raw at 1024); R closed on the meter
- **R1** (`experiments/2026-10-04_r1/RESULTS.md`): full factorial over the 8 channels the chain reads (V, C0, t, D, K4,
  K22, K31, S), ARC k = 4's statistics substituted, 8 nets at n = 256 and n = 128. The core everywhere is **K4 + K31**
  (×0.64 / ×0.60); the covariance and K22 follow by themselves. ARC's κ₄ split by birth block: the closure's residual
  is in the **young (≤ 2 layers)** blocks — young exact + closure for the old = the whole κ₄-world gain (n = 256:
  ×0.61 / 0.47 / 0.60 vs ×0.60 / 0.43 / 0.54). Split by tree type (exact source edits of ARC's step,
  `r1k4split.py`): the **3+2 tree (κ₃ handed on through an edge)** is indispensable, paths matter, stars less, the
  repeated-slice correction (where 10-03's T3 transfer lives) is secondary — why `t3k4` / `zlegs` never paid.
  Kill rule R1 passed. `chain.py`: oracle keys `K31_layers`, `k4add`, `K31add` (off = bitwise; gate 3.7533e-08).
- **R2** (`experiments/2026-10-04_r2/RESULTS.md`): `chain.py k4young` — the young trees built from the chain's OWN
  state, carried 2 layers, read into K4d and K31 beyond the closure's pattern. Tree builder = ARC's blocks to 1e-16;
  `young4.py` (legs, full width) = the FactoredTensor4 version to 1e-14. n = 256: **×0.664 / 0.668 / 0.622 / 0.941** —
  kill rule (×0.85 on both nets) passed; the first κ₄ carrier that improves consistently across networks.
- **R3 step 1** (`experiments/2026-10-04_r3/RESULTS.md`): 1024 × 16, 10 baked MLPs: K4d + K31 **×0.701 (10/10)**,
  K4d only ×0.824, age 1 ×0.815; with the production band ×0.72 / ×0.84. Cost estimate (to be metered): the readers
  cost about what they buy in our representation — best variant K4d-only with the path's coinciding-hub piece and the
  3+2 through a restructured D pairing, ×0.863 raw at ≈ +9% FLOPs → ≈ ×0.94 adjusted (3.7e-9 → 3.5e-9 displayed).
- **R3 step 2 — the port decided on the meter; R CLOSED by its kill rule.** The user chose (evening): write-up first,
  then a time-boxed port with the rule "metered FLOPs ≤ +10% and projection < 3.55e-9". Done: write-up §6.6 rewritten
  (the young fourth cumulant, measured; summary and §7 updated). The production form of the reader
  (`young4.Young4Chain`, only what k3z holds; = the exact reader at n = 128) at 1024 with the band, 10 MLPs: 3+2 age 1
  ×0.905 raw, ages 1–2 ×0.848 (10/10). Metered in k3z with exact-shape stubs (`experiments/2026-10-04_r3/meter/`):
  +7.40% / +14.79% FLOPs — the 3+2 needs a pairing at the last layer (+4.27% per age) that k3z never computes, and the
  last layer's 3+2 cannot be dropped (×0.886) nor stand alone (×1.02). Best projection 3.63e-9 (displayed 3.6e-9) > 3.55e-9
  → no port. Write-up §6.6 carries the metered price.
- Harness notes: zsh runs background jobs at nice +5 (`setopt NO_BG_NICE`); three parallel k = 4 runs at n = 256
  (≈20 GB each) slow everything ×10; four processes writing one JSON lose arms (one file per arm next time).

## 2026-10-03 (day, Mac) — more carriers closed; the last layer at full width; the ablation finished on the Mac
- Cloud #12's container stopped after 4 arms (no commits for 12 h; no credit left). Branch merged (B + 4 arms of A); the
  remaining 12 arms run on the Mac (`experiments/2026-10-03_evidence/mac_runs.sh`, `runs_mac/`, Mac-vs-Mac; chunks of
  ≤4 arms because background tasks are capped at 2 h). Mac is deterministic (base = the 10-02 k3y run bitwise); Mac vs
  cloud per-network raw differs by up to ±4% (float32 rounding), FLOPs by ≤3e-6.
- Write-up §5.2 (`writeup/make_s52.py --write`): cuts ×1.433, C0·D ×1.303, E2 ×1.137, paths ×1.099, band ×1.098,
  feed ×1.093, K31 level ×1.071, L5 ×1.052, colcut ×1.051, marg2 ×1.020, corrector ×1.013 (×1.010 out of its sample),
  csym+binv ×1.013; the last four (marg, k31, star, c0k22) running.
- Write-up §6.6: the next cumulant order (×3–7 at small width, a consistent set); summary and §7 updated.
- Research, all closed (NEGATIVE §74 + RESULTS): feed fed k=4's true (2,1,1) slice ×0.870 (×0.754 / ×1.059); the
  last layer alone — k=4's statistics ×0.572 at n=256 (both nets), the transfer at the last layer ×1.00; at 1024 with
  MC oracles the exact last-layer moments ×0.76 / ×0.82 but their feature-predictable half buys ×0.99–1.01 even in
  sample; the inherited transfer with exact propagators is collinear with the closure's var² (R² 0.71 → 0.74); no
  shared rank-1 κ₄ factor beyond the one-factor scale mixture.
- Intel: the leaders' FLOPs per network track ours (corr 0.85–0.97 → dead-neuron compaction, a chain with cuts); their
  advantage grows on our hardest networks (corr −0.34 … −0.56 with our raw).

## 2026-10-03 (night, Mac) — the κ₃ → κ₄ transfer found and carried; it does not pay alone (NEGATIVE §74)
- Cloud #12 (its final word, pasted by the user): no more tasks (credit nearly gone); A runs sequentially and
  auto-pushes RESULTS.md after every arm (4 of 17 done: corrector ×1.013 / ×1.010 out of sample, band ×1.098, cuts
  ×1.433); `resume.sh` if the container restarts. Gate note: cloud vs Mac adjusted −0.03/−0.82/−0.45% on MLPs 0–2
  (float32 x86/MKL vs arm64) — ratios are cloud-vs-cloud.
- The (2,1,1) class of κ₄post (the largest part of the κ₄ diagonal's pattern) is, to first order, the full κ₃ tensor
  handed to κ₄ at every ReLU (T3, Leonov–Shiryaev; corr 0.62–0.81) plus its gated inheritance (T3 + G 0.81–0.90).
  `chain.py` gained `t3k4` (one-step projection, "exact"/"mf") and `zlegs` (doubled legs riding W∘W, "nodiag" with
  the b = c exclusion); gate k4memo 3 = 3.7533e-08 with both off. Results: K4d corr with k=4 0.80 → ≈0.85, final MSE
  ×0.89 / ×1.00 (one step) and ×1.11 / ×1.06 (inherited) at n = 128 / 256. NEGATIVE §74.

## 2026-10-02 (late evening, Mac) — "what are we missing?": the next cumulant order, measured at small width
- **The user's direction (after k3z = 333534 = 3.7357e-9, #29):** finish the write-up, and in parallel find the method
  the top-3 use that we have not applied.
- **Intel:** no method posts on the forum since 09-22. The submission pages publish per-network final AND all-layer MSE:
  the leaders' errors are a near-constant ×0.57–0.80 of ours on both (marius ×0.57 / ×0.60; J2W ×0.74 / ×0.87 —
  late-layer heavy) → a better STEP at every layer, not only a cheaper old tier. mliston computes 3e10 FLOPs in setup.
- **`experiments/2026-10-03_korder/` (RESULTS.md):** ARC's reference ladder at small width against MC ground truth.
  Our chain = ARC k=3 AUGMENT level; **ARC k_max=4 (full κ₄ tensor, same Edgeworth budget as k=3 AUGMENT) is ×3
  better at n=128/L=8 and ×6–7 at n=256/L=8** — the next order is big and the leaders sit a small step into it.
  Substituting k=4's statistics into our chain: channels alone little (κ₄ diagonal the largest, ×0.74 upstream),
  jointly ×0.46 upstream; one of our own steps on k=4 inputs already gives back ×1.33. The chain's existing κ₄-birth
  readers (k4b, all tree parts, any depth) capture none of it. Gate: our `full` = ARC k=3 SIMPLE exactly.
- **Working hypothesis [U]:** the missing method is the fourth cumulant carried consistently (ARC's k=4 costs ≈L²n⁴;
  the leaders must have a cheap partial form). Next: decompose k=4's κ₄ diagonal into the index classes of the
  post-activation κ₄ (the diag and (3,1) classes are bivariate → O(n³) projections), then a cheap carrier.

## 2026-10-02 (afternoon → evening, Mac) — D/K22 closed (§73); decision: write-up + crumbs; k3z; write-up v4 drafted
- **The step's two pair inputs against MC oracles** (`experiments/2026-10-02_dslice/`, NEGATIVE **§73**): D = κ₃[i,i,j]
  is 7% wrong on the computed block (corr 0.997, no level error) and makes 73–78% of the born covariance error; exact D
  at layers 9 + 12 ×0.885 / ×0.931 (MLP 0 / 1), but any PART alone is network-dependent (computed block ×0.900 /
  ×1.027); ≈85% of its error is the k = 3 truncation. K22's closure is 20% wrong (corr 0.62–0.70); exact K22 at two
  layers ×0.970 on MLP 1 and **×1.102 on MLP 0**; pair features predict it better out of sample (R² 0.45–0.61 vs
  0.35–0.45), which cannot help when the exact value itself hurts. `pathD="full"` ×0.983 raw for +2.2–3.8% FLOPs (my
  first pricing divided by the budget instead of our FLOPs and said +0.7% — corrected). **The chain's errors compensate
  each other; only joint exactness pays.** `chain.py` gained an oracle key `K22_layers` (off = bitwise; gate 3.7533e-08).
- **DECISION (user): write-up as the main track, exact score crumbs in passing.**
- **k3z** (`src/estimator_k3z.py`) = k3y + Winograd's 15-addition Strassen form + level 5 inside the C_pre halves:
  FLOPs ×0.99397, max rel move 1.5e-6, wall-neutral; warm-run gate bitwise vs k3y with both off (a cold first predict
  bills the workspace-construction reshapes, ≈0.4% — compare warm runs only). 100-MLP run started 15:55.
- **Cloud #12** (same session as #11, branch `claude/laughing-hopper-61l0bn`): B done — on today's research base
  (B9 + marg2, torch, 10 MLPs) allages ×0.959, thin384 ×0.980, k4k3 0.5 ×0.970, **stack ×0.921** (9/10), raw 1.985e-8;
  A (production ablation of k3y on MLPs 0–31, 16 arms) running.
- **Write-up v4: first complete draft** (e182e1f; 20 pages after figures): §0–§9 + references; three figures
  (`writeup/make_v4_figures.py`: cost map, graded history, random-projection check + rank payoff); §7's cost-free
  ceiling filled from #12 B (raw ≈1.87e-8 projected: still ×1.25–1.7 behind the leaders' raw). One overstatement fixed
  on the way: the 0.89–1.09 identity check was made on network 0 only (one seed) — the text said two networks.
  Waiting on the user: §8 (LLM use), the author line, the channel, release intent. Waiting on #12 A: §5.2.
- **The parity rule, measured** (`experiments/2026-10-02_parity/`, ~3 min on the Mac — cloud #12's optional task C is
  no longer needed): the κ₃-diagonal error, an odd-order projection of Δκ₃ through the independent W, has **no coherent
  level** — z = mean/SE over 28 (layer, network) pairs: mean z² 0.93, max |z| 2.29 — while the variance error sits
  3–9 SE from zero at depth (its tr ΔC level) and the κ₄ closure −6 … −11. Per-neuron features see ≤ ≈40% of the κ₃
  error at depth, exactly through the two forms the identity allows (own t 21–30%, μ_pre 9–17%). Write-up §6.4 + §0.
- **k3z = 333534 graded 3.7357e-9** (projected ≈3.735e-9; the tenth port on its projection): raw 2.0317e-8, util
  18.33%, 0 failed, wall median 91.2 / max 98.8 s, residual median 0.310 / max 0.331 s. The first 100-MLP run had been
  killed by the 30-minute background-task limit (rerun with `timeout: 7200000`). The write-up now designates 333534
  (`writeup/switch_to_k3z.py`: header, §0, §1, §3 table + §3.8 item, §4 floor/telemetry/cost map from the namespaced
  k3z, §5.1 table, §6, §9, Figures 1–2).
- **The user's write-up decisions:** author Artyom Kuznetsov (AIcrowd Artyom_Mazur); e-mail + forum post after 17 Oct,
  but nothing sent or posted until the user says so; code released with the write-up in a separate public MIT repo;
  §8's clause on ideas waits for the user's list of their own ideas.
- **Board arithmetic for the crumbs:** the board stores 2 significant digits and keeps the earlier entry on a tie, so
  k3y (3.7573e-9) still shows k3x's 3.8e-9. k3z (≈3.735e-9) shows 3.7e-9; the next displayed step (3.6e-9) needs
  < 3.65e-9, ×0.977 more — beyond every crumb left (reshape ≈0.3%, deeper csym ≈0.5% with residual risk, mpair L5
  ×0.993 with wall risk). After k3z, crumbs cannot move the public rank.

## 2026-10-02 (day, Mac) — cloud #11 merged: the variance channel is closed with a mechanism
- **Cloud #11 finished and was merged** (31f30e1; branch `claude/laughing-hopper-61l0bn`). Re-gated after the merge:
  `ablate.py k4memo 3` = 3.7533e-08 (the new `postc`/`prec` hooks are bitwise when off). NEGATIVE **§72**:
  the next layer's variance error is 95% the random projection w_iᵀΔC₀w_i of the off-diagonal post-covariance error
  through independent weights (a theorem, verified 0.89–1.09); exact off-diagonal is worth ≈×0.55–0.68 at every layer
  (upper bound) but no cheap structure carries it — low rank (40–65% of the energy) ×0.93–1.07, sparsity needs 20% of
  the pairs, O(n²) linear models R² ≤ 0.06 out of sample against a ceiling of 0.87–0.98, 65–76% carried through
  earlier W. The four 1e7 covariance oracles (layers 5/8/11/14, float32) are in `experiments/2026-10-02_cov-error/`.
- **Where that leaves the score:** every accuracy channel has now been taken apart with a mechanism (tails §68, pair
  expansion #10, per-neuron correctors §70/§71, variance §72); the κ₄ corrector is the one learned lever and ships;
  the cost axis is closed on the meter and the wall budget is spent (k3y: 97.6 s max). What is left for the score is
  ≈0.6% of exact crumbs (recurring reshape billing, Winograd's 15-addition form).

## 2026-10-02 (night, Mac, Opus 5.5, new chat) — k3w graded on its projection; k3x packaged by the rule; the joint corrector closed
- **333388 (k3w) graded 3.8576e-9** (projected ≈3.86e-9 — the seventh port in a row on its projection): raw 2.0313e-8
  (= k3v's ×1.00004, exact as designed), util 18.93%, 0 failed, residual max 0.288 s, **wall median 84.2 s, max 92.2 s**
  (MLP 0; every other MLP ≤ 88.0 s). Board: rank 31 (displayed 3.9e-9).
- **The k3x rule fired** (wall max 92.2 ≤ 95 s): `src/estimator_k3x.py` default site list now includes `dpair`.
  Checklist: static import scan clean · `whest validate` passes · gate (`experiments/2026-10-02_k3x/gate.py`): with
  k3w's sites bitwise k3w with equal FLOPs at 1024×16 and 128×8; defaults FLOPs ×0.98547, util 18.86%, max rel move
  1.15e-6 · packaged `submission-20261001-193115.tar.gz` (sha bb7e0603…), `estimator.py` byte-identical (sha
  03887a16…), `validate-package` ok. 100-MLP local run (wrapper with the same kwargs) in progress before submitting.
  Grader wall per L5 site is worse than the local ratio suggested: k3w added +12.7 s median on the grader for +3.6 s
  locally (×3.5, not ×2.8), so k3x is expected at ≈99 s median / ≈107 s max of 120.
- **The joint corrector, tested and closed (NEGATIVE §71, corrector RESULTS Part 4).** In sample the variance
  projection, harmful alone, helps together with κ₄ (var+κ₄ ×0.885 / ×0.900 vs κ₄ ×0.916 / ×0.918; gate exact joint
  ×0.2060 / ×0.3024). Out of sample (k4loo protocol, 10 MLPs): variance held-out R² **0.004**, var alone ×1.019 (0/10),
  var+κ₄ ×0.980 (9/10) vs κ₄ alone ×0.974 (10/10). Also: κ₄ pooled over layers 2–15 is worse than 8–15 (R² 0.248 vs
  0.361; ×0.982 vs ×0.974). The shipped corrector stays as it is.
- **Leaders' telemetry fingerprint** (`notes/INTEL-peers-2026-10-01.md` §5): everyone spends 77–95 s of wall for
  0.58–0.79 of our FLOPs; J2W and marius run at the residual cap (0.37 s medians); mliston's setup takes 4.1 s (the
  frozen GRU of forum 18170), marius 0.75 s and J2W 0.38 s also load data.
- **333398 (k3x) graded 3.8069e-9** (projected ≈3.80e-9), wall max 95.2 s / median 90.7 s, residual max 0.281 s,
  0 failed — the eighth port on its projection; dpair cost the grader +6.4 s median (×1.7 of local, not ×3.5).
- **Two more exact FLOP savings → k3y** (`experiments/2026-10-02_csym/`): **csym** (C_pre = Wᵀ(CW) is symmetric:
  compute the upper block row and O22, copy O21 = O12ᵀ; −0.67%, no wall) and **binv** (flopscope bills `solve` as LU
  + getrs although R is triangular: one explicit R⁻¹ per band entry applied by Strassen; −0.59%, no wall; numerically
  clean on all 100 networks). 100 MLPs ×0.98749 of k3x, 100/100, 0 fails. **Submitted as 333402.** Located and
  parked: recurring reshape billing ≈0.33%, Winograd's 15-addition form ≈0.25%. Closed: an R-only QR (cond(P) grows
  to 7e4; squaring it is fatal in float32).
- **333402 (k3y) graded 3.7573e-9** (projected ≈3.76e-9; the ninth port on its projection) — the new best. Wall max
  97.6 s, residual max 0.310 s. **mpair** (L5 on the members' compressed pairing) closed on wall: ×0.99278 for +7.65 s
  local (0.094 %/s). The wall budget is spent.
- **Cloud #11 is running** on `claude/laughing-hopper-61l0bn` (its own name, not `cloud/cov-error`): gates reproduce
  (k4memo 3, k3l_t 10, B9 2.19667e-08), `chain.py` hooks `postc`/`prec` (off = bitwise), covariance oracle baking.

## 2026-10-01 (evening → night, Mac, Opus 5.5) — mistakes analysed; selective L5; the first learned lever that works
- **Started with an honest review of the day's seven errors** (`notes/METHOD-LESSONS-2026-10-01.md`): one pattern —
  claims stronger than the evidence (model extrapolations, upper bounds presented as estimates, a secondary-source
  paraphrase that nearly redirected two weeks of work). Three rules came out of it, then a fourth after two shipping
  defects (below).
- **The κ₄ pattern axis:** in sample, the projection of the κ₄ error onto own+**pair** features is worth ×0.916 /
  ×0.918 (own features alone: nothing). Cross-MLP on two networks it failed (×1.00–1.08); with oracles for MLPs 2–9
  baked, a 10-fold leave-one-out gave ×0.974 offline (10/10) and ×0.963 online (4/4). The two-MLP failure was a
  sample-size failure.
- **Selective Strassen L5, priced by measurement:** profiling full L5 per namespace gave each contraction's FLOPs saved
  per second of backend added; **k3t** = L5 on transport + C/path transport + C_pre: 10 MLPs ×0.9738, 100 MLPs
  ×0.9741, **graded 333357 = 3.9554e-9**. Its contended screen had broken the residual cap on 4 and 7 of 10 MLPs —
  residual is a property of the machine's load.
- **k3v = k3t + the learned κ₄-diagonal corrector:** features recorded on the PRODUCTION trajectory (the runner's own
  weights and seeds), ridge λ=1e-3, s=0.5, layers 8–14 (the last layer has no C0/D/K22 in production). Leave-one-out
  through the real estimator ×0.980 (8/10); 100-MLP run: 87 fresh MLPs ×0.9878 (69/87) after 7 load-induced residual
  breaches (Apple XProtect at 96% of a core, `sysmond`, another Claude Code project). **First submission (333367)
  REJECTED** — "Prohibited file in submission": an `import numpy` in the never-executed recording branch. Fixed,
  verified bitwise against the rejected file, **resubmitted 333368 = 3.9102e-9: raw ×0.9880 on the grader vs ×0.9873
  out of sample locally.**
- **Shipping defects found and fixed:** k3t fails `whest validate` (a leaf scratch built for a Strassen depth the small
  contract-check width does not have — the grader's smoke test is full-size, so 333357 graded); k3v's numpy import.
  `scripts/prepkg_check.sh` (static scan for prohibited imports) + `whest validate` are now the checklist.
- **The corrector's limits:** for the variance and κ₃ the same trick HURTS even in sample (×1.03–1.10); 30 training
  networks give the same held-out R² as 10 (0.353 vs 0.354); a small MLP reaches R² 0.42 vs ridge 0.36 (+≈0.3% est.).
- **More exact L5 sites:** band members (k3w, −1.32% FLOPs) and the D pairing (k3x, −1.45%), both bitwise-gated.
  Grader wall for k3v: median 71.5 s, max 81.2 s → k3x estimated ≈107 s of 120 — the closest to the cap.
- **Cloud #10 merged** (no free accuracy left in the pair expansion; marg2 the best of three substitutes); **#9B
  merged** (the catalogue is a local optimum). **Brief #11 written:** the structure of the post-covariance error ΔC,
  the one object nobody has measured, aimed at the variance reserve (×0.57/×0.77).

## 2026-10-01 (morning, Mac) — grades + residual calibration; k3q submitted; peers per MLP; marg2 ported; L5 is a no-op without a lower recursion floor
- **333028 (k3o) = 5.1127e-9** (raw 2.1494e-8, util 23.78%) and **333029 (k3p) = 4.5644e-9** (raw 2.2067e-8, 20.65%),
  both *exactly* on the local projection (5.12 / 4.57). Board rank 69 → **40**. Public per-MLP ledgers and telemetry
  read from the submission pages with the React-props snippet (no key needed).
- **The residual calibration we were waiting for:** grader residual max **0.1783 s** (k3o) and **0.2330 s** (k3p),
  mean 0.174 / 0.225, **0 of 100 exhausted** both times, against the Mac local-runner max 0.248 / 0.311 s ⇒ the grader
  runs at **0.72–0.75 × the Mac local runner** (the old estimate was 0.79). k3q's local max 0.336 s therefore projects
  to ≈0.25 s of the 0.4 s cap, the hold condition (≲0.26 s) was met, and **k3q was submitted as 333276** after
  re-verifying the tarball (estimator.py byte-identical to `src/estimator_k3q.py`, sha 8ebea33e…, `validate-package` ok).
- **Cloud #9A merged** (6358e06, `cloud/tail-accuracy`): the tails explain **none** of the ×1.8 gap; the error is born
  in the kink **core**. Its one lever is **marg2**. Cloud #8's last two commits merged too (64a1418).
- **Peer analysis, 5 leaders × 50 public MLPs** (`notes/INTEL-peers-2026-10-01.md`): the metric is exactly
  `adjusted = raw × FLOPs/2.2e12`; their per-MLP error correlates **+0.69…+0.76** with ours while the leaders correlate
  **+0.73…+0.83** with each other, and their advantage is a **near-constant factor** (ratio CV 8–9%). ⇒ no leader has a
  mechanism for a special class of networks; the gain is global accuracy. Our raw is flat at ≈2.15e-8 and top-5 needs
  it **roughly halved** at any reachable util (table in the note).
- **marg2 ported** (`src/estimator_k3r.py`, switch `marg2=θ`, default off): k3r with defaults is **bitwise k3q with
  equal FLOPs** at 1024×16 and 128×8. First attempt was NOT bitwise (max|d| 7e-6) because I had regrouped the
  coefficient sum `w + (c5·t/6 + c6·K4d/24)` instead of the original left-to-right accumulation — float32 rounding,
  amplified over 16 layers. Fixed by keeping the original accumulation and selecting with an exact `c·keep + w·drop`
  (×1.0 and ×0.0 are exact on both sides). **10-MLP screen on top of k3q: raw ×0.9805, 9/10 better, FLOPs ×1.00000**
  — the cloud's torch value was ×0.9811, so the port reproduces it to 0.06%.
- **Strassen L5 is a no-op as shipped:** `_Strassen.__init__` breaks out of the recursion at `m < 128`, so
  `strassen_levels=5` stops at the same 64×64 leaf. L4 and L5 wrappers measured **bit-identical on 3 MLPs** (same
  FLOPs 4.4074e11, same raw, same 19375 calls). k3r therefore adds `strassen_min` (default 128 = k3q); the honest L5
  test is `strassen_levels=5, strassen_min=64` (leaf 32×32, 7⁵ groups) and is a FLOPs ×7/8 trade against residual,
  wall and the 8 GB RSS cap. Also measured: the Mac's residual reading of the *same* config varies 0.309–0.377 s
  between runs, i.e. ±35 ms of noise — another reason the grader telemetry is the only real measurement.
- **Step 4 (re-reading the leaders' Phase-1 write-ups) answered a different question than the one it was set.** The
  tail question is closed by our own #9A, so the question became "who has an accuracy mechanism we do not". The
  answer is **mliston (#5 on the board today), topic 18170**: a frozen **GRU (~52k params) trained offline on a few
  thousand random MLPs from the challenge distribution to predict the closure's *centering error*** from the
  closure's own propagating moment state (mean, variance, 3rd/4th cumulants, gate probabilities) — "an on/off probe
  on 20 MLPs cut raw MSE by **17–19%**". This is NOT 504aldo's dead end #10 (that was *output*-level features,
  corr < 0.09) nor our §57 residual analysis (≤ 0.03): it corrects the moments *mid-chain*, which is exactly where
  #9A says all of our error lives. Recorded as the one live accuracy direction from the public record, with
  attribution (18170, mliston). Also from 18170: their 8.62e-8 Phase-1 best was declared invalid by its author (a
  flopscope `tensordot` accounting hole) — not a lever, a disclosure. pranay212's dead/on/kink chassis (18173) is a
  sampling design and is moot now that the tails are known to be inert.
- **Cloud #10 brief written, then REWRITTEN after the user's decision.** First version: the analytic kink-core
  variance hunt (archived, `notes/CLOUD-BRIEF-10-01-analytic-archive.md`). Then I put the fork to the user with the
  numbers — exact engineering tops out at ≈3.6–3.9e-9 and top-5 needs raw halved — and **the user chose the learned
  moment corrector, trained on self-generated MLPs**. `notes/CLOUD-BRIEF.md` is now that brief, built so its tasks 1–2
  can kill the track in a day without any baking.
- **k3q graded: 333276 = 4.1279e-9** (raw 2.0914e-8, util 19.74%, 0 failed), residual max **0.2471 s** against the
  0.25 s I had projected from the k3o/k3p calibration, wall max 63.7 s of 120. All three ports landed on their
  projections (5.12/4.57/4.14 → 5.1127/4.5644/4.1279). Rank ≈32.
- **L5 measured and CLOSED (§69).** `strassen_levels=5` had been a **no-op all along** (`_Strassen.__init__` breaks the
  recursion at `m < 128`): L4 and L5 wrappers are bit-identical on 3 MLPs. With `strassen_min=64` (leaf 32×32, 7⁵
  groups): util ×0.9389, adjusted **×0.9380**, residual unchanged — but **wall ×2.2** (46.9 s local → ≈140 s graded
  against a 120 s cap) and **RSS 7.21 GB of 8**. A 32×32 leaf has no BLAS efficiency left. It would have been the
  largest remaining lever and it cannot ship. §56's claim is now a measurement with a second, independent blocker.
- **Corrector task 1 run here, not in the cloud** (`experiments/2026-10-01_corrector/`), because both 1e7 oracles of
  MLP 0 were already on this machine — now committed (1.3 MB) so the cloud starts at task 2. On base B9:
  - the moment error is **real, not oracle noise**: corr(eA,eB) between the two independent oracles rises with depth
    to **0.94 (var), 0.99 (κ₃), 0.99 (κ₄)** — from layer 8 on, 94–99% of it reproduces;
  - a **linear** model on 21 features the chain already computes explains, at L15, **R² = 0.512 (κ₃)** and
    **0.561 (κ₄)** in sample (p/n bias ≈2%; the two oracles agree to 0.015), i.e. ~52–57% of the *explainable* error.
    The variance channel is much harder (R² ≈ 0.10 of a 0.94 ceiling).
  - ⚠️ **the leverage is inverted**: κ₄ is easiest to predict but has the smallest oracle ceiling (×0.78); variance has
    the biggest (×0.64), a rel error of only 0.1–0.2%, and is the least predictable. **R² is not the deciding number**
    — the payoff through the chain, out of sample, is. That is task 2.
  - Two bugs caught on the way, both worth remembering: the file was first called `signal.py` and shadowed the stdlib
    `signal` module (torch imports it through `subprocess`); and the first ridge returned R² = 0.000 in all 96 cells
    because six *layer-level* features are constant **within** a layer, which made the design matrix singular.
- **`chain.py` gained `mcorr`** (off by default, the old path bitwise): an ONLINE corrector callable of the chain's own
  live state, so the correction stays self-consistent after the trajectory moves — a recorded-trajectory correction
  would be exactly the kind of thing NEGATIVE §57 forbids. `experiments/2026-10-01_corrector/xmlp.py` is the
  fit-on-A-apply-to-B harness; MLP 1's two oracles are baking.
- **E3 deliberately reordered, and said so.** Priced at util ×0.953–0.96 with raw ×1.002 → ≈3.90e-9 from k3r's ≈4.05e-9
  (rank 32 → 31), against 2–4 h of surgery in the most intricate part of `estimator_k3*.py` (the transposed leg layout,
  the band, the matched cut's existing row/column maps). The cross-MLP corrector measurement decides the strategy for
  the remaining 16 days and costs an hour once the oracles land, so it goes first; E3 immediately after.

## 2026-09-30 (≈01:00, Mac) — k3o and k3p submitted; the strategy question; #9 as a two-session brute force
- The user allowed me to submit through the terminal: **k3o → 333028, k3p → 333029** (`whest submit ... --yes`),
  tarball shas re-checked before upload (cdcea273…, e3e0c1a1…). k3q's 100-MLP run in progress (`k3q_n100.json`).
- The user asked whether accuracy can be brute-forced to reach top-5 faster. Answer (in the chat): keep shipping the
  exact stack (sure ×0.70); turn the cloud into a parallel brute force aimed where the evidence is — every chain term ×
  tail × threshold (Prompt A, session #8) and the whole switch catalog re-screened on the post-E2 base + a threshold
  grid (Prompt B, a new session), survivors confirmed on 32 fresh MLPs. Honest odds unchanged: the stack + a lucky
  ×0.9 from the search ≈3.2–3.4e-9 (rank ~15–20); top-5 needs a ×1.5+ accuracy mechanism nobody has found in our
  representation.

## 2026-09-30 (night, Mac) — k3p (band) and k3q (E2) ported; cloud #8's verdict merged; #9 = the tail-accuracy hunt
- **k3p = k3o + the transported band** (METHOD §26, `experiments/2026-09-30_k3p/`): Pᵀ rides the leg transport as r
  extra rows; members materialised by one product; compressed D pairing; pre entry through a QR of P; no
  re-orthonormalisation. band=None bitwise k3o; vs the cloud's torch `bandz` 2–7e-7 per layer. 10-MLP screen of four
  bands: (5,448,rr) best, ×0.8905 of k3o (10/10). **100 MLPs 4.6839e-9 @20.693% = ×0.8928 of k3o, ×0.8484 of k3m,
  100/100; peak RSS 4.5 GB. Packaged `submission-20260929-190455.tar.gz`** (byte-identical; the subprocess runner zeroed
  1 of 3 MLPs at 0.432 s — it reads ≈2× the grader; expected grader ≈0.21–0.24 s). Sent to the user.
- **Residual time:** ≈20–40 µs per flopscope call; Strassen 51% of calls, `_nonlin` 23% (~266 calls/layer). Two plain
  products instead of Strassen saved 16 ms for +0.45% of B — reverted. Paired Strassen combinations don't help:
  flopscope logs reshape/moveaxis as calls. Transposed views in matmul are not the cost (measured). Calls are.
- **k3q = k3p + E2** (METHOD §27, `experiments/2026-09-30_k3q/`): rectangular D pairing without saturated rows +
  the κ₄ level restored through the next Ssum by matrix-vector products. satrow=None bitwise k3p; vs the cloud's torch
  `sep` 2–7e-7 (the level part alone moves the output 5e-4…1e-3, so both parts are verified). 10-MLP screen on top of
  k3p: θ=2.5 ×0.8990 (10/10, raw ×0.943), 3.0 ×0.954, 2.0 ×1.097 — the cloud's curve. Default 2.5; 100-MLP run
  launched ≈00:40.
- **Cloud #8 merged (0c080b7):** stack = 3.79e-9 projected (3.47e-9 with L5), ×0.703 of k3m; top-5 gap ×1.8 is accuracy;
  E4 dead → NEGATIVE §67; the tails are where the marginal Edgeworth series breaks down. **CLOUD-BRIEF #9 written:** the
  tail-regime accuracy hunt (regime attribution with MC oracles, E2-style restrictions, tail-safe readouts).

## 2026-09-29 (late night, Mac) — k3o: the matched dead cut ported; cloud #8 running
The user pasted the #8 prompt into a new cloud session and said "continue". Cloud #8 is live (`cloud/port-battery`,
first commit fbc1bd5: chain.py switches `bandz` / `satrow_k4` / `recv` / `agecut`, anchors exact, task-1 grid running).
- **k3o = k3m + the matched dead-side column cut, ported** (`src/estimator_k3o.py`, `experiments/2026-09-30_k3o/`, METHOD
  §25). The legs are stored **transposed as one wide stack of hub rows** (a block of w_s rows per source in a Bᵀ and an Aᵀ
  region), so the transport is one right-shared Strassen product over row chunks and the D pairing one contraction over
  all hub rows — per-source widths cost nothing extra. New `_Strassen.rshared` (mirror of `shared`); Strassen padding
  rows must be exactly zero (quadrant mixing would leak garbage).
- **Self-checks:** colcut=None == k3m to ≤1.3e-6 on 1024×16 (FLOPs ×1.0020); the production cut == torch `colcut` to
  2–6e-7 per layer at −2.5 / −1.5 / −1.0 while the cut's own effect is 4e-5…1.3e-3.
- **10-MLP screen on the production path:** matched −2.5 = raw ×0.9937 (torch ×0.9928), util 23.79% (×0.956; the cost
  model had said ×0.940), **adjusted ×0.950 on 10/10**; matched −2.25 = raw ×1.030, util 22.70%, adjusted ×0.939 (10/10).
  The two differ by 1.1% — below the grader's resolution; **−2.5 is the primary** (better raw AND cost), −2.25 a
  candidate for the second designation.
- **Residual time is unchanged** (0.245/0.264 vs k3m re-run today 0.243/0.279 s; the 09-26 k3m numbers were a quieter
  machine). The ~1 200 extra calls cost flopscope overhead, which is not residual.
- **100 MLPs: 5.2464e-9 @23.792%, ×0.9503 vs k3m on 100/100**, raw ×0.9943 (79/100), 0 failures, residual
  0.228/0.248 s. **Packaged: `submission-20260929-173629.tar.gz`** (tarball file byte-identical to src, sha256 cdcea273…;
  subprocess 3 MLPs bit-identical per MLP, residual 0.32–0.33 s). **Expected on the grader ≈5.12e-9. The user submits.**

## 2026-09-29 (night, Mac) — cloud check, live board, the cloud's next task rewritten around the top-5 gap
The user: continue; check the cloud session (it seemed to have run Prompt A) and give it a new task; **we must go for
top-5**.
- **Cloud:** the old session had pushed one more commit to `cloud/mst` after our merge (3573cc7, 14:51 UTC): f re-measured
  on the full 10 MLPs the brief required — **27.5–28.2% at layers ≥8, pooled 27.76%** (3-MLP screen: 27.2–28.0%), verdict
  unchanged. Merged (a2a954a); NEGATIVE §66 and CLOUD-SESSIONS #7 updated. The session has nothing left to do — close it.
  It is not reachable from the desktop session tools (only local sessions are listed), so the user pastes the prompt.
- **Live board (15:40 UTC, 225 entries):** #1 J2W 1.70e-9 (1.62e-8, 10.5%) · #2 Luna 1.90e-9 (1.42e-8) · #3 suliman
  2.00e-9 (1.58e-8) = marius_binner 2.00e-9 (**raw 1.26e-8**, 15.9%, submitted 2 h earlier) · **#5 mliston 2.10e-9**
  (1.70e-8, 12.4%) · #16 3.0e-9 · #29 4.0e-9. **We are "Artyom_Mazur"** (profile `artyom_mazur`, confirmed on the 332403
  page — the 09-26 entry's "handle artiem_kuznietsov" does not match what the board shows) at 5.40e-9, **rank 69**. The crowd (~15 teams at
  2.0–3.0e-9) sits at raw 1.3–2.0e-8 and util 10–16%: ×1.1–1.7 more accurate AND ×1.6–2.4 cheaper than us.
- **The arithmetic, stated plainly:** top-5 needs ×2.57 today, ≈×2.8–2.9 by the close. Every live exact lever stacked
  (matched cut ×0.918, band ≈×0.88, E2 ≈×0.955, E3 ≈×0.96–0.98, L5 ×0.9165 once util < 17%) is ≈×0.69 → ≈3.7e-9
  (today's rank ≈24). Top-5 needs a further ≈×1.8 that no known lever supplies — most plausibly the leaders' accuracy
  (raw 1.26–1.62e-8), whose mechanism is unpublished.
- **CLOUD-BRIEF rewritten** for session #8: the port battery + the stack frontier (band float32/lazy QR, honest pricing,
  E2/E3/E4, greedy stack re-measured at each step, matched cut at −2.25 and −2.5, L5 admissibility, a plain verdict on
  the gap to 2.1e-9). Tried to read a field-techniques site named in a forum post (INTEL §4): blocked by the tool's
  safety classifier (URL from untrusted content) — left for the user.
- Upstream check: no new commits in ARC's `mlp_cumulant_propagation`; starter kit and flopscope only docs (09-16).
- Considered and priced out on paper (not sent to the cloud): **depth-targeted accuracy** — the ×1.25 accuracy stack
  switched on only in the last K layers, since only the final layer is scored. It fails on arithmetic: the stack's cost
  (§61: allages + pathD ≈ +12.8 points, k4k3 ≈ 4 n³ per source-layer) scales with the number of live sources, which is
  largest exactly in the last layers (12–15 sources), so K = 4 costs ≈40% of it for ≈60–70% of the benefit, and even
  K = 1 is ≈×1.065 util for ≈×1.025 raw.

## 2026-09-29 (late evening, Mac) — documentation pass: everything from this chat is in the repo; the cloud's MST merged
The user asked to save everything done in this chat into the project context, every method tried, and the cloud's work.
- 🔒 **Found and merged a cloud branch no brief was waiting for: `cloud/mst`** (commits 11:41/11:52 UTC). The old
  session had run Prompt A (the Slice-Markov tail) before the Mac withdrew it — and closed it exactly as the stride
  test predicted: the old tail's own all-distinct share is **27.2–28.0%, flat in a0** (the brief's hypothesis refuted);
  a0 = 10 ×1.20, a0 = 12 ×1.016; **dominated by k3m at every a0**. Its a0 = 1 anchor reproduced §62's `td` mode to the
  last digit after catching a Δ-diagonal double count. → **NEGATIVE §66.** Lesson recorded: check `git ls-remote` before
  writing a brief — a withdrawn prompt may already be running.
- **NEGATIVE-RESULTS:** new section **G** (cost-boundary phase, §55–§66) with an intro; **§63** (jumped tier), **§64**
  (post-hoc corrector), **§65** (terms closed on accuracy at the floor, incl. `corr` exact < split384 and the
  `corr="full"` artefact of the ceiling screen), **§66** (MST); §45 marked as corrected; §62 points to §66; a current
  statement of the boundary heads "The boundary, stated once".
- **`notes/CLOUD-SESSIONS.md` (new):** all seven cloud sessions (#1 verification … #7 MST) + the planned #8 port battery,
  with task, verdict, branch, merge status, NEGATIVE entry, and the lessons (anchors caught two real bugs; do not let the
  cloud edit `src/`; withdrawn prompts may be running). Linked from CLOUD.md and CLAUDE.md. CLOUD-BRIEF updated with MST.
- **METHOD §19–§24:** dead-row cut (k3j/k3m), K31 × 0.65 (k3l), k3n and why k3m stays designated, the dead-side column
  cut and its consistency mechanism, the transported band, where the method stands; results table through k3n.
- **SPEC:** when the symmetric discount applies (aliasing only), einsum wall ×70, Strassen with no level limit
  (sponsor 09-11), and a **grader-vs-local ratio table** (adjusted 0.976, raw fs/torch 0.920, residual 0.753, wall
  3.1–3.9×, resolution 0.4–0.5%, fitted scalars at ⅓ strength).
- **`chain.py` switch table** (chain RESULTS §0): every switch added since k3i — paths/pathD/pathrank/pairx, deadcut,
  level multipliers, ssp/band, varfix, tailx, colcut, satrow, stride — plus the torch configs of k3l/k3m/k3n.
- **FRONTIER** 09-29 block; **CLAUDE.md** CURRENT STATE rewritten (user priority: score > paper; step → report →
  approval; cloud always in view), the 09-28 block archived below it, state tables (k3n graded, k3m designated), repo
  map with every new folder, conventions (private repo, attribution, the pgrep self-match trap, screen sizes).
- Earlier in this pass (commit 89762b2): ideation artifacts saved; RESULTS for coherent-knobs, residual (+analyze.py),
  satrow; the k3m-port cloud folder brought in with NOTE-merge.md; NEGATIVE §55 with the cloud's real-meter numbers.

## 2026-09-29 (evening, Mac) — ideation workflow done; the band basis survives E1; the jump idea closes
- **Ideation workflow completed** (7 lenses → adversarial vetting → synthesis → completeness critic; 16 agents, 4.5M
  tokens). Synthesis, sober: **P(#1) < 0.5%, P(top-5) ≈ 1–2%**, realistic finish ≈4.2–4.7e-9 if the live cost
  levers are ported. It corrected me twice: Prompt B's band arms were wrong (no spectral gap near 320), and the
  saturated-row cut needs its κ₄ level restored. The critic found the one idea class every lens missed —
  **jumping** the old tier's legs k layers through one shared operator, with a slice rule between checkpoints.
- ❌ **The jump is closed** (`experiments/2026-09-29_stride/`, `chain.py stride`, self-check 2.2e-15). One
  un-reset layer of the old tier's slice rule costs ×1.5–×3.5 for every a0 < 10: (4,2,mst) ×3.51, (6,2,Donly)
  ×1.49, (8,2) ×1.18–1.22; only (10,2) is free (×0.995), and that tier is ~17% of source-layers. The all-distinct
  content is load-bearing **per source, per step**, for every source younger than ~10 layers. This also predicts
  the cloud's MST (a0 2–5, no reset) fails, so that prompt was withdrawn.
- ✅ **E1: the transported band basis survives on accuracy** (`experiments/2026-09-29_band/`): +0.69% raw at
  (6,448), +2.46% at (6,384), +10.9% at (6,320). **Its cost hinges on one term** — the per-layer coefficient
  rotation k3k needed because it re-fitted its basis: with it the band is cost-neutral, without it (a lazy QR;
  Wᵀ(PZ) = (WᵀP)Z so Z never changes) util drops to **20.45% → ×0.87**. Float32 conditioning of an un-orthonormalised
  P is the open question; tasked to the cloud.
- ❌ Saturated-row cut, naive: ×1.0094 at 3.0 (6/10, spread ×0.84–1.35), ×1.34 at 2.5 — needs the κ₄ level restored.
- 📋 **Cloud re-tasked**: the old session is to be closed; one new session runs the **port battery** (band in float32
  with lazy QR, honest pricing, E2/E3/age thresholds, the combined stack and whether it crosses util 17% where
  Strassen L5 re-opens). Priority is the score, per the user.

## 2026-09-29 (Mac + cloud + forum) — the κ₃ state is not closable (a principle); a working dead-column lever; the field replicates our walls
- 🔒 **Cloud: (t, D) is not a sufficient κ₃ state — the cost axis closes on a PRINCIPLE** (`experiments/2026-09-28_state-closure/`,
  merged, **NEGATIVE §62**). Exact index-coincidence decomposition of the next diagonal, self-checked by brute force to
  1e-16: the **all-distinct** term is **28.8%** per layer; a (t, D)-state chain is **×25.6** end-to-end; a closer must
  capture **≥93%**; the best oracle closer on any O(n²) basis captures **19.5%**, and re-fitting the t/D coefficients
  0.16% because the missing part is **orthogonal** to both (D-coefficient fits to 1.000). Why κ₄ closes and κ₃ cannot:
  κ₄ enters the readout at second order, κ₃ *is* the readout. The cloud caught its own first-run error (wrong layer's W,
  a bogus 138%) with the transport self-check and now `raise`s if it is worse than 1e-10.
- ✅ **A working cost lever: the dead-side birth-COLUMN cut** (`experiments/2026-09-29_colcut/`, `chain.py colcut`,
  NEGATIVE §48 revised). The transpose of the dead-row cut. Symmetric cut is bad (|α|>3: ×1.12 while removing 0.03% of
  the d2-weighted mass — a 400× proxy failure); **split by side, all of it is saturated parents** (κ₃ passes *through* a
  near-identity neuron, so its column carries transported content). **Dead side at −2.5: raw ×0.9928 on 10/10.**
  **Mechanism = consistency** (a row-gated neuron should not keep birthing κ₃ on the column axis), and it **predicted**
  that matched row+column cuts beat row-only at every threshold, growing with the gated fraction — **confirmed at all
  five** (−3.0…−2.0: 0.9985, 0.9970, 0.9928, 0.9790, 0.9475). The interaction moves the optimum from −2.5 to **−2.25,
  projected ×0.918 (≈4.95e-9)**, util modelled. Port scoped: needs per-source widths → legs as one wide (n × Σw)
  matrix, where transport and pairing are single products; a two-class shortcut captures only 29–54%.
- 🔍 **Forum intelligence** (`notes/INTEL-forum-2026-09-29.md`, ~40 public posts read):
  - **504aldo independently mapped the same walls** (topic 18218): memoryless (2,1)/(3,) closure 45× worse, "the lost
    part is 18–23% of D21 energy, high-rank, not predictable (+0.02 R²)" ≡ our §62; CP cap needs 8n cols for 3–14%
    where 2% is needed ≡ §58; learned corrections see corr < 0.09 ≡ ours ≤ 0.03; and their row 21 states our mechanism
    in their own words. **Two teams, different code, same walls.**
  - **The sharpest constraint on #1:** 504aldo's bill *without* the old tier is 0.150·B, and the leaders sat at
    0.149/0.152/0.164·B with better raw — **whatever carries the leaders' old-source content costs ~nothing**, and it is
    not a compression of the source representation. J2W is now at 10.5% with raw 1.62e-8, **more accurate than our
    chain at unlimited cost (≈1.72e-8)** — so the #1 method is not a cheaper version of ours.
  - **Sampling is dead in Phase 2** (three teams: MC rate at width 1024, best ~3e-6; control-variate hybrids need
    33–300× reduction, realise 1.5×). It won Phase 1 (256×32), where closures hit a ~7e-7 wall.
  - **Attribution owed:** our §57 coherence mechanism was published first by jamesrahenry in Phase 1 (18097; census
    18157 item 1c, "coherent corrections amplify ~16:1"). Credited in the write-up.
  - The leaders do not publish ("score prizes reward silence": 864/1405 graded submissions undocumented).
- ✍️ Write-up §6 now carries the principle (§62) and the independent replication, with the prior-work credit.
- 🧭 Ideation workflow (7 lenses × adversarial vetting × synthesis × completeness critic) relaunched after the
  previous run died on a usage limit; running.

## 2026-09-28 (night, Mac + cloud) — the cost axis closes for the fifth time; the write-up becomes the live prize track
- ⚠️ **k3n graded 5.3977e-9 — 0.13% WORSE than k3m's 5.3906e-9** (332872; 100/100, 0 failures, residual
  0.155/0.164 s, wall 59.2/63.7 s, util 24.834%), although it was better on 66/100 of our local MLPs.
  Full transfer chain: torch-10 **×1.0089** → flopscope-100 **×1.0030** → grader **×0.9987**. The gain did not
  merely shrink, it **changed sign** on a different draw of MLPs. Recorded as **NEGATIVE §60** with the rule
  it implies: the local→grader discrepancy is ≈0.4–0.5% on a 50-MLP mean, so **a lever worth <2–3% cannot be
  validated by a submission**, and scalar tuning on our baked set is below the instrument's resolution.
  **k3m (332403) remains the best graded and is the submission the write-up will designate.**
- **Cloud closed the last cost idea on its first number** (`experiments/2026-09-28_tail-sampling/`, merged,
  **NEGATIVE §59**): unbiased importance sampling of the tail's CP terms has relative error ≈ κ/√r with
  **κ = 62.08 at n = 1024, i.e. κ/√n = 194%** against a 3–5% budget — the stopping rule's threshold exceeded
  10×, stable across a0 ∈ {1..4} × layer ∈ {11,13,15}, formula verified against real draws to 1–2%, and
  variance-reduced variants no better than 86.2% at r = 2n. Reaching 5% would need r ≈ 1.5e6 columns against
  the R = 15360 the tail contains.
  **The mechanism is the keeper, and it corrects my own reasoning:** §16's coherence is a property of the
  **diagonal readout t**, while sampling variance is set by the **Frobenius norm of the tensor**, and in *that*
  norm the tail's terms are nearly orthogonal (κ/√R = 0.50 against 0.008 for coherence, 1.0 for orthogonality).
  I had the right mechanism and applied it in the wrong norm. **Same-sign t and near-orthogonal T are not in
  conflict** — t is one diagonal, sampling damages the whole tensor.
- **NEGATIVE §61:** every live accuracy lever is unaffordable at 24.8% util. `k4k3` (the best, ×1.0396) costs
  **≈4 n³ products per source per layer**; `thin384` ≈+5 points of B for ×1.0079; `pathD="full"` / `allages`
  ≈+12.8 points. `k4b` alone IS free (O(n²)) but is monotonically harmful on the shippable base. So the whole
  ×1.11 stack is conditional on a cost breakthrough that is now closed **five times** (§54, §55, §56, §58, §59).
- **New cloud task — the question under all five failures** (`notes/CLOUD-BRIEF.md`, branch
  `cloud/state-closure` (the brief said `cloud/closure-test`)): every attempt tried to *represent* the κ₃ tensor more cheaply; none asked whether it
  must be represented at all, since the nonlinear step reads only **t** and **D**. **Is (t, D, C, κ₄) a
  sufficient state?** Precedent: our κ₄ *is* already closed in O(n²) (the scale-mixture), so the analogue for
  κ₃ is not exotic. Designed with a control arm (B4) that separates "the slices are insufficient" from "the fit
  found a bad optimum", and with Part 3 (the 100-MLP flopscope ablation table for the write-up) to be done
  regardless, because it is deadline work with certain value.
- 🔑 **Strategic reframe from a careful re-read of SPEC §6.** The **Algorithmic Contribution prize is $20,000**,
  judged by ARC from *code + write-up*, in a **separate category** — one participant can take a score prize and
  the algorithmic prize. Score prizes are decided **only** by a private re-evaluation on fresh MLPs, where
  "solutions overfitted to specific MLPs/seeds are penalized". And ARC's stated bias (guidelines #18041) is
  *mechanistic understanding, negative results and ablations welcome; less interested in heavy sampling,
  fitted constants, performance optimization, opaque LLM-optimized code.*
  **That is close to a description of what this project has actually produced**, and it is why the write-up is
  a live prize track rather than a consolation. It also says what to de-emphasise: our fitted constants — one
  of which (§60) graded worse than the value it replaced, which we will report rather than hide, since §6 warns
  that unhedged dubious claims can get an entry skipped entirely.
- ✍️ **Write-up v4 started.** v3 describes the Part I method at 7.07e-8 and is 13× out of date, so v4 is a
  rewrite. `writeup/v4-PLAN.md` holds the claim ladder (C1–C13, each with its evidence and its status) and the
  section structure. **`writeup/v4-s6-boundary.md` is drafted in full (~1750 words)** — the boundary section,
  which is the part we think is the contribution: the five-instance cost-model mechanism, coherence as the test
  for any cheap correction (with the staleness trap stated), and the instrument's resolution. §5 waits on the
  cloud's 100-MLP ablation table; nothing else is blocked.

## 2026-09-28 (evening, Mac) — the cost-free ceiling drops to 1.724e-9; k3n packaged; k4b is an interaction
- ✅ **k3n on 100 MLPs: 5.5045e-9 @24.892%, ×1.0030 over k3m, better on 66/100, 0 failures**, residual
  0.196/0.226 s. Packaged (`submission-20260928-161016.tar.gz`), expected grader ≈5.372e-9 — free and strictly
  better, but it will not change the displayed rank. **Its real value is the transfer factor it measured:** a
  scalar fitted on 10 torch MLPs came through at **1/3 strength** on the 100-MLP flopscope set (×1.0089 →
  ×1.0030, breadth 9/10 → 66/100). Every fitted constant from now on is quoted with that ≈3× discount;
  structural levers are not affected. (`experiments/2026-09-28_k3n/`)
- 🔑 **The cost-free ceiling moved from 1.92e-9 to 1.724e-9** (`experiments/2026-09-28_accuracy-free/`).
  Re-opened every term ever rejected as *"does not pay"* — that phrase always meant a cost/benefit at 25% of
  budget, never an accuracy verdict. On top of the ×1.11 stack, three more live levers:
  **`pathD="full"` ×1.0252**, **`k4k3` ×1.0396 at its interior optimum 0.5**, **`k4b` birth depth ×1.0878 at
  k4b=5**, plus `thin384` ×1.0079 and `k31mult` 0.65 → 0.55. Structural-only combination **×1.0878 → floor
  1.763e-9**; with the fitted pair **×1.1122 → floor 1.724e-9**, and **every arm improves 10/10 MLPs**.
  Board #1 is 1.70e-9, so the floor is now worth rank 1–2 rather than 2–3.
- **Closed on ACCURACY, not on cost** (a distinction this project had not drawn before): `aug` (ARC's dense
  AUGMENT diagrams) **×0.463** — it double-counts what we derived independently (`c0d`, `pairx`, `k31pat`,
  the path births); `e3` **×0.944**, which *corrects* §45 from "a real mechanism that does not pay" to simply
  wrong on the modern base; `edge2` ×0.981; `tet=4`, `q22x` neutral.
- **A methodological catch worth keeping:** `corr="split1024rr"` (the exact correction leg) is **worse** than
  `split384rr`. The randomized rank is non-monotone past 192 — the truncation is acting as regularisation, not
  only as an approximation. Do not assume "more rank is better" here.
- ⚠️ **`k4b` looked like a free win and is not.** With `k4bparts=("sd",)` the n³ branches are skipped, so extra
  birth depth is O(n²) ≈ 0.03% of budget — free. But on the **shippable** base (k3n config) it is
  **monotonically worse**: k4b = 1/2/3/5/8/15 → ×1.000/0.995/0.989/**0.977**/0.965/0.963. It only pays
  together with all-age path births; with `e3age=1` the deeper κ₄ births are inconsistent with a κ₃ that has
  no path content at those ages. **So nothing from the stack is shippable before the cost breakthrough.**
- **Net strategic effect: the cost breakthrough is now worth more, and it is the only thing left.** Everything
  else on the accuracy axis is either already in the stack, or dead on accuracy, or conditional on the stack.

## 2026-09-28 (Mac + cloud) — the ceiling measurement reframes everything; the tail closes except for one crack; k3n banked
- 🔑 **The decisive measurement of the week** (`experiments/2026-09-28_ceiling/`, 10 MLPs): **if cost were free,
  the accuracy stack we have always refused ON COST is ×1.11**, better on **9 of 10** MLPs —
  knobs ×1.0089, `allages` ×1.0809, `thin192` ×1.0154, `deadcut` off ×1.002. At the measured flopscope/torch
  ratio 0.920 that is raw **1.92e-8**, so **at the 10% multiplier floor the score is ≈1.92e-9 = rank 2–3**
  (board: #1 1.80e-9, #2 1.90e-9). Conservatively without `allages` (the only expensive lever): 2.11e-9.
  **So the plan is not "improve accuracy" and not "improve cost" — it is: reach the floor, then spend the
  freed budget back on accuracy.** Below 10% of budget, FLOPs are worth nothing and accuracy is worth 1:1.
- `allages` read ×1.026 on 3 MLPs and ×1.081 on 10 — **the fourth time a 3-MLP screen mis-ranked a 5–10%
  effect** in this project (§51). Also flagged: the `corr="full"` arms of the screen are an artifact —
  `chain.py` takes a separate branch there in which our path/age switches are inactive (two different configs
  returned identical values, in a shorter runtime). `thin192` is the honest proxy and it saturates at ≈×1.02.
- **Cloud closed the tail merge** (`experiments/2026-09-26_old-tail/`, merged; NEGATIVE §58). Accuracy is not
  the obstacle: the tail is a **soft** channel (1% relative error → 0.1–1.1% raw, budget 3–5%), with **no √n
  discount**, and a **rank-n block exists** (n=64 converged: 4.29e-4) — the tail is not incompressible.
  The construction is the obstacle, three ways: the only O(n²) construction (keep the r heaviest columns)
  gives **62.8% at real n=1024**; one Gram fitting pass costs **8.0 units against a total merge budget of 8.8
  units/layer**, and 400 passes reach 4.32% and are still falling (≈400× gap); and **CP has no nested
  optimum**, so a warm start from a converged block jumps to 48.7% — worse than cold selection.
  **Fourth instance of one mechanism** (§54 band entry, §55 pathrank basis, §56 L5 memory traffic, §58 the
  merge): *a cost model that prices the saved product and not the construction it needs says yes every time.*
- ⚠️ **The one crack, and the next cloud task:** everything cheap that was tried is **biased** (drop the light
  columns), which loses mass coherently — §16's failure mode. **Unbiased importance sampling** (p ∝ term
  weight, reweight by g/(r·p)) is O(n²), needs no passes and is **nested by construction**, so all three
  objections miss it. Predicted error ≈ (Σ‖term‖)/(√r‖T‖) — one cheap number decides it. Brief with a
  stopping rule: `notes/CLOUD-BRIEF.md`, branch `cloud/tail-sampling`.
- ✅ **k3n banked** (`src/estimator_k3n.py`): k3m + `feedmult 0.8` + `k22c 0.0`, two scalar levels found by
  `experiments/2026-09-26_coherent-knobs/` (scan + refine, 10 MLPs). Torch ×1.0089 on **9/10** MLPs at **zero
  extra FLOPs** (−0.002 points: k22c=0 skips one n² product). Self-checks: `k3n(1.0, 1.0)` is **bitwise** k3m
  with equal FLOPs (25.169% of B); defaults differ by 1.4e-5 relative. 100-MLP run in flight; expected
  ≈5.34e-9 on the grader.
- Also measured and worth keeping: the grader's wall is **90% real kernel time** (55.2 s of 60.9), so no
  op-count trick can buy back the Strassen L5 lever (§56).

## 2026-09-26 (midday, Mac) — k3m graded 5.3906e-9 (best); Strassen L5 and the variance channel both closed; top-5 is now provably a cost problem
- **332403 (k3m) → 5.3906e-9**, predicted 5.386e-9 (0.09% off). Grader final-layer MSE 2.1652e-8, util 24.835%,
  **100/100 MLPs, 0 failures**, residual 0.153–0.178 s of 0.400, predict wall 57.5–75.3 s of 120 (mean 60.9).
  New ratios: grader/local **0.9764**, wall 3.14× local on the mean and **3.9× on the worst MLP**.
- ❌ **Strassen L5 closed** (`experiments/2026-09-26_strassen5/`, NEGATIVE §56). The meter pays — util
  25.169% → **23.067%** (×0.9165, less than the modelled ×0.88) with a 1.5e-6 relative output change and an
  acceptable residual — but **wall 19.7 s → 38.4 s (×1.95)**, i.e. ≈147 s on the grader's worst MLP against a
  120 s cap, and peak RSS 7.81 GB of 8 GB. `chunk 8` fixes neither (38.2 s, residual 0.364 s). Mechanism: level 5
  runs 20 element-wise combine passes over a 1.1 GB buffer — memory-bandwidth bound, not layout-bound (§49).
- ❌ **The variance channel is closed to cheap corrections** (`experiments/2026-09-26_variance/`, NEGATIVE §57).
  Added `varfix` / `varfix_late` to `chain.py` (default off; `ablate.py k4memo 3` still 3.7533e-8).
  - `diagnose.py`: rms relative variance error 3.2e-4 at layers 0–6 — **exactly the 1e7 oracle noise floor**, so
    the error is invisible there — growing to 1.85e-3 at layer 15 with a +4.5e-4 bias. A least squares of that
    error on (α, α², log var, κ₄/var², t/σ³, Σ|D|) gives **R² = 0.098**.
  - `scan.py`: `var ← var(1+c)` gives **MSE ≈ 2.4e-8 + 5c²**, minimum at **c ≈ 0 to within 3e-5**; c = 1e-4 costs
    ×2.7, c = 1e-3 costs ×200. The variance LEVEL is already calibrated to ≈1e-5 — level fixes closed a third time.
  - `decompose.py`: every low-parameter correction is **worse** than doing nothing (level ×0.59, level + log var
    ×0.54, full 7-feature fit ×0.535). **Mechanism: coherence.** A smooth function of smooth observables is
    coherent across neurons and the readout sums it coherently, while the ×1.54 lives in the incoherent residual,
    which the readout discounts by √n — measured, 1.8e-3 rms pattern ≡ a coherent 4.1e-5, ratio 44 ≈ √n.
  - ⚠️ Method trap worth an hour: multiplicative factors measured on the base trajectory are **stale from layer 1**
    (touching var moves the chain's own var); at this stiffness the drift alone reads as ×0.658 for an "exact"
    arm, which is NOT a contradiction of the ×1.54 oracle arm (that substitutes absolute values). Only
    state-dependent corrections are testable — and only those are implementable.
- **Cloud session (user-launched) worked on the pathrank port and confirmed it is dead on the real meter**
  (branch `claude/friendly-mendel-a6a3yf`): factored path leg at rank 384 = **×1.0495** of the dense leg,
  at full rank 47.5% of B vs 25.169%. Its torch rank sweep at deadcut −2.5: rank 192 → 2.3768e-8, 128 → 2.3840e-8,
  64 → 2.3857e-8 (base 2.3462e-8), i.e. +1.3…+1.7% raw — so even a hypothetical free basis would not pay.
  Its `src/estimator_k3m.py` is parked (name clash with the Mac's submitted k3m); only its experiment folder is
  evidence. **Not merged into main.**
- 🔴 **The strategic conclusion, in arithmetic.** Board top-5 = ≤2.30e-9. Our cost (24.835%) with the board's best
  raw (1.54e-8) is **3.82e-9 ≈ rank 17**; our raw (2.165e-8) at the leaders' 13% is 2.81e-9 ≈ rank 8, and at the
  10% floor **2.17e-9 ≈ rank 3**. **Accuracy cannot reach top-5; only cost can**, and every known cost switch is
  now closed. So the whole remaining budget goes to the **old-tail representation** (FRONTIER (i), symmetric-CP
  cubes), which is what `notes/CLOUD-BRIEF.md` now tasks the cloud with — Stage A measures how accurate the tail
  must be (drop / rank-truncate / perturb its D and t contributions separately), Stage B builds the merge.

## 2026-09-26 (morning, Mac) — k3l graded 5.5984e-9 (rank 73); the grader now hides half the test set; k3m packaged; pathrank withdrawn
- **332253 (k3l) → 5.5984e-9**, exactly the predicted ≈5.57e-9 (grader/local = **0.9756**). Final-layer MSE
  2.1293e-8, all-layers 8.14e-9, util 26.29%, **100/100 MLPs, 0 failures**. Telemetry (public submission page, no
  login): residual **0.1476–0.1623 s, mean 0.1558** against a 0.400 s cap; predict wall 57.5–64.3 s, mean 60.9,
  against 120 s. grader/local-runner residual = **0.753**. Board: rank 73, handle `artiem_kuznietsov`.
- ⚠️ **Grader schema 2.0: the test set is split 50 public + 50 private** (`notes/SPEC.md`, new subsection). The
  ranked number is the mean over the 50 public MLPs; the final score at Phase-2 close is over all 100 with the
  private half sealed. Per-MLP **telemetry still covers all 100**, only the private scores are hidden. Earlier
  submissions were graded on 100. Consequence: the iteration signal is noisier by ≈√2 and the final rank is decided
  on a half we never see — prefer levers that improve 99–100 of our own 100 MLPs over levers that win on the mean.
- **The board moved up while we sat at 5.6e-9:** #1 J2W 2.00e-9, #2 suliman_tadros 2.10e-9, **#2 Luna 2.10e-9**
  (from 2.60e-9, five hours earlier), #4 marius_binner / mliston 2.30e-9, #6 a_s6 2.50e-9. **Top-5 is now ≤2.30e-9**,
  not 2.50e-9.
- ✅ **k3m packaged** (`submission-20260926-061347.tar.gz`) = `src/estimator_k3m.py`, a **standalone** copy of k3l
  with `deadcut=-2.5` (sha256 543905c6…; the file inside the tarball is byte-identical to `src/`). Two bitwise
  self-checks with equal FLOPs (`experiments/2026-09-26_k3m/k3m_check.py`): `k3m(-3.0) == k3l()` at 26.622% of B and
  `k3m() == k3l(-2.5)` at 25.169%. So the 100-MLP numbers measured yesterday through the wrapper carry over
  unchanged: **5.5209e-9 @ 24.894%**, 0 failures, residual 0.197/0.227 s. Subprocess 3 MLPs: 5.715e-9 @24.935%,
  residual 0.29–0.38 s (subprocess reads ≈2× the grader). **Expected grader ≈5.386e-9** → ≈ rank 60. Awaiting the
  user's submit.
- ❌ **`pathrank` withdrawn before the port** (`notes/NEGATIVE-RESULTS.md` §55) — an arithmetic result, not a run.
  The cost model priced the truncated path leg as a pure volume reduction (q = ρ/n) of the one transport it needs,
  which silently assumes the rank-ρ factors already exist. Counting the basis as well: `Y Ω` + `Z = Qᵗ Y` (2q) plus
  `Wᵗ Q` + `(Wᵗ Q) Z` (2q) = **4q = 1.0 units at ρ=256 — exactly the cost of the dense transport**, before
  `fnp.linalg.qr` (≈0.125 units); a power iteration makes it 5q–6q. And no cheap deterministic basis exists:
  with the closure's K22, `Y = d1 Dᵗ d3 + d2 D d2 + (1+v) d2 C0² d3 + rank-1`, where the rank-1 part contracts to
  O(n²) for free and everything else needs a low-rank D / C0², closed by §18/§20/§22. Same failure mode as §54
  (shared subspace): the model priced the saved product, not the basis it needs.
- ✅ **Strassen L5 is back on the table** — the residual headroom is much larger than the shipping rule assumed
  (grader 0.156 s vs 0.400 cap; grader/local = 0.753 → local-runner max ≤0.35 s is safe). Modelled: billing
  0.636 → ≈0.560, util 24.894% → ≈21.9%, **≈4.86e-9 local / ≈4.74e-9 grader** on top of k3m, with *zero* accuracy
  cost. Blockers, all Mac-only: the `m < 128` guard in `_Strassen.__init__` (L5 needs leaf 32), RSS (the L5 level
  region is ≈2.2 GB on top of k3h's 3.95 GB against an 8 GB cap → `chunk` may have to drop to 8), and wall time
  (32×32 leaf matmuls are BLAS-inefficient; local wall must stay ≲39 s).
- **Cloud brief written** (`notes/CLOUD-BRIEF.md`): a paste-ready prompt for the **variance-pattern** channel
  (the ×1.54 single-channel lever), branch `cloud/variance-pattern`, three to four fresh 1e7 MC oracles, no timing
  dependence, no writes to the shared notes. Chosen over L5 (Mac-only decision) and over pathrank (dead).

## 2026-09-25 (late evening, Mac) — deadcut −2.5 confirmed on 100 MLPs: 5.5209e-9 (×0.962)
- The cloud's best single lever, measured on the production path (`runs/v_k3l_cut25.py`, a two-line subclass;
  `src/estimator_k3l.py` untouched): **adjusted 5.5209e-9**, final 2.2172e-8 (+1.53%), util 24.894%, all-layers
  8.363e-9, **0 failures**, residual mean 0.197 / max 0.227 s, wall 19.4 s. **99/100 MLPs better on adjusted**
  (raw is worse on 90/100 — the lever trades accuracy for utilisation, and the trade pays almost everywhere).
- **Residual improved** (0.197/0.227 against k3l's 0.207/0.239): fewer alive rows to contract. So the lever is
  free on the constraint that actually caps Strassen depth. Expected on the grader ≈5.36e-9.
- **The projection held to 0.6%** (predicted 5.488e-9 from a 10-MLP torch screen + the modelled utilisation),
  which validates both the calibrated cost model and the torch→flopscope accuracy shift for future screening.
- Not packaged: `src/estimator_k3m.py` (this + path-leg rank 256) is being built in a cloud session and
  supersedes it.

## 2026-09-25 (evening, Mac) — accuracy re-attributed on k3l: variance is the lever, and the oracle method hit its limit
Ran in parallel with the cloud's cost-frontier sweep. Full write-up:
`experiments/2026-09-26_accuracy/RESULTS.md`. MLP 0, base `k3l_t` (self-check 2.4074e-08 = the logged value).
- **Re-attribution was needed:** the standing numbers came from `oracle_subst3.py` on **k3f**, before k3i/k3j/k3l took
  raw from 2.80e-8 to 2.31e-8.
- **Oracle substitutions:** variance ×1.43, κ₄ diag ×1.30, κ₃ diag ×1.09, κ₃+κ₄ ×1.69, **variance+κ₃+κ₄ ×5.02**,
  C ×0.96, D ×0.85, C0 ×1.03. Product of the singles 2.03 vs 5.02 together → **strong super-additivity**, which is the
  mechanism behind this project's history of ~×1.05–1.15 per lever.
- **A sanity arm broke, and it mattered.** `var0+C0` substitutes exactly what `C` does, so they must agree; they
  differ by 48% (1.68e-8 vs 2.50e-8). Cause: `mc_oracle_0.npz` (N=1e7, variance noise 4.5e-4) and
  `mc_oracle_all_0.npz` (N=4e6, 7.1e-4) differ by **6.6e-4 relative — pure MC noise, the quadrature sum of the two**
  — and that alone moves the final MSE by half. **So "exact C is worse than the chain" is an artefact of oracle
  noise, not evidence of error cancellation**, and every arm fed from the 4e6 file (C, C0, D) is noise-dominated.
- **Response curve** along the noise direction (`var_sensitivity.py`): a sharp bowl, minimum at t≈0.25 with
  **1.56e-8 (×1.54)**, rising to 4.45e-8 at t=−1 and 7.01e-8 at t=+2. The ML combination of the two oracles sits at
  t*=0.29 — the MSE minimum lands where statistics says the truth is, so the final MSE is effectively a detector of
  the true variance, and both oracles are already off the bottom. Baked GT noise (1e9 samples, ~1.6e-10) is not the
  limit.
- **Consequences:** (a) the **variance pattern** is the biggest single accuracy channel (×1.5) and was absent from the
  candidate list; (b) **κ₃ is nearly exhausted (×1.09)**, so the κ₃-driven G_off lever is no longer candidate #1;
  (c) the scale that matters for the variance is 1e-4–1e-3 relative, which is also a warning for cost work that
  truncates transports; (d) to quantify any single channel again we need two independent 1e7 oracles (different
  seeds) and the spread between them as the error bar, or N≈2e8 for a real 1e-4 oracle.
- **Error bars, from a second independent 1e7 oracle** (`mc_oracle.py 0 1e7 777`, 16 min; the script now takes an
  optional seed and writes its own file). Pooled values with the A/B spread: **variance ×1.54 (1.09), κ₄ ×1.31
  (1.01), κ₃ ×1.14 (1.07), κ₃+κ₄ ×1.76 (1.01), all three ≥8.21× (1.13)**. Every arm reproducible, so the
  single-channel attribution stands and the noise problem was specific to the N=4e6 file, as diagnosed.
- **The pooled column is the real find:** halving the oracle noise moved the joint arm 5.02/4.46 → **8.21**, so
  every ceiling measured this way is a **lower bound set by the oracle**. Exact diagonals mean raw ≤2.93e-9 —
  adjusted 7.7e-10 at today's utilisation, 2.6× better than the board's #1. The accuracy axis is nowhere near
  exhausted; we simply cannot reach it yet. (FRONTIER's standing "×5.5 with exact var+κ₃+κ₄" is superseded.)
- `chain.py` gained one switch, `var0_layers` (exact variance substituted early instead of at the readout), default
  off — every existing configuration is bitwise unchanged, as promised to the parallel cloud session.

## 2026-09-25 (cloud session #1) — the cloud agrees with the Mac, exactly; two setup bugs found
Scope of the session was verification only (CLOUD.md §6), no research and no submission.
- **The chain reproduces to every printed digit.** `ablate.py k4memo 3` → **3.7533e-08** (the standing
  self-check) and `ablate.py k3l_t 10` → **2.3135e-08**, with MLPs 0/1/2/7/9 at
  2.4074 / 2.1572 / 2.2646 / 2.3867 / 2.8053e-08 — identical to the Mac's numbers in CLOUD.md §6 at the
  `%.4e` resolution, i.e. < 5e-5 relative, two orders below the 1e-3 gate. This is on x86_64 + MKL
  against the Mac's arm64 + Accelerate, `torch.set_num_threads(3)` untouched on both. The remaining
  five MLPs (not yet on the Mac): 3: 2.3714e-08 · 4: 2.5874e-08 · 5: 1.9805e-08 · 6: 1.9206e-08 ·
  8: 2.2540e-08. Time does not carry, as §2 says: 17–19 s/MLP vs ~11 s/MLP on the Mac.
- **`--data mini10` = 36 s**, all three sha256 checks OK (`weights_0 d524e662eca3d117`,
  `weights_9 3af7229b6cc196af`, `gt_10 557e6fcc891474bb`), `baked/` 641 MB, gt (10,16,1024).
- **Setup script, step 2–3: clean.** All three upstreams cloned into `/opt/arc/external` at the exact
  pins (33cf0c05 / 3e1e3b5 / 93d091a); the whest env came from the starter kit's uv project
  (whestbench 0.16.1, flopscope 0.12.1, numpy 2.2.6) — the PyPI fallback was not needed.
  Warm re-run of steps 0–3 + `whest validate` on `estimator_k3l.py`: 2 s, 4/4 OK.
- **Bug 1 — step 4 (torch) never worked in the cloud.** `download.pytorch.org/whl/cpu/torch/` answers
  200, but its wheel hrefs point at `download-r2.pytorch.org`, which the allow-list did not cover →
  `CONNECT tunnel failed, response 403`, an empty `.venv-torch` and no symlink, masked by the wrapper's
  `|| true; exit 0`. Both arms of step 4 hit the same host. Worked around with plain PyPI
  (`torch==2.8.0`, 68 s, `2.8.0+cu128`, 6.4 GB). CLOUD.md §4 now lists the host; **the script itself
  still needs the PyPI third arm**, otherwise the next environment rebuild repeats this — the in-session
  install lives in `/opt/arc` but the filesystem snapshot is taken from the setup script's own run.
- **Bug 2 — `.gitignore` missed both symlinks.** `external/` and `.venv-torch/` carry a trailing slash,
  which does not match a symlink, so under `--prefix /opt/arc` both sat untracked and one `git add -A`
  would have committed them into the tree CLOUD.md §8 says must stay clean. Slashes dropped.
- Setup duration at session start could not be recovered: the script writes no log and the snapshot
  normalises every mtime to one stamp. Worth adding a timestamped log line if that number matters.
- **Both follow-ups landed on the Mac the same evening** (see the next entry up): step 4 got its PyPI
  third arm plus a loud failure when the venv does not import, and the script now appends timestamped
  lines to `$PREFIX/setup.log`, so the next session can answer the duration question.

---

## 2026-09-25 (evening) — repo committed for the first time; second-machine (cloud) setup built
- **First commit** of the project: 594 files, 23.5 MB (code, notes, every JSON run report, write-up v3).
  `external/`, `*.npy`/`*.npz`, tarballs and venvs stay out. No AIcrowd credential exists in the tree;
  submitting remains a manual step from this Mac. Nothing pushed yet — the private GitHub repo is the user's call.
- **`scripts/cloud_setup.sh`** rebuilds the environment anywhere: the three upstream repos at the exact commits
  (`scripts/external_pins.txt`), the whest env (**fails loudly if whestbench ≠ 0.16.1 or flopscope ≠ 0.12.1** —
  a moved meter would invalidate every FLOP number in these notes), a light `.venv-torch` (torch 2.8.0 + numpy
  only: `chain.py`/`ablate.py` need nothing else, so the 4 GB ARC env is opt-in), then data + `whest validate`.
- **`scripts/fetch_baked.py`** rebakes `baked/` from the parquet shards that actually hold the first K MLPs
  (16 per shard, 1.1 GB) instead of materialising the 7 GB split — 10 MLPs from one shard — and checks the result
  by sha256 against this Mac's arrays. Verified: `weights_0`, `weights_9`, `gt[:10]` bit-identical (the dataset
  row stores weights exactly as `MLP.from_row` hands them to `predict()`; no transpose, float32).
- **What a second machine may and may not decide (CLOUD.md §2).** FLOP count `C` is analytic → identical anywhere.
  MSE carries over to ~1e-3 (different BLAS reassociates sums). **Residual time and wall time do not carry over at
  all** — residual is ≈20 µs × ops *on this CPU* and a breach zeroes predictions, so cloud runs pass
  `--no-residual-wall-time-limit --wall-time-limit 600` and every timing-dependent shipping decision stays here.
- Two grader options found that make cloud runs cheap: `whest run --streaming` (iterate N MLPs, no 14 GB on disk)
  and `--no-residual-wall-time-limit` / `--residual-wall-time-limit`.
- Cloud agreement check for the first session: `ablate.py k4memo 3` → 3.7533e-08, `ablate.py k3l_t 10` → 2.3135e-08
  (per-MLP Mac values recorded in CLOUD.md §6; keep `torch.set_num_threads(3)` so the diff stays clean).

## 2026-09-25 (day) — 332247 = 5.939e-9 (rank 69); shared subspace parked; K31 × 0.65 → k3l 5.74e-9 local, packaged
- **332247 (k3j) = 5.939e-9**, rank 69 (504aldo cluster at 5.40e-9 = ranks 53–68); telemetry clean.
- Shared subspace for old sources: torch variants (6:320 best, +1.3%), production mirror with one subspace iteration
  (+1.2%), flopscope port `estimator_k3k.py` exact vs torch — but 27.8% of B vs k3j 26.6% (entry-update overhead).
  Parked (NEGATIVE §54, 2026-09-25_cost RESULTS §5–§7).
- **K31 (closure κ₄ (3,1) slice) × 0.65: ×1.058 in torch (flat optimum 0.6–0.7; level ≈ AUGMENT's 0.54 of MC).** Other
  level scans flat (K22 C0² weight, feed, star birth). → `estimator_k3l.py`: **100 MLPs 5.7387e-9 @26.27%, 98/100 better
  than k3j**, packaged `submission-20260925-075409.tar.gz`. ⏳ the user submits (expected ≈5.57e-9).

## 2026-09-25 (night) — 332170 = 6.728e-9 (rank 72); k3j (dead-row cut) 6.11e-9 local, packaged
- **332170 (k3i v2) = 6.728e-9**, rank 72; telemetry clean (residual 0.135–0.156 s, wall 58–67 s). Grading takes
  ~30 min for our method (100 MLPs × ~60 s / 5 workers), which is normal.
- flopscope probes: rectangular products are billed in proportion, `take` ≈ free; a rectangular workspace Strassen
  (views of the same buffers) is exact (5e-6) and keeps the 0.61 billing.
- **k3j = k3i + dead-row cut (α<−3):** bitwise k3i when off; vs torch 2–6e-7 (stress test at −1.0 exact to 3e-7).
  **100 MLPs 6.1137e-9 @26.27%, final +0.2% vs k3i, 0 fails, residual 0.208/0.251 s.** Packaged
  `submission-20260924-201225.tar.gz`. ⏳ the user submits (expected ≈5.9e-9).
- Shared subspace on k3j config: 5:384 = +1.5% MSE; cost model says ≈21.9% of B (×1.17 net). Other variants not run.
- Stopped all compute at the user's request (01:50).

## 2026-09-24 (late evening) — ARC's AUGMENT reference is ×1.3 better; its cheap part → k3i (6.92e-9 local, packaged)
- **ARC `kind=AUGMENT, factor=True`** (the paper's "factorized augmented" algorithm; ARC's MIT code): MLP 0 2.0563e-8,
  MLP 1 1.9893e-8 vs our k3f 2.6608e-8 / 2.5780e-8 (×1.3). ~31 min and 40–55 GB per MLP (dense n³ (2,1,1) slices).
  `AUGMENT_G` (path legs removed) = 5.69e-7: AUGMENT only works as a consistent whole.
- Against the MC oracle (`aug_moments.py` + `aug_vs_chain.py`): AUGMENT's κ₃ diagonal is exact to the oracle's noise;
  its var is better; its κ₄ is not. With our `paths` switch our κ₃ also reaches the noise floor.
- **The old E3 test had a bug** (half strength, inconsistent correction; NEGATIVE §45 flagged), and **the 3-MLP
  "paths are harmful" verdict was sample noise**: on 10 MLPs all paths give ×1.10 (NEGATIVE §51 corrected).
- Only the path's first readout matters (age ≤1), and its D part rides L1/L2 for free → +1 product/layer. With the
  free star κ₄ birth and the augmented pair diagrams C0³, C0∘K22: 10 MLPs 2.7989e-8 → 2.4441e-8 (×1.145).
- **k3i** (`src/estimator_k3i.py`, METHOD §18): switches off = bitwise k3h; vs torch 2–5e-7. 100 MLPs: v1 (t-only)
  7.0233e-9, **v2 6.9242e-9 @29.82%, 100/100 better than k3h**. Packaged `submission-20260924-135835.tar.gz`
  (subprocess bit-identical). ⏳ the user submits.
- Also measured: κ₄ level ×f compounds (NEGATIVE §52); error budget / structure / sparsity (RESULTS §10).

## 2026-09-24 (evening) — 332142 = 7.401e-9; leaders' telemetry; path births (bug found, still negative)
- **332142 (k3h, Strassen L4) = 7.401e-9** (final 2.560e-8, 28.91%), 100/100 MLPs, residual 0.126–0.135 s, predict
  wall 55.7–60.4 s. ≈ rank 70. Board: #1 J2W 2.00e-9, #5 Luna 2.60e-9, #10 ≈3.1e-9.
- **Leaders' public telemetry** (read from their submission pages, `experiments/2026-09-24_k3-chain/leaders/`):
  | who | raw | util | kernel wall | residual |
  |---|---|---|---|---|
  | J2W #1 | 1.61e-8 | 12.7% | 93 s | 0.33 s |
  | suliman #2 | 1.63e-8 | 13.8% | 81 s | 0.32 s |
  | marius #3 | 1.56e-8 | 14.7% | 77 s | 0.35 s |
  | a_s6 #4 | 2.00e-8 | 12.6% | 91 s | 0.36 s |
  | Luna #5 | 1.67e-8 | 15.5% | 38 s | 0.15 s |
  | lode_dockx | 3.69e-8 | 9.8% | 61 s | 0.35 s (all-layers MSE 0.70: only the final layer is predicted) |
  | us | 2.56e-8 | 28.9% | 53 s | 0.13 s |
  - The leaders are 1.53–1.65× more accurate than us **on every public MLP** (per-MLP ratio min 1.22, max 2.0); log
    errors correlate 0.67–0.79 with ours and 0.74–0.87 among themselves. Most have data-dependent FLOPs (±5% per MLP).
  - Most leaders run slow kernels (3–4 GFLOP/s billed) with residual near the 0.4 s cap; Luna's profile matches ours.
- **ARC reference AUGMENT (factor=True)** on MLPs 0–2 launched (`aug_ref.py`, ~40–55 GB RSS: dense n³ (2,1,1)
  slices). Result pending at the time of writing.
- **Bug in the old E3 test (§45):** `chain.py` carried E3 at half the reference strength but subtracted the full birth
  slice. New `paths` switch = all factorable AUGMENT path births as ONE extra leg Sym(T·Y ⊗ 3A ⊗ B), verified vs ARC's
  code to 5e-16. On k3f (3 MLPs, 2.6983e-8): E3 2.7973e-8, all paths 2.7449e-8 — worse (NEGATIVE §51).
- κ₄ level ×f at every layer compounds (NEGATIVE §52): the current level is at the optimum.
- **Error budget of k3f vs MC (MLP 0, `err_budget.py`):** per-layer readout injections at depth: var 2.2–3.2e-9,
  κ₃ 1.6–2.0e-9, κ₄ 1.0–1.3e-9; κ₄ level 0.95, κ₃ level 0.995, var slope 1.0002 (rel err 1.9e-3 at layer 15). The final
  error has no common-mode bias and simple per-neuron features explain 1–4% of it (`err_structure.py`): it is the
  accumulated mean-channel error.
- **Activation sparsity (`sparsity_probe.py`, MLP 0):** α<−3 ("dead", d1<1.4e-3) is 0% at layers 0–2, 8% at 8, 18% at
  15. Dropping dead rows from the CONTRACTIONS (not the outputs — their own t_i matters, ~4e-9) would save ≈12–20% of
  the per-source products, data-dependently. Candidate cost lever.

---

## 2026-09-24 (afternoon) — top-5 goal; 332128 = 8.286e-9; κ₃-step diagnosis; k3h (Strassen L4) 7.67e-9 packaged
- The user set the goal to **top-5** (≈2.6e-9) and described the working style; saved in `CLAUDE.md` and in memory.
- **332128 (k3f) = 8.286e-9** (≈ rank 77). Grader telemetry, read from the page data: residual 0.106–0.116 s on all
  100 MLPs (≈0.79 × local), predict wall ≈36 s.
- **The κ₃ one-step test with exact inputs** (`mc_k3step.py`) leaves a residual of 3–5% of |t′|, correlated 0.43 with
  the chain's κ₃ error. The step itself is truncated.
  - **E3** (edge × κ₃(2,1) tree) is a real mechanism: coef ≈0.9 against a theory value of 1, R² 0.33 at depth. In the
    chain it gives ×1.057 at +50% cost, so it does not pay.
  - A "memoryless κ₃" chain is closed: the all-distinct pass-through fits the amplitude form with R² ≤ 0.39.
- **The variance error channel is the off-diagonal covariance** (`var_attrib.py`), i.e. coherent transport of K11
  errors. Those come from D (corr 0.993 with MC) and κ₄ patterns.
- **Young κ₄ births:** the star diagonal is free and worth ×1.017 on 10 MLPs. Not shipped yet.
- **Strassen:**
  - residual ≈20 µs per flopscope op, independent of batch shape;
  - L4 at chunk 8 would cost +0.1 s residual.
  - **k3h:** the D pairings are summed inside the recursion (only the sum is needed; the leaf is matmul + sum in a
    reused scratch region; einsum is 70× slower). L4 at chunk 16 fits in 2.7 GB with L3's call count.
  - **100 MLPs: 7.6733e-9 @28.90%**, MSE = k3f, residual max 0.203 s local.
  - Packaged `submission-20260924-092524.tar.gz`. ⏳ the user submits.
- **Shared subspace for old sources** (torch accuracy test): age ≥5 at rank 384 costs +1.3% MSE; ranks 512/640 or
  age ≥8 are lossless; there is a cliff below 384. Cost model: ≈×1.16 net. It is the next engineering step.
- Closed (NEGATIVE §45–47): E3 does not pay; memoryless κ₃; young κ₄ births beyond the free star-diagonal. Also: the
  kink-only columns trick (cost-weighted kept fraction 0.93, not worth it).

## 2026-09-24 (day) — 332066 graded 9.562e-9 (rank 82); k3f 8.59e-9 packaged; the tail and the pattern problem
- **332066 (k3e) = 9.562e-9** on the public 50 (final 2.96e-8, 0/50 failures, private 50 completed). Rank 82; the #10
  threshold is ≈3.1e-9. The 504aldo MIT-copy cluster sits at 5.4e-9 (raw 2.13e-8 at 25%).
- **Old-source tail vs the amplitude mode** (positive homogeneity → κ₃ ∝ Sym(μ⊗C)). The old part of t follows μᵢvarᵢ
  with R² 0.92; the old part of D follows the amplitude form with R² 0.87. As a replacement for old sources, even the
  per-layer oracle fit is 17× worse (NEGATIVE §39). The cost of the tail stays.
- **Oracle headroom on the k3e/k3f config (MLP 0):**

  | exact | final MSE |
  |---|---|
  | κ₃ | 2.29e-8 |
  | κ₄ | 2.29e-8 |
  | variance | 2.18e-8 (k3f: 1.88e-8) |
  | κ₃+κ₄ | 1.49e-8 |
  | **var+κ₃+κ₄** | **4.8e-9** |

  The MC full C and the MC D at 4e6 samples are too noisy to serve as oracles; their pair noise adds up coherently.
- **The rest of the first-order scale-mixture set:**
  - marg + K31 together: ×1.06 on 10 MLPs (each alone is neutral or worse);
  - the κ₄→κ₃ feed P2 = Sym(h⊗D1C0D1), made O(n²) in the source representation through J(D1C0D1)Jᵀ = B diag(d1b) Aᵀ;
  - MC check (`mc_feed.py`): the closure's feed matches MC with corr 0.97, but it is not what limits κ₃;
  - all together: **×1.125 on 10 MLPs, all better.**
  - The feed channel is the lever 504aldo reported (forum #18218, post 5); ours is derived independently and
    attributed.
- **`src/estimator_k3f.py`:**
  - switches off: bitwise equal to k3e; against torch: 2–4e-7;
  - the port check caught a double-count bug in the torch `freefull` + feed reference; the split branch was correct;
  - residual trims are bitwise identical, including `Wt·C` fused into the leg stack.
- **k3f on 100 MLPs: adjusted 8.5916e-9, final 2.6547e-8, 32.36%, 0 fails, residual max 0.167 s, 100/100 better
  than k3e.** Packaged `submission-20260924-065809.tar.gz` (subprocess check bit-identical, residual 0.18–0.195 s,
  RSS 3.55 GB). ⏳ the user submits.
- Closed today (NEGATIVE §39–§44):
  - the amplitude tail closure;
  - P1 (v × tree);
  - "the feed limits κ₃";
  - marginally corrected leg gating;
  - κ₃ level rescaling (the error is the pattern);
  - richer κ₄ pattern features (corr 0.89 but no readout gain).
- **Strassen L4 assessment:** ≈×1.12 on cost. It is blocked by residual time (+50 ms at chunk 8) and memory (≈4 GB at
  chunk 16) until the ops per `_rec` level are cut.

---

## 2026-09-24 (night, cont.) — κ₄: the missing diagram (edge × κ₃) + a scale-mixture closure of the pattern → k3e
Full record: `experiments/2026-09-24_k3-chain/RESULTS.md` §7; method: METHOD §15; dead ends: NEGATIVE-RESULTS §34–§38.
- New MC oracle `mc_oracle31.py` (4e6 samples, layers 6/10/14; pre (3,1) and (2,1) slices, post slices, the
  common-scale statistic q). One-step test `k31_step.py` reproduces the 0.955 deficit at layer 14 on independent MC
  data, then tests candidates one by one. **C0·D = ½(a2b3Dᵀ + a3b2D)∘C0 closes it: 0.999/0.999/1.004, corr 0.999.**
  Nothing else moves it: marginal Edgeworth coefficients, (3,1) slice, K22², C0·K22, ρ³…ρ⁶, t·C0, κ₃² blocks.
- FRONTIER A1 (pure scale mixture) **falsified by sign** (`scale_mixture_test.py`): the pre (2,2) level is radial (two
  estimates of v agree to 3 digits), but higher scale orders lower the post (2,2).
- In the chain C0·D alone is **worse** (4.64e-8 vs 3.75e-8): it raises the κ₄ level to 0.91 of MC, but the chain's κ₄
  PATTERN is wrong (corr 0.23–0.53), so the error grows. MC κ₄ᵢ correlates 0.86 with varᵢ² (`k4_pattern.py`).
- **Scale-mixture closure** (K4dᵢ = 3v varᵢ², K22ᵢⱼ = v(varᵢvarⱼ + 2C0²), level v from memo22) + C0·D: 3 MLPs 3.0522e-8
  (all better); **10 MLPs 3.5198e-8 → 3.0975e-8 (×1.136), thin-E production form 3.5550e-8 → 3.1496e-8 (×1.129)**,
  9/10 MLPs better.
- The levels in O(n²) (uᵀXu plus the mean-field trace) reproduce the exact ones to 4 digits, so the 2 dense κ₄ products
  per layer are dropped.
- **`src/estimator_k3e.py`:** switches off → bitwise equal to k3d; vs the torch chain at full thin rank 2–5e-7 (the
  changes themselves are 1e-4–2e-3); **32.30% of B** (k3d 34.22%).
- **k3e on 100 MLPs: adjusted 9.7622e-9** (k3d 1.2260e-8, ×1.26), final 3.0224e-8, 0 fails, residual max 0.168 s
  under load, 96/100 MLPs better. **Packaged `submission-20260923-201554.tar.gz`** (subprocess check bit-identical,
  residual 0.17–0.21 s). ⏳ the user submits (k3d stays as a fallback package).

---

## 2026-09-24 (after midnight) — k3d: residual time 0.215 → 0.137 s at slightly lower cost; packaged
- **k3c L3 on 100 MLPs** (`runs/k3c_L3_n100.json`): adjusted 1.2335e-8, final 3.5835e-8, util 34.42%, 0 fails, but
  residual mean 0.199 / **max 0.215 s** — misses our ≤0.2 s criterion. Not shipped. Why the margin matters: prize
  ranking comes only from the Private Re-evaluation; one residual breach there zeroes that MLP and ruins the mean.
- **A/B, 6 MLPs, load ≈2.6:** base products through Strassen 1.2326e-8 @34.42%, residual 0.250/0.272 s; plain
  (`base_strassen=False`) 1.3007e-8 @36.33%, 0.182/0.190 s. Residual also rises with machine load (0.199 → 0.25 s).
- **Sanity checks:** the old `k3c_L0_n3.json` (MSE 5e-6) predates the factor-3 fix; re-run L0 equals L3 on MLPs 0–2
  (`runs/k3c_L0_n3_recheck.json`). `strassen_paths.py`: L0/L3nb/L3 predictions differ only by float32 noise
  (final rms 1.2e-6, max 7e-6).
- **Where residual time goes** (`prof/run_sections.py`, instrumented copy of k3c; ms/MLP): `_nonlin` 47–67, κ₄ block
  37–45, base C 33–36, transport 28, left+paired 26–30, birth 15–17, **teardown after `return` 39–45** (frees of
  ≈2.5 GB workspaces). Residual ∝ the number of flopscope calls (≈12–28 µs each); an n×n temporary costs only
  ≈0.01 ms more than `out=` (`prof/temp_cost.py`) — the "temporaries" hypothesis was wrong. Aside: `out=` with
  symmetric-tagged inputs costs ≈5 ms of flopscope *overhead* per op (wall only).
- **k3d (`src/estimator_k3d.py`):** (1) the 4 base products per layer as 2 paired Strassen calls, [Wt, Wsq^T] @ [C, Xk]
  then @ (that)^T (C, Xk symmetric); layer 0 one product (C = I); last layer diagonals only; (2) workspaces cached on
  the instance (the grader keeps one instance per worker; the FAQ allows instance state as an optimisation if each
  predict is correct alone). Self-checks (`k3d_check.py`): `batch_base=False` **bitwise equal to k3c**, also with the
  workspace reused over MLPs A,B,A (so nothing is read before written); batched = k3c ± float32 noise.
- **k3d on 100 MLPs** (`runs/k3d_L3_n100.json`): **adjusted 1.2260e-8**, final 3.5833e-8, util 34.22%, 0 fails,
  **residual mean 0.127 / max 0.137 s** (2.9× headroom), wall 11.6 s. RSS over 8 predicts 2.95 → 3.49 GB, saturating.
- **Packaged `submission-20260923-191338.tar.gz`** (file identical to src; `--runner subprocess` 3 MLPs bit-identical,
  residual 0.17–0.19 s). ⏳ the user submits.

---

## 2026-09-24 (night) — Context consolidated for a new chat; k3c L3 100-MLP validation
The user asked for everything to be wrapped into long-term project context, including **which methods failed and
why**. Done:
- `experiments/2026-09-24_k3-chain/RESULTS.md`: the complete record of the k=3 phase (every config and number,
  scripts, flopscope rules). The oracle and diagnostic outputs that had only been printed are recovered into
  `logs/` (compare_oracle_0, oracle_subst_0, k22_*, k4level_0, k4_diag_k3xedge, k3c_L4_n100_summary).
- `notes/METHOD.md` **Part II** (§8–§14): the k=3 chain, memo22 κ₄, thin E, buffered/Strassen layout, cost model,
  results.
- `notes/NEGATIVE-RESULTS.md` **§F** (§16–§33): 18 dead ends of the k=3 phase with mechanisms. The common one: the
  diagonal readout sums positive-weight contributions, so any truncation loses its tail coherently.
- `notes/FRONTIER.md` (new): the target arithmetic (both axes must move), the κ₄ per-step deficit and the
  scale-mixture / radial-κ₆ hypothesis with a test plan, κ₃×edge, cost-class ideas, the recommended order.
- `CLAUDE.md`: rewritten as state tables (submissions / packages / estimators / runs / next actions). The old
  STATUS blocks are in `notes/CLAUDE-2026-09-24-status-archive.md`.
- Wu et al. PDF saved to `notes/references/`.
- k3c L3 on 100 MLPs (quiet machine): ⏳ running at the time of writing; the result goes into the CLAUDE.md state table.

---

## 2026-09-24 (afternoon/evening) — Thin correction legs (6→4 products/source, lossless); an MC oracle finds κ₄ is the accuracy lever
User: "top-10 is the bet — reason about global correctness, verify, study the maths". Paper (arXiv 2605.05179, read
§1–9): its factorized algorithm is O(L²n^K) like ours; no cheaper trick there. Global target: adjusted ≤3.5e-9 ⇔
raw ≤3.5e-8 at ≤10% budget (our raw is already there) or raw ~1.7e-8 at ~20%.

**Cost — thin E (verified, `src/estimator_k3c.py`).** Exact split of the (2,1) correction:
`dK21 = diag(alpha) C0 diag(d1) + R`, `alpha = w(1,2) − 2 P1 d1`. The first part is carried for FREE as `3 B diag(alpha)`
(a column rescaling of the tree leg); R is numerically low-rank. Self-checks exact (`freefull`, `fcfull` = 3.7533e-8).
R dropped: ×10.5 worse; R rank 16 (SVD): 3.7742e-8 (×1.006); randomized range finder 16+8 cols: 3.7679e-8. Per source-
layer 4 dense products instead of 6 (= 504aldo's "7→4 with thin n×16 residuals", derived independently). flopscope port
had a double-counted factor 3 (caught by the full-rank self-check). **k3c on 3 MLPs: L3 1.369e-8 at 36.3% (wall 10 s),
L4/chunk 8 1.253e-8 at 33.2% (wall 17.8 s, residual 0.24 s)**, final MSE unchanged (3.77e-8).
Closed on the way: full-rank tree legs needed (rank-256 U: ×2; only births ≥8: ×1.3); D not low-rank (rank 256: ×2);
per-source low rank of the HUB leg A is what hurts (A only: 5.53e-8; B only rank 256 from age 4: ×1.08) — the hub
enters Σ_p A·(g B²) with positive weights, so its discarded tail adds coherently.

**Accuracy — MC oracle (`mc_oracle.py`, 1e7 samples, MLP 0; `compare_oracle.py`, `oracle_subst.py`).**
- Readout with EXACT per-neuron moments: Gaussian 1.8e-7, +κ₃ 2.2e-8, **+κ₃+κ₄ 1e-10** at layer 15 → the one-step
  Edgeworth readout is essentially exact once κ₄ᵢᵢᵢᵢ is right.
- Chain vs MC: mean rel err 2e-4; variance slope 1.0006; κ₃ diag corr 0.998, slope 1.00, rel 6%; **κ₄ diag corr only
  0.23–0.53 and slope 0.54 at layer 15** (the chain has half the κ₄ level at depth).
- Substitution (baked GT, MLP 0, base 3.50e-8): exact κ₄ diag at all layers **2.23e-8 (×1.57)**; exact κ₃+κ₄ diag at
  all layers **1.67e-8 (×2.1)** — the leaders' raw level (1.64e-8).
- κ₄ mechanisms vs MC: tree births (paths) raise corr slightly, stars wrong; **κ₃×edge κ₄ (d₂d₁³κ₃C) raises the corr of
  κ₄ diag to 0.53–0.71** (a real mechanism, dropped by SIMPLE) but not the level.
- (2,2)-slice oracle (`mc_oracle22.py`, 4e6): MC κ₄ diag mean = 3 × mean κ₄[iijj] at every layer; the chain's (2,2)
  mean is right at layer 3 and falls to 70% by layer 15 — and the deficit arises in the NONLINEAR step (post-ReLU (2,2):
  0.94/0.84/0.71/0.69 at layers 6/10/13/14, corr 0.89–0.99), not in transport (non-direct pairings average out over
  W). Not explained by: κ₃² pair diagrams, higher Gaussian ρ terms (ρ⁴…ρ⁶). "κ₄[iijj] ∝ C_ij" (504aldo's fitted
  closure) is NOT what the MC shows (C is sign-random; (2,2) variation is ~30% explained by C²).
- Level rescaling by oracle ratios is unstable (κ₄ sits in a feedback loop) — discarded as a test.
- **Decisive one-step decomposition (`k22_step.py`, layer 14):** fed the MC pre-activation κ₄ (2,2) slice, the
  nonlinear step gives the post (2,2) at **95.5% of MC (corr 0.978)**; with the chain's own inputs 68.7%. So the step
  loses ~4.5% per layer and it COMPOUNDS (0.955^10 ≈ 0.63; observed 0.69 at layer 15). The residual correlates best
  with Φ_aΦ_b·K22_pre (coef ≈0.06): the κ₄ pass-through through ReLU is ~6% too weak. Interpretation [U]: the
  pre-activations are close to a Gaussian SCALE MIXTURE (z_i | y ~ N(·, Σ_a W²y²) with a common random scale s =
  ‖y‖²-like); with nonzero means E[y²|s] is nonlinear in s, so terms in κ₃(s) (radial κ₆) and Var(s)² appear at
  second order; v = Var(s)/s̄² ≈ 0.022 at layer 15 makes them a few % per step. Next: a scalar recursion for the
  common scale's higher cumulants (radial κ₆), verified against the oracle.
- **k3c L4 on 100 MLPs FAILED 4 MLPs on residual time (max 0.453 s)** while MC oracles ran in parallel (mean wall 23 s
  vs 18 s quiet). L4 has no margin under load → not shipped. L3 re-run on a quiet machine.

---

## 2026-09-24 (morning) — Submission 331962 graded 1.915e-8: public rank 89. The per-source family is exhausted
- **331962 (Strassen L2) graded 1.915e-8** (final 3.43e-8, all-layers 1.36e-8, 55.87%, 0/50 fails). Grader wall
  25.8–29.4 s/MLP (2.9× local, not 5×), residual share of the bar 0%. **Public rank 89.** Leaders' raw MSE is only
  ~2× ours (1.6–2.0e-8); their edge is budget (~13–18% vs our 50–56%).
- **Strassen depth on the real runner (3 MLPs):** L3 1.879e-8 at 50.07% (wall 14.2 s, residual 0.19 s); L4 chunk 8
  1.703e-8 at 45.4% (wall 25 s → ~73 s on the grader: too close to the 120 s cap). Estimator peak RSS: L2 2.2 GB,
  L3 3.2 GB, L4/8 3.1 GB, L4/4 2.2 GB (the runner's 10.5 GB is the dataset, not us). L3 packaged:
  `submission-20260923-082030.tar.gz`.
- **Compressing only very old sources (Tucker, r=128, k_new=32; radial-κ₄ baseline 4.25e-8):** young=10 4.32e-8,
  young=8 4.76e-8, young=6 6.37e-8 (r=64: 8.57e-8), young=5 9.66e-8. Cost saving at young ≥ 8 is only 0–11% → no
  net gain.
- **Can old-source content be regenerated from young?** (`old_vs_young.py`, MLP 0) At layer 15 sources older than 4
  carry 69% of the norm of κ₃[iii] (older than 6: 53%) and correlate with the young part only 0.6–0.8 (residual
  fraction of a per-layer linear fit 0.4–0.6). No.
- **Higher tetrachoric terms** (ρ³…ρ⁶ in every pair cumulant): 3.775e-8 vs 3.753e-8. No.
- **Conclusion:** within the per-source family the cost floor is ~45–50% with Strassen, and every cheapening costs
  ≥1.5× accuracy. Top-10 (≤3.5e-9) needs a different representation of the third cumulant.

---

## 2026-09-24 (early) — Own k=3 chain in flopscope: 2.50e-8 adjusted on 100 MLPs (2.87x the submitted 331944)
- **More ablations of the faithful chain (3 MLPs):** memoryless κ₄ (2,2)-slice transport through W∘W instead of
  the radial scalar: **3.75e-8 vs 4.25e-8 (1.13×, all 3 MLPs)** — kept. Second-order Edgeworth κ₃² alone: 4.37e-8
  (worse). Cheap dense AUGMENT diagrams alone: 6.6e-8 (worse; inconsistent truncation). (2,1) slice only from the
  newest y sources: y8 6.2e-8, y4 3.9e-7. Low-rank ΔK21 in the correction leg: rank 128 → 2.0e-7. Full-C tree legs
  without the E leg: 3.2e-7. **Every cheapening of the faithful chain costs ≥1.5×.**
- **Error localisation** (`inject.py`, exact means injected at layers 0..u): 4.25e-8 total; u=7 3.66e-8; u=11
  2.45e-8; u=13 1.52e-8; u=14 8.0e-9. **The error is made by the per-step truncation at the deep layers** (≈6e-9
  per layer at 12–15), not by early births.
- **Port to flopscope** (`src/estimator_k3b.py`), float32 check vs float64: 3.7552e-8 vs 3.7533e-8. flopscope
  gotchas found: (1) **arrays are immutable** — no item assignment; write through `out=`; (2) `fnp.zeros` is
  tagged symmetric, so `out=` into its slices fails — use `fnp.empty`; (3) `as_symmetric` validation and
  `stack/concatenate` of large non-contiguous views cost *untracked* (residual) time; (4) freeing arrays of
  hundreds of MB costs ~6 ms residual each. Design that fixes it: all legs in one preallocated buffer ordered
  [B,E,A] per source, one batched matmul for all transports, one batched matmul + sum for the (2,1) slice,
  every temporary written with out= into preallocated workspaces. Residual per MLP 0.07–0.11 s.
- **Grader residual ≈ local residual:** on submission 331944's page the time-breakdown bar shows "penalty" ≈1% of
  7.2 s ≈ ≤72 ms against 77 ms locally; the 5× slower grader wall time is backend/overhead only.
- **100 MLPs, no Strassen:** adjusted **2.5037e-8**, final 3.5447e-8, all-layers 1.3871e-8, 70.6% of B, 0 fails,
  max residual 0.068 s, wall 2.3 s.
- **Workspace Strassen** (per-level preallocated buffers, combos and recombination via out= into quadrant views):
  2 MLPs: L2 adjusted 2.05e-8 at 55.9% (residual 0.175 s, wall 9.7 s); L3 1.84e-8 at 50.1% (residual 0.22 s,
  wall 15 s → ~75 s on the grader, cap 120 s). L2 is the safe first submission.

---

## 2026-09-23 (night) — New goal: top-10 with margin. Own SIMPLE-k=3 chain built and matched to the reference
User set the goal: **top-10 with margin** (#10 = 3.5e-9 adjusted today; we are at 7.07e-8, a ~20× gap; ~29× in
raw MSE). That is a different method class, so work moved to `experiments/2026-09-24_k3-chain/`.

- **Read ARC's reference SIMPLE k=3 nonlinear step and wrote it out as explicit formulas** (`step_check.py`,
  `chain.py`): single-index power moments with κ₃ and κ₄ Edgeworth terms (p = 1..4), pair power cumulants
  Q(p,q) for (1,1), (2,1), (2,2), Leonov–Shiryaev conversion to K(2,1), K(2,2); κ₄ kept as ONE radial scalar
  `a4 = 3(Σκ4_iiii + Σκ4_iijj)/(n(n+2))`, read back at the next layer as κ4_iiii = 4·a4, κ4_iijj = 4·a4/3
  (metric 2 after the linear step). Verified against mlp_kprop by teacher forcing: every dense slice to 1e-15.
- **Our own full chain in the source representation** (per birth: A = T, B = T·diag(d1)C0 with C0 the
  zero-diagonal covariance, E = T·V with V = diag(ΔK3) + 3ΔK21ᵀ the exact-repeated-slice correction, hub g = d2):
  **matches mlp_kprop per layer to 1e-16 and reproduces 4.2509e-8 on baked MLPs 0–2 exactly**, 7 s/MLP in
  float64 torch vs 12 s for the reference. Cost: 6 products per source-layer (3 transports + 3 for the (2,1)
  slice) ≈ 660 products ≈ 64% of B (the reference: 148%).
- **Ablations of the faithful chain (3 MLPs, final MSE; full = 4.25e-8):** no κ₄ scalar 4.54e-7 (10.7× worse);
  correction leg only diagonal 1.74e-6 (41×); no correction 2.80e-6; window J=8 1.50e-7, J=6 3.51e-7, J=4
  8.68e-7, J=3 1.26e-6, J=2 1.83e-6. **Every ingredient is load-bearing and old sources are needed.**
- **Tucker compression of old sources** (shared basis P n×r transported like a leg, symmetric core G r³, HOSVD
  refresh; self-check: exact at r = n): young=1/r=64 1.43e-6, young=1/r=128 6.91e-7, young=2/r=128 2.75e-7.
  Same failure mode as the covariance-subspace dead end: the diagonal readout sums the discarded residuals
  coherently. Consistent with #18218 ("shared subspace = 1.3× lever, cliff below").
- **Reading:** within the per-source paradigm the cost floor is ~25% of B (504aldo). Leaders (1.64e-8 raw at
  ~13%) must use a different representation. The other axis is accuracy: raw ~1e-8 at ~25% would also be top-10
  (2.5e-9). κ₄ is the obvious lever (10.7× from a single scalar). Measuring AUGMENT (matrix-valued κ₄) next.

---

## 2026-09-23 (evening) — Write-up v3 drafted; the September baseline was mis-transcribed; one over-claim corrected
- **Submission 331944 is the best one** (7.070972551635606e-08; verified on its public page: 50/50 public
  MLPs, 0 failed, final MSE 4.721e-7, 14.98%). This is the ID the write-up cites.
- **⚠️ The "September baseline 4.058e-6" was wrong.** A new 100-MLP run of our k=2 + tetrachoric
  (`experiments/2026-09-23_writeup-ablations/report_k2.json`) gives **3.8502e-6**, and it is *bit-identical*
  to the stored September report `experiments/2026-09-16_baked-gt/k1_mini.json`, whose own log also printed
  3.8502e-6. No logged run ever produced 4.058e-6 as a split mean; it matches **MLP 0's** value (4.0576e-6),
  so it was almost certainly one MLP's number copied as the mean. Consequences:
  - gain of the shipped method over **our** k=2 baseline: **5.36× adjusted, 8.03× raw** (not 5.65×);
    over the kit's gain-only covariance propagation (re-run here: **4.0504e-6** = kit docs, and per MLP it
    matches ARC's float64 reference k=2 to 3 s.f. on MLPs 0–4 — a clean adapter check): **5.64× / 8.45×**;
  - Edgeworth J=1 over our baseline: 1.69× (not 1.78×);
  - "tetrachoric is neutral alone" (NEGATIVE-RESULTS §14) was wrong: it is a **5.2% gain alone**, 7.7% at J=1, and
    in the shipped configuration switching it off costs 3.1% final / 14.6% all-layers (100-MLP ablation).
  Historic entries below are left as written; the tables in CLAUDE.md / METHOD.md are corrected.
- **Over-claim corrected.** "Our readout beats the reference's (175× vs 121×)" is not a fair comparison:
  in `state_split3.py` the "readout" is our first-order Edgeworth term fed the *reference chain's exact*
  diagonal κ₃, and the 175× arm also has ground-truth upstream means. On the reference's own upstream state,
  our first-order term is 1.5× *worse* than the reference's nonlinearity step (6.41e-8 vs 4.25e-8). METHOD §0
  now says this; the write-up states it with a [U] on the cause.
- **Edgeworth check re-done with a logged script** (`edgeworth_mc_check.py` → `logs/edgeworth_mc_check.txt`):
  the earlier "40M-sample MC, removes 61–87%" had no surviving script. Exact test on a Gamma family (closed
  form via incomplete gamma, itself confirmed by 4e7-sample MC to <1 s.e.): the term removes **96–99%** of the
  Gaussian-closure error at skew ≤ 0.1 and 86–97% at 0.2–0.3; residual ∝ skew², zero correction at a = 0.
- **Clips never bind** (3 MLPs, all layers): max |skew| = 0.17 (clip 1.0), max off-diagonal |ρ| = 0.60
  (clip 0.999).
- **Write-up v3** (`writeup/algorithmic-contribution.md`, v2 archived as `…v2-2026-09-22.md`), rebuilt around the
  method, with five new figures from logs (`experiments/2026-09-16_kprop-arc/make_figures_v3.py` →
  `writeup/figures/f1..f5`, table twins in `DATA-v3.md`); the old k=3 figures moved to Appendix A.
- **New optional lever measured: level-batched Strassen** (`experiments/2026-09-23_multisource/
  batched_strassen_test.py` + `.log`). Each Strassen level runs once for a whole stack of products (one batched
  `fnp.matmul` at the leaves), so Python-call count is O(levels) instead of O(7^levels) — the thing that capped
  us at level 1. flopscope bills the batched matmul exactly (batch × m·n·(2k−1)); ratio vs direct 0.88 / 0.78 /
  0.70 / 0.64 / 0.61 at levels 1–5, max rel. diff ≤ 8e-6. Wall time grows (local, 11 products of 1024²:
  0.08 / 0.23 / 0.38 / 0.71 / 1.48 s at levels 1–5; the grader ran ~5× slower than local in 331944), so
  levels 3–4 are the safe range. Projection: level 4 + skipping the unused last-layer D21 + Strassen on the
  base covariance → ~11% of budget → adjusted ≈ 5.2e-8 (1.35×); level 5 plus D21 on the newest 3 sources could
  cross under the 10% floor (≈ 4.7e-8, 1.5×). Not implemented — see the plan in CLAUDE.md.

---

## 2026-09-23 (afternoon) — The method is at a characterised local optimum; the next class is the barrier
Continued down the planned route (improve the ReLU transport beyond the `d1^3` pass-through) and it
closed, from two independent directions.

- **ARC's own reference answers the question.** `factor_k3.py`'s docstring states that the augmented
  Edgeworth term set "includes diagrams with no O(n)-rank factorization (e.g. **the covariance
  triangle and kappa_3 x edge products**), which this implementation DROPS". Those are exactly the two
  terms I had derived as the next correction. They are unaffordable for the reference too, and the
  reference still reaches 121× without them — so **they are not the gap**. Independently, my own
  contraction analysis puts the triangle at `O(n^4)`, which settles it.
- **The birth magnitude is not the gap either.** At layer 1 (where the input is exactly Gaussian, so
  the birth model is the whole story) our regenerated source correlates 0.998 with truth but carries
  only 0.918 of the magnitude, and 0.918^16 = 0.25 — so a compounding deficit looked plausible.
  Swept a multiplicative factor on the born source (`birth_scale.py`, 3 MLPs):

  | scale | 0.90 | **1.00** | 1.05 | 1.089 | 1.15 | 1.25 | 1.40 |
  |---|---|---|---|---|---|---|---|
  | vs scale 1 | 0.92× | **1.00×** | 0.98× | 0.93× | 0.82× | 0.63× | 0.39× |

  **The optimum is exactly at the derived value.** Rescaling toward the "missing" magnitude makes it
  worse. Worth stating in the write-up: the method has no fitted constant, and fitting one does not help.

**Standing position.** Graded 7.070972551635606e-08, 5.74× the September baseline, at 14.98% of budget.
Everything cheap has been taken and the boundaries are now measured on all sides:

| lever | verdict |
|---|---|
| more sources (J) | saturates at 4–5 with the covariance correction |
| 4th-cumulant path term | 1.04–1.06× for 4 points of budget → net worse |
| diagonal variance correction | harmful (0.22×) |
| D21 on a subset of sources | 3% |
| higher-order birth diagrams | no O(n)-rank factorisation; reference drops them too |
| rescaling the birth | optimum is the derived value |
| low-rank propagators | real Lyapunov concentration, but sources live 4 layers and the contractions need A, B materialised → 1.4% |
| deeper Strassen | meter says 0.78 at level 2, but residual time hits 81% of the cap |
| cumulant-order extrapolation | wrong sign |
| covariance subspace truncation | 3424× worse at m=512 |

The remaining 14× to top-10 needs the full cumulant-tensor chain, which our own cost model puts near
92% of budget even with Strassen. That is the barrier, and it is a cost barrier, not an accuracy one.

---

## 2026-09-23 (midday) — Matrix-decomposition route explored; Lyapunov concentration is real but does not pay
Fourth submission graded: **7.070972551635606e-08** (local 7.1794e-8; grader 1.5% better, fourth
consecutive agreement). **5.74×** the September baseline.

User asked to look at matrix decompositions, ranks and tensor contractions. The natural target: the
dominant cost is transporting each source's legs, and `A_m` is the **product of (age) random
matrices**, whose singular values spread exponentially (Lyapunov). Unlike CP truncation of the
cumulant tensor — where every published attempt failed — this is a **matrix**, so SVD truncation is
optimal (Eckart–Young). There is also a structural gift: `B = A U` with `U` fixed at birth, so
`A = X Y^T` implies `B = X Z^T` and only `X` needs transporting, at `O(n^2 r)`.

**The concentration is real** (singular energy of `A_m` by age, 1 MLP):

| age | top-32 | top-64 | top-128 | top-256 | top-512 |
|---|---|---|---|---|---|
| 1 | 0.112 | 0.210 | 0.376 | 0.623 | 0.894 |
| 2 | 0.222 | 0.386 | 0.617 | 0.860 | 0.990 |
| 3 | 0.315 | 0.516 | 0.756 | 0.943 | 0.999 |
| 4 | 0.394 | 0.615 | 0.840 | 0.975 | 1.000 |

**And truncation is nearly free in accuracy:** rank 512 from age 2 gives 0.98×, rank 256 from age 3
gives 0.97×, the schedule 512@2 + 256@3+ gives 0.94×. `fnp.linalg.qr` and `svd` exist, so a
randomized range finder is implementable, and it was implemented and validated.

**But it does not pay, for two compounding reasons:**
1. **Sources only live 4 layers.** Truncating at age 3 leaves exactly ONE layer that uses the
   factored form, while the sketch itself costs ≈0.9 of a matmul. Measured utilization fell only
   14.98% → 14.54% and the score moved 7.93e-8 → 7.82e-8 on the same 6 MLPs, i.e. **1.4%**.
2. **The contractions need `A` and `B` materialised anyway.** `kappa3[i] = 3 sum_a d2_a A[i,a] B[i,a]^2`
   is elementwise in (i,a); doing it in factored form costs `O(n r^3)`, worse than materialising.
   Materialising costs `2 n^2 r` each, which cancels what the factored D21 product saves.
   Reverted to the full-rank estimator.
- **Also tested and marginal:** applying the D21 covariance term to only the newest k of 4 sources —
  best is k=3 at 8.298e-8 against k=4's 8.558e-8 on 3 MLPs, a 3% gain. Not worth a change.
- **Standing conclusion:** at J=4 we sit at 14.98% of budget, paying multiplier 0.1498 instead of
  0.1. Getting the same accuracy under the floor is still worth ~1.5× and is the best identified
  lever, but neither deeper Strassen, nor fewer corrected sources, nor low-rank propagators reach it.

---

## 2026-09-23 (morning) — The covariance correction WORKS after all: my earlier test conflated two effects
Third submission graded: **id from 2026-09-23 04:22, score 9.901564582002446e-08** — first time below
1e-7, and the grader beat our local 1.02115e-7 by 3.1% (third consecutive agreement).

**Then a real reversal.** `covcorr.py` had concluded the covariance correction hurts. It applied the
diagonal and off-diagonal parts of the same correction **together**, and they have opposite signs of
effect. Split apart at J=4 (`vardiag.py`, 3 MLPs):

| applied | final_layer_mse | vs none |
|---|---|---|
| nothing | 1.2264e-6 | 1.00× |
| diagonal only | 5.7037e-6 | **0.22× (harmful)** |
| **off-diagonal only** | **6.7418e-7** | **1.82×** |
| both | 2.2084e-6 | 0.56× |

So the off-diagonal `Delta cov[i,j] = (1/2)(d2_i d1_j D21[i,j] + d1_i d2_j D21[j,i])` is a genuine win
and the diagonal must keep the exact rectified-Gaussian variance. A second split
(`offdiag_sweep2.py`): of the two leg assignments, keeping **only the first term** measures *better*
(2.15× at J=4) than keeping both (1.82×) **and costs half as much**. Accuracy saturates at J=4–5 and
degrades past it, matching the κ₃ sweep.

| J, terms | 3,t1 | **4,t1** | 5,t1 | 6,t1 | 8,t1 |
|---|---|---|---|---|---|
| raw vs none | 1.66× | **2.15×** | 2.15× | 2.00× | 1.71× |
| adjusted | 8.78e-8 | **8.56e-8** | 1.02e-7 | 1.25e-7 | 1.78e-7 |

**Shipped: `n_sources=4, strassen_levels=1, cov_correction=True`.** Authoritative on 100 MLPs:
**adjusted 7.1794e-8**, final 4.79346e-7 (2.06×), all-layers 2.0347e-7 (2.63×), utilization 14.977%,
**0/100 failures**, max residual 0.0809 s of the 0.4 s cap. **1.42×** the previous submission and
**5.65×** the September baseline of 4.058e-7. Rank 100 on the public board is 5.08e-8, so we are now
within 1.4× of appearing on it.

**The obvious next gain is the floor.** At 14.98% the multiplier is 0.1498 instead of 0.1, so ~1.5× of
the score is going to the meter. Running the same accuracy at ≤10% of budget would score 4.79e-8 —
inside the public top 100. Deeper Strassen alone does not get there (L=2 → 13.3%, and residual time
would roughly triple), and fewer sources lose more accuracy than they save. Candidate: transport J=4
sources but apply the D21 term to only the newest 2–3, or stack the per-source matmuls into one large
product to cut Python-call overhead and afford a deeper recursion.

---

## 2026-09-23 (late night) — kappa4 ruled out; Strassen confirmed on the meter; J=4 shipped
- **4th cumulant: measured, marginal, rejected.** Extended the verified cumulant rule to
  `Delta mu += (1/24)(alpha^2-1) phi kappa4 / sigma^3`. Validated the leading κ₄ tree diagrams at
  width 64 against 2e7 MC samples (`k4_diagrams.py`): the STAR term is identically zero wherever
  `alpha = 0` (its hub carries `d3 = -alpha*phi/sigma^2`), and the PATH diagram alone correlates
  **0.985** with the true κ₄. Control: the κ₃ prefactor fits 3.30 against the derived 3.0 with
  correlation 0.997, so the derivation is sound and the 10% is higher-order at that small width.
  The path template reuses matrices the κ₃ machinery already carries, so it costs one extra matmul
  per source per layer. **But end-to-end it buys only 1.04–1.06× while costing 4 points of budget**,
  which pushes past the floor and makes the adjusted score worse (1.512e-7 → 1.912e-7). A third κ₃
  source beats a κ₄ term. Closed.
- **Strassen-Winograd verified on the meter.** The organisers confirmed it is permitted. flopscope
  really does bill the op tree: ratios **0.88 / 0.78 / 0.70 / 0.65** at 1/2/3/4 levels, with max
  relative deviation 3.6e-6 in float32 — three orders below our accuracy (`strassen_test.py`).
- **But recursion depth is capped by residual time, not by the meter.** Each level multiplies
  Python-level fnp calls by ~7. Measured across 20 MLPs at J=4: level 2 costs **0.325 s residual
  against a 0.4 s cap (81%), consistently and not as a warm-up artefact**. The grader runs on one
  physical core, so that is an unacceptable risk — a breach zeroes every prediction. Level 1 runs at
  0.06–0.13 s. **Shipped level 1.**
- **New configuration: `n_sources=4, strassen_levels=1`.** Authoritative on 100 MLPs:
  **adjusted 1.02115e-7**, final 9.87674e-7, all-layers 5.36054e-7, utilization 10.339%,
  0/100 failures, max residual 0.059 s. That is **1.14×** the previous submission and **3.97×** the
  September baseline. Packaged and validated.
  Same-4-MLP control: J=3/L=0 1.2812e-7, J=3/L=1 1.2812e-7, **J=4/L=1 1.0834e-7**, J=4/L=2 1.0479e-7.
- **Note on the floor:** utilization is 10.34%, just *over* the 10% multiplier floor, so 3.4% of the
  score is lost to the multiplier. Level-2 Strassen would return us to 9.39% but costs the residual
  budget. A mixed-depth scheme (level 2 on ~27% of the matmuls) would land under the floor at ~0.14 s
  residual — a 3.4% gain, left as a micro-optimisation.

---

## 2026-09-23 (night, later) — Second submission graded; covariance correction derived and found premature
- **Submission id 331892 graded: 1.1359980192082731e-07** (local 1.16775e-7, grader 2.8% better — the
  pipeline reproduces for the second time). Cumulative **3.57×** over the September baseline of 4.058e-7
  and **1.96×** over submission 331885. Account `artiem_kuznietsov`.
- **Covariance correction derived from the verified rule, and it HURTS at every J.** The cumulant rule
  `Delta(cumulant) = sum_k (1/k!) E[d^k f] kappa_k` reproduces the MC-verified Edgeworth mean term exactly
  when `E[d^3 ReLU] = -alpha*phi/sigma^2`, so the 2-point case gives
  `Delta cov[i,j] = (1/2)(d2_i d1_j D21[i,j] + d1_i d2_j D21[j,i])`, `D21[i,j] = kappa3[i,i,j]`, and the
  source decomposition makes D21 two extra matmuls per source per layer
  (`D21 = 2*M1@B^T + M2@A^T`). Measured (`covcorr.py`):

  | J | mean only | mean + covariance |
  |---|---|---|
  | 1 | 1.61× | 1.17× |
  | 3 | 3.16× | 1.82× |

  Worse at every J, and it also doubles the cost. **Consistent with the map**, where injecting the *exact*
  k=3 covariance alongside k=2 means scored 0.70× — a covariance of one quality paired with means of
  another degrades the iterate. The covariance is the *last* ingredient, not the next one: it pays only
  once the means are near-exact (the map's 175× arm used ground-truth means).
- **Where we actually stand inside the mean class.** Ceiling with exact means + our covariance + our
  readout = **28×**. Our multi-source κ₃ model saturates at **4.5×** (J=8; it degrades at J=15 because the
  leading Wick diagram is a poor model for old sources). So ~6× remains inside this class, which would land
  near adjusted 2e-8 ≈ rank 85 — still short of top-10 at 3.5e-9.
  Note our self-consistent κ₃ (4.25× at J=15) **beats the exact-κ₃-from-another-trajectory oracle (2.98×)**:
  consistency with our own state matters more than pointwise exactness.
- **Next lever: the 4th cumulant.** The same rule gives `Delta mu = (1/24) (alpha^2-1) phi kappa4 / sigma^3`
  for the mean. 504aldo report a memoryless κ₄ regeneration halving their error at no cost, which is the
  single best-evidenced cheap gain left. Derive the rectified-Gaussian κ₄ in closed form, verify against MC
  as was done for κ₃, then regenerate and transport it on the existing source machinery.

---

## 2026-09-23 (night) — The decisive map: what each ingredient is worth, and a 2x from summing sources

### The map (`state_split2.py`, `state_split3.py`) — this is the centrepiece
Injected reference/ground-truth quantities at layers 0..14 only, so the scored layer is always
computed by us. **Corrects an earlier over-claim of ours: the covariance is NOT irrelevant.** The
earlier arms that said so had paired a k=3 covariance with k=2 MEANS — a mismatched state.

| upstream mean | upstream cov | our Edgeworth readout | final_layer_mse | vs k=2 |
|---|---|---|---|---|
| ours | ours | no | 5.1494e-6 | 1.00× |
| k=3's | ours | no | 2.5538e-7 | 20.2× |
| **ground truth** | ours | no | **2.2229e-7** | **23.2×** |
| ground truth | ours | yes | 1.8388e-7 | 28.0× |
| k=3's | k=3's | no | 2.4293e-7 | 21.2× |
| k=3's | k=3's | yes | 6.4131e-8 | 80.3× |
| **ground truth** | **k=3's** | **yes** | **2.9370e-8** | **175×** |
| full k=3 | | | 4.2509e-8 | 121× |

1. **A perfect final readout alone is worth 1.08×.** Fixing only the last layer barely matters, which
   kills the idea of computing the final-layer cumulant by a cheap O(L) backward pass.
2. **The three ingredients multiply.** Our Edgeworth readout is worth 1.08× on a bad state and 3.8×
   on a good one.
3. **Our readout beats the reference's**: ground-truth means + k=3 covariance + our readout gives
   2.94e-8 against full k=3's 4.25e-8. That is our asset, not a borrowing.
4. **Hard ceiling of any mean-only method: 23.2×** (raw ≈2.2e-7 → adjusted ≈2.2e-8 ≈ rank 85). Top-10
   is unreachable without correcting the covariance too, which needs the n² slice D21 and therefore
   genuine cumulant transport.

### Summing over all sources (`multisource.py`) — a real 2×, and a bug in our own code
The shipped estimator regenerated κ₃ from the **previous layer only**, but a source is born at every
ReLU layer and transported forward, so a depth-l pre-activation carries l of them. Carrying a window
of the J newest (2 matmuls per source per layer; birth is free because A starts as the identity):

| J | 1 | 2 | 3 | 4 | 6 | 8 | 15 |
|---|---|---|---|---|---|---|---|
| raw vs k=2 | 1.61× | 2.37× | 3.16× | 3.94× | 4.43× | 4.54× | 4.25× |
| adjusted | 2.97e-7 | 2.01e-7 | **1.64e-7** | 1.61e-7 | 1.90e-7 | 2.22e-7 | 3.07e-7 |

Accuracy saturates near 4.5× and *degrades* at J=15 — the leading Wick diagram is a poor model for
old sources. **A bug was caught by self-check:** the first run applied the ReLU linearisation twice
at birth, which showed up as J=1 scoring 1.28× where the shipped estimator already scored 1.78×.
Fixed; the table above is post-fix.

**Ported to `src/estimator.py` as `n_sources=3`.** On 5 mini MLPs: **adjusted 1.2955e-7 against the
submitted 2.5032e-7 — 1.93×** — at **9.10% of budget, still under the 10% floor**, 0 failures,
residual 0.0439 s against the 0.4 s cap. Full 100-MLP run in progress; previous submission
(id 331885) scored 2.2219e-7 on the grader.

---

## 2026-09-23 (later) — Submitted; then two non-standard shortcuts tested and both killed
User asked to think non-standard rather than default to the engineering grind, with the goal of
actually reaching the leaders. Two cheap shortcuts were tested first. Both died fast, which is the
point of testing them first. Full detail:
[../experiments/2026-09-16_kprop-arc/RESULTS-shortcuts.md](../experiments/2026-09-16_kprop-arc/RESULTS-shortcuts.md).

- **Richardson extrapolation across cumulant ORDER — dead.** If the `(L/n)^k` error law held per
  neuron, the step `mu^(k) − mu^(k−1)` would predict the residual error, and `k=1` costs 0.07% of
  `k=2` so the correction would be free. Measured: correlation 0.17, and the fitted slope has the
  **opposite sign** to theory (+0.027 vs −0.0159). Applying the theoretical correction makes it 8%
  worse; even an oracle-fitted slope buys 1.03–1.08×. The law governs aggregate MSE; per-neuron error
  directions decorrelate between orders. This closes "resummation", which our own RESEARCH.md had
  flagged as organizer-endorsed and unexplored.
- **Low-dimensional subspace — spectrum concentrates, dynamics do not.** The pre-activation
  covariance concentrates hard with depth: 90% of the energy in **145 of 1024** directions by layer
  15, participation ratio 511 → 65, numerically rank-deficient from layer 13. Promising, because the
  k=3 cost is concentrated in exactly those deep layers. But projecting the chain onto the top-m
  **exact** eigenspace destroys it: m=512 is **3424× worse**, m=200 is 16662× worse (self-check at
  m=n reproduces the chain). **Mechanism:** the mean reads `sigma_i = sqrt(C_ii)` and the diagonal
  aggregates every eigendirection, so discarded directions bias sigma downward for all n neurons
  coherently. Variance concentration does not imply low-dimensional dynamics when the readout is a
  diagonal functional.
- **Target restated precisely.** The multiplier floors at 0.1 below 10% of budget, and **every
  current leader sits above the floor**, paying 13–25% instead of 10%. So the unclaimed optimum is
  raw ≈2e-8 at ≤10% of budget, which wins outright; raw ≤3.5e-8 at ≤10% is enough for top-10. ARC's
  reference k=3 already gives raw 4.1e-8, only 17% above that. **The whole gap to the frontier is a
  ~15× cost reduction, not accuracy.**
- **Next:** the remaining route is the K=3 port. Porting ARC's own MIT reference is legitimate (it is
  the organizers' code, not a participant's submission); 504aldo's repo is not. Spike continues with
  a flopscope cost model of the transport before any build.

---

## 2026-09-23 — A constant-free analytic mean correction, derived, verified and shipped into the estimator
User confirmed AIcrowd registration (the 2 Oct hard gate is cleared). Full detail:
[../experiments/2026-09-16_kprop-arc/RESULTS-edgeworth.md](../experiments/2026-09-16_kprop-arc/RESULTS-edgeworth.md).
Machine kept light throughout: 3 torch threads, `--max-threads 4` for the grader, `nice -n 10`, one MLP at a
time, CPU only.

Starting point: yesterday's finding that the whole k=2 error is a MEAN error. So fix the mean.
- **The correction.** Derived the first-order Edgeworth term for the ReLU mean,
  `mu_post = mu*Phi(a) + sigma*phi(a) − s*sigma*a*phi(a)/6` with `s` the standardized skew of the
  pre-activation. **Verified against 40M-sample Monte Carlo.** Also derived the closed-form third cumulant of
  a rectified Gaussian and verified it **twice**: against MC, and against the ARC reference chain at layer 0
  (where the pre-activations are exactly Gaussian) to **7.8e-16**.
- **Route 1, transport the diagonal: DEAD.** The memoryless approximation
  `kappa3_pre[i,i,i] ≈ sum_a W[a,i]^3 kappa3_post[a,a,a]` carries only **1.7% of the true magnitude at
  depth** (correlation ~0.45). The third cumulant is made by off-diagonal cross terms. Independently
  reproduces and sharpens the corresponding dead-end row of forum write-up #18218.
- **Route 2, regenerate it from the covariance: WORKS.** Those cross terms are a functional of the
  mean and covariance the k=2 chain already carries. The leading connected Wick diagram, contracted with the
  next layer's weights, collapses to **one matmul per layer**:
  `Vt = cov_prev @ (w*d1_prev)`, `kappa3 = 3*sum_c (w*d2_prev)*Vt**2`. Coefficients `d1 = Phi(a)`,
  `d2 = phi(a)/sigma` checked against the reference `wick.py` to 0.00e+00. Correlation with truth **0.998 at
  layer 1**, decaying to 0.64 at depth — and it is most accurate exactly where yesterday's blame curve says
  the error is generated most (layers 1–3).
- **End-to-end (torch, 3 MLPs, baked GT).** Self-check: the uncorrected arm reproduces the reference k=2
  exactly (4.5018e-6). Corrected: **3.086e-6, a 1.67× gain**, at 3.82% of budget. Oracle arm (perfect
  diagonal): 1.726e-6, 2.98×. So the regeneration captures 56% of what is available, and **the whole
  mean-only Edgeworth direction has a hard ceiling near 3×** against full k=3's 110×. That is a useful
  negative: the k=3 gain does not live in the diagonal readout.
- **Ported into [`src/estimator.py`](../src/estimator.py)** behind `edgeworth=True`. `whest validate` passes.
  On 5 mini MLPs: **adjusted 2.503e-7**, utilization **3.82%**, effective compute 8.406e10 — matching the
  hand calculation of 8.4e10 exactly, and still under the 10% multiplier floor, so **the extra compute is
  free on the metric**. Zero fitted constants; the only constant is a skew clip that never binds at 1024×16
  (measured |skew| ≤ 0.1) and exists so the width-4 validator probe cannot blow up.
- **AUTHORITATIVE RESULT, full public `mini` split (100 MLPs), 0/100 failures:**
  **adjusted 2.2767e-7** (was 4.058e-7) = **1.78×**; final 2.2767e-6; all-layers 1.1591e-6 (1.85×);
  utilization 3.823%, multiplier still floored at 0.1. Per-MLP best/worst 1.231e-7 / 3.348e-7, so the
  gain is not carried by outliers. Packaged as `submission-20260922-192502.tar.gz`. Report:
  [../experiments/2026-09-23_edgeworth/RESULTS.md](../experiments/2026-09-23_edgeworth/RESULTS.md).
- **Ablation with a real surprise:** switching the old tetrachoric covariance term OFF costs **7.7%**
  (2.2767e-7 → 2.4527e-7), so it stays on. Measured alone in September it was neutral, and arm C showed
  covariance-only fixes actively hurt — **once the mean is corrected the same covariance improvement
  pays**. One mechanism explains all three: a better covariance is only useful paired with a mean of
  matching quality. That is the constructive side of "accuracy ⊥ stability".
- **SUBMITTED AND GRADED (2026-09-22 UTC).** `whest login` as `artiem_kuznietsov`, then
  `whest submit submission-20260922-192502.tar.gz --watch`. **Submission id 331885, grader score
  2.2219498214326448e-07**, no failures. Our local `mini` number was 2.2767e-7, so the grader came out
  **2.5% better** — local and remote agree closely (the two are different MLP sets from the same
  distribution: the live score uses the 50 public MLPs of the Phase-2 suite, `mini` is the 100 openly
  released verification examples). This is the evaluated submission the algorithmic-contribution
  write-up will point at; its ID is now in the write-up header.
- **Positioning:** 1.67× beats the published covariance + whitened-antithetic-MC blend (#18200, 1.33×) and is
  purely analytic, which is the profile ARC says it rewards. It does **not** approach top-10.

---

## 2026-09-22 (night) — USER COMMITTED ("я решаю идти", "I am going for it"). New science: the k=2 error is a MEAN error, and a vector may fix it
Two new experiments, every endpoint self-check exact. Full write-up:
[../experiments/2026-09-16_kprop-arc/RESULTS-layer-blame.md](../experiments/2026-09-16_kprop-arc/RESULTS-layer-blame.md).
Machine deliberately kept light (3 torch threads, `nice -n 10`, one MLP at a time, k=3 towers freed
immediately; CPU only, no MPS/GPU memory touched). ≈30 s per MLP.

- **`layer_blame.py` — per-layer oracle injection (3 MLPs).** Self-checks: base 5.1494e-6 = full k=2 ✓;
  GT mean injected at l=15 → 0 exactly ✓; k=3 state at l=15 → 4.25e-8 = full k=3 ✓; cov-only at l=15 →
  base exactly ✓.
  1. **The entire k=2 final-layer error is carried by the propagated MEAN.** Arm C (covariance replaced by
     k=3's, mean untouched) **never helps at any depth and is up to 16% WORSE** (C/base peaks at 1.16 at
     layers 2–3). Arm A (mean made exact) falls monotonically 1.00 → 0. Arm B ≈ arm A throughout.
  2. → **One mechanism explains three published near-nulls:** our tetrachoric correction (neutral at width
     1024), #18200's Hermite-2 covariance correction (−1.2%, i.e. worse), #18214's exact bivariate ReLU
     covariance (+7.4% only). Covariance is the wrong lever at 1024×16. Also a clean, controlled instance
     of #326024's "accuracy ⊥ stability" claim, which was argued at depth 32 without this decomposition.
  3. **No hot layer:** marginal blame ≈7e-7 at layers 1–3 decaying to ≈2e-7 at 9–15 (errors born early are
     amplified by the remaining depth). Structural reason why layer-adaptive cumulant order fails.
  4. **Layer 0 contributes exactly zero** (+8e-10) — it must, the input is exactly Gaussian there. A free
     validation of the whole measurement.
- **`edgeworth_probe.py` — can a vector replace the tensor?** Derived the first-order Edgeworth term for the
  ReLU mean, `mu_post ≈ mu*Phi(a) + sigma*phi(a) − kappa3_diag*a*phi(a)/(6*sigma²)`, and **verified it
  against 40M-sample Monte Carlo** (removes 61–87% of the closure error at a≠0; vanishes at a=0 as it must,
  where the residual is a 4th-cumulant effect). Validated `get_dslice((3,))` against an explicit dense
  tensor at width 16 (max diff 8.9e-16). With the TRUE diagonal kappa3 from the k=3 chain the correction
  **improves the one-step mean readout 3.7×–8.8× at every layer**, ratio exactly 1.00 at layer 0 (another
  free check). At the final layer it reaches **6.1e-8 against the full k=3 chain's 4.1e-8 — within 1.5× of
  a method that pays O(L²n³)** to carry the whole cross-cumulant tensor. Pre-activation standardized skew
  is small (0.008 → 0.091 with depth), the regime where a first-order Edgeworth term is expected to hold.
- **Honest limit:** the oracle supplies the diagonal. Whether it can be *propagated* cheaply is unproven.
  504aldo's dead-end table reports a memoryless diagonal-slice chain at raw 9.1e-7 — 45× worse than their
  engineered chain, but ~5× better than our 4.5e-6, which is consistent with our ratios.
- **Next (designed, not run):** an O(n²)-per-layer diagonal scheme with **zero fitted constants** — closed-form
  rectified-Gaussian "birth" skew (verify against MC before building), memoryless transport
  `sum_a W[a,i]^3 * k3[a]`, Edgeworth readout at every layer. Expected landing ~1e-6 raw → adjusted ~1e-7,
  a ~4× gain on our current submission; **not** a top-10 method. Entirely our own derivation, with no
  dependency on another participant's code (see the source-authenticity rule below).

---

## 2026-09-22 (late) — OFFICIAL RULES read in full (v14): two corrections + a hard gate we did not know about
Read the binding rules page end to end (§1–§14) instead of relying on forum summaries. Distilled into
[SPEC.md](SPEC.md) ("OFFICIAL RULES — full read"). Three things change what we do:
1. ⛔ **Registration & Team Freeze = 2 Oct 2026 23:59 UTC.** "No new individual or Team registrations will be
   accepted after that deadline, even though the Competition Period continues through October 31." If the
   user is not already registered (rules accepted on the Competition Site, v14), **everything else is moot**.
   Cannot be verified from this machine — the in-app browser is not signed in as the user. **Confirm first.**
2. ✅ **Designation is UP TO TWO submissions**, not one (§6). Default if you do not choose = your two best
   public entries. → we can designate a safe generalizing k=2 AND an aggressive entry. Docs said "one"; fixed.
3. ⚠️ **Source authenticity (§6 verification): the code must not be "derived from another Participant's
   submission."** Combined with §4 (submissions "authored substantially by other Participant(s)" = violation),
   **building on 504aldo's MIT repo is unsafe for prize purposes** even though §5.7 permits public pre-existing
   code. The ~19 identical 5.40e-9 entries sit in that position. Written clearance from
   arc-whestbench@aicrowd.com would be needed; their Strassen clearance was addressed to *their* submissions.
Also newly pinned: Phase 1 participation is not required (each phase is self-contained, own pool, own private
re-run); score prize and algorithmic prize are separate categories and one participant can win both;
**community-contribution awards $500–$5,000** exist with no application process; the grader suite is 100 hidden
MLPs = 50 public (live board) + 50 withheld, and prizes come from a *third*, fresh suite; **we retain IP** (our
own paper is fine) but any place prize requires an OSI open-source release within 30 days, kept up 3 years, and
grants ARC a broad perpetual licence; flopscope dtype rates confirm **float16 bills the same as float32**.

---

## 2026-09-22 — Context re-actualized: live rules re-verified, figure data re-logged, PDF pipeline ready
Fresh-account handoff check. Everything below verified live on AIcrowd / the forum today.
- **Dates & rules (forum #18197, #18041):** submissions close **17 Oct 2026 23:59 UTC**; **team/registration freeze
  2 Oct**; **write-up deadline 24 Oct 2026 23:59 UTC** (one week after close; the evaluator closes with the phase);
  **10 submissions/day/team**. Phase-2 pool $100k incl. **$20k Algorithmic Contribution**.
- **Algorithmic-Contribution rules:** "a PDF technical write-up + exactly ONE submission ID — both mandatory";
  send to arc-whestbench@aicrowd.com or post on the forum. Judged at ARC's discretion on *mechanistic insight*
  (leaderboard score alone does not decide); **negative results / ablations / measurement studies explicitly welcome**
  (a Phase-1 "characterizing a systematic bias" write-up is cited as a model); LLM use must be disclosed and
  unvalidated claims hedged ("the LLM gave this explanation, which we did not validate…") or ARC may skip the entry.
  → **BLOCKER for the write-up: we need a real AIcrowd submission ID.** No `whest login` key is stored on this
  machine, so the k=2 tarball has **not** been submitted from here. User action.
- **Leaderboard (2026-09-22):** 997 participants, 22,173 submissions. #1 J2W 2.10e-9 (raw 1.64e-8, 13.1% util),
  #10 3.5e-9, #100 5.08e-8 (the board lists only the top 100). The 5.4e-9 / raw-2.13e-8 copy-cluster = 19 identical
  entries at rank 42. Our k=2 (adjusted 4.06e-7) is ~8× outside the top-100 cut — expected; it is the generalizing
  baseline for the write-up, not a ranking play.
- **New Phase-2 write-ups since our research pass:** **#18200** (submission 328274): fixed 0.75/0.25 blend of
  gain-covariance propagation + *whitened antithetic MC* (X=[Z;−Z], whitened by G^{-1/2} fused into layer 1;
  4,038 samples; 9.345% util) → adjusted **4.05e-7 → 3.04e-7 (−24.85%)** on mini; branch-error correlation r=0.038;
  MC alone 1.07e-6. Their full Hermite-2 covariance correction was **−1.2% (worse)** — independently confirms our
  "tetrachoric ≈ neutral at 1024". **#18219**: below 10% util the board is a pure-MSE ranking; QMC at d=1024 collapses
  to the MC n⁻¹ rate. No organizer announcements since the Phase-2 launch (full category scan). Details: RESEARCH.md.
- **⚠️ MAJOR (read later today): forum #18218 (504aldo, 10 Sep) — a factorized K=3 chain engineered INTO budget:
  raw 2.13e-8 / adjusted 5.4e-9 at 0.25·B** (algebraic 7→4 restructuring, memoryless κ₄ regeneration with 16 fitted λ,
  age-gated shared subspace, Strassen–Winograd; MIT code = the 19-entry copy-cluster). Their 25-row dead-end table
  overlaps our three negatives (CP-cap needs ≥8n; dropping sources 1000× worse; marginal fixes cap at 4–5e-6) and also
  kills MC-control-variate hybrids (best 1.5×, need 33–300×), learned correctors (<0.09 corr) and the 1/n-literature
  route. **Our v1 thesis "k=3 accurate XOR affordable" was over-claimed** — true only for the reference implementation
  under naive levers. Open frontier sharpened: leaders 1.64–1.71e-8 at 0.13–0.16·B, below the K3+memoryless-κ₄
  ceiling (~1.9e-8), at the cost of the young tier alone. Full notes: RESEARCH.md (bottom).
- **Write-up rewritten to draft v2** (`writeup/algorithmic-contribution.md`; v1 kept as `…v1-2026-09-19.md`): honest
  positioning vs #18218, claim labels [M]/[D]/[U], LLM-use disclosure, the 5 figures embedded, numbers refreshed from
  the logged re-runs (k=3 = 4.10e-8 on 5 MLPs; rank-cap 128…4096; both adaptive sweeps in full). PDF rendered:
  `writeup/algorithmic-contribution.pdf`.
- **Re-ran every measurement script with logs kept** (`run_figure_data.sh` → `logs/*.txt`; the originals were never
  logged): kink mean 0.700 ✓; k=3 depth scaling ×4.27/4.13/4.06 ✓; breakdown 47.8%+47.5% ✓; κ₃ eff-rank 885–1024 ✓;
  ladder on 5 MLPs k1 3.33e-4 / k2 4.50e-6 / **k3 4.10e-8** ✓ (12.4 s/MLP in torch); rank-cap + layer-adaptive
  re-runs pending at time of writing (see logs). Everything reproduces the 09-17/09-19 numbers.
- **Write-up production pipeline (no LaTeX on this Mac):** `experiments/2026-09-16_kprop-arc/make_figures.py`
  → `writeup/figures/` (PNG+PDF + `DATA.md` table twins); `writeup/build_pdf.sh` = python-markdown → HTML →
  **WeasyPrint** (via `uvx`, brew pango) → `writeup/algorithmic-contribution.pdf`.

**Next — a fork for the user (not decided):**
- **A (lock-in, ~1 day):** USER runs `whest login` + `whest submit` (k=2 as is, or k=2 + the #18200 whitened-antithetic-MC
  blend for −25%) → submission ID into draft v2 → final proofread → file PDF + ID (email or forum) before 24 Oct.
  Honest measurement study; modest contribution (partly redundant with #18218).
- **B (research shot, the remaining ~3 weeks):** attack the sharpened open problem — κ₄-class accuracy (≤1.9e-8) at
  ≤0.16·B without per-source dense legs. Starting points: 504aldo's MIT chain + findings log as the base, our
  reference-side diagnostics (energy concentrates in late layers: 99% in 128 directions at layer 15). High risk; this
  is where a real paper would come from. Recommended: do A first, then B with whatever time remains.

---

## 2026-09-19 (evening) — Layer-adaptive k also dead → cheap-lever investigation COMPLETE
Built a manual layer-adaptive loop, **self-verified at both endpoints** (m=0→4.502e-6=full k2 ✓; m=16→3.705e-8=full k3 ✓).
- **Forward** (`layer_adaptive.py`, k=3 first m layers): k=3-early barely helps (m=4: 3.6e-6, ×1.2 @ 10.6%). 24× gain only appears at m→16.
- **Reverse** (`reverse_adaptive.py`, k=3 last j layers from Gaussian): also barely helps (j=8: 3.3e-6, ×1.4 @ 37.6%). Gain only at j→16.
- → **k=3 must be tracked through ALL 16 layers**; dropping κ₃ anywhere (early or late) loses the accuracy. Verifies
  "transport is the crux": the accumulated full-depth cross-cumulant structure produces the accurate final marginal. Irreducible O(L²n³).
- **Best affordable operating point ≈ pure k=2** (~4e-7 adjusted); layer-adaptive m=3–4 gives only ~4.1e-7 (×1.1), not worth the complexity.

**COST-LEVER INVESTIGATION COMPLETE — all three ruled out with clean evidence:** low-rank (κ₃ full-rank),
on/off/kink (70% kink), layer-adaptive (needs all-depth k=3). ARC's cumulant propagation is accurate XOR affordable at 1024×16.
**Leaders' 1.7e-8 @ ~20% is outside this framework** (efficient large-depth estimator = ARC's open problem, or learned closure, or partial public-overfit).
- **Deployable base (banked):** clean k=2 = `src/estimator.py`, adjusted 4.06e-7, validated + generalizing. Ready to designate.
- **Remaining (harder) shots at the top, all deferred/untested:** learned closure-corrector shipped as data (Phase-1 gave only +17–22%);
  reverse-engineer the leaders; 1/n effective theory (user deferred). Plus private-re-run reshuffle wildcard.
- **The research itself is the Algorithmic-Contribution deliverable:** a complete, self-verified compute-accuracy map of cumulant
  propagation at the Phase-2 (wide/moderate-depth) regime + evidence ruling out low-rank/kink/layer-adaptive.

---

## 2026-09-19 (later) — DOUBT-AND-VERIFY killed the two easy cost levers (major recalibration)
Told to implement rank-cap while doubting/verifying. Research REFUTED the plan (good — saved a doomed build):
- **κ₃ is NOT low-rank.** Per-layer SVD of the 3rd-cumulant factor: R grows 2048→32768 (=2n·(l+1)) but **effective
  rank stays ~n=1024** every layer; energy spread across hundreds of directions (top-128 = 62% at layer 0).
- **Crude rank-cap DESTROYS accuracy** (`rcap_test.py`): full k=3 = 3.7e-8; R_max=1024 → **2.9e-6 (79× worse)**,
  R_max=4096 → 8.5e-7 (23× worse). Column-dropping isn't valid CP-tensor compression; κ₃ is genuinely full multilinear rank.
- **on/off/kink also weak** (`kink_diag.py`): mean fraction |α=μ/σ|<2 = **0.70** (≈100% early, ~42% at layer 15);
  firmly on/dead only ~38% even at the deepest layer. He-init keeps μ≈0,σ≈1 → most neurons sit at the ReLU elbow.
- **Even best-case orthogonal re-compression to R=n saves only ~L≈8×** (2nL→n) → 148%/8 ≈ 18% budget → adjusted
  ~7.6e-9 ≈ top-45, NOT top-10. So the compression lever cannot reach the top even if it worked.

**RECALIBRATION (honest):** ARC's full cumulant propagation is ~irreducibly O(L²n³); it gives 4.2e-8 at 148% (or
best-case ~18% compressed → ~top-45). **Reaching top-10 (≤3.6e-9) / #1 (2.1e-9) requires a method fundamentally more
efficient than ARC's published cumulant propagation** — which is exactly the large-depth-efficiency problem ARC's
paper leaves OPEN, and what the leaders have evidently cracked. Our two easy levers are dead.
- **Untested shots at the top:** (1) **layer-adaptive k** (early layers are cheap in the O(L²) profile — does k=3-early
  capture the gain? needs a careful manual-loop test); (2) **1/n effective-theory correction** (Roberts–Yaida) — a
  cheaper closure than full tensors; research bet. Plus the private-re-run reshuffle wildcard.
- **Solid fallback (deployable, generalizing):** k=2 (2.4%, ~#120) or k=2 + control-variate/RQMC (~4e-7 per Phase-1 =
  ~#100), + a strong **Algorithmic-Contribution write-up** — our measured cost/accuracy map + ruling out low-rank/kink
  IS a genuine contribution (extends #326024's "map of the wall" to the Phase-2 regime).

---

## 2026-09-19 — Cost structure of k=3 pinned + the #1 path identified
- **Leaderboard moved:** new #1 **J2W (Numascale AS) 2.10e-9** (final 1.64e-8, ~12.8% budget, 295 entries, active);
  #2 marius 2.4e-9, #3 suliman 2.5e-9 (stale). Big **5.40e-9 / final 2.13e-8 cluster persists** (~17 identical =
  copied shared solution, ~25% budget). Top tightly packed 2.1–3.6e-9; best affordable final MSE ~1.64e-8. 988 participants.
- **Companion-paper theory (arXiv:2605.05179), verified against our measurements:** MSE ~ **(L/n)^K** (our ladder drops
  ~64–107× per k, matching n/L=64); **factorized cost = O(L²·n^K)**. Depth is ARC's *explicit open problem*; no
  resummation / optimal-K frontier in the paper → unclaimed.
- **Verified locally:** k=3 depth-scaling is **×4.06–4.27 per depth-doubling = O(L²) confirmed.** And **95% of k=3 cost
  is the 3rd-cumulant linear step** (`FactoredTensor3.get_dslice` 47.8% + `contract_W` 47.5%); Wick/nonlinear step is 3%.
- **KEY code find (`factor_k3.py`):** the 3rd cumulant is an **explicit low-rank tensor** `T=Sym(Σ_{r=1}^R A1_{i,r}A2_{j,r}A3_{k,r})`,
  factors `(n,R)`. The O(L²) is **rank R growing ~n per layer.** → **Capping R (low-rank re-compression per layer)
  is a direct, code-level cost lever**: O(L²n³) → O(L·R_max·n²). If R_max keeps ~4e-8 accuracy, k=3 fits ≪ budget.
- **Non-obvious external sources for the levers:** Roberts–Yaida–Hanin *Principles of Deep Learning Theory*
  (arXiv:2106.10165) — systematic **1/n effective theory**, deviations from Gaussian set by **L/n** (= our small param
  1/64) → a cheap perturbative correction to the mean; low-rank cumulant **CP-decomposition** (arXiv:2508.18408, 1911.03813)
  → theory for rank-capping.
- **THE #1 PATH (theoretical):** rank-capped factored k=3 (+AUGMENT / on-off-kink / 1-over-n correction) to land
  final ≤~1.5e-8 at **≤10% budget** (multiplier floored at 0.1) → **adjusted ≤1.5e-9 < current #1 (2.1e-9)**. And it
  GENERALIZES (private re-run). Hard but concrete. Next experiment: implement R-cap on FactoredTensor3, measure
  accuracy vs R_max & cost.

---

## 2026-09-17 — ARC reference measured at 1024×16: the accuracy/cost ladder (KEY DATA)
Cloned ARC `mlp_kprop` (MIT), built + **validated** an adapter (vs MC at n=64; k=2 reproduces our covariance
baseline at 1024×16; ARC's FLOP counter reproduces flopscope on k=2). Full results in
[experiments/2026-09-16_kprop-arc/RESULTS.md](experiments/2026-09-16_kprop-arc/RESULTS.md). Measured ladder:
- k=1 mean: **3.3e-4**; k=2 cov: **4.5e-6 @ 2.4% budget** (= our baseline); k=3: **4.2e-8** (≈leaderboard-top raw MSE!)
- **BUT k=3 costs 3.25e12 = 148% of budget → over-budget → FAIL.** BASE variant is no cheaper.
- Wide regime is a real tailwind (k=3 works here, unlike Phase-1 depth-32). The blocker is COST, not accuracy.
- **Reframed problem:** get ~k=3 accuracy (~4e-8) at ≤~20% budget (leaders do ~1.7-2.1e-8 @ 15-25%). Our wedges
  are now concretely = "cheapen the 3rd-cumulant correction ~10×": **low-rank 3rd cumulant (eff. rank ~2.7) is
  the top lever**, then on/off/kink, learned corrector, layer-adaptive/partial k=3, resummation.
- Phase-2 rules verified (#18197): ALL compute must be in flopscope (no numpy/scipy/torch/threads/FFI at inference);
  data files (shipped weights) OK. Prizes = private re-run of one designated submission (build to GENERALIZE).
- NEXT: prototype low-rank 3rd cumulant (torch) — accuracy vs full k=3 and cost — then port affordable estimator to fnp.

---

## 2026-09-16 (evening) — Deep research pass (competitor + literature review)
Read the corpus in depth → [RESEARCH.md](RESEARCH.md), refined [PLAN.md](PLAN.md). Game-changers:
- **Prizes = PRIVATE re-run** of a designated submission, not the public board → the 2.5e-9 top is partly
  public-overfit and won't carry. **Build to generalize.**
- **The wall (from #326024, the strongest write-up):** mean-prop is a unit-gain feedback loop; **"accuracy ⊥
  stability"** — naive-more-accurate closures diverge with depth; three exits (unbiased/contracting/truth-inject)
  measured closed *at depth 32*. ~99% of error is mean/variance TRANSPORT (final non-Gaussianity ~2%). Score
  bounded by analytic-anchor bias; input-space variance reduction "doesn't survive fusion".
- **Phase-2 tailwind:** depth 32→16 halves the wall (k=3 still ~8× at depth 16); width 1024 strengthens wide-theory;
  **ARC's MIT code `mlp_kprop` already beats the high-order memory wall** (harmonic/diagonal-slice compact tensors).
- Mapped ~20 DEAD ENDS to avoid (in RESEARCH.md). Companion paper arXiv:2605.05179.
- **Thesis:** resummed cumulant prop + on/off/kink + low-rank backward-subspace correction, as a CV anchor.
  Wedges each unexplored/organizer-endorsed (resummation, low-rank).
- **Next:** verify Phase-2 rule changes; clone `mlp_kprop`; **experiment #1 = run k=2/3/4 cumulant prop at 1024×16
  on baked GT** vs our 4.06e-6, to see the free base before innovating.

---

## 2026-09-16 (pm) — Leaderboard reality check → METHOD PIVOT
Looked at the live Phase 2 leaderboard (955 participants, 21.6k submissions, top updated hours ago). Hard truth:
- Top `adjusted_final_layer_score` = **2.5e-9** (final_layer_mse ~1.7e-8, ~13–15% budget, ~450× vs sampling).
  Big **shared-method cluster** at 5.4e-9 (final 2.13e-8 repeated across ~13 users) — a copy-able known solution.
- OUR covariance+tetrachoric estimator (adjusted ~4.05e-7) is **~160× off the top**, ~75× off the cluster,
  and would rank **~#120 / 164** — below the copy-the-known-method tier. Raw-MSE gap ~240×.
- This is a **method-class gap, not tuning.** Pure structural moment propagation (vectors A/B) caps ~200× short.
- Frontier method (confirmed via Phase-1 write-ups on the forum): analytic moment/**cumulant** (Hermite) propagation
  used as a **control variate** + **randomized quasi-Monte-Carlo (RQMC)** sampling of the residual → ~1000×
  variance reduction over plain MC. Variants: "intra-network control variate", "MC-prefix / analytic-suffix hybrid".
- Good news: (1) methods are **public** (Algorithmic-prize write-ups); (2) our tetrachoric/cumulant work is exactly
  the control-variate **anchor** — reusable, not wasted; (3) Phase 2 regime change (1024×16, residual *capped* not
  priced) means some Phase-1 tricks won't transfer → room for a Phase-2-specific method (paper angle).

### Pivot
- Primary approach is now the **hybrid control-variate + RQMC** (was "vector C"); the Hermite/cumulant closure (A/B)
  becomes its anchor, not a standalone.
- Next: pull 2–3 top Phase-1 write-ups, extract the recipe, port to Phase 2, reproduce the ~5e-9 cluster, then push.
- Target: **Algorithmic Contribution prize + top-10** (realistic); not #1 outright.
- **Confirmed on baked GT** (`experiments/2026-09-16_baked-gt/`): our K=1 estimator scores **adjusted 4.058e-7**
  on real 1024×16 (final 4.058e-6, all-layers 2.150e-6, 2.35% budget). Tetrachoric ≈ neutral at 1024 (matches
  the width-sweep extrapolation). Would rank **~#120–130 / 164**. final>all-layers → bias concentrates at depth.
- Research-grounded brainstorm done: the field's wall = analytic-anchor mean-bias (sampling capped). Three
  unexploited angles → **self-consistent (fixed-point) cumulant closure + 1/width (RMT) leading term +
  offline spectral learned-debiaser, used as a control-variate anchor**. Full landscape/plan discussed in chat;
  fold into PLAN.md once the direction is locked. Next: read top write-ups (45pp PDF #326024 + 2–3) to verify novelty.

---

## 2026-09-16 — Vector A (tetrachoric covariance) implemented + width-sweep finding
- Derived + **verified** the tetrachoric/Hermite series for the ReLU covariance (PLAN.md, vector A).
  Key insight: the existing gain method IS the n=0 term. Implemented in `src/estimator.py` as `n_corr_terms`.
- Probe (`experiments/2026-09-13_tetrachoric-probe/`, RESULTS.md): high-sample MC ref at reduced width,
  no dataset. Paired ΔMSE vs baseline (negative = helps); K=1 captures ~all of it (K=2,3 add <1e-8):
    - width 256: **+1.27e-6** (HURTS — anti-CLT, non-Gaussianity dominates)
    - width 512: **−1.76e-6** (helps ~8%)
    - width 768: **−0.42e-6** (helps, ≈ local noise floor)
  → The correction **flips from hurting to helping** as width → competition regime. Strong evidence it
    helps at 1024, but the benefit shrinks with width → likely a *modest* gain at 1024, not a knockout.
- Set default `n_corr_terms=1` (data-driven). Ceiling of A = exact moments UNDER the Gaussian
  assumption; beyond that needs vector B (non-Gaussian / skew correction).

### Revised next actions
- [ ] **Critical path:** enable baked 1e9 GT at 1024×16 (Stage-3 dataset) to (a) confirm A's real gain
      and (b) read per-layer `all_layers_mse` to choose the next lever. Local MC can't resolve ~1e-7 at 1024 cheaply.
- [ ] Then decide A-only vs A+B; start the writeup — the sign-flip-with-width result is a clean paper figure.
- [ ] Re-verify Phase 2 deadline (~17 Oct) + Algorithmic Contribution prize rules on AIcrowd.

---

## 2026-09-12 — Beachhead set up
- Read the full starter-kit spec (problem setup, scoring, estimator contract, algorithm ideas,
  code patterns, covariance + mean examples). Distilled into [SPEC.md](SPEC.md).
- Confirmed toolchain: `uv 0.11.14` (arm64), starter-kit `.venv` already synced. M2-class Mac, no GPU needed.
- Smoke test passed: `examples/02_mean_propagation.py` runs; flopscope counts 86,639,616 FLOPs (matches docs).
- Established the target: beat **covariance propagation** `final_layer_mse = 4.05e-6` (only bundled
  example that beats sampling). Strategy + ranked attack vectors in [PLAN.md](PLAN.md).
- Scaffolded project: `src/estimator.py` (working copy = covariance baseline), `notes/`, `experiments/`.
- git init at `projects/arc-white-box`; `external/` (vendored upstream, own git) is gitignored.

### Next actions
- [ ] Stage 3 authoritative eval: reproduce covariance 4.05e-6 on `mini` dataset; log per-layer error.
      (Downloads public HF dataset `arc-whestbench-public-2026@v2-phase2` — ~confirm before first run.)
- [ ] Decide direction: leaderboard-max (hybrid C) vs paper-max (exact bivariate ReLU, A). See PLAN.md.
- [ ] Re-verify Phase 2 deadline + Algorithmic Contribution prize rules on AIcrowd.
