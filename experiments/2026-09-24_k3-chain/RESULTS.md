# k=3 chain phase (2026-09-23 night → 2026-09-24) — complete record

Goal set by the user on 2026-09-23: **top-10 with margin** (public #10 = 3.5e-9 adjusted, #1 = 2.10e-9 with raw
final MSE 1.64e-8). This folder holds everything from the move to the k=3 cumulant chain: our own implementation of
ARC's reference algorithm, the flopscope ports, ~70 ablations, and a Monte-Carlo oracle.

Unless stated otherwise: baked N=1e9 ground truth of the public `mini` split, MLPs 0–2 (weights in
`../2026-09-16_kprop-arc/baked/`), width 1024, depth 16, final-layer MSE averaged over the 3 MLPs.

---

## 0. Files

| file | what |
|---|---|
| `step_check.py` | explicit formulas of the reference SIMPLE k=3 nonlinear step, checked slice by slice against `mlp_kprop` (teacher forcing) — **all dense slices to 1e-15** |
| `k4_probe.py` | how the reference stores κ₄: one radial scalar per layer (see §1) |
| `chain.py` | **our torch implementation of the full chain in the "source" representation**, float64, with every switch used below (`run_chain`, `run_chain_tucker`). Self-check `python chain.py 128 6`: per-layer means vs `mlp_kprop` to 1e-16 |
| `ablate.py` / `ablate32.py` | driver: `python ablate.py <config> 3` → `logs/ablate_<config>.txt`. Config names below are keys of its `CONFIGS` dict |
| `inject.py` | exact (baked GT) means injected at layers 0..u → where the error is made |
| `old_vs_young.py` | how much κ₃ comes from old sources and whether it can be predicted from young ones |
| `mc_oracle.py` | **Monte-Carlo oracle**: per-neuron pre-activation mean, var, κ₃, κ₄ at every layer (+ post means). `runs/mc_oracle_0.npz` (1e7 samples, MLP 0, ~18 min on 3 threads) |
| `mc_oracle22.py` | MC oracle for pair statistics: covariance and κ₄[i,i,j,j] at chosen layers; `post` variant for post-ReLU. `runs/mc_oracle22_0.npz`, `runs/mc_oracle22post_0.npz` (4e6 samples) |
| `compare_oracle.py` | chain vs oracle per layer (mean, var, κ₃ diag, κ₄ diag) + readout-truncation check |
| `oracle_subst.py` | final MSE when oracle statistics are substituted into the chain |
| `k4_diag.py`, `k22_analyze.py`, `k22post.py`, `k22_step.py`, `k4level.py` | κ₄ diagnostics (§6) |
| `fs_vs_torch*.py` + `fs_compare*.py` | flopscope estimator vs torch chain, layer by layer (self-check of every port) |
| `profile_*.py`, `op_timing*.py`, `out_test.py`, `sym_debug*.py`, `mem_probe.py` | flopscope residual-time / memory / API probes (§4) |
| `runs/` | grader-format reports of the flopscope estimators (`k3b_*` … `k3h_*`), variant wrappers `v_*.py`, oracle npz (gitignored) |
| **added 2026-09-24 (sessions 3–4)** | |
| `mc_oracle31.py` | MC at layers 6/10/14: pre C, κ₃[iij], κ₄[iijj], κ₄[iiij]; post C, κ₃[iij], κ₄[iijj]; common-scale q stats → `runs/mc_oracle31_0.npz` (4e6) |
| `mc_oracle_all.py` | MC C and κ₃[iij] at ALL layers → `runs/mc_oracle_all_0.npz` (4e6; too noisy as a pair-slice oracle, NEGATIVE §50) |
| `mc_feed.py` + `feed_analyze.py` | MC κ₄→κ₃ feed contraction vs the scale-mixture closure (§8) |
| `mc_k3step.py` + `k3step_analyze.py` + `k3step_diagrams.py` | **κ₃ one-step test with exact inputs** and candidate diagrams (E3, triangle) (§9) |
| `k31_step.py` | one-step test of the post (2,2) κ₄ slice with MC inputs and candidate terms (§7) |
| `scale_mixture_test.py` | exact scale-mixture average vs first order (§7) |
| `k4_pattern.py`, `k4_pattern2.py` | what predicts the per-neuron κ₄ (σ⁴ etc.) and the readout MSE of each predictor |
| `compare_oracle2.py`, `k22_level.py`, `chain_vs_mc.py` | chain vs MC per layer for any config (κ₃/κ₄/var/D/C slopes and correlations) |
| `oracle_subst2.py`, `oracle_subst3.py` | oracle headroom on the k3e / k3f configs (exact var, κ₃, κ₄, C, D) |
| `old_tail.py`, `passd_closure.py` | the old-source tail vs the amplitude form (§8, §9) |
| `var_attrib.py` | variance error: diagonal vs off-diagonal channel, and input attribution (§9) |
| `k3_level_test.py` | κ₃ level rescaling test (NEGATIVE §43) |
| `k3d_check.py`, `k3e_check.py`, `k3f_check.py`, `k3f_fuse_check.py`, `k3g_check.py`, `k3h_check.py` | port self-checks (bitwise vs predecessor; vs torch at full thin rank) |
| `strassen_paths.py`, `mem_probe2.py`, `residual_profile.py`, `prof/*` | Strassen path agreement, RSS over consecutive predicts, residual profiling (sections, temporaries, views, batch shape, einsum), `prof/grader_telemetry.js` (read grader telemetry from the submission page) |

**`chain.py` `run_chain` switches** (all off by default = the faithful reference). The pre-existing ones are `window`,
`k4`, `corr`, `edge2`, `aug`, `tet`, `k4b`, `k4star`, `srank`, `dlr`, `oracle`, `k4k3`, `q22x`, `inject`. New ones:

| switch | effect |
|---|---|
| `c0d` (True or e.g. `"22"`, `"11 22"`) | edge×κ₃ term in Q(p,q), optionally per channel |
| `k4pat`, `k22pat`, `k4lvl` | σ⁴ κ₄ closure, (varvar + 2C0²) pair closure, O(n²) levels |
| `marg` | marginal coefficients |
| `k31pat` | κ₄ (3,1) closure |
| `k4feed="P2"` / `"P1P2"` | κ₄→κ₃ feed |
| `margleg` | corrected gating (harmful) |
| `e3`, `e3age` | E3 births |
| `k4bparts` | subset of κ₄ births: `pd`, `p22`, `sd`, `s22` |
| `shared_sub=(age, r, …)` | shared-subspace projection of old sources |
| `tail="amp"` / `"oracle"` / `"oracle_t"` / `"oracle_D"` (with `window`) | old-tail closures |
| `oracle` keys | `C_layers`, `C0_layers`, `D_layers`, `t_scale`, `D_scale` in addition to `k3`/`k4`/`var_layers` |
| **added 2026-09-24 evening → 2026-09-29** (k3i … k3n and the cost-boundary research; all off/neutral by default, both gates exact with them off: `ablate.py k4memo 3` = 3.7533e-08, `ablate.py k3l_t 10` = 2.3135e-08) | |
| `paths=("e3","e3t","k22","cc")`, `e3age`, `pathD` (`"full"`/`"merge"`), `pathsub`, `pathfold` | augmented **path births** as one extra leg T·Y per source (§10; k3i). `e3age=1`: the path leg lives one layer; `pathD="merge"` = production's cheaper (2,1) pairing; `pathfold` folds it into the tree leg (catastrophic, NEGATIVE §53) |
| `pathrank=k` | rank-k SVD of the path leg at birth — NEGATIVE §55 (dead on the real meter) |
| `pairx=("k22",)` | AUGMENT's dense pair diagram edge × κ₄(2,2) inside `nonlin_slices` |
| `deadcut=thr`, `deadC` | **dead-ROW cut**: rows of neurons with α = μ/σ < thr zeroed in every leg after gating and in D (k3j −3.0, **k3m −2.5**); `deadC` also drops them from the next covariance product |
| `k31mult`, `feedmult`, `k22c`, `sdmult`, `k4scale`, `k4mult` | level multipliers: K31 closure (k3l 0.65), κ₄→κ₃ feed (k3n 0.8), C0² weight of the K22 pattern (1 = varvar + 2C0²; k3n 0), star-diagonal κ₄ birth, κ₄ births, per-layer κ₄ level test (§52) |
| `d21_ages`, `dlr`, `srank` | only the newest sources feed D; rank-dlr D (sensitivity); per-source low rank from a given age (NEGATIVE §22) |
| `ssp=(a0, r, q, over)`, `ss_iter`, `ss_over` | **band / shared subspace** of old sources (production mirror of k3k, NEGATIVE §54): sources of age ≥ a0 projected on a top-r Rayleigh–Ritz basis. `ssp=(a0, r, 0, 0)` = the **transported** band of `experiments/2026-09-29_band/` (E1: alive) |
| `varfix`, `varfix_late` | variance-diagonal corrections `diag(C) → diag(C)·g`, g an (L, n) array or `callable(l, v, mu_pre)`; early = before the source loop, late = after the oracle block, before `nonlin_slices` (`experiments/2026-09-26_variance/`, NEGATIVE §57) |
| `tailx=(a0, mode)` | **cloud, old-tail Stage A:** transform the tail's (t, D) contribution of sources older than a0 — `exact`, `dropD`, `dropT`, `rankD`, `diagD`, `scaleT`, `noiseT`, `scaleTD`, `noiseTD` (NEGATIVE §58) |
| `colcut=c` or `(lo, hi)` | **birth-COLUMN cut** at source birth, a_birth = μ/√diag(C): keep columns with lo ≤ α ≤ hi (or \|α\| ≤ c); masks B, ER, B2 columns and per-column dK3, alpha, g, d3, d1b, h, g3 — the column is removed exactly. **Dead side `(−2.25, inf)` + `deadcut −2.25` is the live matched cut** (`experiments/2026-09-29_colcut/`) |
| `satrow=thr` | saturated-row cut of the D pairing before `nonlin_slices`: D rows with α_now > thr zeroed (naive form; needs the κ₄ level restored — `experiments/2026-09-29_satrow/`) |
| `stride=(a0, k, mode)` | **jumped old tier**: sources ≥ a0 read exactly only at checkpoints (every k layers, first and last layer), in between (τ, Δ) pushed by the exact repeated-index rule; `mode="Donly"` keeps τ exact. `(4, 1)` reproduces the base to 2.2e-15. Closed: NEGATIVE §63 |
| (monkeypatch, not a switch) | the cloud's MST (`experiments/2026-09-29_mst/mst.py`) and state-closure (`experiments/2026-09-28_state-closure/`) patch `run_chain` / `read_slices` from outside; `chain.py` was not edited for them |

**The production configurations in torch** (keys of `ablate.py` `CONFIGS`): `k3l_t` = k3l (10 MLPs **2.3135e-8**, the
standing 10-MLP gate); k3m = `k3l_t` with `deadcut=-2.5` (**2.3462e-08** on the colcut base, see
`experiments/2026-09-29_colcut/pairs.py BASE`); k3n = k3m + `feedmult=0.8, k22c=0.0`.

**The k3f/k3h configuration in torch** is `ablate.py` `feedP2_mk` (3 MLPs 2.6983e-8, 10 MLPs 2.7989e-8).

Note: `.npz` oracle files are gitignored — regenerate with `mc_oracle.py 0 1e7` and
`mc_oracle22.py 0 4e6 3,7,11,14,15` (+ `... 6,10,13,14 post`).

---

## 1. The reference algorithm, made explicit (verified)

SIMPLE k=3 (Wu et al. 2605.05179 / `mlp_kprop` with `kind=SIMPLE, factor=True`) per layer:
- **Linear step:** μ ← Wᵀμ, C ← WᵀCW exact; κ₃ (factored) multiplied by Wᵀ on every factor; κ₄ = one radial scalar a4
  with metric set to the average E[WWᵀ] = 2I.
- **Nonlinear step (dense part):** single-index power moments `P_p = w(0,p) + w(3,p) t/6 + w(4,p) K4d/24` (p = 1..4),
  pair power cumulants `Q(p,q) = a1 b1 C0 + ½ a1 b2 Dᵀ + ½ a2 b1 D + ½ a2 b2 C0² + ¼ a2 b2 K22` with
  `a_k = w(k,p)_i, b_k = w(k,q)_j`, `w(k,p) = E[∂^k ReLU(Z)^p]`. Conversion (Leonov–Shiryaev):
  `K21 = Q21 − 2P1 Q11`, `K22 = Q22 − 2P1_j K21 − 2P1_i K21ᵀ − 2Q11² − 4P1_iP1_j Q11`, `K3/K4` diag from P's.
- **κ₄ radial projection:** `a4 = 3(Σ K4_iiii + Σ_{i≠j} K22_ij)/(n(n+2))`; read back next layer as
  κ₄[iiii] = 4·a4, κ₄[iijj] = 4·a4/3 (metric 2).
- **κ₃ (1,1,1) part:** pass-through (every factor gated by d1 = Φ(a)) + tree birth
  `Sym(U ⊗ 3I ⊗ U diag(d2))`, U = diag(d1)·C0 (C0 = pre-activation covariance with ZERO diagonal) + a correction block
  `Sym(V ⊗ I ⊗ I)`, `V = diag(ΔK3) + 3ΔK21ᵀ` that makes the repeated-index slices (3,), (2,1) exact.
All verified: `step_check.py` (1e-15 per slice), `chain.py` (means 1e-16), baked MLPs 0–2 → **4.2509e-8 exactly**
(config `full`).

**Source representation (ours):** per birth m: A = T (transport from birth, identity at birth), B = T·U,
E = T·V, hub g = d2. κ₃ = Sym(B ⊗ 3A ⊗ B diag g) + Sym(E ⊗ A ⊗ A). Readouts:
`t_i = 3 Σ_p g A B² + Σ_p E A²`; `D = L1 Bᵀ + L2 Aᵀ + L3 Eᵀ` with `L1 = 2A∘B∘g, L2 = B²∘g + ⅔E∘A, L3 = ⅓A²`.
Cost 6 products per source-layer (3 transports + 3 for D) ≈ 660 products ≈ 64% of B (the reference: 148%).

## 2. Accuracy experiments on the faithful chain (3 MLPs; `full` = 4.2509e-8, radial κ₄)

| config | what | final MSE | vs full |
|---|---|---|---|
| **k4memo** | κ₄ (2,2) matrix + diag, transported by direct pairings through W∘W | **3.7533e-8** | **1.13× better** |
| no_k4 | no κ₄ at all | 4.54e-7 | 10.7× worse |
| corr_diag | correction leg only ΔK3 (no ΔK21) | 1.74e-6 | 41× worse |
| corr_none | no correction leg | 2.80e-6 | 66× worse |
| J4_corrdiag | window 4 + diagonal-only correction | 2.63e-6 | — |
| J8 / J6 / J4 / J3 / J2 | keep only the newest J sources | 1.50e-7 / 3.51e-7 / 8.68e-7 / 1.26e-6 / 1.83e-6 | 3.5–43× worse |
| edge2 / k4memo_edge2 | + single-index κ₃² Edgeworth term (radial / memo22 base) | 4.37e-8 / 3.87e-8 | worse |
| aug / k4memo_aug | + dense AUGMENT pair diagrams (C0³, t·C0, C0·D, K4d·C0, C0·K22) | 6.59e-8 / 6.23e-8 | worse |
| tet3 / tet4 / tet6 | + ρ³…ρ⁶ Gaussian terms in every pair cumulant | 3.775e-8 | no gain |
| k4b1 / k4b1_nostar / k4b2 / k4b15 | + all-distinct κ₄ births (tree paths 4XC0Xᵀ, stars) from the newest 1/2/all sources | 3.93 / 3.71 / 4.13 / 4.62 e-8 | ≈ noise or worse |

**Error localisation** (`inject.py`, exact means injected at layers 0..u; `logs/inject_simple.txt`): 4.25e-8 total;
u=7 3.66e-8; u=11 2.45e-8; u=13 1.52e-8; u=14 8.0e-9. **The error is made by the per-step truncation at layers
12–15** (≈6e-9 per layer), not by early births.

## 3. Cost experiments on the faithful chain (base = k4memo 3.7533e-8 unless noted)

| config | idea | final MSE | verdict |
|---|---|---|---|
| tk_y1_r32 / y1_r64 / y1_r128 / y2_r64 / y2_r128 / y2_r256 / y3_r128 | Tucker: fold sources older than y into one shared basis P (n×r) + symmetric core G (r³) (radial κ₄ base 4.25e-8) | 2.04e-6 / 1.43e-6 / 6.91e-7 / 6.16e-7 / 2.75e-7 / 8.14e-8 / 1.38e-7 | fails for young folds |
| tk_y10_r128 / y8 / y6 / y6_r64 / y5 | fold only very old sources | 4.32e-8 / 4.76e-8 / 6.37e-8 / 8.57e-8 / 9.66e-8 | y≥10 lossless but saves ≤11% cost |
| d21y8 / y6 / y4 / y2 | (2,1) slice only from the newest y sources (diag from all) | 6.17e-8 / 1.37e-7 / 3.85e-7 / 1.02e-6 | old sources needed in D |
| lr8 / lr16 / lr32 / lr64 / lr128 (C0-tree) | rank-k ΔK21 in the correction leg | 1.10e-6 / 8.07e-7 / 4.21e-7 / 1.86e-7 / 2.03e-7 | fails (full-rank term inside) |
| fullC_none / fullC_diag | tree from FULL C, no / diagonal-only correction | 3.29e-7 / 3.16e-7 | fails |
| fcfull | full-C tree + exact correction | 3.7533e-8 | = base (self-check) |
| fclr16 / fclr64 | full-C tree + rank-k ΔK21′ | 4.59e-8 / 4.03e-8 | works partly |
| **freefull** | exact split ΔK21 = diag(α)C0diag(d1) + R, α = w(1,2) − 2P1d1 | 3.7533e-8 | = base (self-check) |
| freelr0 | drop R | 3.95e-7 | R needed |
| **freelr16 / freelr64** | R rank 16 / 64 (SVD) | **3.7742e-8 / 3.7581e-8** | **lossless** |
| **split16 / split16rr / split8rr / split24rr** | production form: E = A diag(ΔK3) + 3B diag(α) + thin T R_kᵀ (rr = randomized range finder, k+8 probes, 1 power step) | 3.7743e-8 / **3.7679e-8** / 3.793e-8 / 3.79e-8 | **lossless — shipped in k3c** |
| ulr1024 / ulr256 / ulr128 | rank-k tree leg U (correction exact) | 3.7533e-8 / 7.55e-8 / 9.87e-8 | B must be full rank |
| ulr256from8 / ulr128from11 / ulr256from4 | low-rank U only for deep births | 4.85e-8 / 4.10e-8 / 6.07e-8 | still lossy |
| dlr256 / dlr128 / dlr64 | replace D by its best rank-k approximation | 7.75e-8 / 1.36e-7 / 2.41e-7 | D is not low-rank |
| sr4_256 / sr4_128 / sr3_256 / sr6_128 / sr2_512 | per-source low rank (project A,B,E on top-r of [A,B]) from age a | 6.27e-8 / 1.75e-7 / 1.17e-7 / 6.64e-8 / 5.41e-8 | lossy |
| sp_sr4_256 / …A / …B | same in production form, projecting A+B / only A / only B | 5.50e-8 / 5.53e-8 / **4.06e-8** | **the hub leg A is the sensitive one** |
| old_vs_young.py | predict old-source κ₃ from young | at layer 15 sources older than 4 carry 69% of ‖t‖; corr with young 0.6–0.8 | cannot regenerate |

float32 check (`ablate32.py k4memo`): 3.7552e-8 vs 3.7533e-8 in float64 — float32 is fine.

## 4. flopscope ports

| estimator | design | result |
|---|---|---|
| `src/estimator_k3.py` | first port, per-source Python loops, batched Strassen with concatenate | **residual time 6.6 s** on the grader-format runner → MLPs zeroed. Superseded. |
| `src/estimator_k3b.py` | buffered: all legs in one preallocated buffer [B,E,A per source]; one batched call per layer; workspace Strassen (`_Strassen`, out= into quadrant views) | 100 MLPs: L0 2.5037e-8 @70.6%; **L2 1.9802e-8 @55.9% → submitted 331962, graded 1.915e-8 (rank 89)**; L3 1.879e-8 @50.1% (3 MLPs) → packaged `submission-20260923-082030.tar.gz`, NOT submitted (superseded by k3c) |
| `src/estimator_k3c.py` | k3b + thin correction legs (H n×24 transported, Ut fixed; α-part free) + Strassen on the base products | 3 MLPs: **L3 1.369e-8 @36.3% (wall 10 s, residual 0.15 s)**; L4/chunk 8 1.253e-8 @33.2% (wall 17.8 s, residual 0.24 s). **100 MLPs L4 (`runs/k3c_L4_n100.json`, `logs/k3c_L4_n100_summary.txt`): util 30.9%, 4 MLPs (4–7) FAILED on residual time (0.408/0.411/0.422/0.453 s, wall 30–33 s)** under CPU contention (MC oracles running in parallel) → failed MLPs score 1.0 and the split mean becomes 3.4e-2; L4 not safe. (This run started before `mm()` routed the base products through Strassen.) **100 MLPs L3 (`runs/k3c_L3_n100.json`, base products through Strassen): adjusted 1.2335e-8, final 3.5835e-8, all-layers 1.4100e-8, util 34.42%, 0 fails, residual mean 0.199 s / max 0.215 s, wall 12.1 s** (misses the ≤0.2 s criterion). A/B on 6 MLPs under load ≈2.6 (`runs/k3c_L3_ab6.json`, `runs/k3c_L3nb_ab6.json`): base Strassen on 1.2326e-8 @34.42%, residual 0.250/0.272 s; off (`base_strassen=False`) 1.3007e-8 @36.33%, residual 0.182/0.190 s. L0 re-check (`runs/k3c_L0_n3_recheck.json`): final MSE equal to L3 on MLPs 0–2 (the old `k3c_L0_n3.json`, 5e-6, predates the factor-3 fix); `strassen_paths.py`: L0/L3nb/L3 predictions differ by float32 noise only (final rms 1.2e-6, max 7e-6) |
| `src/estimator_k3d.py` | k3c + the 4 base products per layer as 2 paired Strassen products ([Wt, Wsq^T] @ [C, Xk], then @ (that)^T; C, Xk symmetric), layer 0 one product, last layer diagonals only; workspaces cached on the instance (no frees inside predict) | `k3d_check.py` (`logs/k3d_check.txt`): `batch_base=False` is **bitwise equal to k3c**, also with the workspace reused over MLPs A,B,A; batched: util 34.216% (k3c 34.421%), residual 0.185–0.205 s vs 0.25 s (same load), predictions differ from k3c by float32 noise (final rms 1.0e-6 = the L0-vs-L3 spread). **100 MLPs (`runs/k3d_L3_n100.json`): adjusted 1.2260e-8, final 3.5833e-8, all-layers 1.4099e-8, util 34.22%, 0 fails, residual mean 0.127 / max 0.137 s, wall 11.6 s** (per-MLP final MSE / k3c: 0.998–1.002). Packaged `submission-20260923-191338.tar.gz`; the packaged file via `--runner subprocess` (3 MLPs, `runs/k3d_pkg_subproc_n3.json`): bit-identical MSE, residual 0.17–0.19 s. RSS over 8 consecutive predicts on one instance (`mem_probe2.py`, `logs/mem_probe2_k3d_8.txt`): 2.95 → 3.49 GB, growth saturating (k3c: 2.96 → 3.40 GB over 3) |

**flopscope rules learned (all measured):**
- Arrays are **immutable**: no `x[i] = ...`; write through `out=` (works for matmul, multiply, add, subtract, sum).
- `fnp.zeros` is tagged symmetric; `out=` into its slices raises "out symmetry does not match". Use `fnp.empty`.
- `as_symmetric` validates with a tolerance (fails on 1e-6 float asymmetry) — symmetrise `0.5(X+Xᵀ)` first; its
  `isclose` costs time.
- **Residual (untracked) time comes from:** concatenating large non-contiguous views (~7 ms per call), `stack`
  (~0.8 ms per 12×4 MB), freeing arrays of hundreds of MB (~6 ms per 384 MB), and Python overhead per call
  (~10–20 µs). Fix: preallocated buffers + out=, O(1) calls per layer, contiguous copies (`fnp.copy`) before concat.
- Batched matmul with broadcasting `(1,h,h)@(S,h,h)` is billed exactly S·h²(2h−1); `swapaxes` is free.
- The grader runs **2.9× slower than local in wall time** (331962: 25.8–29.4 s vs 9.35 s) but its residual share was
  0% (≤0.14 s): residual ≈ local. The **cap is 0.4 s residual and 120 s wall**; local residual must stay ≲0.2 s
  *under load*.
- Estimator peak RSS: k3b L2 2.2 GB, L3 3.2 GB, L4/chunk 8 3.1 GB (the runner's own 10.5 GB is the dataset).
- **Residual per section, k3c L3** (`prof/run_sections.py`, instrumented copy; ms per MLP, load ≈2.5): base C products
  33–36, κ₄ block 37–45 (≈30 of it the base Strassen), `_nonlin` 47–67, leg transport 28, left+paired 26–30, birth
  15–17, **frame teardown after `return` 39–45** (freeing the ≈2.5 GB of workspaces; L0: 10). L0 total 0.12 s.
- Residual scales with the **number of flopscope calls** (≈12–28 µs each, incl. view slicing), not with temporaries:
  an n×n temporary costs only ≈0.01 ms more than `out=` (`prof/temp_cost.py`). A Strassen L3 call on 1–2 matrices
  ≈0.9 ms residual, on a leg chunk ≈1.5 ms.
- `out=` with **symmetric-tagged inputs** (e.g. `fnp.ones`) costs ≈5 ms of flopscope *overhead* per n×n op (not
  residual; only wall).
- The grader keeps one estimator instance per worker for many MLPs (`subprocess_worker.py` loop; FAQ: instance state
  is an allowed optimisation if every predict is correct on its own) → caching workspaces removes the frees.

## 5. Monte-Carlo oracle (MLP 0)

`compare_oracle.py` (`logs/compare_oracle_0.txt`):
- **Readout with EXACT per-neuron moments** (layer 15, MSE vs MC post-mean): Gaussian 1.78e-7; +κ₃ 2.18e-8;
  **+κ₃+κ₄ 9.7e-11**; +κ₃² 8.1e-11. The one-step Edgeworth readout is essentially exact once κ₄ᵢᵢᵢᵢ is right.
- **Chain vs MC (per layer):** pre-activation mean rel. err 2e-4; variance slope 1.0006, rel 1.9e-3 at layer 15
  (MC noise 4.5e-4); **κ₃ diag corr 0.998, slope 1.00, rel 6.8% (noise 0.9%)**; **κ₄ diag corr 0.23–0.53, slope 0.98 →
  0.54 from layer 2 to 15**.
- **Substitution** (`oracle_subst.py` → `logs/oracle_subst_0.txt`; baked GT, MLP 0, base = split24rr+memo22 3.5035e-8): κ₄ exact at last layer 3.27e-8; var exact at last
  3.25e-8; κ₃ exact at last 3.17e-8; κ₃+κ₄+var at last 2.72e-8; **κ₄ exact at all layers 2.23e-8 (×1.57)**;
  **κ₃+κ₄ exact at all layers 1.67e-8 (×2.1)** — the leaders' raw level.

## 6. κ₄ diagnostics — what is wrong with κ₄ and why

- **Models vs MC κ₄ diag** (`k4_diag.py` → `logs/k4_diag.txt`, `logs/k4_diag_k3xedge.txt`; corr/slope at layer 15): radial 0.00/0.53; memo22 0.23/0.54;
  memo22 + tree paths 0.34/0.55; + stars −0.23/0.48 (stars formula wrong or harmful);
  **memo22 + κ₃×edge (d₂d₁³κ₃C) 0.53/0.59**, + paths 0.58/0.60. κ₃×edge is a real mechanism (corr at layer 13 rises
  0.24 → 0.67) but does not fix the level. Formula: `Δκ4[iiii] = 12 Σ_a W_ai d2_a K_ai (C0 D1 W)_ai`,
  `K_ai = κ3_pre[a, v_i, v_i]`, v_i = D1 W[:,i], computed from current and next-layer legs (≈3 extra products/source
  in production — not yet ported).
- **(2,2) slice** (`k22_analyze.py` → `logs/k22_analyze_0.txt`): MC κ₄ diag mean = **3.0 × mean κ₄[iijj]** at every layer (radial-like level);
  the chain's pre (2,2) mean vs MC: 1.00 (l3), 0.94 (l7), 0.84 (l11), 0.72 (l14), 0.70 (l15). Pair variation is
  ~30% explained by C0² (R² 0.30 at l15), NOT by C (C is sign-random; 504aldo's fitted "κ₄[iijj] ∝ C_ij" is not what
  the MC shows).
- **Post-ReLU (2,2)** (`k22post.py` → `logs/k22post_0.txt`, incl. q22x and tet variants): chain/MC 0.938 (l6), 0.841 (l10), 0.714 (l13), 0.687 (l14), corr 0.89–0.99 → the
  deficit lives in the NONLINEAR step (transport is unbiased: non-direct pairings average to zero over the fresh W).
  κ₃² pair diagrams (D·Dᵀ, D², t·D) and ρ⁴…ρ⁶ do not move it.
- **One-step decomposition** (`k22_step.py` → `logs/k22_step_14.txt`, layer 14): with MC pre-activation κ₄ inputs the step gives **95.5% of
  MC (corr 0.978)**; with the chain's own inputs 68.7%. The step loses ~4.5% per layer and it compounds
  (0.955^10 ≈ 0.63). Residual best explained by Φ_aΦ_b·K22_pre (coef ≈0.06): the κ₄ pass-through is ~6% weak.
  Interpretation [U]: pre-activations are near a **Gaussian scale mixture** (z | y ~ N(·, Σ_a W²y²), common scale
  s); with nonzero means E[y²|s] is nonlinear in s, so κ₃(s) (radial κ₆) and Var(s)² terms enter at second order;
  v = Var(s)/s̄² ≈ κ₄/(3σ⁴) ≈ 0.022 at layer 15.
- `k4level.py` (→ `logs/k4level_0.txt`): the MC/chain ratio of the mean κ₄ diag level per layer is 1.00 (l0–4) →
  1.05 (l6) → 1.12 (l9) → 1.26 (l12) → 1.44 (l15). Rescaling κ₄ by these ratios every layer is **unstable**:
  iteration 0 gives 6.3e-7, iteration 1 7.7e-8, iteration 2 2.6e-5, with residual ratios oscillating. κ₄ is in a
  feedback loop, so this was discarded as a test.

## 7. κ₄ phase 2 (2026-09-24, after midnight): the missing diagram and the scale-mixture closure

**MC data** (`mc_oracle31.py 0 4e6 6,10,14` → `runs/mc_oracle31_0.npz`, `logs/mc_oracle31_0.txt`, 222 s): pre-activation
C, κ₃[iij], κ₄[iijj], κ₄[iiij] (the (3,1) slice), post-activation C, κ₃[iij], κ₄[iijj], and the common-scale statistic
q = (1/n)Σᵢ dzᵢ²/varᵢ: Var(q) 0.025/0.038/0.053, κ₃(q) 0.0031/0.0084/0.0202 at layers 6/10/14.

**One-step test** (`k31_step.py` → `logs/k31_step_0.txt`, `_0b`, `_0c`; MC inputs = MC μ, C, t, K4d, K22; self-check vs
`chain.nonlin_slices` 0.0e+00). Post (2,2) pair-mean ratio step/MC (corr):

| inputs / added term | layer 6 | layer 10 | layer 14 |
|---|---|---|---|
| chain's own inputs | 0.941 (0.987) | 0.844 (0.957) | 0.686 (0.888) |
| MC inputs, chain D (reproduces `k22_step.py`) | 0.977 | 0.965 | **0.955** (0.978) |
| + MC D | 0.976 | 0.963 | 0.956 (0.974) |
| + marginal Edgeworth corr. of pair coefficients (t, K4d) | 0.969 | 0.956 | 0.951 — wrong way |
| + κ₄ (3,1) slice (MC) | 0.979 | 0.966 | 0.959 |
| + K22² / + C0·K22, C0²·K22 | 0.976 / 0.977 | 0.963 / 0.963 | 0.956 / 0.956 |
| + ρ³…ρ⁶ (tet6) | — | 0.963 | 0.956 |
| + t·C0 / + κ₃² D-blocks | — | 0.960 / 0.960 | 0.953 / 0.951 |
| **+ C0·D: ½(a2b3 Dᵀ + a3b2 D)∘C0 in every Q** | **0.999** | **0.999** (0.999) | **1.004** (0.999) |
| MC inputs with the chain's D + C0·D | 1.000 | 1.003 | 1.006 |
| chain's own inputs + C0·D | 0.964 | 0.882 | 0.737 |

K11 (slope 1.000, relerr 0.3%) and K21 (slope 1.001) are exact with MC inputs with or without C0·D. **The missing
per-step term is the second-order diagram edge × κ₃(2,1) slice** (it was inside AUGMENT, §26, together with t·C0 and
K4d·C0, which bias the covariance). The chain's D is accurate (corr with MC D 0.995–0.997, slope 0.98–0.99).

**Pure scale-mixture hypothesis (FRONTIER A1) — falsified by sign** (`scale_mixture_test.py` → `logs/scale_mixture_test_0.txt`).
v from q (minus its Gaussian part 2Σρ²/n²) = 0.0119/0.0172/0.0215 equals v from K22/(varᵢvarⱼ) = 0.0119/0.0171/0.0214
(the pre (2,2) level IS radial); scale skewness 1.9/3.0/4.8. Exact scale-mixture averaging / first order in v =
0.996/0.986/0.977 — it LOWERS the post (2,2), while the measured deficit needs ×1.025/1.039/1.046.

**In the chain** (`ablate.py`, 3 MLPs unless noted; self-check `k4memo` = 3.7533e-8 reproduced):
- `k4memo_c0d` (all Q) 4.6428e-8 ✗; by channel: Q22 only 3.7595e-8 (0), **Q11 only 3.6494e-8**, Q21 only 4.9784e-8 ✗.
- Why the level fix alone does not help (`k22_level.py` → `logs/k22_level_0*.txt`, `compare_oracle2.py`): with C0·D the
  chain's pre K22 level goes 0.70–0.72 → **0.91–0.92 of MC** at layers 14–15, κ₃ unchanged (corr 0.998), but the
  readout error caused by the chain's κ₄ GROWS (layer 15: 1.87e-9 → 4.2e-9): the chain's per-neuron κ₄ pattern is
  wrong (corr with MC 0.23–0.53), so a higher level adds error. (Q22-only moves the post (2,2) by just 1%: the one-step
  gain comes through the Leonov–Shiryaev terms −2P1·K21, −2Q11².)
- **The κ₄ pattern** (`k4_pattern.py` → `logs/k4_pattern_0.txt`): MC κ₄ᵢ correlates with **varᵢ²** 0.67 → 0.86 (layers
  2 → 15; MC noise/cv 0.04–0.5), the chain's memo22 κ₄ᵢ only 0.23–0.53. Readout error with 3v·varᵢ² at the best
  level 1.1e-9 vs the chain's 1.9–2.0e-9 at layers 10–15. Mechanism: memo22 transports through W∘W (direct pairings)
  and loses the signed structure; for a scale mixture κ₄ᵢ = 3v·(Wᵀ C̃ W)ᵢᵢ², i.e. the square of the pre-variance.
- **Scale-mixture closure** `k4pat` (K4dᵢ = 3v·varᵢ², level from memo22) + `k22pat` (K22ᵢⱼ = v(varᵢvarⱼ + 2C0ᵢⱼ²)):
  3 MLPs: k4pat 3.5125e-8, k4pat+c0d 3.1406e-8, k4pat+c0d11 3.6105e-8, k4pat+k22pat 3.2597e-8,
  **k4pat+k22pat+c0d 3.0522e-8** (all three MLPs better: 3.36→2.84, 3.99→3.23, 3.91→3.08).
- **10 MLPs:** k4memo 3.5198e-8 → **k4pat+k22pat+c0d 3.0975e-8 (×1.136)**; production-like split24rr 3.5550e-8 →
  **3.1496e-8 (×1.129)**; per-MLP ratios 0.79–1.09 (9/10 better; MLP 4 worse by 7–9%). c0d without the Q21 channel:
  3.1728e-8. **(3,1) closure K31 = 3v varᵢ C0ᵢⱼ: harmful** (4.9079e-8 with the closure, 4.0593e-8 alone).
- **O(n²) levels** (`k4lvl`): sum(M) = uᵀXu exactly (u = row sums of W∘W), tr(M) with the mean-field Gₐᵦ = uₐuᵦ/n off
  the diagonal and the exact Gₐₐ = Σᵢ W⁴ₐᵢ: 3.0974e-8 (vs 3.0975e-8 exact levels); split24rr 3.1496e-8 (identical)
  → the two dense κ₄ products per layer are not needed.

**flopscope port `src/estimator_k3e.py`** (`k3e_check.py` → `logs/k3e_check_fs.txt`, `logs/k3e_check_torch.txt`):
`closure=False, c0d=False` is bitwise equal to k3d (1024×16); on n=128 depth 6 at full thin rank the flopscope outputs
match `chain.py` (freefull + the same switches) to 2–5e-7 while the changes themselves are 1e-4–1.8e-3. Cost
**32.30% of B** (k3d 34.22%), residual unchanged (0.178 vs 0.176 s, direct harness).

**k3e on 100 MLPs** (`runs/k3e_L3_n100.json`, machine load 7 → 2.4 during the run): **adjusted 9.7622e-9**, final
3.0224e-8, all-layers 1.1299e-8, util 32.30%, 0 fails, residual mean 0.150 / p95 0.166 / max 0.168 s, wall 11.1 s.
Per-MLP final MSE k3e/k3d: mean 0.847, median 0.838, range 0.655–1.179, **96/100 better**; all-layers ratio 0.803.
RSS over 5 predicts (`logs/mem_probe2_k3e_5.txt`) 3.13 → 3.60 GB, saturating. Packaged
`submission-20260923-201554.tar.gz` (file identical to src, sha256 daaf4970…); `--runner subprocess`, 3 MLPs
(`runs/k3e_pkg_subproc_n3.json`): bit-identical MSE, 0 fails, residual 0.170/0.214/0.194 s.
Residual cost model (`prof/view_cost.py`): each flopscope op ≈11.7 µs of residual (plus ≈80 µs of flopscope overhead);
a view (slice) ≈1 µs. Residual ≈ 12 µs × number of ops.

## 8. Phase 3 (2026-09-24 day): what limits κ₃/variance, the old tail, and the rest of the scale-mixture set → k3f

**Grader:** 332066 (k3e) = **9.562e-9** public (final 2.960e-8, all-layers 1.11e-8, 32.30%, 50/50, 0 failures;
private 50 also completed), grader wall 34–37 s/MLP (3.1× local). Public rank **82** (#1 2.10e-9, #10 ≈3.1e-9; a
19-entry cluster at 5.4e-9 / raw 2.13e-8 = 504aldo's public MIT chain).

**Old-source tail vs the amplitude mode** (`old_tail.py` → `logs/old_tail_0.txt`; chain = k3e config, corr freefull).
ReLU nets are positively homogeneous, so a common amplitude z = a·u passes every layer; it gives
κ₃ᵢ = 1.5 v μᵢ varᵢ and κ₃[iij] = (v/2)(2μᵢC0ᵢⱼ + μⱼ varᵢ). The κ₃ diagonal carried by sources older than 3 layers
(66–79% of |t| at layers 11–14) is fitted by μᵢvarᵢ alone with R² 0.91–0.93 (young-t regressor 0.33–0.47);
the old (2,1) slice by the amplitude form with R² 0.85–0.88; coefficient ratio D/t = 1/3 as predicted; the old
fraction of the amplitude (0.47 → 0.82) tracks "amplitude variance born before l−K". **But not usable**: window K
+ absorbed tail (`tail="amp"`) K=3 6.0e-7, K=4 4.2e-7, K=6 2.2e-7 (no tail K=3 1.27e-6; full 3.09e-8), and even the
per-layer ORACLE fit (`tail="oracle"`) K=3 5.2e-7 (t only 3.4e-7, D only 2.3e-7), K=6 2.1e-7.

**Oracle headroom on the k3e configuration** (`oracle_subst2.py` → `logs/oracle_subst2_0.txt`, MLP 0, chain 2.90e-8):
exact κ₄ 2.29e-8, exact κ₃ 2.29e-8, κ₃+κ₄ 1.49e-8, κ₃+κ₄ at layers 8–15 1.65e-8, **var exact 2.18e-8, var+κ₄ 1.50e-8,
var+κ₃ 1.26e-8, var+κ₃+κ₄ 4.79e-9**, all three at the last layer only 2.17e-8. The variance (0.19% rel. error) is
a lever as big as κ₃ or κ₄, and the three interact strongly.

**Covariance terms of the scale mixture** (one-step, `logs/k31_step_0.txt`: marginal corrections + MC K31 together
cut the K11 relerr 0.3% → 0.1% at layer 14; each alone does not). In the chain (`ablate.py`, 10 MLPs, base k3e =
split24rr + closure + c0d, 3.1496e-8): `k3e_marg` 2.9813e-8 (two MLPs worse), **`k3e_marg_k31` 2.9722e-8 (×1.06,
none worse than 1.003)**; channel Q11 only (marg11+k31_11, 3 MLPs) 3.1430e-8 worse.

**κ₄ → κ₃ feed** (all-distinct κ₃ births from the κ₄ (2,1,1) slice, Δκ₃[xyz] = ½ d2ₓ d1ᵧ d1_z κ₄[xxyz], with the
closure κ₄[aabc] = v(varₐC_bc + 2C_abC_ac): P1 = v × tree, P2 = Sym(h ⊗ D1C0D1), h = v/2 d2 var; J(D1C0D1)Jᵀ =
B diag(d1_birth) Aᵀ, so O(n²) in the source representation). 3 MLPs: P1 3.1088e-8, P2 2.9677e-8, P1P2 3.0075e-8.
MC check (`mc_feed.py` 4e6, `feed_analyze.py` → `logs/feed_analyze_0.txt`): the closure's contracted feed matches MC
(corr 0.973–0.979, slope 0.95–0.98); the feed is 6–9% of |t|; the chain's κ₃ error (rel 5–6%) does NOT correlate
with it (0.04 / 0.00) → the κ₃ error is elsewhere. **10 MLPs: `feedP2` 3.0638e-8; `feedP2_mk` (P2 + marg + K31)
2.7989e-8 (×1.125, all 10 better, worst 0.980)**; `feedP12_mk` 2.8468e-8.
Bug found by the port self-check: the torch `freefull` + feed branch first double-counted P2's birth-time (2,1)
slice (α moved before R); fixed; split-branch numbers (all 10-MLP runs) were correct.

**flopscope `src/estimator_k3f.py`** (k3e + marg + k31 + feed P2; `k3f_check.py` → `logs/k3f_check_*.txt`):
switches off → bitwise equal to k3e; vs torch at full thin rank 2–4e-7 for marg / k31 / feed / all (the changes:
1e-4–8e-4). Cost 32.37% of B. Residual trims (`k3f_fuse_check.py` → `logs/k3f_fuse_check.txt`): t/6 and K4d/24 once,
unused Wick coefficients skipped, and `Wt·C` riding the leg transport in the free newborn-A slot (−1 Strassen call
per layer) — all bitwise identical to the untrimmed k3f; residual 0.193–0.207 → 0.182–0.185 s (direct harness).

## 9. Phase 4 (2026-09-24 afternoon): the κ₃ step error, the variance channel, κ₄ births — after 332128

**Grader:** 332128 (k3f) = **8.286e-9** (final 2.560e-8, all-layers 9.75e-9, 50/50, 0 failures), ≈ rank 77.

**κ₃ one-step test with exact inputs** (`mc_k3step.py` 4e6 at layers 5/9/13 → `k3step_analyze.py`, `logs/k3step_analyze_0.txt`).
The next-layer diagonal t′ splits into two parts. The repeated-index part is measured from MC post K3/K21 (|R3| 0.43 /
0.31 / 0.31 against |t′| 0.45 / 0.32 / 0.32). The rest is the all-distinct part T (0.13 / 0.086 / 0.081). SIMPLE's
all-distinct terms on MC inputs are pass-through (E[u³] minus repeats: 0.110 / 0.075 / 0.075), tree (0.070 / 0.039 /
0.027) and feed P2 (0.005 / 0.004 / 0.003). They leave a **residual of 5.2% / 3.7% / 2.9% of |t′|**. The chain's own
error is 4.1% / 5.5% / 7.3%, and the correlation between residual and chain error is 0.43–0.45. MC noise ≈2e-3, well
below the residual (1e-2). **The κ₃ step itself is truncated.**
- Candidate diagrams (`k3step_diagrams.py`): **E3 = edge × κ₃(2,1) tree**, ½ d3ₐ d1_b d1_c C_ab κ₃[aac], contracted
  as 3Σₐ W′ₐᵢ d3ₐ[(C0W̃)ₐᵢ(D0W̃)ₐᵢ − Σ_b W̃²_bᵢC0ₐ_bD0ₐ_b]. It explains R² 0.09 / 0.27 / 0.33 of the residual with
  coef 0.92 / 0.90 / 0.87 (theory: 1). The transposed slice explains little; the Gaussian triangle is negligible.
- **E3 in the chain** (a source leg B2 = D1·Dᵀ at birth, hub ½d3; t += 3Σ d3·A∘B∘B2; three D pairings; its (2,1) birth
  slice is subtracted from dK21). 3 MLPs: full 2.552e-8 (×1.057), age ≤1 2.640e-8, ≤2 2.626e-8, ≤4 2.592e-8.
  It costs +2 products per source-layer (+50% budget) → **does not pay**.
- **All-distinct pass-through vs the amplitude form** (`passd_closure.py`): R² 0.09 / 0.22 / 0.39, coef ≈ v/2 at
  depth. A memoryless κ₃ chain with this closure would carry ~18% t error, so it is closed.

**Where the variance error comes from** (`var_attrib.py` → `logs/var_attrib_0.txt`). δvar_{l+1} splits into two
channels:
- diagonal Σ W′² δK2: norm 1.6e-3 / 1.0e-3 / 1.5e-3;
- off-diagonal covariance: norm **8.4e-3 / 8.6e-3 / 1.04e-2**.

So the variance error is the coherent transport of post-covariance (K11) errors. The step's K11 is exact on MC inputs,
so the K11 error comes from the chain's inputs to the first-order terms, mainly the κ₃ (2,1) slice D (corr 0.993 with
MC, i.e. ~12% pattern error) and κ₄. δK2 by input: var 0.8, t 0.3, K4 0.15 correlation at depth.

**Young κ₄ births on top of the σ⁴ closure** (`k4b`, newest source; 3 MLPs, base 2.6983e-8):

| births | 3 MLPs | cost |
|---|---|---|
| full | 2.5821e-8 | 3 products/layer |
| diagonal (paths + stars) | 2.5997e-8 | 1 product |
| **star diagonal only** | 2.6158e-8 | **0 products** |
| pairs only | 2.6730e-8 | |
| k4b=2 full | 2.5587e-8 | |

On 10 MLPs: star-diagonal 2.7512e-8 (**×1.017, free**, 8/10 better); diagonal 2.7348e-8 (×1.023, 1 product).

**Grader telemetry of 332128 (k3f), read from the submission page's data:** residual_wall_time_s mean 0.110, min 0.106,
max 0.116 over all 100 MLPs (5 workers; flopscope ops run on a "flopscope server"); participant predict wall ≈36 s.
**Grader residual ≈ 0.79 × local runner (0.139) ≈ 0.6 × subprocess runner (0.18–0.195), and very stable.**

**Deeper Strassen:**
- L4 with chunk 8 on k3f: util 29.04%, but local residual 0.25–0.30 s and wall 20 s. Not shipped.
- Two diagnostics:
  - ops with two quadrant views as inputs looked expensive in a microbenchmark (0.1–0.2 ms), but rewriting the
    combos with single-view inputs (`estimator_k3g.py`, bitwise identical) did not lower the residual;
  - the residual scales with op count at ≈20 µs per op, regardless of batch shape (`prof/batch_cost.py`).
- **k3h (`src/estimator_k3h.py`):** the D pairings need only Σ_k L_k R_kᵀ. Strassen is bilinear, so the leaf products
  are summed before recombination. The leaf is `matmul` into a scratch view of the free level L−2 region, then `sum`;
  flopscope's einsum bills the same but is 70× slower (`prof/einsum_speed.py`). Buffers are shared between modes
  (X1, X2 with chunk slots, X3 with one slot per level), so **L4 at chunk 16 = 2.7 GB of workspace with the same
  number of calls as L3**.
- **k3h L4 on 100 MLPs** (`runs/k3h_L4_n100.json`): **adjusted 7.6733e-9**, final 2.6549e-8 (per-MLP ratio to k3f
  0.997–1.003), **util 28.90%**, 0 fails, residual mean 0.173 / max 0.203 s (local, load ≈3.7), wall 17.1 s.
  RSS 3.71 → 3.95 GB over 4 predicts. Packaged `submission-20260924-092524.tar.gz`: file identical to src; subprocess
  3 MLPs bit-identical, residual 0.24–0.33 s (expected on the grader: ~0.14–0.20 s).

**Shared subspace for old sources** (`shared_sub`, torch, 3 MLPs, base k3f 2.6983e-8). The legs A, B, ER of sources
with age ≥ a are projected onto the top-r eigenvectors of the band's summed Gram, re-projected each layer.

| a : r | final MSE |
|---|---|
| 5 : 640 | 2.6985e-8 |
| 8 : 384 | 2.6985e-8 |
| 5 : 512 + 8 : 384 | 2.7026e-8 |
| **5 : 384** | **2.7328e-8 (+1.3%)** |
| 5 : 384 + 8 : 224 | 2.7616e-8 |
| 6 : 256 | 2.8621e-8 |
| 4 : 384 | 2.9180e-8 |
| 5 : 256 | 3.2866e-8 |
| 3 : 384 | 4.1477e-8 |
| 4 : 256 | 5.47e-8 |
| 5 : 192 | 5.47e-8 |

A shared basis works where per-source bases failed (NEGATIVE §21). Cost model [U]: an old source in a basis P
(n×r) costs ≈8n²r. That is materialising A and B (4n²r) plus pairings summed in the compressed space (4n²r),
≈1.5 products at r=384, against 2.44 at Strassen L4. The band also pays for transporting P and updating the basis
(≈1.4 products per layer). **Net for 5 : 384 ≈ −4.5% of B at +1.3% MSE ≈ ×1.16.**

## 10. Phase 5 (2026-09-24 evening): 332142, the leaders' telemetry, ARC's AUGMENT reference, path births

**Grader:** 332142 (k3h) = **7.401e-9** (final 2.560e-8, 28.91%), 100/100, residual 0.126–0.135 s, wall 55.7–60.4 s.

**Leaders' public telemetry** (`leaders/per_mlp_final.py`, `leaders/extract.js`): J2W 1.61e-8 @12.7% (kernel 93 s,
residual 0.33 s), suliman 1.63e-8 @13.8%, marius 1.56e-8 @14.7%, a_s6 2.00e-8 @12.6%, Luna 1.67e-8 @15.5% (kernel 38 s,
residual 0.15 s — our profile), lode_dockx 3.69e-8 @9.8% (only the final layer predicted). Per public MLP the leaders are
1.53–1.65× more accurate than us (min 1.22, max 2.0); corr of log errors with ours 0.67–0.79.

**ARC's own reference, `kind=AUGMENT, factor=True`** (`aug_ref.py`; 31 min and 40–55 GB per MLP because of the dense
n³ (2,1,1) slices): **MLP 0 final 2.0563e-8** (SIMPLE 3.7046e-8, our k3f 2.6608e-8). Per layer (GT floor ≈7e-10) AUGMENT
has ≈0.6× our excess error at layers 4–9 and 0.77–0.8× at 12–15 — its advantage starts EARLY.
- `AUGMENT_G` (AUGMENT with the path legs removed, κ₄ = Sym(G⊗M) kept): **5.69e-7** — AUGMENT only works as a consistent
  whole (the pK→K conversion of the (2,1,1) slice uses the full pK₁₁₁); not a valid ablation.
- Cost of the reference: 72n³L² vs SIMPLE's 30n³L² (paper, Table 1), i.e. ~2.4× SIMPLE.

**Path births in our chain** (`paths` switch; NEGATIVE §51): 3 MLPs on k3f: E3 2.7973e-8, all 2.7449e-8 (base 2.6983e-8).
Per layer on MLP 0 they give only a small early gain (layer 5: 4.13e-9 → 3.75e-9; AUGMENT 2.90e-9).

**κ₄ level ×f at every layer** (`k4mult`): 0.95 → 3.49e-8, 1.05 → 7.61e-8, 1.10 → 2.56e-7: compounding, current = optimum.

**Error structure** (`err_budget.py`, `err_structure.py`, `perlayer.py`): per-layer readout injections at depth: var
2.2–3.2e-9, κ₃ 1.6–2.0e-9, κ₄ 1.0–1.3e-9; the final error has no common mode and 1–4% R² on per-neuron features.
**Sparsity** (`sparsity_probe.py`): dead neurons (α<−3) 18% at layer 15; dropping them from contractions ≈ −12–20% of
per-source products.

**Continued (same evening): AUGMENT dissected, k3i.**
- AUGMENT on MLP 1: 1.9893e-8 (k3f 2.5780e-8) → the ×1.3 holds.
- `aug_vs_chain.py` (MLP 0, vs MC 1e7): AUGMENT's κ₃-diagonal readout injection is 6–10e-10 at every layer, which is the
  MC oracle's own noise floor (8.7e-10 at layer 0, where κ₃ ≡ 0); k3f 1.3–2.0e-9 at depth. Var rel. error at layer 15:
  AUGMENT 1.32e-3, k3f 1.87e-3. κ₄ diagonal: AUGMENT corr 0.87 / level 0.93, k3f 0.86 / 0.95 (no better).
  With all path births our κ₃ injection also drops to the noise floor (0.8–1.0e-9) and var to 1.53e-3.
- ARC's augmented term list (`get_all_terms_iso(3, 4, augment=True)`): for the pair slices (1,1), (2,1), (2,2) it adds
  C·C² (C³), t·C, K4d·C, C·D, C·K22 and the (3,1) slice to SIMPLE's C, C², D, K22. We had all but C³ and C·K22.
- 10 MLPs (torch, `ablate.py`), base k3f 2.7989e-8:

  | config | 10 MLPs | note |
  |---|---|---|
  | paths all ages (`k3f_pall`) | 2.5499e-8 | 8/10 better (0–2 was the unlucky subset) |
  | paths age ≤1 / ≤2 / ≤4 | 2.5748 / 2.5846 / 2.5679e-8 | first readout carries the gain |
  | paths rank-64 Y, all ages | 2.6766e-8 | low rank loses it |
  | e3t only | 2.8109e-8 | |
  | ≤1, t only / merged D / full D | 2.5817 / 2.5780 / 2.5748e-8 | |
  | ≤1 full + star κ₄ birth | 2.4788e-8 | star: ×1.039 here (×1.017 on k3f) |
  | ≤1 full + C0∘K22 | 2.5403e-8 | |
  | ≤1 full + star + C0∘K22 | 2.4495e-8 | |
  | k3f + star + C0∘K22 (no paths) | 2.7374e-8 | the paths are the key |
  | all ages full + star + C0∘K22 | 2.3607e-8 | ×1.186, but +2 products per source-layer |
  | **k3i** = ≤1 t-only + star + C0∘K22 | **2.4919e-8** | 10/10 better; +1 product/layer |
  | k3i without the Y∘U subtraction | 2.6383e-8 | keep the subtraction |
  | **k3i merge** (+ merged D) | 2.4664e-8 | same cost |
  | **k3i merge + C0³** (tet3) | **2.4441e-8** | ×1.145; = src/estimator_k3i.py v2 |
- flopscope `src/estimator_k3i.py` (`k3i_check.py`): all switches off → bitwise k3h (28.939%); vs torch at n=128,
  full thin rank: 2–5e-7 per layer while the changes are 1e-5–5e-4. Default: 29.855% of B.
- **100 public MLPs, k3i t-only version (`runs/k3i_n100.json`): adjusted 7.0233e-9, final 2.3561e-8, all-layers
  8.84e-9, util 29.81%, 0 fails, residual mean 0.176 / max 0.203 s, wall 17.4 s; per-MLP final / k3h: mean 0.888,
  98/100 better.** v2 (merge + C0³) running.
- **v2 on 100 MLPs (`runs/k3i_v2_n100.json`): adjusted 6.9242e-9, final 2.3221e-8, util 29.82%, 0 fails, residual
  0.199 / 0.245 s (load ≈5), 100/100 better than k3h.** Packaged `submission-20260924-135835.tar.gz` (sha256 235f6149…);
  `--runner subprocess` 3 MLPs bit-identical, residual 0.28–0.38 s.
- Path leg folded into the tree leg at age 2 (`pathfold`): 1.1530e-7 — catastrophic (NEGATIVE §53).
- **Dead-row cut** (`deadcut`, zero the rows of dead neurons α<thr in every leg after gating and in D; `deadC` also in
  the next covariance): 10 MLPs on k3i2 (2.4441e-8): thr −3 **2.4516e-8 (+0.3%)**, −3 + C 2.4619e-8, −2.5 2.4821e-8.
  Cost model on MLP 0's dead fractions (transports × alive(l−1), pairings × alive(l)²): **−13.5% of per-source products
  (≈ −12% of B, ×1.14 adjusted)** at −3. Needs rectangular Strassen with row compaction in production.
- **AUGMENT's κ₄ G vs our closure vs MC** (`compare_g.py`, MLP 0, MC 4e6 at layers 6/10/14; `aug_moments.py AUGMENT 0 G`):
  K31 corr AUGMENT 0.70/0.74/0.81 vs closure 0.66/0.70/0.79, but level AUGMENT **0.54** of MC vs closure 0.74–0.78;
  K22 corr 0.64/0.70/0.77 vs 0.61/0.67/0.76, level ≈0.97 both. MC C alone correlates 0.66–0.75 with MC K31.
  **The dense G is not better than the scale-mixture closure**, so AUGMENT's remaining edge over k3i is not its κ₄.
  Candidates left: path births at all ages (×1.04 in our chain; but on MLP 1 all-age paths hurt ours, 2.69e-8, while
  AUGMENT gets 1.99e-8 there), and consistency terms we have not identified.

## 11. Levels of the closure's off-diagonal κ₄ slices on k3j (2026-09-25; torch, 10 MLPs, base k3j_t 2.4516e-8)
| K31 multiplier | 0 | 0.3 | 0.5 | **0.6** | **0.7** | 0.8 | 1.0 | 1.3 |
|---|---|---|---|---|---|---|---|---|
| final MSE (e-8) | 2.7296 | 2.4299 | 2.3325 | **2.3147** | **2.3176** | 2.3414 | 2.4516 | 2.7749 |
At 0.7 all 10 MLPs are better or equal. The closure's K31 level is 0.74–0.78 of MC (`compare_g.py`); ×0.7 puts it at
≈0.53 of MC, the level ARC's consistent AUGMENT truncation produces (0.54) → read as "consistent truncation", not a free
fit. The κ₄→κ₃ feed level is at its optimum (×0.7 2.4626e-8, ×1.3 2.5057e-8). Chosen: K31 × 0.65 (→ k3l).
On the k3l base (K31 × 0.65; torch 10 MLPs **2.3135e-8**): K22 closure's C0² weight c (pattern only, level kept) 0 →
2.3061, 0.5 → 2.3095, 2 → 2.3229 (flat, keep 1); κ₄→κ₃ feed ×0.7 → 2.3042, ×1.3 → 2.3878; star κ₄ birth ×0.5 → 2.3139,
×2 → 2.3808. **No other level lever**: only K31 was off. Per MLP vs AUGMENT: MLP 0 2.4074 (AUG 2.0563), MLP 1 2.1572
(AUG 1.9893).
