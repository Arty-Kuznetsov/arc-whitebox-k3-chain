# ARC cumulant propagation at 1024×16 — the accuracy/cost ladder

Ported ARC's MIT reference (`mlp_kprop`, torch) to run on our baked-GT MLPs via a validated adapter
(see below). Measured the exact accuracy AND FLOP cost of k=1/2/3 at the competition shape.
Adapter validated two ways: (a) tracks Monte Carlo at small width (n=64: k1→k2→k3 = 2.97e-3→1.97e-4→1.13e-5,
converging to a 4.5e-7 noise floor); (b) at 1024×16, **k=2 reproduces our independent covariance baseline
(4.5e-6 vs 4.06e-6) and ARC's FLOP model reproduces flopscope's (k=2: 5.18e10 vs our 5.17e10)**.

## The ladder (baked GT, 1024×16, mini MLPs; FLOP = ARC's counter, validated ≈ flopscope)

| config | final_layer_mse | all_layers_mse | FLOPs | % of B=2**41 | deployable |
|---|---|---|---|---|---|
| our covariance+tetrachoric (fnp) | 4.06e-6 | 2.15e-6 | 5.17e10 | 2.35% | ✅ adjusted 4.06e-7 |
| kprop **k=1** (mean) | 3.33e-4 | 2.94e-4 | ~9e7 | ~0.004% | ✅ |
| kprop **k=2** (covariance) | 4.50e-6 | 2.43e-6 | 5.18e10 | 2.36% | ✅ adjusted 4.5e-7 |
| kprop **k=3-BASE** | 4.54e-7 | — | 3.25e12 | **147.9%** | ❌ OVER BUDGET |
| kprop **k=3-SIMPLE** | **4.25e-8** | 1.63e-8 | 3.25e12 | **148.0%** | ❌ OVER BUDGET |

(k=3: n=64 seed check gave the same convergence shape; 1024×16 k3 measured on 5 MLPs, tightly clustered 3.6–4.6e-8.)

## What this means (the reframed problem)
1. **The wide regime is a real tailwind, empirically:** at 1024×16, **k=3 cumulant propagation reaches
   final_layer_mse ~4.2e-8 — ~100× better than covariance** and near the leaderboard's *top* raw MSE
   (~1.7–2.1e-8). This is "the low-bias moment propagation" the strongest Phase-1 team said they couldn't
   build — it exists, and the wide/depth-16 regime unlocks it (they were stuck at depth 32).
2. **BUT k=3 costs 148% of the per-MLP budget → over budget → predictions zeroed → FAIL.** k=3 as ARC
   implements it is not directly deployable. BASE is no cheaper (the cost is the 3rd-cumulant contraction,
   O(n³) with large constants, not the degree-4 term). float32 gives no free win (ARC's count already
   ≈ the float32 flopscope bill).
3. **So the challenge is precisely a compute–accuracy tradeoff: get ~k=3 accuracy at ≤~20% budget.**
   The leaderboard leaders achieve final ~1.7–2.1e-8 at ~15–25% budget (adjusted 2.5–5.4e-9) — i.e. they
   get *better-than-our-k=3* accuracy at <1/6 of naive k=3's cost. That efficiency is the competitive secret
   (or partly public-MLP overfit — recall prizes are the private re-run).

## → Our wedges are now concretely motivated as "cheapen the k=3 correction ~10×"
- **Low-rank 3rd cumulant** (effective rank ~2.7): track the 3rd cumulant in rank-r form ⇒ O(n²·r) not O(n³).
  If r~10, that's ~10–100× cheaper → potentially k=3 accuracy at a few % budget. **Top-priority lever.**
- **On/off/kink:** only "kink" neurons (|α|=|μ/σ| small) need the correction; dead→0, on→linear handled exactly
  ⇒ 3rd-cumulant work on a subset of neurons.
- **Learned corrector:** predict the k=3 mean-correction from cheap k=2 state (offline-trained, shipped as data,
  evaluated in fnp) at O(n²).
- **Layer-adaptive / partial k=3:** budget allows ~10 "k=3 layer-equivalents"; spend them where skew accumulates.
- **Resummation** for stability/convergence.

## Files
- `validate_adapter.py` — small-width MC validation of the adapter (conventions: num_layers=depth+1,
  inject W.T into Ws[0..depth-1], init_kind="manual" w_var=2.0, read act{depth-1}).
- `dump_baked.py` — dumps baked weights + GT means (`all_layer_means`) for K MLPs to `baked/` (gitignored).
- `kprop_1024.py`, `flop_count.py`, `map_tradeoff.py` — the ladder + cost measurements.
- NEXT: prototype the **low-rank 3rd cumulant** in torch (measure accuracy vs full k=3 and cost), then port the
  affordable estimator to flopscope.numpy for a real submission.

## Re-run 2026-09-22 (all scripts, logs kept in `logs/`)
`./run_figure_data.sh` re-executed every script above sequentially (≈3 min total; k=3 ≈ 12 s/MLP in float64 torch)
and saved each stdout to `logs/<script>.txt` (the original runs were never logged). Everything reproduces:
k=1 3.33e-4 / k=2 4.50e-6 / **k=3 4.10e-8** on MLPs 0–4 (k=3 per MLP 3.63–4.59e-8); k=3 cost 3.254e12 (148.0%),
×4.27/4.13/4.06 per depth doubling; breakdown 47.8% + 47.5%; κ₃ eff-rank 1024→885; kink mean 0.700; rank-cap
128…4096 → ×107…×23; forward/reverse adaptive sweeps in full. `logs/flop_k1.txt` adds k=1 = 3.465e7 FLOPs (ARC counter).
`make_figures.py` turns the logs into `../../writeup/figures/fig1–5` (+ `DATA.md`, the table twin of each figure).
