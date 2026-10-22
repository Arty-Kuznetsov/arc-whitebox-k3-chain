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

## B. Today's cost-free ceiling (torch chain, float64, the 10 baked MLPs, COST IGNORED; `b_ceiling.py`, `resB/`)
Base = B9 + marg2 (#9A arm `marg2_sat_m2.0`, reproduced bitwise per MLP: 2.15516e-08). Levers refused on cost only.
