# A third-cumulant chain at its boundary: code, measurements and journal

This repository holds the code and the research record behind submission **333534** to the ARC White-Box Estimation
Challenge 2026 (Phase 2), by Artyom Kuznetsov (AIcrowd `Artyom_Mazur`), and behind the Algorithmic Contribution
write-up `writeup/v4.pdf`, posted on the challenge forum as topic 18249
(https://discourse.aicrowd.com/t/18249). Everything here is released under the MIT licence (see `LICENSE`).

The estimator predicts the per-neuron mean post-ReLU activation of a random He-initialised ReLU network (width 1024,
depth 16, Gaussian input) within a metered FLOP budget. It is a cumulant-propagation chain closed at the third
cumulant. Its skeleton is the SIMPLE k = 3 algorithm of Wu et al. (arXiv:2605.05179) and their MIT reference
implementation `mlp_cumulant_propagation`, re-derived and re-implemented for the `flopscope` meter. What we added is in
the write-up, §3, and every addition can be switched off to recover its predecessor bitwise.

## What is where

The layout is the one we worked in, because the scripts find each other by relative paths.

| path | contents |
|---|---|
| `src/` | every production estimator, oldest to newest: `estimator.py` (Part I, submission 331944) and `estimator_k3b.py` to `estimator_k3z.py`, plus the probe `estimator_k4a.py` (§7.3). **`estimator_k3z.py` is the designated submission 333534, byte for byte.** Each file stands alone and imports only `flopscope`, `whestbench` and the Python standard library. |
| `experiments/<date>_<topic>/` | one folder per measurement: a `RESULTS.md` with every number, the scripts that produced it and their small outputs. `2026-09-24_k3-chain/` holds the float64 research chain (`chain.py`, every term a switch; `ablate.py`, named configurations; `young4.py`, the young-κ₄ reader; `mc_oracle.py`, the per-neuron Monte-Carlo oracle). `2026-09-29_port-battery/pb.py` holds the configurations most later folders import. Oracle files (`*.npz`) and the baked networks are not included; the scripts and `scripts/fetch_baked.py` regenerate them. |
| `scripts/` | `setup.sh` (environments at the pinned versions, the data, a contract check), `fetch_baked.py` (the first public networks, rebuilt from the public dataset and checked by sha256), `external_pins.txt` (the exact upstream commits), `poll_sub.py` (reads a graded submission's public page and its per-network telemetry), `shape_grid.sh` and `prepkg_check.sh` (the checks we ran before every upload). |
| `notes/` | the research journal `WORKLOG.md` (dated, newest first), the catalogue of negative results `NEGATIVE-RESULTS.md` (N1–N76, each with its mechanism and its folder; the notes
and the `RESULTS.md` files cite entry 54 as §54, the write-up as N54), `METHOD.md` (the method as it evolved) and `SPEC.md` (rules, metric and billing as we read them). |
| `writeup/` | the write-up's section sources, the build script, the figure scripts, the figures and `v4.pdf`. |

## Where the analysis is

Most of the write-up is about why this family of estimators behaves as it does. The folders behind those results:

| result (write-up section) | folder |
|---|---|
| per-neuron errors are random projections of joint errors; odd orders have no coherent level (§6.4, Propositions 1–2) | `2026-10-02_cov-error`, `2026-10-02_parity`, `2026-10-07_randdir` |
| what a per-neuron correction can learn is set by the truncation order (§6.4) | `2026-10-13_coherence-law` |
| the share of κ₃ its two slices cannot carry, derived and checked on the true tensor (§6.2, Proposition 3) | `2026-10-13_prop3`, `2026-10-14_vm` (`prop3_truth.py`) |
| the skew law κ₃ᵢ ≈ c(l)·μᵢ, its first-layer constant and a closure that carries it (§6.4) | `2026-10-14_skew` |
| the slope as the partial trace's projection on the mean, the input radius as a scale mode, other activations (§6.4) | `2026-10-14_vm` |
| where the next order's accuracy lives; age-truncated k = 4 in ARC's own code (§7) | `2026-10-03_korder`, `2026-10-04_r1` … `_r3`, `2026-10-14_k4age` |

`scripts/error_anatomy.py` measures the coherent and learnable shares of any estimator of this challenge from its outputs
alone, on the ten public networks with their 10⁹-sample truth. `scripts/build_oracle_dataset.py` assembles the
Monte-Carlo oracles we measured against.

## How to reproduce

`scripts/setup.sh` clones the pinned upstream repositories into `external/`, builds the grading environment
(`.venv-whest`, whestbench 0.16.1 with flopscope 0.12.1) and the research environment (`.venv-torch`, torch and numpy
only), downloads the first ten public networks into `experiments/2026-09-16_kprop-arc/baked/` (about 1.1 GB) and runs
the contract check. It needs `uv` and git, and it never touches AIcrowd.

```bash
bash scripts/setup.sh                     # add --n-mlps 30 for the thirty networks some measurements use
```

The designated estimator, on the grader's own runner:

```bash
.venv-whest/bin/whest validate --estimator src/estimator_k3z.py
# all 100 public networks against the baked 1e9-sample ground truth; about 65 minutes on a laptop at 4 threads
.venv-whest/bin/whest run --estimator src/estimator_k3z.py \
    --dataset hf://aicrowd/arc-whestbench-public-2026@v2-phase2 --split mini --runner local --max-threads 4 --format json
```

FLOP counts are exact on any machine. Wall and residual time are not: the grader's numbers for every submission are on
its public page, and `python scripts/poll_sub.py 333534` prints them.

The research chain gates itself before anything is measured with it:

```bash
cd experiments/2026-09-24_k3-chain
PYTHONPATH=. ../../.venv-torch/bin/python ablate.py k4memo 3     # must print 3.7533e-08
PYTHONPATH=. ../../.venv-torch/bin/python ablate.py full 3       # must print 4.2509e-08, ARC's k = 3 reference
```

Every `RESULTS.md` names the script and the command behind its numbers, and the write-up's §10 lists where each
headline claim can be checked and how long it takes. Where a measurement used a Monte-Carlo oracle, the oracle's seed
and sample count are recorded next to it; `experiments/2026-09-24_k3-chain/mc_oracle.py` regenerates the per-neuron
oracles (about 18 minutes per 10⁷ samples), and each folder that used another oracle contains its generator.

## Credits

- Wu et al., *Estimating the expected activations of random neural networks* (arXiv:2605.05179), and the MIT reference
  implementation: the algorithm this chain is built on.
- 504aldo (AIcrowd forum topic 18218): an independent catalogue of the same walls; their post also named the κ₄ → κ₃
  channel. No code of theirs was used.
- jamesrahenry (forum topics 18097 and 18157): the coherence mechanism, first published in Phase 1.
- mliston (forum topic 18170): the idea of an offline-trained corrector of the chain's own moments.
- hyojun_kwon (forum topic 18154): accuracy against stability of an iterated closure, the precursor of our consistency rule.
- il.miscusi (forum topic 18248) and kai_leal_gonzalez (topic 18246) published related results on the scale mode and on a
  conditional skew identity independently, on 2026-10-15 (write-up §6.4). Susupro (topic 18254) measured the exact k = 4 at
  1024 × 16 and a corrector that reads through W, which extend our §7 and §6.4.

## Use of language models

The research was directed by the author. Claude Code (Anthropic) wrote the code and the drafts under that direction,
ran the experiments and kept the journal. The write-up's §9 gives the full account, including the errors the model
made. No number in the write-up or in this repository comes from a model's statement: each comes from a logged run.

## How to cite

Artyom Kuznetsov, *A third-cumulant chain at its boundary: consistency, random projections, the price of carrying κ₃,
and where the next order's accuracy lives*. Algorithmic Contribution write-up, ARC White-Box Estimation Challenge 2026,
Phase 2, submission 333534 (2026). https://discourse.aicrowd.com/t/18249
