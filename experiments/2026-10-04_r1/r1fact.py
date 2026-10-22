"""R1 (notes/PLAN-2026-10-04.md): is there a SMALL consistent set of the next order? Full factorial over the eight
pre-activation channels our chain reads, substituted from ARC's k_max=4 SIMPLE run (full kappa4; x0.13-0.17 of our
chain's final MSE at n=256, L=8) at layers 1 .. L-1-SKIP (SKIP=1: the last step is the chain's own, so every arm
measures UPSTREAM channels; SKIP=0 includes the last layer's readout):
  V   variance (diag C), substituted early: legs, closure patterns and the step all see it    (oracle var0_layers)
  C0  off-diagonal pre-activation covariance, early                                           (C0_layers)
  t   kappa3 diagonal k3[iii]                                                                  (k3_layers)
  D   kappa3 (2,1) slice k3[iij]                                                               (D_layers)
  K4  kappa4 diagonal k4[iiii]                                                                 (k4_layers)
  K22 kappa4 (2,2) slice k4[iijj]                                                              (K22_layers)
  K31 kappa4 (3,1) slice k4[iiij] -- the pair expansion's K31 term (closure: 3 k31mult v var_i C0_ij)  (K31_layers)
  S   kappa4 (2,1,1) slice k4[aabc] feeding the kappa4 -> kappa3 births (closure: v var_a C_bc)   (dense3 feedS)
Final-layer MSE vs Monte-Carlo (1e7 samples, korder.py's cache; MC noise var/N reported, to subtract).
    env PYTHONPATH=$ARC/src nice -n 10 uv run --project $ARC python r1fact.py <n> <depth> <seed> [SKIP] [arms]
arms: "all" (default, the 256 subsets) or a comma list of names like "V+t+K4,K4+S".
Output: res/fact_n<n>_L<depth>_s<seed>_skip<SKIP>.json (resumable: arms already there are skipped).
"""
import itertools
import json
import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("R1_THREADS", "3")))
from r1lib import CHANNELS, HERE, Timer, chain_kwargs, final_mse, k4_stats, mc, name, run_chain, weights, BASE  # noqa

n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
SKIP = int(sys.argv[4]) if len(sys.argv) > 4 else 1
ARMS = sys.argv[5] if len(sys.argv) > 5 else "all"
N = int(1e7)
layers = set(range(1, depth - SKIP))
out_f = HERE / "res" / f"fact_n{n}_L{depth}_s{seed}_skip{SKIP}.json"
(HERE / "res").mkdir(exist_ok=True)
res = json.loads(out_f.read_text()) if out_f.exists() else {"n": n, "depth": depth, "seed": seed, "skip": SKIP,
                                                              "layers": sorted(layers), "arms": {}}
if ARMS == "all":
    subsets = [set(c) for r in range(len(CHANNELS) + 1) for c in itertools.combinations(CHANNELS, r)]
else:
    subsets = [set() if a == "base" else set(a.split("+")) for a in ARMS.split(",")]
todo = [s for s in subsets if name(s) not in res["arms"]]
Ws = weights(n, depth, seed)
T = Timer()
mu_mc, var_mc = mc(Ws, seed, N)
res["mc_noise"] = float(var_mc.mean() / N)
print(f"seed {seed}: MC ready ({T():.0f} s), noise {res['mc_noise']:.3e}; {len(todo)} arms to run", flush=True)
if todo:
    T = Timer()
    need_S = any("S" in s for s in todo)
    st, mu4 = k4_stats(Ws, s211_layers=(layers if need_S else ()))
    res["k4S"] = float(((mu4[-1] - mu_mc[-1]) ** 2).mean())
    res["k4_secs"] = T()
    print(f"  k4S {res['k4S']:.3e} ({T():.0f} s)", flush=True)
    for s in todo:
        T = Timer()
        e = final_mse(Ws, chain_kwargs(s, st, layers), mu_mc)
        res["arms"][name(s)] = e
        b = res["arms"].get("base")
        print(f"  {name(s):34s} {e:.4e}" + (f"  x{(e - res['mc_noise']) / (b - res['mc_noise']):.3f} (noise-sub)"
                                             if b else "") + f"  [{T():.1f} s]", flush=True)
        out_f.write_text(json.dumps(res, indent=1))
out_f.write_text(json.dumps(res, indent=1))
