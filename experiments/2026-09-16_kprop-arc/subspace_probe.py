"""Does the pre-activation covariance live in a low-dimensional subspace?

The k=3 transport costs O(n^2 * R) per layer. If the covariance (and therefore the directions the
cumulants actually occupy) had effective rank m << n, every contraction could be done in an
m-dimensional basis and the whole chain would collapse in cost. This is the one lever that would
change the cost class rather than the constant.

Cheap and decisive: eigendecompose the pre-activation covariance at each layer and report the
spectrum's effective rank. Also reports the participation ratio, a basis-free effective dimension.
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, 10)

stats = []
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=2, kind=Kind.SIMPLE,
                   output_all=True, up_to_layer=f"act{depth-1}")
    per = []
    for l in range(depth):
        C = Kb[f"pre{l}"][2].to_tensor()
        ev = torch.linalg.eigvalsh(C).flip(0).clamp_min(0)
        tot = float(ev.sum())
        cum = torch.cumsum(ev, 0) / tot
        k90 = int((cum < 0.90).sum()) + 1
        k99 = int((cum < 0.99).sum()) + 1
        pr = float(tot ** 2 / (ev ** 2).sum())          # participation ratio
        cond = float(ev[0] / ev[ev > 0][-1])
        per.append((k90, k99, pr, cond, float(ev[0]), float(ev[-1])))
    stats.append(per)
    del Kb, mlp
    print(f"  mlp {i} done", flush=True)

S = np.mean(stats, axis=0)
print(f"\n=== Pre-activation covariance spectrum ({N_MLPS} MLPs, n={n}) ===")
print(f"{'l':>2} | {'dims for 90%':>12} {'dims for 99%':>12} {'particip. ratio':>16} "
      f"{'cond':>9} | {'lam_max':>9} {'lam_min':>9}")
for l in range(depth):
    k90, k99, pr, cond, lmax, lmin = S[l]
    print(f"{l:>2} | {k90:>12.0f} {k99:>12.0f} {pr:>16.1f} {cond:>9.1f} | {lmax:>9.3f} {lmin:>9.2e}")
print(f"\nIf 'dims for 99%' stays near n={n} the covariance is essentially full rank, every direction")
print("carries comparable weight, and there is no cheap subspace to run the cumulant transport in.")
