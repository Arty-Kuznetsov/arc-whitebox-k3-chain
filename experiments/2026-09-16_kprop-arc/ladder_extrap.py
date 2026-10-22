"""Richardson extrapolation across CUMULANT ORDER: is the error ladder geometric per neuron?

Measured ladder at 1024x16: k1 3.33e-4, k2 4.50e-6, k3 4.10e-8. Ratios 74x and 110x, close to the
companion paper's (L/n)^k law with n/L = 64. If that law holds PER NEURON and not merely in
aggregate, then with e_k = mu^(k) - mu_true and e_k = c_i * q^k,

    mu^(2) - mu^(1) = c_i*q*(q-1)        =>    e_2 = (mu^(2) - mu^(1)) * q/(q-1)
    corrected:  mu* = mu^(2) + (mu^(2) - mu^(1)) * q/(1-q),     q = L/n = 1/64.

k=1 costs 3.5e7 FLOPs against k=2's 5.2e10, i.e. 0.07% -- essentially free. If the per-neuron
correlation is high this buys accuracy for nothing, and the same test at the next rung says whether
k=3 + extrapolation reaches k=4 accuracy.

What is measured here, per MLP and per rung:
  corr(delta, e)   -- delta = mu^(k) - mu^(k-1), e = mu^(k) - mu_true. Is the step a proxy for the error?
  beta_hat         -- least-squares slope of e on delta. Theory says q/(q-1) = -1/63 for q=1/64.
  mse after        -- final-layer MSE after subtracting beta*delta, with beta THEORETICAL and beta FITTED.

Run:  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
      env PYTHONPATH="$ARC/src" uv run --project "$ARC" python ladder_extrap.py [n_mlps]
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
gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
Q = 1.0 / 64.0                      # L/n, the companion paper's small parameter
BETA_TH = Q / (Q - 1.0)             # = -1/63, the theoretical slope of e on delta

rows = []
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    mus = {}
    for k in (1, 2, 3):
        Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=k, kind=Kind.SIMPLE,
                       factor=(k >= 3), output_all=True, up_to_layer=f"act{depth-1}")
        mus[k] = np.asarray(Kb[f"act{depth-1}"][1].to_tensor())
        del Kb
    g = gt[i][-1]
    per = {}
    for k in (2, 3):
        delta = mus[k] - mus[k - 1]
        e = mus[k] - g
        corr = float(np.corrcoef(delta, e)[0, 1])
        beta_hat = float((delta @ e) / (delta @ delta))
        mse0 = float((e ** 2).mean())
        mse_th = float(((e - BETA_TH * delta) ** 2).mean())
        mse_fit = float(((e - beta_hat * delta) ** 2).mean())
        per[k] = (corr, beta_hat, mse0, mse_th, mse_fit)
    rows.append(per)
    print(f"  mlp {i}: k2 corr={per[2][0]:+.3f} beta={per[2][1]:+.5f} | "
          f"k3 corr={per[3][0]:+.3f} beta={per[3][1]:+.5f}", flush=True)
    del mlp

print(f"\n=== Richardson extrapolation across cumulant order ({N_MLPS} MLPs) ===")
print(f"theoretical slope q/(q-1) with q = L/n = 1/64:  {BETA_TH:+.6f}\n")
print(f"{'rung':>22} | {'corr':>7} {'beta_hat':>10} | {'mse before':>11} "
      f"{'after (theory)':>15} {'after (fitted)':>15} | {'gain th':>8} {'gain fit':>9}")
for k in (2, 3):
    c = np.mean([r[k][0] for r in rows]); b = np.mean([r[k][1] for r in rows])
    m0 = np.mean([r[k][2] for r in rows]); mt = np.mean([r[k][3] for r in rows])
    mf = np.mean([r[k][4] for r in rows])
    print(f"{'mu^%d - mu^%d' % (k, k-1):>22} | {c:>7.3f} {b:>10.5f} | {m0:>11.3e} "
          f"{mt:>15.3e} {mf:>15.3e} | {m0/mt:>7.2f}x {m0/mf:>8.2f}x")
print("\nIf corr is high and beta_hat sits near the theoretical slope, the ladder is geometric per")
print("neuron and the step between two cumulant orders predicts the remaining error. k=1 is free,")
print("so a working k2 rung is accuracy for no compute. A working k3 rung would mean k=3 + one")
print("extrapolation reaches k=4-class accuracy without ever forming a 4th-cumulant tensor.")
