"""Corrected split: upstream MEAN quality vs final READOUT quality.

state_split.py had a flaw: injecting the k=3 mean at every layer overwrote the final layer too, so
that arm trivially returned k=3's own answer. Here the injection stops one layer short, and the last
layer is always computed by us. Arms:

  upstream x readout, with upstream in {k2, k3} and readout in {gauss, edge}
    upstream k2  -- our own k=2 means all the way
    upstream k3  -- k=3's post-ReLU means injected at layers 0..depth-2 (covariance always ours)
    readout gauss-- mu*Phi(a) + sigma*phi(a)
    readout edge -- the same plus the Edgeworth term using the TRUE kappa3 diagonal of the final
                    pre-activation, taken from the k=3 reference

Reference points: plain k=2 = 4.50e-6 on MLP 0 (self-check of the k2/gauss arm) and full k=3 = 4.25e-8.
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
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)


def chain(Wms, upstream, readout, omu, ok3):
    mu, C = torch.zeros(n), torch.eye(n)
    for l in range(depth):
        Wm = Wms[l]
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt()
        al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        mu_g = mu_pre * Ph + sig * ph
        if l == depth - 1 and readout == "edge":
            mu = mu_g - ok3[l] * al * ph / (6.0 * var)
        else:
            mu = mu_g
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        C = torch.outer(Ph, Ph) * C_pre
        C.diagonal().copy_((ez2 - mu_g * mu_g).clamp_min(0.0))
        if upstream == "k3" and l < depth - 1:      # never overwrite the scored layer
            mu = omu[l].clone()
    return mu


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
ARMS = [("k2", "gauss"), ("k2", "edge"), ("k3", "gauss"), ("k3", "edge")]
res = {a: [] for a in ARMS}
k3ref = []
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE,
                   factor=True, output_all=True, up_to_layer=f"act{depth-1}")
    omu = [Kb[f"act{l}"][1].to_tensor().clone() for l in range(depth)]
    ok3 = [Kb[f"pre{l}"][3].get_dslice((3,)).clone() if 3 in Kb[f"pre{l}"] else torch.zeros(n)
           for l in range(depth)]
    del Kb
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    k3ref.append(float(((omu[-1] - g) ** 2).mean()))
    for a in ARMS:
        res[a].append(float(((chain(Wms, a[0], a[1], omu, ok3) - g) ** 2).mean()))
    print(f"  mlp {i}: " + "  ".join(f"{a[0]}/{a[1]}={res[a][-1]:.3e}" for a in ARMS)
          + f"  (k3={k3ref[-1]:.3e})", flush=True)
    del Wms, mlp, omu, ok3

b = float(np.mean(res[("k2", "gauss")]))
print(f"\n=== Upstream mean quality x final readout quality ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK k2/gauss = {b:.4e} (reference k=2) ; full k=3 = {np.mean(k3ref):.4e}\n")
print(f"{'upstream':>9} {'readout':>8} | {'final_layer_mse':>16} | {'vs k=2':>9}")
for a in ARMS:
    v = float(np.mean(res[a]))
    print(f"{a[0]:>9} {a[1]:>8} | {v:>16.4e} | {b/v:>8.2f}x")
print(f"{'k=3':>9} {'k=3':>8} | {np.mean(k3ref):>16.4e} | {b/np.mean(k3ref):>8.2f}x")
print("\nk3/gauss isolates what perfect upstream means are worth on their own; k2/edge isolates what")
print("a perfect final readout is worth on its own; k3/edge is what having both would give.")
