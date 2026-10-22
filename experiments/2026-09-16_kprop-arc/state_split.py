"""Where does the 110x of k=3 actually live: in the mean, or in the covariance?

Two earlier measurements look contradictory and the difference decides the whole strategy.
  - layer_blame arm C: injecting the k=3 covariance at ONE layer never helps (up to 16% worse).
  - edgeworth_chain oracle arm: the exact kappa3 diagonal fed to the Edgeworth mean term at EVERY
    layer buys only 2.98x, against full k=3's 110x.
Neither tests "the covariance is right at every layer". If that is the missing piece, it matters:
the kappa3 slice the covariance needs (D21[a,b] = kappa3[a,a,b]) is an n x n object that the Wick
expansion regenerates from the covariance in O(n^2), the same trick already shipped for the diagonal.

Four arms of the same k=2 chain, differing only in what is overwritten with the k=3 reference at
EVERY layer:
  none  -- plain k=2.                          SELF-CHECK: must reproduce the reference k=2.
  mean  -- post-ReLU mean <- k=3's.
  cov   -- post-ReLU covariance <- k=3's.
  both  -- both.                               SELF-CHECK: must approach full k=3.
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


def chain(Wms, mode, omu, ocov):
    mu, C = torch.zeros(n), torch.eye(n)
    out = []
    for l in range(depth):
        Wm = Wms[l]
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt()
        al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        mu = mu_pre * Ph + sig * ph
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        var_post = (ez2 - mu * mu).clamp_min(0.0)
        C = torch.outer(Ph, Ph) * C_pre
        C.diagonal().copy_(var_post)
        if mode in ("mean", "both"):
            mu = omu[l].clone()
        if mode in ("cov", "both"):
            C = ocov[l].clone()
        out.append(mu.clone())
    return torch.stack(out)


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
MODES = ("none", "mean", "cov", "both")
res = {m: [] for m in MODES}
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
    ocov = [Kb[f"act{l}"][2].to_tensor().clone() for l in range(depth)]
    del Kb
    g = torch.tensor(gt[i], dtype=torch.float64)
    k3ref.append(float(((omu[-1] - g[-1]) ** 2).mean()))
    for m in MODES:
        res[m].append(float(((chain(Wms, m, omu, ocov)[-1] - g[-1]) ** 2).mean()))
    print(f"  mlp {i}: none={res['none'][-1]:.3e} mean={res['mean'][-1]:.3e} "
          f"cov={res['cov'][-1]:.3e} both={res['both'][-1]:.3e} (k3={k3ref[-1]:.3e})", flush=True)
    del Wms, mlp, omu, ocov

b = float(np.mean(res["none"]))
print(f"\n=== What the k=3 information buys, injected at EVERY layer ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK 'none' = {b:.4e} (reference k=2 4.50e-06 on MLP 0); "
      f"full k=3 = {np.mean(k3ref):.4e}\n")
print(f"{'injected':>10} | {'final_layer_mse':>16} | {'vs k=2':>9}")
for m in MODES:
    v = float(np.mean(res[m]))
    print(f"{m:>10} | {v:>16.4e} | {b/v:>8.2f}x")
print(f"{'full k=3':>10} | {np.mean(k3ref):>16.4e} | {b/np.mean(k3ref):>8.2f}x")
print("\nIf 'cov' is large, the covariance carried through depth is the missing piece and a cheap")
print("O(n^2) Wick regeneration of D21 is the next lever. If 'cov' ~ 1, the gap is higher-order")
print("terms in the mean and the readout correction itself has to get better.")
