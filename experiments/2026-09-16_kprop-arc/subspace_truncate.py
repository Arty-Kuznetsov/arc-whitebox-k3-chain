"""Is the shrinking covariance subspace REAL, i.e. can the chain be run inside it?

subspace_probe.py showed the pre-activation covariance concentrates with depth (90% of the energy in
145 of 1024 directions by layer 15; participation ratio 511 -> 65). If that subspace is real, the
expensive deep layers -- which is exactly where the k=3 transport cost lives, since its rank grows as
2n(l+1) -- could be run in an m-dimensional basis instead of n.

Idealized test: run the k=2 chain but project the post-ReLU covariance onto its top-m eigenvectors at
every layer, and score the final-layer mean against baked GT. Exact eigenvectors are used on purpose:
this measures whether the INFORMATION is there, not whether a cheap basis can be found. If truncation
at m ~ 150-300 is near-lossless the lever is real and the next question is how to get a cheap basis.
If it is not, the spectrum is a red herring and the k=3 cost cannot be cut this way.

Self-check: m = n must reproduce the untruncated chain exactly.
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)


def chain(Wms, m):
    """k=2 chain; if m < n the post-ReLU covariance is projected to its top-m eigenspace."""
    mu = torch.zeros(n)
    C = torch.eye(n)
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
        if m < n:
            C = 0.5 * (C + C.T)
            ev, U = torch.linalg.eigh(C)
            idx = ev.argsort(descending=True)[:m]
            C = (U[:, idx] * ev[idx]) @ U[:, idx].T
    return mu


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])
MS = [n, 512, 300, 200, 150, 100, 64]
res = {m: [] for m in MS}
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for m in MS:
        res[m].append(float(((chain(Wms, m) - g) ** 2).mean()))
    print(f"  mlp {i}: full={res[n][-1]:.3e}  m=200 -> {res[200][-1]:.3e}", flush=True)
    del Wms, mlp

base = float(np.mean(res[n]))
print(f"\n=== Covariance truncated to its top-m eigenspace every layer ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK m = n = {n}: {base:.4e}  (expect the reference k=2, 4.50e-06 on MLP 0)\n")
print(f"{'m':>6} | {'final_layer_mse':>16} | {'vs full':>8}")
for m in MS:
    v = float(np.mean(res[m]))
    print(f"{m:>6} | {v:>16.4e} | {v/base:>7.2f}x")
print("\nvs full ~ 1 means the covariance really does live in m directions and the deep, expensive")
print("layers of a k=3 chain could be run in that basis. Much larger than 1 means the small")
print("eigen-directions still drive the mean and the spectrum is not an exploitable subspace.")
