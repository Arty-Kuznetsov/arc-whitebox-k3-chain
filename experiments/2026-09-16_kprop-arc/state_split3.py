"""Final arbitration: does the covariance matter when PAIRED with correct means?

state_split2 gave: perfect upstream means + our covariance = 20.2x, +perfect final readout = 23.6x,
against full k=3's 121x. The residual 5.1x can only come from the covariance. Earlier arms that said
"covariance is irrelevant" injected a k=3 covariance alongside k=2 MEANS -- a mismatched pair. This
arm injects both, and also injects the baked GROUND-TRUTH means, to bracket the ceiling of any
mean-only method.

Upstream (layers 0..depth-2) injections; the scored last layer is always computed by us.
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


def chain(Wms, inj_mu, inj_cov, edge, omu, ocov, ok3, gtmu):
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
        mu = mu_g - ok3[l] * al * ph / (6.0 * var) if (edge and l == depth - 1) else mu_g
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        C = torch.outer(Ph, Ph) * C_pre
        C.diagonal().copy_((ez2 - mu_g * mu_g).clamp_min(0.0))
        if l < depth - 1:
            if inj_mu == "k3":
                mu = omu[l].clone()
            elif inj_mu == "gt":
                mu = gtmu[l].clone()
            if inj_cov == "k3":
                C = ocov[l].clone()
    return mu


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
ARMS = [("none", "none", False), ("k3", "none", False), ("gt", "none", False),
        ("gt", "none", True), ("k3", "k3", False), ("k3", "k3", True), ("gt", "k3", True)]
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
    ocov = [Kb[f"act{l}"][2].to_tensor().clone() for l in range(depth)]
    ok3 = [Kb[f"pre{l}"][3].get_dslice((3,)).clone() if 3 in Kb[f"pre{l}"] else torch.zeros(n)
           for l in range(depth)]
    del Kb
    gtm = [torch.tensor(gt[i][l], dtype=torch.float64) for l in range(depth)]
    g = gtm[-1]
    k3ref.append(float(((omu[-1] - g) ** 2).mean()))
    for a in ARMS:
        res[a].append(float(((chain(Wms, a[0], a[1], a[2], omu, ocov, ok3, gtm) - g) ** 2).mean()))
    print(f"  mlp {i} done", flush=True)
    del Wms, mlp, omu, ocov, ok3, gtm

b = float(np.mean(res[("none", "none", False)]))
print(f"\n=== Paired injections, layers 0..{depth-2}; last layer always computed ({N_MLPS} MLPs) ===")
print(f"SELF-CHECK plain = {b:.4e} (reference k=2) ; full k=3 = {np.mean(k3ref):.4e}\n")
print(f"{'mean':>6} {'cov':>6} {'edge':>6} | {'final_layer_mse':>16} | {'vs k=2':>9}")
for a in ARMS:
    v = float(np.mean(res[a]))
    print(f"{a[0]:>6} {a[1]:>6} {str(a[2]):>6} | {v:>16.4e} | {b/v:>8.2f}x")
print(f"{'--':>6} {'--':>6} {'k=3':>6} | {np.mean(k3ref):>16.4e} | {b/np.mean(k3ref):>8.2f}x")
print("\n'gt none False' is the hard ceiling of any method that only fixes the MEAN.")
print("If 'k3 k3' lands near full k=3, the covariance matters once paired with correct means.")
