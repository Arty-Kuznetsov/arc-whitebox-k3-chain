"""Is the k=2 mean error fixable by a DIAGONAL third-cumulant correction?

layer_blame.py showed the whole k=2 final-layer error is carried by the propagated MEAN
(fixing the covariance alone does nothing). The k=2 post-ReLU mean update
    mu_post = mu_pre*Phi(a) + sigma_pre*phi(a),      a = mu_pre/sigma_pre
is EXACT for Gaussian pre-activations. Its leading error in the pre-activation's own third
cumulant is the Edgeworth term (derived + MC-verified separately):
    mu_post ~= mu_pre*Phi(a) + sigma_pre*phi(a) - kappa3_diag * a * phi(a) / (6*sigma_pre**2)
where kappa3_diag[i] = kappa3_pre[i,i,i]. That needs only a VECTOR per layer, not the
n x n x n tensor, so if it captures a large share of the mean error the whole expensive
cross-cumulant transport might be avoidable.

This probe answers the upper-bound question only: taking the pre-activation state and the
TRUE diagonal kappa3 from the k=3 reference chain, does the Edgeworth readout beat the
Gaussian readout against the baked N=1e9 ground-truth means?

Step 0 validates get_dslice((3,)) against an explicit dense tensor at small width.

Run:  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
      env PYTHONPATH="$ARC/src" uv run --project "$ARC" python edgeworth_probe.py [n_mlps]
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
phi = lambda a: np.exp(-0.5 * a * a) / SQRT2PI
Phi = lambda a: 0.5 * (1.0 + torch.erf(torch.tensor(a) / np.sqrt(2.0)).numpy())


def build(W, d=depth, width=n):
    m = MLP(input_dim=width, hidden_dim=width, output_dim=width, num_layers=d + 1,
            nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(d):
        m.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    return m


# ---------------- Step 0: validate the diagonal accessor at small width ----------------
ns, ds = 16, 3
rng = np.random.default_rng(0)
Ws = rng.normal(0.0, np.sqrt(2.0 / ns), (ds, ns, ns))
Kb = mlp_kprop(build(Ws, ds, ns), {1: torch.zeros(ns), 2: torch.eye(ns)}, k_max=3,
               kind=Kind.SIMPLE, factor=True, output_all=True, up_to_layer=f"act{ds-1}")
K3 = Kb[f"pre{ds-1}"][3]
dense = np.asarray(K3.to_tensor())
diag_dense = np.array([dense[i, i, i] for i in range(ns)])
diag_slice = np.asarray(K3.get_dslice((3,)))
err = float(np.abs(diag_dense - diag_slice).max())
print(f"STEP 0  get_dslice((3,)) vs explicit diagonal at n={ns}: max|diff| = {err:.3e}"
      f"   {'OK' if err < 1e-12 else 'MISMATCH -- STOP'}")
print(f"        (scale check: max|kappa3_iii| = {np.abs(diag_dense).max():.3e})\n")
if err >= 1e-12:
    sys.exit(1)

# ---------------- Step 1: the probe at the competition shape ----------------
gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])
rows = []
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    Kb = mlp_kprop(build(W), {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE,
                   factor=True, output_all=True, up_to_layer=f"act{depth-1}")
    per_layer = []
    for l in range(depth):
        pre = Kb[f"pre{l}"]
        mu = np.asarray(pre[1].to_tensor())
        var = np.maximum(np.asarray(pre[2].to_tensor()).diagonal(), 1e-12)
        sig = np.sqrt(var)
        k3d = np.asarray(pre[3].get_dslice((3,))) if 3 in pre else np.zeros(n)
        a = mu / sig
        gauss = mu * Phi(a) + sig * phi(a)
        edge = gauss - k3d * a * phi(a) / (6.0 * var)
        k3mean = np.asarray(Kb[f"act{l}"][1].to_tensor())
        g = gt[i][l]
        per_layer.append((
            float(((gauss - g) ** 2).mean()), float(((edge - g) ** 2).mean()),
            float(((k3mean - g) ** 2).mean()), float(np.abs(a).mean()),
            float(np.abs(k3d).mean()), float((np.abs(k3d) / (sig ** 3)).mean()),
        ))
    rows.append(per_layer)
    del Kb
    print(f"  mlp {i} done", flush=True)

R = np.mean(rows, axis=0)
print(f"\n=== Edgeworth diagonal-kappa3 readout probe ({N_MLPS} MLPs, mean) ===")
print("At each layer: take the k=3 chain's PRE-activation state, read the post-ReLU mean two ways,")
print("score both against the baked N=1e9 GT mean of that layer.\n")
print(f"{'l':>2} | {'Gaussian mse':>13} {'Edgeworth mse':>14} {'ratio':>7} | "
      f"{'k=3 chain mse':>14} | {'mean|a|':>8} {'mean|skew|':>10}")
for l in range(depth):
    gm, em, km, ma, _, sk = R[l]
    print(f"{l:>2} | {gm:>13.3e} {em:>14.3e} {gm/em if em else float('inf'):>7.2f} | "
          f"{km:>14.3e} | {ma:>8.3f} {sk:>10.4f}")
print("\nratio > 1 = the diagonal-kappa3 correction improves the one-step readout at that layer.")
print("Compare 'Edgeworth mse' with 'k=3 chain mse': how much of the full k=3 accuracy a")
print("VECTOR-only correction recovers, against a tensor the full chain pays O(L^2 n^3) to carry.")
