"""End-to-end: k=2 covariance propagation + Edgeworth mean correction, measured on baked GT.

Chain state is (mu, C) exactly as in k=2. The ONLY change is the post-ReLU mean, which gains the
first-order Edgeworth term in the pre-activation's own diagonal third cumulant:

    mu_post = mu_pre*Phi(a) + sigma*phi(a) - kappa3_diag * a * phi(a) / (6*sigma^2)

Three arms share the same chain code (one switch), so differences are attributable:
  none   -- no correction. MUST reproduce the reference k=2 (4.50e-6 on MLP 0). Self-check.
  wick   -- kappa3_diag REGENERATED from the previous layer's mean+covariance, one extra
            (n,n)@(n,n) matmul per layer, zero fitted constants. This is the deployable arm.
  oracle -- kappa3_diag taken from the k=3 reference chain. Upper bound of the whole idea.

Run:  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
      env PYTHONPATH="$ARC/src" uv run --project "$ARC" python edgeworth_chain.py [n_mlps]
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


def chain(Wms, mode, oracle=None):
    """Returns the per-layer post-ReLU means. mode in {'none', 'wick', 'oracle'}."""
    mu = torch.zeros(n)
    C = torch.eye(n)
    prev = None                      # (C_pre, d1, d2) of the previous layer
    out = []
    for l in range(depth):
        Wm = Wms[l]                                   # Wm[i, a] = W_challenge[l][a, i]
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt()
        al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)

        if mode == "none" or l == 0:
            k3 = torch.zeros(n)                       # layer 0: pre-activations exactly Gaussian
        elif mode == "oracle":
            k3 = oracle[l]
        else:                                          # 'wick': regenerate from the previous state
            C_prev, d1_prev, d2_prev = prev
            V = (Wm * d1_prev) @ C_prev                # the one extra matmul, ~2.1e9 FLOPs
            k3 = 3.0 * ((Wm * d2_prev) * V * V).sum(dim=1)

        mu_new = mu_pre * Ph + sig * ph - k3 * al * ph / (6.0 * var)
        # covariance: the standard k=2 gain method, untouched
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        mu_gauss = mu_pre * Ph + sig * ph
        var_post = (ez2 - mu_gauss * mu_gauss).clamp_min(0.0)
        C = torch.outer(Ph, Ph) * C_pre
        C.diagonal().copy_(var_post)

        prev = (C_pre, Ph, ph / sig)
        mu = mu_new
        out.append(mu.clone())
    return torch.stack(out)


gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
res = {m: {"final": [], "all": []} for m in ("none", "wick", "oracle")}
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]

    Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE,
                   factor=True, output_all=True, up_to_layer=f"act{depth-1}")
    oracle = [Kb[f"pre{l}"][3].get_dslice((3,)).clone() if 3 in Kb[f"pre{l}"] else torch.zeros(n)
              for l in range(depth)]
    del Kb

    g = torch.tensor(gt[i], dtype=torch.float64)
    for m in ("none", "wick", "oracle"):
        P = chain(Wms, m, oracle)
        res[m]["final"].append(float(((P[-1] - g[-1]) ** 2).mean()))
        res[m]["all"].append(float(((P - g) ** 2).mean()))
    if i == 0:
        print(f"SELF-CHECK  arm 'none' on MLP 0 = {res['none']['final'][0]:.4e}"
              f"   (reference k=2 = 4.50e-06)\n")
    print(f"  mlp {i}: none={res['none']['final'][-1]:.3e}  wick={res['wick']['final'][-1]:.3e}"
          f"  oracle={res['oracle']['final'][-1]:.3e}", flush=True)
    del Wms, mlp, oracle

B = 2 ** 41
FLOPS = {"none": 5.18e10, "wick": 5.18e10 + 15 * 2.147e9, "oracle": float("nan")}
print(f"\n=== k=2 + Edgeworth mean correction, baked GT, {N_MLPS} MLPs ===")
print(f"{'arm':>7} | {'final_layer_mse':>16} {'all_layers_mse':>15} | {'vs k=2':>7} | "
      f"{'% of B':>7} {'adjusted':>10}")
b = float(np.mean(res["none"]["final"]))
for m in ("none", "wick", "oracle"):
    f = float(np.mean(res[m]["final"])); a = float(np.mean(res[m]["all"]))
    fl = FLOPS[m]
    pct = 100 * fl / B if fl == fl else float("nan")
    adj = f * max(0.1, fl / B) if fl == fl else float("nan")
    print(f"{m:>7} | {f:>16.4e} {a:>15.4e} | {b/f:>6.2f}x | {pct:>6.2f}% {adj:>10.3e}")
print("\n'wick' FLOPs = measured k=2 cost + 15 extra (n,n)@(n,n) matmuls (2n^3-n^2 each).")
print("Both stay under the 10% multiplier floor, so the extra compute is free on the metric.")
