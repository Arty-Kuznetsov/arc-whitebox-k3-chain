"""Explicit formulas for the reference SIMPLE k=3 nonlinear step, checked slice by slice against mlp_kprop.

Teacher forcing: at every layer the reference's own pre-activation tower is fed to our formulas, and our
post-activation slices are compared with the reference's act{l} tower. Only dense O(n^2) slices are
checked here (mean, variance, off-diagonal covariance, kappa3 (3,) and (2,1) slices, kappa4 radial core);
the factored all-distinct kappa3 part is checked separately.

Run (from this folder):
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src" uv run --project "$ARC" python step_check.py [n] [depth]
"""
import sys

import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.diagslice import diagslice
from mlp_kprop.wick import relu_wick_coef

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
torch.manual_seed(0)
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
for l in range(depth):
    mlp.Ws[l].weight.data = torch.randn(n, n) * (2.0 / n) ** 0.5
Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE, factor=True,
               output_all=True, up_to_layer=f"act{depth-1}")


def offdiag(M):
    return M - torch.diag(torch.diag(M))


def k4_slices(H):
    return diagslice(H, (4,)), offdiag(diagslice(H, (2, 2)))


def my_step(mu, C, t, D, K4d, K22):
    """Our nonlinear step. Inputs: pre-activation mean, covariance, kappa3 (3,) slice t_i = k3[iii],
    (2,1) slice D_ij = k3[iij] (zero diagonal), kappa4 (4,) slice and (2,2) slice (zero diagonal)."""
    var = torch.diag(C)
    w = {(k, p): relu_wick_coef(mu, var, k, p) for k in range(5) for p in range(1, 5)}
    C0 = offdiag(C)
    # single-index power moments P_p = E[y^p]: Gaussian + kappa3 + kappa4 Edgeworth terms
    P = {p: w[(0, p)] + w[(3, p)] * t / 6 + w[(4, p)] * K4d / 24 for p in range(1, 5)}
    # pair power cumulants kappa(y_i^p, y_j^q), i != j
    def Q(p, q):
        a = lambda k: w[(k, p)][:, None]
        b = lambda k: w[(k, q)][None, :]
        return (a(1) * b(1) * C0
                + 0.5 * a(1) * b(2) * D.T + 0.5 * a(2) * b(1) * D
                + 0.5 * a(2) * b(2) * C0 * C0
                + 0.25 * a(2) * b(2) * K22)
    Q11, Q21, Q22 = Q(1, 1), Q(2, 1), Q(2, 2)
    P1, P2, P3, P4 = P[1], P[2], P[3], P[4]
    K1 = P1
    K2 = P2 - P1 * P1
    K3 = P3 - 3 * P2 * P1 + 2 * P1 ** 3
    K4 = P4 - 4 * P3 * P1 - 3 * P2 * P2 + 12 * P2 * P1 * P1 - 6 * P1 ** 4
    K11 = Q11
    K21 = Q21 - 2 * P1[:, None] * Q11
    K22 = Q22 - 2 * P1[None, :] * K21 - 2 * P1[:, None] * K21.T - 2 * Q11 * Q11 - 4 * P1[:, None] * P1[None, :] * Q11
    return K1, K2, K11, K3, offdiag(K21), K4, offdiag(K22)


print(f"n={n} depth={depth}")
print(" l |   mean    var   cov_off   k3(3,)   k3(2,1)   k4(4,)   k4(2,2)   (max rel err)")
for l in range(depth):
    pre, act = Kb[f"pre{l}"], Kb[f"act{l}"]
    mu, C = pre[1].core, pre[2].core
    if 3 in pre:
        t, D = pre[3].get_dslice((3,)), offdiag(pre[3].get_dslice((2, 1)))
    else:
        t, D = torch.zeros(n), torch.zeros(n, n)
    K4d, K22 = k4_slices(pre[4]) if 4 in pre else (torch.zeros(n), torch.zeros(n, n))
    mine = my_step(mu, C, t, D, K4d, K22)
    ref = (act[1].core, torch.diag(act[2].core), offdiag(act[2].core), act[3].get_dslice((3,)),
           offdiag(act[3].get_dslice((2, 1))), *k4_slices(act[4]))
    errs = [float((a - b).abs().max() / (b.abs().max() + 1e-300)) for a, b in zip(mine, ref)]
    print(f"{l:2d} | " + "  ".join(f"{e:.1e}" for e in errs))
