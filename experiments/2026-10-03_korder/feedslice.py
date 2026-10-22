"""The kappa4 -> kappa3 feed reads the PRE-activation (2,1,1) slice as the scale-mixture closure's
k4pre(a,a,b,c) = v (var_a C_bc + 2 C_ab C_ac). How close is that to ARC k=4's pre (2,1,1) slice? Per layer: corr and
fitted-v R2 of the closure form, of its two terms separately, and how much of the slice is coherent (its projection on
the var_a C_bc direction).  args: n depth seed"""
import sys
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
g = torch.Generator().manual_seed(seed)
Ws = [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
for l in range(depth):
    mlp.Ws[l].weight.data = Ws[l].T.clone()
out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                up_to_layer=f"act{depth - 1}")
idx = torch.arange(n)


def offd(T):
    T = T.clone(); T[idx, idx, :] = 0; T[idx, :, idx] = 0; T[:, idx, idx] = 0
    return T


def fitstats(X, y):
    X = [x.flatten().numpy() for x in X]; y = y.flatten().numpy()
    F = np.stack(X, 1)
    coef, *_ = np.linalg.lstsq(F, y, rcond=None)
    r = y - F @ coef
    return coef, 1 - (r @ r) / (y @ y)


print(f"n={n} L={depth} seed={seed}: k=4 pre (2,1,1) slice vs the closure's form")
for l in range(2, depth):
    pre = out[f"pre{l}"]
    C = pre[2].to_tensor(); var = torch.diagonal(C); C0 = C - torch.diag(var)
    S = offd(pre[4].get_dslice((2, 1, 1)))
    t1 = offd(var[:, None, None] * C0[None, :, :])               # var_a C_bc
    t2 = offd(2.0 * C0[:, :, None] * C0[:, None, :])             # 2 C_ab C_ac
    (c1,), r1 = fitstats([t1], S)
    (c2,), r2 = fitstats([t2], S)
    cc, rr = fitstats([t1 + t2], S)
    c12, r12 = fitstats([t1, t2], S)
    v_k4 = float(pre[4].get_dslice((4,)).mean() / (3.0 * (var * var).mean()))
    print(f"  l={l}: R2 var_a C_bc {r1:.3f} (v {c1:.2e}) | 2 C_ab C_ac {r2:.3f} | closure form {rr:.3f} (v {cc[0]:.2e}) | "
          f"both free {r12:.3f} | v from k=4's diagonal {v_k4:.2e}", flush=True)
