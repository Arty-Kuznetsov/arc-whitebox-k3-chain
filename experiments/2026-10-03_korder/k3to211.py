"""Where does the post-activation kappa4 (2,1,1) slice come from? Expanding y = mu + d1 dz + d2/2 (dz^2 - var) + ...
and applying Leonov-Shiryaev to k(y_a, y_a, y_b, y_c) (a doubled, a, b, c distinct) gives, besides the Gaussian
doubled-hub tree, terms LINEAR in the pre-activation kappa3 and kappa4:
  G  gating        d1_a^2 d1_b d1_c k4pre(a,a,b,c)
  T3 kappa3 transfer   2 var_a d1_a d2_a d1_b d1_c k3pre(a,b,c)        (the all-distinct kappa3 x the self-variance)
  TD D-edge        2 d1_a d2_a d1_b d1_c (C_ab D_ac + C_ac D_ab)      (D[a,c] = k3(a,a,c))
  E3 edge x kappa3     d1_a^2 d2_b d1_c C_ab k3(a,b,c) + (b<->c)       (an off-diagonal edge x all-distinct kappa3)
Compare each, and their sum, with ARC k=4's post (2,1,1) slice; then their projections on K4d(l+1).  args: n depth seed"""
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
Phi = lambda x: 0.5 * (1.0 + torch.erf(x / 2 ** 0.5))


def offd(T):
    T = T.clone(); T[idx, idx, :] = 0; T[idx, :, idx] = 0; T[:, idx, idx] = 0
    return T


def stats(x, y):
    x, y = x.flatten(), y.flatten()
    return float(np.corrcoef(x.numpy(), y.numpy())[0, 1]), float((x @ y) / (x @ x)), float((x @ y) ** 2 / ((x @ x) * (y @ y)))


print(f"n={n} L={depth} seed={seed}: candidate terms vs k=4's post (2,1,1) slice at layer l (corr, slope, R2 of the slice)")
for l in range(1, depth - 1):
    pre = out[f"pre{l}"]
    mu = pre[1].to_tensor(); C = pre[2].to_tensor(); k3 = pre[3].to_tensor()
    var = torch.clamp(torch.diagonal(C), min=1e-12); sd = torch.sqrt(var); a = mu / sd
    d1 = Phi(a); d2 = torch.exp(-0.5 * a * a) / (2 * torch.pi) ** 0.5 / sd
    D = pre[3].get_dslice((2, 1))                                   # [a, c] = k3(a,a,c)
    C0 = C - torch.diag(var)
    k4p = pre[4].get_dslice((2, 1, 1))                              # [a, b, c] = k4pre(a,a,b,c)
    S = offd(out[f"act{l}"][4].get_dslice((2, 1, 1)))               # truth: k4post(a,a,b,c)
    bc = d1[None, :, None] * d1[None, None, :]
    G = offd(d1[:, None, None] ** 2 * bc * k4p)
    T3 = offd(2.0 * (var * d1 * d2)[:, None, None] * bc * k3)
    TD = offd(2.0 * (d1 * d2)[:, None, None] * bc * (C0[:, :, None] * D[:, None, :] + C0[:, None, :] * D[:, :, None]))
    E3 = offd((d1 ** 2)[:, None, None] * (d2[None, :, None] * d1[None, None, :] * C0[:, :, None] +
                                         d1[None, :, None] * d2[None, None, :] * C0[:, None, :]) * k3)
    parts = {"G": G, "T3": T3, "TD": TD, "E3": E3, "G+T3": G + T3, "all4": G + T3 + TD + E3}
    line = f"  l={l}:"
    for k, v in parts.items():
        c, s, r2 = stats(v, S)
        line += f"  {k} {c:+.3f}/{s:.2f}"
    print(line, flush=True)
