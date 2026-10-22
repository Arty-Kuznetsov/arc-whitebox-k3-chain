"""Is the (2,1,1)-class part of k=4's K4d(l+1) the (2,2) class of layer l-1 carried two steps? Path: K22post(l-1)
--W_l--> pre (2,1,1) slice at l [sum_pr W_pa^2 W_rb W_rc K22post(p,r)] --ReLU gating d1_a^2 d1_b d1_c--> post (2,1,1)
at l --W_{l+1}--> K4d(l+1) via 6 sum W_ai^2 W_bi W_ci. Closed form, O(n^3):
  T_i = 6 sum_p Q_pi (K22post (M o M))_pi,  M = W_l diag(d1_l) W_{l+1},  Q = (W_l o W_l) diag(d1_l^2) (W_{l+1} o W_{l+1})
(index coincidences ignored). Compare with the true (2,1,1)-class contribution c211(l+1) of k4classes.py.
args: n depth seed"""
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
off = ~torch.eye(n, dtype=torch.bool)
print(f"n={n} L={depth} seed={seed}: true (2,1,1)-class part of K4d(l+1) vs the (2,2)-two-step carrier (and vs the")
print("  pre-activation (2,1,1) slice at l: how much of it is the projected (2,2) class of l-1)")
for l in range(2, depth - 1):
    Wl, Wn = Ws[l], Ws[l + 1]
    # true (2,1,1)-class contribution to K4d(l+1)
    S = out[f"act{l}"][4].get_dslice((2, 1, 1)).clone()
    S[idx, idx, :] = 0; S[idx, :, idx] = 0; S[:, idx, idx] = 0
    c211 = 6.0 * torch.einsum("ai,abc,bi,ci->i", Wn ** 2, S, Wn, Wn)
    # the carrier from K22post(l-1)
    K22p = out[f"act{l - 1}"][4].get_dslice((2, 2)).clone(); K22p[~off] = 0
    pre = out[f"pre{l}"]
    mu, var = pre[1].to_tensor(), torch.clamp(torch.diagonal(pre[2].to_tensor()), min=1e-12)
    d1 = Phi(mu / torch.sqrt(var))
    M = Wl @ (d1[:, None] * Wn)
    Q = (Wl ** 2) @ ((d1 ** 2)[:, None] * Wn ** 2)
    T = 6.0 * (Q * (K22p @ (M ** 2))).sum(0)
    # the pre (2,1,1) slice at l and its (2,2)-class part
    Spre = pre[4].get_dslice((2, 1, 1)).clone()
    Spre[idx, idx, :] = 0; Spre[idx, :, idx] = 0; Spre[:, idx, idx] = 0
    P22 = torch.einsum("pa,pr,rb,rc->abc", Wl ** 2, K22p, Wl, Wl)
    P22[idx, idx, :] = 0; P22[idx, :, idx] = 0; P22[:, idx, idx] = 0
    cs = float(np.corrcoef(P22.flatten().numpy(), Spre.flatten().numpy())[0, 1])
    sh = float((P22.flatten() @ Spre.flatten()) ** 2 / ((P22.flatten() @ P22.flatten()) * (Spre.flatten() @ Spre.flatten())))
    c = float(np.corrcoef(T.numpy(), c211.numpy())[0, 1]); s = float((T @ c211) / (T @ T))
    print(f"  l={l}: K4d(l+1) (2,1,1)-part vs carrier: corr {c:.3f} slope {s:.3f} | pre (2,1,1) slice at l vs projected (2,2): corr {cs:.3f} (R2 {sh:.3f})", flush=True)
