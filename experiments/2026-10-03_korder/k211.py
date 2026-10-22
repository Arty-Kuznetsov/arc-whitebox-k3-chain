"""Is the (2,1,1) class of k=4's post-activation kappa4 (the largest part of the next kappa4 diagonal's pattern) a
fresh "doubled-hub" birth? Gaussian-order candidate: k4post(a,a,b,c) ~ g'_a C_ab C_ac d1_b d1_c with
g'_a = 2 Phi(alpha_a)(1 - Phi(alpha_a)) (ReLU: f f'' = 0, so only Var f' survives) -- a kappa3-source-like tensor (hub a
twice, leaves b, c through U = diag(d1) C), whose projection on the next layer's K4d is 6 sum_a W_ai^2 g'_a ((C D1 W)_ai)^2,
O(n^3), and through transported kappa3 legs costs O(n w). Compare the slice and its K4d projection with k=4's, per layer.
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
print(f"n={n} L={depth} seed={seed}: (2,1,1) slice of k4post(l-1) vs the doubled-hub birth; their K4d(l) projections")
print("   l | slice: corr  slope | K4d proj: corr  slope | birth-proj share of true (2,1,1)-proj variance")
for l in range(2, depth):
    W = Ws[l]
    pre = out[f"pre{l - 1}"]
    C = pre[2].to_tensor(); mu = pre[1].to_tensor()
    var = torch.clamp(torch.diagonal(C), min=1e-12); a = mu / torch.sqrt(var)
    d1 = Phi(a)
    phi = torch.exp(-0.5 * a * a) / (2 * torch.pi) ** 0.5
    # Leonov-Shiryaev at tree order: k(X,X,Y,Z) = k(X^2,Y,Z) - 2 mu_X k(X,Y,Z) - 2 k(X,Y) k(X,Z), with (f^2)'' = 2 1(z>0)
    gp = 2.0 * (d1 * (1.0 - d1) - phi * (a * d1 + phi)) if (len(sys.argv) > 4 and sys.argv[4] == "ls") else 2.0 * d1 * (1.0 - d1)
    C0 = C - torch.diag(torch.diagonal(C))
    S = out[f"act{l - 1}"][4].get_dslice((2, 1, 1)).clone()          # [a, b, c] = k4post(a,a,b,c)
    S[idx, idx, :] = 0; S[idx, :, idx] = 0; S[:, idx, idx] = 0
    U = C0 * d1[None, :]                                              # U[a, b] = C0_ab d1_b
    Bt = gp[:, None, None] * U[:, :, None] * U[:, None, :]            # birth candidate [a, b, c]
    Bt[idx, idx, :] = 0; Bt[idx, :, idx] = 0; Bt[:, idx, idx] = 0
    x, y = Bt.flatten(), S.flatten()
    c_sl = float(np.corrcoef(x.numpy(), y.numpy())[0, 1]); s_sl = float((x @ y) / (x @ x))
    p_true = 6.0 * torch.einsum("ai,abc,bi,ci->i", W ** 2, S, W, W)
    UW = U @ W                                                        # [a, i] = sum_b C0_ab d1_b W_bi
    p_b = 6.0 * ((W ** 2) * gp[:, None] * UW ** 2).sum(0) - 6.0 * ((W ** 2) * gp[:, None] * ((U ** 2) @ (W ** 2))).sum(0)
    c_p = float(np.corrcoef(p_b.numpy(), p_true.numpy())[0, 1]); s_p = float((p_b @ p_true) / (p_b @ p_b))
    pt = p_true - p_true.mean(); pb_ = (p_b - p_b.mean()) * s_p
    share = 1.0 - float(((pt - pb_) ** 2).sum() / (pt ** 2).sum())
    print(f"  {l:2d} |        {c_sl:.3f}  {s_sl:.3f} |           {c_p:.3f}  {s_p:.3f} | {share:.3f}", flush=True)
