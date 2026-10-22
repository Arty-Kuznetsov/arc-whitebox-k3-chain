"""Where does ARC k=4's next-layer kappa4 diagonal come from? K4d(l)_i = sum_abcd W_ai W_bi W_ci W_di k4post(l-1)_abcd
splits by the index-coincidence class of (a,b,c,d): diag (aaaa), (3,1) (aaab x4), (2,2) (aabb x3), (2,1,1) (aabc x6),
all-distinct (the remainder). The diag, (3,1), (2,2) classes are BIVARIATE (functions of pairs (z_a, z_b)) and their
projections cost O(n^3); (2,1,1) is trivariate; all-distinct needs the tensor. Report each class's share of K4d's
pattern (centered variance) and of its level, and how well the bivariate sum and our closure track the truth.
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
off = ~torch.eye(n, dtype=torch.bool)
print(f"n={n} L={depth} seed={seed}: share of K4d(l) (centered variance across neurons | mean) by class of k4post(l-1)")
print("   l | diag          | (3,1)         | (2,2)         | (2,1,1)       | all-distinct  | corr(bivariate sum, truth)  corr(bivar+211, truth)")
for l in range(2, depth):
    W = Ws[l]
    P = out[f"act{l - 1}"][4]
    tru = out[f"pre{l}"][4].get_dslice((4,))
    d4 = P.get_dslice((4,))
    s31 = P.get_dslice((3, 1)).clone(); s31[~off] = 0      # [a, b] = k4(a,a,a,b)
    s22 = P.get_dslice((2, 2)).clone(); s22[~off] = 0      # [a, b] = k4(a,a,b,b)
    s211 = P.get_dslice((2, 1, 1)).clone()                 # [a, b, c] = k4(a,a,b,c)
    idx = torch.arange(n)
    s211[idx, idx, :] = 0; s211[idx, :, idx] = 0; s211[:, idx, idx] = 0
    c_diag = (W ** 4 * d4[:, None]).sum(0)
    c31 = 4.0 * ((W ** 3) * (s31 @ W)).sum(0)
    c22 = 3.0 * ((W ** 2) * (s22 @ (W ** 2))).sum(0)
    c211 = 6.0 * torch.einsum("ai,abc,bi,ci->i", W ** 2, s211, W, W)
    c_ad = tru - (c_diag + c31 + c22 + c211)
    parts = [c_diag, c31, c22, c211, c_ad]
    tv = ((tru - tru.mean()) ** 2).sum()
    line = f"  {l:2d}"
    for c in parts:
        line += f" | {float(((c - c.mean()) ** 2).sum() / tv):5.2f} {float(c.mean() / tru.mean()):+6.2f}"
    biv = c_diag + c31 + c22
    line += f" | {float(np.corrcoef(biv.numpy(), tru.numpy())[0, 1]):.3f}   {float(np.corrcoef((biv + c211).numpy(), tru.numpy())[0, 1]):.3f}"
    print(line, flush=True)
