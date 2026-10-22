"""Is there a shared rank-1 non-Gaussian component in k=4's kappa4 beyond the one-factor scale mixture? Residuals of
k=4's slices after the closure's form (level v fitted): R4_i = K4d_i - 3 v var_i^2, R22_ij = K22_ij - v (var_i var_j + 2 C_ij^2).
A rank-1 component lam b(x)b(x)b(x)b would make R22 = lam (b^2)(b^2)^T (rank one) and R4 = lam b^4. Report the top
singular share of R22, whether its top vector predicts R4, and its alignment with C's top eigenvector and with mu.
args: n depth seed"""
import sys
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
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
print(f"n={n} L={depth} seed={seed}")
for l in range(2, depth):
    T = out[f"pre{l}"]
    C = T[2].to_tensor().numpy(); var = np.diag(C).copy(); C0 = C - np.diag(var); mu = T[1].to_tensor().numpy()
    K4d = T[4].get_dslice((4,)).numpy(); K22 = T[4].get_dslice((2, 2)).numpy().copy(); np.fill_diagonal(K22, 0)
    pat = np.outer(var, var) + 2 * C0 * C0; np.fill_diagonal(pat, 0)
    v = (K22 * pat).sum() / (pat * pat).sum()                      # one-factor level fitted on the (2,2) slice
    R22 = K22 - v * pat
    R4 = K4d - 3 * v * var * var
    U, S, Vt = np.linalg.svd(R22)
    share1 = S[0] ** 2 / (S ** 2).sum()
    b2 = U[:, 0] * np.sign(U[:, 0].sum())                          # candidate b^2 (up to scale)
    c4 = np.corrcoef(b2 ** 2, R4)[0, 1]                            # does b^4 = (b^2)^2 predict R4?
    ev = np.linalg.eigh(C)[1][:, -1]
    al_C = abs(np.corrcoef(np.sqrt(np.abs(b2)), np.abs(ev))[0, 1]); al_mu = abs(np.corrcoef(np.sqrt(np.abs(b2)), np.abs(mu))[0, 1])
    fitK22 = 1 - (R22 ** 2).sum() / (K22 ** 2).sum()
    print(f"  l={l}: one-factor explains {fitK22:.3f} of K22's energy; residual top-1 singular share {share1:.3f}; "
          f"corr((b^2)^2, R4) {c4:+.3f}; |b| vs |top eig of C| {al_C:.2f}, vs |mu| {al_mu:.2f}", flush=True)
