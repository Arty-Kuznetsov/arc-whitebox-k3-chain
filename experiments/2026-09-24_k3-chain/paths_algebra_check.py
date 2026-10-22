# Check: ARC's factored-AUGMENT (1,1,1) path blocks (factor_k3.py, legs loop, budget 3) minus the SIMPLE tree
# equal Sym(Y (x) 3I (x) U) with U = w1*C0 and Y = d1 D^T d3 + d2 D d2 + 1/2 d2 K22 d3 + d2 C0^2 d3 (chain.py `paths`).
import torch
from mlp_kprop.factor_k3 import FactoredTensor
torch.set_default_dtype(torch.float64); torch.manual_seed(0)
n = 9
w = {k: torch.rand(n) + 0.2 for k in (1, 2, 3, 4, 5)}
def symz(M): M = 0.5 * (M + M.T); M.fill_diagonal_(0.0); return M
C0 = symz(torch.randn(n, n)); K22 = symz(torch.randn(n, n))
D = torch.randn(n, n); D.fill_diagonal_(0.0)          # D[i,j] = kappa3[i,i,j]
WK21 = D / 2                                           # ARC: diagslice(kappa3,(2,1))/2, near (doubled) index first
legs = [(1, 1, C0, 1), (2, 1, WK21, 2), (1, 2, WK21.T, 2), (2, 2, K22 / 4, 2), (2, 2, C0 * C0 / 2, 2)]
I3 = torch.eye(n) * 3
arc = FactoredTensor(n=n, d=3)
for a_l, b_l, s_l, c_l in legs:
    rights = [(a_r, b_r, s_r) for a_r, b_r, s_r, c_r in legs if c_l + c_r <= 3]
    fac1 = w[a_l][:, None] * s_l
    for a_r, b_r, s_r in rights:
        if (a_l, b_l, c_l) == (1, 1, 1) and (a_r, b_r) == (1, 1):
            continue  # the SIMPLE tree, not part of the extra births
        fac3 = w[b_l + a_r][:, None] * w[b_r][None, :] * s_r
        arc.add_factors_((fac1, I3, fac3.T))
d1, d2, d3 = w[1], w[2], w[3]
Y = d1[:, None] * D.T * d3[None, :] + d2[:, None] * D * d2[None, :] + 0.5 * d2[:, None] * K22 * d3[None, :] \
    + d2[:, None] * C0 * C0 * d3[None, :]
U = d1[:, None] * C0
mine = FactoredTensor(n=n, d=3, factors=(Y, I3, U))
Ta, Tm = arc.to_tensor(), mine.to_tensor()
print("max|arc - mine| =", float((Ta - Tm).abs().max()), " max|arc| =", float(Ta.abs().max()))
# the (2,1) slice at birth of Sym(Y (x) 3I (x) U): kappa[i,i,j] = Y_ij U_ij (hub j)
S21 = torch.stack([torch.stack([Tm[i, i, j] for j in range(n)]) for i in range(n)]); S21.fill_diagonal_(0.0)
print("max|(2,1) slice - Y*U| =", float((S21 - Y * U).abs().max()))
print("diag3 max =", float(torch.stack([Tm[i, i, i] for i in range(n)]).abs().max()))
# old chain E3 (g3 = d3/2) vs the reference E3 part alone
Ye3 = d1[:, None] * D.T * d3[None, :]
arc_e3 = FactoredTensor(n=n, d=3)
arc_e3.add_factors_((w[1][:, None] * C0, I3, (w[3][:, None] * w[1][None, :] * WK21).T))      # left C, right (2,1)
arc_e3.add_factors_((w[1][:, None] * WK21.T, I3, (w[3][:, None] * w[1][None, :] * C0).T))    # left (1,2), right C
print("E3: ref vs Sym(Ye3 x 3I x U):", float((arc_e3.to_tensor() - FactoredTensor(n=n, d=3, factors=(Ye3, I3, U)).to_tensor()).abs().max()),
      "| ref vs old chain (Ye3/2):", float((arc_e3.to_tensor() - FactoredTensor(n=n, d=3, factors=(Ye3 / 2, I3, U)).to_tensor()).abs().max()))
