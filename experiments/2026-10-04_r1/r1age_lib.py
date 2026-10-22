"""Closure patterns and fits shared by the R1 age/type splits (r1age.py defines the same functions inline).
The scale-mixture closure k4 = v Sym(C (x) C): K4d ~ 3 var^2, K31 ~ 3 var_i C0_ij, K22 ~ var_i var_j + 2 C0_ij^2,
S = k4[aabc] ~ var_a C0_bc + 2 C0_ab C0_ac (repeated entries zeroed, as ARC's dslices)."""
import torch


def offd_mask(n):
    return ~torch.eye(n, dtype=torch.bool)


def zrep3(S):
    n = S.shape[0]
    idx = torch.arange(n)
    S = S.clone()
    S[idx, idx, :] = 0
    S[idx, :, idx] = 0
    S[:, idx, idx] = 0
    return S


def patterns(C):
    var = torch.diagonal(C)
    C0 = C - torch.diag(var)
    offd = offd_mask(C.shape[0])
    pS = zrep3(var[:, None, None] * C0[None, :, :] + 2.0 * C0[:, :, None] * C0[:, None, :])
    return {"K4d": 3.0 * var * var, "K31": 3.0 * var[:, None] * C0, "K22": (torch.outer(var, var) + 2 * C0 * C0) * offd,
            "S": pS}


def lsfit(X, P):
    return float((X * P).sum() / (P * P).sum())
