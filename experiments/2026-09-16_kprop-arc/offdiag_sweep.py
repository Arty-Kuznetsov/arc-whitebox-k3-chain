"""Off-diagonal covariance correction: how many sources can we afford, and are both terms needed?

vardiag.py: at J=4 the off-diagonal D21 correction alone is worth 1.82x (the diagonal part alone is
0.22x, i.e. harmful -- applying them together had masked both effects). It costs 2 matmuls per source
per layer on top of the 2 that transport the source, so the budget question is sharp.

    D21 = 2 * (A*d2*B) @ B^T   +   ((B*B)*d2) @ A^T
             \____term1____/       \____term2____/

Sweeps J, and at fixed J compares term1-only / term2-only / both, since dropping one halves the cost.
Costs assume one level of Strassen on every source matmul (measured meter ratio 0.88).
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
B_BUDGET = 2 ** 41
MM = n * n * (2 * n - 1) * 0.88          # one Strassen level
K2 = 5.179e10
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)
CLIP = 1.0


def cost(J, mode):
    extra = {"off": 0, "t1": 1, "t2": 1, "both": 2}[mode]
    per = 2 + extra
    mm = sum(per * min(l, J) for l in range(1, depth)) - (depth - 1 if J else 0)
    return K2 + mm * MM


def chain(Wms, J, mode):
    mu, C = torch.zeros(n), torch.eye(n)
    src = []
    for l in range(depth):
        Wm = Wms[l]
        for s in src:
            s[0] = Wm if s[0] is None else Wm @ s[0]
            s[1] = Wm @ s[1]
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt()
        al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        d1, d2 = Ph, ph / sig

        k3 = torch.zeros(n)
        D21 = torch.zeros(n, n) if (mode != "off" and src) else None
        for A, Bm, d2m in src:
            AD = A * d2m
            k3 = k3 + 3.0 * (AD * Bm * Bm).sum(dim=1)
            if mode in ("t1", "both"):
                D21 = D21 + 2.0 * ((AD * Bm) @ Bm.T)
            if mode in ("t2", "both"):
                D21 = D21 + ((Bm * Bm) * d2m) @ A.T

        mu_g = mu_pre * Ph + sig * ph
        mu = mu_g
        if src:
            mu = mu - (k3 / (var * sig)).clamp(-CLIP, CLIP) * sig * al * ph / 6.0
        ez2 = (mu_pre * mu_pre + var) * Ph + mu_pre * sig * ph
        var_post = (ez2 - mu_g * mu_g).clamp_min(0.0)
        C = torch.outer(Ph, Ph) * C_pre
        if D21 is not None:
            C = C + 0.5 * (d2[:, None] * d1[None, :] * D21 + d1[:, None] * d2[None, :] * D21.T)
        C.diagonal().copy_(var_post)

        for s in src:
            s[0] = s[0] * d1[:, None]
            s[1] = s[1] * d1[:, None]
        if J > 0 and l < depth - 1:
            src.append([None, C_pre * d1[:, None], d2.clone()])
            src[:] = src[-J:]
    return mu


gt = np.load(BAKED / "gt.npy")
N = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
CFGS = [(4, "off"), (3, "t1"), (4, "t1"), (5, "t1"), (6, "t1"), (8, "t1"), (6, "off"), (8, "off")]
res = {c: [] for c in CFGS}
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for c in CFGS:
        res[c].append(float(((chain(Wms, *c) - g) ** 2).mean()))
    print(f"  mlp {i} done", flush=True)
    del Wms, mlp

b = float(np.mean(res[(4, "off")]))
print(f"\n=== Off-diagonal covariance correction, cost-aware ({N} MLPs) ===")
print(f"SELF-CHECK J=4 no correction = {b:.4e} (the shipped chain)\n")
print(f"{'J':>3} {'terms':>6} | {'final_layer_mse':>16} {'vs none':>9} | {'% of B':>7} {'mult':>6} {'adjusted':>10}")
best = None
for c in CFGS:
    v = float(np.mean(res[c])); f = cost(*c)
    mult = max(0.1, f / B_BUDGET); adj = v * mult
    print(f"{c[0]:>3} {c[1]:>6} | {v:>16.4e} {b/v:>8.2f}x | {100*f/B_BUDGET:>6.2f}% {mult:>6.3f} "
          f"{adj:>10.3e}{' OVER' if f > B_BUDGET else ''}")
    if best is None or adj < best[1]:
        best = (c, adj)
print(f"\nbest adjusted: J={best[0][0]} terms={best[0][1]} at {best[1]:.4e}")
print("shipped configuration scores 1.0212e-07 locally (grader gave 9.9016e-08)")
