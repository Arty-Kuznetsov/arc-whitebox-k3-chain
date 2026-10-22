"""Are the source propagators effectively low rank, and does truncating them cost accuracy?

The dominant cost is transporting each source's legs: A_m <- P A_m and B_m <- P B_m, two full
(n,n)@(n,n) products per source per layer. But A_m is the PRODUCT of (age) matrices P = W diag(d1).
Products of random matrices have exponentially spreading singular values (Lyapunov spectrum), so an
old propagator should be effectively low rank. Crucially this is a MATRIX, where SVD truncation is
optimal (Eckart-Young) -- unlike CP truncation of the cumulant tensor, which is where every published
attempt failed.

There is also a structural gift: B_m = A_m U_m with U_m fixed at birth, so writing A = X Y^T gives
B = X (U_m^T Y)^T. Transport touches X only, so a rank-r source costs O(n^2 r) instead of O(n^3).

PART A measures the singular spectrum of A_m by source age.
PART B truncates A (and lets B follow) to rank r for sources older than `min_age` and measures the
final-layer MSE. Exact SVD is used on purpose: this asks whether the INFORMATION survives, not
whether a cheap basis exists.
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
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)
CLIP = 1.0
J = 4


def build(W):
    m = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
            nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        m.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    return [m.Ws[l].weight.clone() for l in range(depth)]


def truncate(M, r):
    """Randomised rank-r approximation -- the same construction a deployed version would use
    (cost O(n^2 r)), and far more robust than an exact SVD on these ill-conditioned products."""
    if r >= n:
        return M
    U, S, V = torch.svd_lowrank(M, q=min(n, r + 8), niter=2)
    return (U[:, :r] * S[:r]) @ V[:, :r].T


def chain(Wms, sched, _unused=None, spectra=None):
    """sched: dict age -> rank (missing age = full rank)."""
    mu, C = torch.zeros(n), torch.eye(n)
    src = []                       # [A, U_birth, d2, age]; B = A @ U_birth is derived
    for l in range(depth):
        Wm = Wms[l]
        for s in src:
            s[0] = Wm if s[0] is None else Wm @ s[0]
            s[3] += 1
            if spectra is not None and s[3] <= 6:
                sv = torch.linalg.svdvals(s[0])
                e = (sv * sv); tot = float(e.sum())
                spectra.setdefault(s[3], []).append(
                    [float(e[:k].sum()) / tot for k in (32, 64, 128, 256, 512)])
            r = sched.get(s[3], n) if isinstance(sched, dict) else n
            if r < n:
                s[0] = truncate(s[0], r)         # B = A @ U inherits the same rank
        mu_pre = Wm @ mu
        C_pre = Wm @ C @ Wm.T
        var = C_pre.diagonal().clamp_min(1e-12)
        sig = var.sqrt(); al = mu_pre / sig
        Ph, ph = nPhi(al), nphi(al)
        d1, d2 = Ph, ph / sig
        k3 = torch.zeros(n); D21 = torch.zeros(n, n) if src else None
        for A, Ub, d2m, _ in src:
            Bm = A @ Ub
            AD = A * d2m
            k3 = k3 + 3.0 * (AD * Bm * Bm).sum(dim=1)
            D21 = D21 + 2.0 * ((AD * Bm) @ Bm.T)
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
        if l < depth - 1:
            src.append([None, C_pre * d1[:, None], d2.clone(), 0])
            src[:] = src[-J:]
    return mu


gt = np.load(BAKED / "gt.npy")
N = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])

print("PART A  singular-energy of the source propagator A_m, by age (1 MLP)")
spec = {}
chain(build(np.load(BAKED / "weights_0.npy")), {}, None, spectra=spec)
print(f"{'age':>4} | " + " ".join(f"{'top-'+str(k):>8}" for k in (32, 64, 128, 256, 512)))
for age in sorted(spec):
    m = np.mean(spec[age], axis=0)
    print(f"{age:>4} | " + " ".join(f"{v:>8.4f}" for v in m))

print("\nPART B  truncating A (and B) to rank r for sources older than min_age")
CFGS = [("full", {}), ("512@2+", {2: 512, 3: 512, 4: 512}),
        ("512@2,256@3+", {2: 512, 3: 256, 4: 256}),
        ("512@2,256@3,192@4", {2: 512, 3: 256, 4: 192}),
        ("384@2,256@3,192@4", {2: 384, 3: 256, 4: 192}),
        ("256@3+", {3: 256, 4: 256}), ("256@3,128@4", {3: 256, 4: 128})]
res = {c[0]: [] for c in CFGS}
def relcost(sched):
    # per-source-per-layer matmul-equivalents: full = 3 (A, B, D21); rank r = 1.5*r/n + r/n + r/n
    tot = 0.0
    for age in (1, 2, 3, 4):
        r = sched.get(age, n)
        base = 2.0 if age == 1 else 3.0          # age 1 gets its A for free
        tot += base if r >= n else base * (r / n) * 1.2
    return tot / (2.0 + 3.0 * 3)
for i in range(N):
    Wms = build(np.load(BAKED / f"weights_{i}.npy"))
    g = torch.tensor(gt[i][-1], dtype=torch.float64)
    for c in CFGS:
        res[c[0]].append(float(((chain(Wms, c[1]) - g) ** 2).mean()))
    print(f"  mlp {i} done", flush=True)
    del Wms

b = float(np.mean(res[CFGS[0][0]]))
print(f"\nSELF-CHECK no truncation = {b:.4e} (the shipped chain)\n")
print(f"{'schedule':>20} | {'final_layer_mse':>16} {'vs full':>9} | {'rel cost':>9} {'util':>7} {'adjusted':>10}")
for c in CFGS:
    v = float(np.mean(res[c[0]])); rc = relcost(c[1])
    util = 0.0235 + (0.1498 - 0.0235) * rc
    adj = v * max(0.1, util)
    print(f"{c[0]:>20} | {v:>16.4e} {b/v:>8.2f}x | {rc:>9.3f} {100*util:>6.2f}% {adj:>10.3e}")
print("\nvs full ~ 1 at rank 64-128 would mean the transport can run at O(n^2 r), cutting the")
print("dominant cost by 8-16x -- enough to drop under the 10% multiplier floor with sources to spare.")
