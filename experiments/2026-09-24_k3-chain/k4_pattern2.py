"""Beyond var_i^2: what explains the per-neuron kappa4 diagonal (MC)? Scale-mixture with a per-neuron variance response
rho_i (kappa4_i = 3 v rho_i^2). Candidates: rho = var (pure scale); rho from the MC (2,2) slice, rho_i ~ sum_j (K22_ij +
2 C_ij^2)/var_j (the covariance of dz_i^2 with the common-scale statistic q); rho = var + c * sum_j C_ij^2 / var_j (cheap).
Reports corr with MC kappa4_i and the readout MSE of the best scalar multiple (MC mean, var, kappa3 exact)."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from chain import relu_wick
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz")
src = {}
for f in ("mc_oracle22_0.npz", "mc_oracle31_0.npz"):
    d = np.load(HERE / "runs" / f)
    for l in d["layers"]:
        src[int(l)] = (d[f"C{int(l)}"].astype(np.float64), d[f"k22_{int(l)}"].astype(np.float64))
for l in sorted(src):
    C, K22 = src[l]; n = C.shape[0]; off = ~np.eye(n, dtype=bool)
    mv, vv, k3, k4 = (mc[k][l].astype(np.float64) for k in ("mean", "var", "k3", "k4"))
    K22o = K22.copy(); np.fill_diagonal(K22o, 0.0); C0 = C - np.diag(np.diag(C))
    rho_mc = ((K22o + 2 * C0 ** 2) / vv[None, :]).sum(1) / n            # off-diagonal part of Cov(dz_i^2, q) * n
    g2 = (C0 ** 2 / vv[None, :]).sum(1) / n
    w, _, _ = relu_wick(torch.tensor(mv), torch.tensor(vv))
    base = (w[(0, 1)] + w[(3, 1)] * torch.tensor(k3) / 6).numpy(); gk = (w[(4, 1)] / 24).numpy(); ref = mc["post_mean"][l]
    def best(p):
        r = ref - base; c = float((gk * p) @ r / ((gk * p) @ (gk * p)))
        return float(((base + gk * c * p - ref) ** 2).mean())
    feats = {"var^2": vv ** 2, "rho_mc^2": rho_mc ** 2, "var*rho_mc": vv * rho_mc}
    # cheap: rho = var + c g2, fit c by correlation scan
    bestc, bestr = 0.0, -1
    for c in np.linspace(-5, 20, 101):
        r_ = np.corrcoef(k4, (vv + c * g2) ** 2)[0, 1]
        if r_ > bestr: bestc, bestr = c, r_
    feats[f"(var+{bestc:.2f} g2)^2"] = (vv + bestc * g2) ** 2
    out = " | ".join(f"{k}: corr {np.corrcoef(k4, p)[0,1]:.3f} rd {best(p):.2e}" for k, p in feats.items())
    print(f"l{l:2d} exact-k4 rd {best(k4):.2e} | {out}")
