"""Which missing second-order diagram explains the all-distinct kappa3 one-step residual (k3step_analyze.py)?
Candidates, all contracted into the next-layer diagonal with MC inputs:
  E3 (edge x kappa3(2,1) tree): 1/2 d3_a d1_b d1_c C_ab kappa3[aac] ->
       3 sum_a W'_ai d3_a [ (C0 W~)_ai (D0 W~)_ai - sum_b W~_bi^2 C0_ab D0_ab ],  W~ = diag(d1) W', D0_ac = kappa3[aac]
  E3T: the same with D0 transposed (kappa3[acc], i.e. the c-doubled slice)
  TRI (Gaussian triangle): d2_a d2_b d2_c C_ab C_bc C_ca -> tr((X C0)^3), X = diag(W'_:,i d2)  (subset of neurons)
Reports the residual's norm and least-squares fits on the subset."""
import math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); ks = np.load(HERE / "runs" / "mc_k3step_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy").astype(np.float64)
rng = np.random.default_rng(0)
nrm = np.linalg.norm
for l in [int(v) for v in ks["layers"]]:
    Wp = W[l + 1]; n = Wp.shape[0]
    C = ks[f"C{l}"]; var = np.diag(C).copy(); C0 = C - np.diag(var)
    Dz = ks[f"Dz{l}"]; k3z = np.diag(Dz).copy(); D0 = Dz - np.diag(k3z)
    K21y = ks[f"K21y{l}"]; K3y = ks[f"K3y{l}"]; K21y0 = K21y - np.diag(np.diag(K21y))
    mu = mc["mean"][l]; s = np.sqrt(var); a = mu / s
    Phi = 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2))); phi = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    d1, d2, d3 = Phi, phi / s, -a * phi / s ** 2
    v = mc["k4"][l].mean() / (3 * (mc["var"][l] ** 2).mean())
    tp = mc["k3"][l + 1]
    R3 = (Wp ** 3).T @ K3y + 3 * ((Wp ** 2) * (K21y0 @ Wp)).sum(0)
    Wt = d1[:, None] * Wp
    passd = ks[f"u3{l}"] - (Wt ** 3).T @ k3z - 3 * ((Wt ** 2) * (D0 @ Wt)).sum(0)
    V = C0 @ Wt
    tree = 3 * ((Wp * d2[:, None]) * (V * V - (C0 * C0) @ (Wt * Wt))).sum(0)
    hx = Wp * (d2 * var)[:, None]
    feed2 = 1.5 * v * (hx.sum(0) * (Wt * V).sum(0) - 2 * (hx * Wt * V).sum(0))
    res = (tp - R3) - passd - tree - feed2
    UD = D0 @ Wt; UDT = D0.T @ Wt
    E3 = 3 * ((Wp * d3[:, None]) * (V * UD - (C0 * D0) @ (Wt * Wt))).sum(0)
    E3T = 3 * ((Wp * d3[:, None]) * (V * UDT - (C0 * D0.T) @ (Wt * Wt))).sum(0)
    sub = rng.choice(n, 128, replace=False)
    Ct = torch.tensor(C0)
    TRI = np.zeros(len(sub))
    for k, i in enumerate(sub):
        x = torch.tensor(Wp[:, i] * d2)
        XC = x[:, None] * Ct
        TRI[k] = float(torch.trace(XC @ XC @ XC))
    def fit(y, cols):
        X = np.stack(cols, 1); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
        return 1 - (r @ r) / (y @ y), c
    print(f"layer {l}->{l+1}: |res| {nrm(res):.3e}  |E3| {nrm(E3):.3e}  |E3T| {nrm(E3T):.3e}  |TRI|(sub, scaled) {nrm(TRI)*math.sqrt(n/len(sub)):.3e}")
    for name, cols in (("E3", [E3]), ("E3T", [E3T]), ("E3+E3T", [E3, E3T])):
        r2, c = fit(res, cols); print(f"   full: res ~ {name:7s} R2 {r2:.3f} coef {np.round(c, 3)}")
    for name, cols in (("TRI", [TRI]), ("E3+E3T+TRI", [E3[sub], E3T[sub], TRI])):
        r2, c = fit(res[sub], cols); print(f"   sub : res ~ {name:10s} R2 {r2:.3f} coef {np.round(c, 3)}")
