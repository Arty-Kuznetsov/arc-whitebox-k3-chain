"""Analysis for mc_k3step.py: decompose the next-layer kappa3 diagonal into repeated-index (MC) + all-distinct parts and
compare the all-distinct truth with SIMPLE's pass-through + tree + feed evaluated on exact (MC) inputs."""
import math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); ks = np.load(HERE / "runs" / "mc_k3step_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy").astype(np.float64)
_, info = run_chain([torch.tensor(w) for w in W], record=True, k4="memo22", corr="split24rr", k4pat=True, k22pat=True,
                    c0d=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2")
nrm = np.linalg.norm
for l in [int(v) for v in ks["layers"]]:
    Wp = W[l + 1]; n = Wp.shape[0]
    C = ks[f"C{l}"]; var = np.diag(C).copy(); C0 = C - np.diag(var)
    Dz = ks[f"Dz{l}"]; k3z = np.diag(Dz).copy(); Dz0 = Dz - np.diag(k3z)
    K21y = ks[f"K21y{l}"]; K3y = ks[f"K3y{l}"]; K21y0 = K21y - np.diag(np.diag(K21y))
    mu = mc["mean"][l]; s = np.sqrt(var); a = mu / s
    d1 = 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2))); d2 = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / s
    v = mc["k4"][l].mean() / (3 * (mc["var"][l] ** 2).mean())
    tp = mc["k3"][l + 1]
    R3 = (Wp ** 3).T @ K3y + 3 * ((Wp ** 2) * (K21y0 @ Wp)).sum(0)
    T = tp - R3
    Wt = d1[:, None] * Wp
    passd = ks[f"u3{l}"] - (Wt ** 3).T @ k3z - 3 * ((Wt ** 2) * (Dz0 @ Wt)).sum(0)
    V = C0 @ Wt
    tree = 3 * ((Wp * d2[:, None]) * (V * V - (C0 * C0) @ (Wt * Wt))).sum(0)
    hx = Wp * (d2 * var)[:, None]
    feed2 = 1.5 * v * (hx.sum(0) * (Wt * V).sum(0) - 2 * (hx * Wt * V).sum(0))
    res = T - passd - tree - feed2
    ch = info[l + 1]["t"].numpy(); err = tp - ch
    c = lambda x, y: np.corrcoef(x, y)[0, 1]
    print(f"layer {l} -> {l+1} (v = {v:.4f}): |t'| {nrm(tp):.3e} | repeated R3 {nrm(R3):.3e} | all-distinct T {nrm(T):.3e}")
    print(f"   pass {nrm(passd):.3e}  tree {nrm(tree):.3e}  feedP2 {nrm(feed2):.3e}  (P1 = v*tree {nrm(v*tree):.3e})")
    print(f"   residual T - pass - tree - feedP2: {nrm(res):.3e} ({nrm(res)/nrm(tp):.3f} of |t'|); with P1 too: "
          f"{nrm(res - v*tree):.3e}; chain error |t' - t_chain| {nrm(err):.3e} ({nrm(err)/nrm(tp):.3f})")
    print(f"   corr(residual, chain error) {c(res, err):.3f}; corr(residual, tree) {c(res, tree):.3f}; corr(residual, pass) {c(res, passd):.3f}; "
          f"fit residual ~ tree: coef {float(res @ tree / (tree @ tree)):.3f}")
