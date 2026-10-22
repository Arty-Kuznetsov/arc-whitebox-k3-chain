"""Compare the MC kappa4->kappa3 feed F_i (mc_feed.py) with the scale-mixture closure prediction, and with the chain's
kappa3 diagonal error at the next layer (chain = k3e configuration)."""
import math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); fd = np.load(HERE / "runs" / "mc_feed_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
mu_c, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], k4="memo22", corr="split24rr", k4pat=True,
                       k22pat=True, c0d=True, k4lvl=True, record=True)
for l in fd["layers"]:
    l = int(l)
    mu, var, k4 = mc["mean"][l], mc["var"][l], mc["k4"][l]
    s = np.sqrt(var); a = mu / s
    d2 = np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / s
    v = k4.mean() / (3 * (var ** 2).mean())
    C = fd[f"C{l}"].astype(np.float64); Szu = fd[f"Szu{l}"].astype(np.float64); Su2 = fd[f"Su2{l}"]
    Wn = W[l + 1].astype(np.float64)
    ksm = v * (np.diag(C)[:, None] * Su2[None, :] + 2 * Szu ** 2)
    kmc = fd[f"k4zu{l}"].astype(np.float64)
    Fmc = fd[f"F{l}"]; Fsm = 1.5 * ((Wn * d2[:, None]) * ksm).sum(0)
    t_mc = mc["k3"][l + 1]; t_ch = info[l + 1]["t"].numpy()
    err = t_mc - t_ch
    c = lambda x, y: np.corrcoef(x, y)[0, 1]
    sl = lambda x, y: float(x @ y / (x @ x))
    print(f"layer {l} -> {l+1}: v = {v:.4f}")
    print(f"  kappa4(z_x,z_x,u_i,u_i): MC vs closure corr {c(kmc.ravel(), ksm.ravel()):.3f}, slope MC/closure {sl(ksm.ravel(), kmc.ravel()):.3f}")
    print(f"  F (feed into t_{l+1}): |F_mc| {np.linalg.norm(Fmc):.3e}, |F_sm| {np.linalg.norm(Fsm):.3e}, corr {c(Fmc, Fsm):.3f}, "
          f"slope {sl(Fsm, Fmc):.3f}")
    print(f"  t_{l+1}: |t_mc| {np.linalg.norm(t_mc):.3e}; chain error |t_mc - t_chain| {np.linalg.norm(err):.3e} "
          f"(rel {np.linalg.norm(err)/np.linalg.norm(t_mc):.3f}); corr(err, F_mc) {c(err, Fmc):.3f}, corr(err, F_mc - F_sm) "
          f"{c(err, Fmc - Fsm):.3f}; |F_mc - F_sm| {np.linalg.norm(Fmc - Fsm):.3e}")
    print(f"  regression of the chain error on F_mc: slope {sl(Fmc, err):.3f}, residual fraction "
          f"{np.linalg.norm(err - sl(Fmc, err) * Fmc) / np.linalg.norm(err):.3f}")
