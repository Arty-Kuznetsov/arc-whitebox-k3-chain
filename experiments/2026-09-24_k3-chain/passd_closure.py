"""Can the ALL-DISTINCT pass-through of the pre-activation kappa3 (the only place the old kappa3 content enters the next
diagonal) be closed cheaply? MC 'passd' (mc_k3step) vs candidate closures, contracted with w~_i = d1 o W'_:,i:
  amp  : kappa3 ~ Sym(mu (x) C): 3[(w~.mu)(w~^T C0 w~) - 2 sum_a w~_a^2 mu_a (C0 w~)_a]
  ampD : the all-distinct part of a kappa3 whose (2,1) slice equals the MC D, built as Sym(D-derived rank structure)?
         (not available cheaply; skipped)
  diag-tree : 3 sum_a w~_a (d2-free) ... (the pure tree of the pre-activation: sum_a W'... ) 
Also reports how passd compares to the full t' and whether a scalar multiple of 'amp' tracks it across layers."""
import math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); ks = np.load(HERE / "runs" / "mc_k3step_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy").astype(np.float64)
nrm = np.linalg.norm
for l in [int(v) for v in ks["layers"]]:
    Wp = W[l + 1]
    C = ks[f"C{l}"]; var = np.diag(C).copy(); C0 = C - np.diag(var)
    Dz = ks[f"Dz{l}"]; k3z = np.diag(Dz).copy(); D0 = Dz - np.diag(k3z)
    mu = mc["mean"][l]; s = np.sqrt(var); a = mu / s
    d1 = 0.5 * (1 + np.vectorize(math.erf)(a / math.sqrt(2)))
    Wt = d1[:, None] * Wp
    passd = ks[f"u3{l}"] - (Wt ** 3).T @ k3z - 3 * ((Wt ** 2) * (D0 @ Wt)).sum(0)
    V = C0 @ Wt
    amp = 3 * ((Wt * mu[:, None]).sum(0) * (Wt * V).sum(0) - 2 * ((Wt ** 2) * mu[:, None] * V).sum(0))
    # alternative cheap regressors: the repeated-index pass-through parts and (w~.mu)^3-like terms
    rep = (Wt ** 3).T @ k3z + 3 * ((Wt ** 2) * (D0 @ Wt)).sum(0)
    wm = (Wt * mu[:, None]).sum(0)
    def fit(y, cols):
        X = np.stack(cols, 1); c, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ c
        return 1 - (r @ r) / (y @ y), c
    r_amp, c_amp = fit(passd, [amp])
    r_rep, c_rep = fit(passd, [rep])
    r_all, c_all = fit(passd, [amp, rep, wm ** 3, wm * (Wt * V).sum(0)])
    v = mc["k4"][l].mean() / (3 * (mc["var"][l] ** 2).mean())
    print(f"layer {l}->{l+1}: |passd| {nrm(passd):.3e} (|t'| {nrm(mc['k3'][l+1]):.3e}); R2 amp {r_amp:.3f} (coef {c_amp[0]:.4f}, "
          f"v/2 = {v/2:.4f}); R2 repeated-pass {r_rep:.3f} (coef {c_rep[0]:.3f}); R2 amp+rep+wm3+wm*q {r_all:.3f}")
