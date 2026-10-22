"""What predicts the per-neuron pre-activation kappa4 diagonal? MC (mc_oracle_0, 1e7 samples) against:
chain   the chain's own K4d (memo22 + split24rr),
rad     the radial scale-mixture model 3 v var_i^2 with v from the MC (2,2) level (v = mean K22_ij/(var_i var_j)),
radc    3 v var_i^2 with the chain's own level (v fitted to the chain's K4d mean),
and the readout MSE vs the MC post-mean when each predictor replaces MC's kappa4 in an otherwise exact readout
(MC mean, var, kappa3), plus the MSE with the best single scalar multiple of each predictor.
MC noise of the kappa4 estimate per neuron ~ sqrt(24/N) var^2 (Gaussian approximation)."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain, relu_wick
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); N = float(mc["N"])
pairs = {}
for f in ("mc_oracle22_0.npz", "mc_oracle31_0.npz"):
    d = np.load(HERE / "runs" / f)
    for l in d["layers"]:
        pairs[int(l)] = d[f"k22_{int(l)}"].astype(np.float64)
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], k4="memo22", corr="split24rr", record=True)
print(" l | MC k4 cv  noise/cv | corr(MC, chain) corr(MC, var^2) | v_MC  v_chain | readout MSE: MC k4 | chain | chain*c* (c*) | "
      "rad(v_MC) | rad*c* | zero k4")
extra = []
for l in range(2, len(info)):
    mv, vv, k3, k4 = (mc[k][l].astype(np.float64) for k in ("mean", "var", "k3", "k4"))
    ref = mc["post_mean"][l]
    w, _, _ = relu_wick(torch.tensor(mv), torch.tensor(vv))
    base = (w[(0, 1)] + w[(3, 1)] * torch.tensor(k3) / 6).numpy()
    g = (w[(4, 1)] / 24).numpy()                       # readout sensitivity to kappa4
    rd = lambda k: float(((base + g * k - ref) ** 2).mean())
    def best(p):                                       # best scalar c for c * p in the readout
        r = ref - base
        c = float((g * p) @ r / ((g * p) @ (g * p)))
        return rd(c * p), c
    ch = info[l]["K4d"].numpy()
    v2 = vv * vv
    cv = k4.std() / k4.mean(); noise = np.sqrt(24 / N) * v2.mean() / k4.std()
    vmc = None
    if l in pairs:
        K = pairs[l]; n = K.shape[0]
        vmc = float((K / np.outer(vv, vv))[~np.eye(n, dtype=bool)].mean())
    vch = float(ch.mean() / (3 * v2.mean()))
    radmc = 3 * vmc * v2 if vmc is not None else None
    bc, cc = best(ch); br, cr = best(3 * v2)
    radch = 3 * vch * v2                               # sigma^4 pattern at the chain's own level (no oracle)
    extra.append((l, rd(radch), cr / vch if vch else 0, rd(ch)))
    print(f"{l:2d} | {cv:.3f} {noise:.3f} | {np.corrcoef(k4, ch)[0,1]:.3f} {np.corrcoef(k4, v2)[0,1]:.3f} | "
          f"{(vmc if vmc is not None else float('nan')):.4f} {vch:.4f} | {rd(k4):.2e} | {rd(ch):.2e} | {bc:.2e} ({cc:.2f}) | "
          f"{(rd(radmc) if radmc is not None else float('nan')):.2e} | {br:.2e} | {rd(0 * k4):.2e}")
print("\n l | readout MSE: chain K4d | sigma^4 pattern at the chain's level (radc) | best level / chain level")
for l, a_, c_, b_ in extra:
    print(f"{l:2d} | {b_:.2e} | {a_:.2e} | {c_:.2f}")
