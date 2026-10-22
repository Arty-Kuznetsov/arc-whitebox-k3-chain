"""Per-layer chain (k3f config) vs MC: variance, kappa3 diagonal t, kappa3 (2,1) slice D, off-diagonal covariance.
slope = <chain, MC>/<chain, chain> (unbiased by MC noise), corr, and the variance error split into bias and pattern."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); ma = np.load(HERE / "runs" / "mc_oracle_all_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
K3F = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2", margleg=True)
Cs = []
orig = ch.nonlin_slices
def hook(mu, C, t, D, K4d, K22, **kw):
    Cs.append(C.clone()); return orig(mu, C, t, D, K4d, K22, **kw)
ch.nonlin_slices = hook
mu, info = ch.run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record=True, **K3F)
ch.nonlin_slices = orig
sl = lambda c, m: float(c @ m / (c @ c))
cr = lambda c, m: float(np.corrcoef(c, m)[0, 1])
print(" l | var: rel.bias  rel.RMS (MC noise) | t: slope corr | D: slope corr | C0: slope corr")
n = W.shape[1]; off = ~np.eye(n, dtype=bool)
for l, r in enumerate(info):
    v, vm = r["var"].numpy(), mc["var"][l]
    bias = float(np.mean(v / vm - 1)); rms = float(np.sqrt(np.mean((v / vm - 1) ** 2)))
    s = f"{l:2d} | {bias:+.2e} {rms:.2e} ({np.sqrt(2 / float(mc['N'])):.1e})"
    if l >= 1:
        s += f" | {sl(r['t'].numpy(), mc['k3'][l]):.4f} {cr(r['t'].numpy(), mc['k3'][l]):.4f}"
        Dc, Dm = r["D"].numpy()[off], ma[f"D{l}"].astype(np.float64)[off]
        s += f" | {sl(Dc, Dm):.4f} {cr(Dc, Dm):.4f}"
    Cc, Cm = Cs[l].numpy()[off], ma[f"C{l}"].astype(np.float64)[off]
    s += f" | {sl(Cc, Cm):.5f} {cr(Cc, Cm):.5f}"
    print(s)
