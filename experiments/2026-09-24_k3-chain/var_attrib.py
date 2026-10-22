"""Where does the per-neuron variance error come from (k3f chain, MLP 0)? At layers l = 5, 9, 13:
  delta var_{l+1} (chain - MC)  =  sum_a W'^2_ai delta K2_a  (diagonal channel)  +  rest (off-diagonal covariance channel)
with K2_MC from mc_k3step (post variance, 4e6). delta K2_a is then attributed to the chain's per-neuron input errors
(mu, var, t, K4d vs MC 1e7) through the analytic sensitivities of K2 = P2 - P1^2 (finite differences of the step)."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz"); ks = np.load(HERE / "runs" / "mc_k3step_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
K3F = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2")
rec = []
orig = ch.nonlin_slices
def hook(mu, C, t, D, K4d, K22, **kw):
    o = orig(mu, C, t, D, K4d, K22, **kw)
    rec.append(dict(mu=mu.numpy().copy(), var=torch.diagonal(C).numpy().copy(), t=t.numpy().copy(),
                    K4d=(K4d.numpy().copy() if K4d is not None else np.zeros_like(mu.numpy())), K2=o["K2"].numpy().copy()))
    return o
ch.nonlin_slices = hook
ch.run_chain([torch.tensor(w, dtype=torch.float64) for w in W], **K3F)
ch.nonlin_slices = orig

def K2f(mu, var, t, k4):
    w, _, _ = ch.relu_wick(torch.tensor(mu), torch.tensor(var))
    P1 = w[(0, 1)] + w[(3, 1)] * torch.tensor(t) / 6 + w[(4, 1)] * torch.tensor(k4) / 24
    P2 = w[(0, 2)] + w[(3, 2)] * torch.tensor(t) / 6 + w[(4, 2)] * torch.tensor(k4) / 24
    return (P2 - P1 * P1).numpy()
nrm = np.linalg.norm
for l in [int(v) for v in ks["layers"]]:
    r = rec[l]; Wp = W[l + 1].astype(np.float64)
    K2mc = ks[f"Vy{l}"]
    dK2 = r["K2"] - K2mc
    dvar_next = rec[l + 1]["var"] - mc["var"][l + 1]
    diag_ch = (Wp ** 2).T @ dK2
    off_ch = dvar_next - diag_ch
    ref = {k: mc[m][l] for k, m in (("mu", "mean"), ("var", "var"), ("t", "k3"), ("K4d", "k4"))}
    base = K2f(ref["mu"], ref["var"], ref["t"], ref["K4d"])
    parts = {}
    for k in ("mu", "var", "t", "K4d"):
        args = dict(ref); args[k] = r[k]
        parts[k] = K2f(args["mu"], args["var"], args["t"], args["K4d"]) - base
    step_err = K2mc - base          # error of the step itself with MC inputs (plus MC noise)
    print(f"layer {l}->{l+1}: |dvar_{l+1}| {nrm(dvar_next):.3e} (rel {nrm(dvar_next)/nrm(mc['var'][l+1]):.2e}); "
          f"diagonal channel {nrm(diag_ch):.3e}, off-diagonal channel {nrm(off_ch):.3e}; corr(dvar, diag) {np.corrcoef(dvar_next, diag_ch)[0,1]:.3f}")
    print(f"   |dK2| {nrm(dK2):.3e} (rel {nrm(dK2)/nrm(K2mc):.2e}); attribution of dK2 by input error: " +
          ", ".join(f"{k} {nrm(v):.2e} (corr {np.corrcoef(v, dK2)[0,1]:+.2f})" for k, v in parts.items()) +
          f"; step-with-MC-inputs error {nrm(step_err):.2e}")
