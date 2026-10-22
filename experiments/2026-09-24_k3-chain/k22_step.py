"""Decompose the post-(2,2) kappa4 deficit at layer 14: feed the nonlinear step the chain's own inputs vs the MC pre-activation
kappa4 inputs (K22 and K4 diag), compare the post (2,2) slice with the MC post (2,2) at layer 14."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
pre = np.load(HERE / "runs" / "mc_oracle22_0.npz"); post = np.load(HERE / "runs" / "mc_oracle22post_0.npz")
mc = np.load(HERE / "runs" / "mc_oracle_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
calls = []
orig = ch.nonlin_slices
def hook(mu, C, t, D, K4d, K22, **kw):
    calls.append((mu.clone(), C.clone(), t.clone(), D.clone(), None if K4d is None else K4d.clone(), None if K22 is None else K22.clone(), kw))
    return orig(mu, C, t, D, K4d, K22, **kw)
ch.nonlin_slices = hook
ch.run_chain(Ws, k4="memo22", corr="split24rr")
ch.nonlin_slices = orig
off = lambda M: M[~np.eye(M.shape[0], dtype=bool)]
l = 14
mu, C, t, D, K4d, K22, kw = calls[l]
ref = off(post[f"k22_{l}"].astype(np.float64))
def run(**over):
    args = dict(mu=mu, C=C, t=t, D=D, K4d=K4d, K22=K22); args.update(over)
    r = orig(args["mu"], args["C"], args["t"], args["D"], args["K4d"], args["K22"], **kw)
    x = off(r["K22"].numpy()); return x.mean() / ref.mean(), np.corrcoef(x, ref)[0, 1]
K22mc = torch.tensor(pre[f"k22_{l}"].astype(np.float64)); K22mc.fill_diagonal_(0.0)
K4mc = torch.tensor(mc["k4"][l])
Cmc = torch.tensor(pre[f"C{l}"].astype(np.float64)); tmc = torch.tensor(mc["k3"][l]); mumc = torch.tensor(mc["mean"][l])
print("post (2,2) at layer 14: ratio chain/MC of the pair-mean, and corr")
for name, over in [("chain inputs", {}), ("MC K22_pre", dict(K22=K22mc)), ("MC K22_pre + MC K4 diag", dict(K22=K22mc, K4d=K4mc)),
                   ("MC K22, K4, C", dict(K22=K22mc, K4d=K4mc, C=Cmc)), ("MC K22, K4, C, t, mu", dict(K22=K22mc, K4d=K4mc, C=Cmc, t=tmc, mu=mumc))]:
    r, c = run(**over)
    print(f"  {name:28s} ratio {r:.3f}  corr {c:.3f}")

# residual structure with exact kappa4 inputs
r = orig(mumc, Cmc, tmc, D, K4mc, K22mc, **kw)
w, d1, d2 = ch.relu_wick(mumc, torch.diagonal(Cmc))
resid = ref - off(r["K22"].numpy())
Cd = torch.diagonal(Cmc)
C0 = (Cmc - torch.diag(Cd)).numpy()
feats = {
    "const": np.ones_like(resid),
    "Phi_a*Phi_b": off(np.outer(d1.numpy(), d1.numpy())),
    "K22pre": off(K22mc.numpy()),
    "Phi Phi K22pre": off(np.outer(d1.numpy(), d1.numpy()) * K22mc.numpy()),
    "C0^2": off(C0 ** 2),
    "w22w22 C0^2": off(np.outer(w[(2, 2)].numpy(), w[(2, 2)].numpy()) * C0 ** 2),
    "m1_a m1_b": off(np.outer(w[(0, 1)].numpy(), w[(0, 1)].numpy())),
    "w12 w12 (=4 m1 m1)": off(np.outer(w[(1, 2)].numpy(), w[(1, 2)].numpy())),
    "K4_a*K4_b": off(np.outer(K4mc.numpy(), K4mc.numpy())),
}
print(f"\nresidual (MC - chain with exact inputs): mean {resid.mean():.3e}  (MC mean {ref.mean():.3e})  std {resid.std():.3e}")
for k, f in feats.items():
    c = np.corrcoef(f, resid)[0, 1]
    b = float(f @ resid / (f @ f))
    print(f"  corr with {k:22s} {c:+.3f}   single-feature fit coef {b:.3e}")
