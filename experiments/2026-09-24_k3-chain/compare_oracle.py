"""Compare the chain's per-layer pre-activation statistics with the Monte-Carlo oracle (mc_oracle.py)."""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
N = float(mc["N"])
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], k4="memo22", corr="split24rr", record=True)
def st(a, b):
    c = np.corrcoef(a, b)[0, 1]
    slope = float(a @ b / (b @ b))
    rel = float(np.sqrt(((a - b) ** 2).mean() / (b ** 2).mean()))
    return c, slope, rel
print(f"MLP {i}, MC N={N:.0e}. For each quantity: corr(chain, MC), slope chain/MC, relative RMS diff")
print(" l |  mean_pre (rel)   |  var: corr slope rel   |  k3: corr slope rel   |  k4: corr slope rel  | MC noise: var, k3 (rel)")
for l, r in enumerate(info):
    mv, vv, k3, k4 = mc["mean"][l], mc["var"][l], mc["k3"][l], mc["k4"][l]
    a = st(r["mu_pre"].numpy(), mv)
    b = st(r["var"].numpy(), vv)
    c = st(r["t"].numpy(), k3) if l > 0 else (0, 0, 0)
    d = st(r["K4d"].numpy(), k4) if l > 1 else (0, 0, 0)
    nv = np.sqrt(2 / N); nk3 = np.sqrt(15 / N) * np.mean(vv ** 1.5) / np.sqrt((k3 ** 2).mean() + 1e-30)
    print(f"{l:2d} | {a[2]:.2e}          | {b[0]:.4f} {b[1]:.4f} {b[2]:.2e} | {c[0]:.3f} {c[1]:.3f} {c[2]:.3f} | {d[0]:.3f} {d[1]:.3f} {d[2]:.3f} | {nv:.1e} {nk3:.3f}")

# ---- readout truncation vs state error, per layer: plug MC moments into the Edgeworth readout
from chain import relu_wick
print("\nreadout check: MSE of post-mean vs MC post-mean (MC noise floor ~ var/N)")
print(" l | Gaussian(MC mu,var) | +k3 | +k3+k4 | +k3+k4+k3^2 | chain's own | MC noise floor")
for l in range(len(info)):
    mu_t = torch.tensor(mc["mean"][l]); var_t = torch.tensor(mc["var"][l])
    k3_t = torch.tensor(mc["k3"][l]); k4_t = torch.tensor(mc["k4"][l])
    w, _, _ = relu_wick(mu_t, var_t)
    ref = mc["post_mean"][l]
    g0 = w[(0, 1)].numpy()
    e3 = (w[(0, 1)] + w[(3, 1)] * k3_t / 6).numpy()
    e4 = (w[(0, 1)] + w[(3, 1)] * k3_t / 6 + w[(4, 1)] * k4_t / 24).numpy()
    e6 = (w[(0, 1)] + w[(3, 1)] * k3_t / 6 + w[(4, 1)] * k4_t / 24 + w[(6, 1)] * k3_t ** 2 / 72).numpy()
    own = mu[l].numpy()
    f = lambda a: float(((a - ref) ** 2).mean())
    print(f"{l:2d} | {f(g0):.3e} | {f(e3):.3e} | {f(e4):.3e} | {f(e6):.3e} | {f(own):.3e} | {float(mc['var'][l].mean()) / N:.1e}")
