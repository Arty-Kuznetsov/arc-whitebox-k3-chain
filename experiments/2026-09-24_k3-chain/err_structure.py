"""Structure of the chain's final-layer error (baked GT): common-mode bias, and how much of the per-neuron error is
explained by simple per-neuron features of the chain's own final pre-activation stats (mu, sigma, alpha, t, K4d)."""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain, relu_wick
HERE = Path(__file__).resolve().parent
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")
kw = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True)
for i in [int(x) for x in sys.argv[1].split(",")]:
    W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
    mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record=True, **kw)
    for l in (11, 13, 15):
        e = mu[l].numpy() - gt[i][l]
        r = info[l]; m = r["mu_pre"].numpy(); v = r["var"].numpy(); s = np.sqrt(v); a = m / s
        t = r["t"].numpy(); k4 = r["K4d"].numpy()
        w, d1, d2 = relu_wick(r["mu_pre"], r["var"])
        feats = {"1": np.ones_like(m), "d1": d1.numpy(), "phi*s": (d2 * r["var"]).numpy(), "d3*t": (w[(3, 1)] * r["t"]).numpy() / 6,
                 "d4*k4": (w[(4, 1)] * r["K4d"]).numpy() / 24, "a*phi": (a * d2.numpy() * s)}
        X = np.stack(list(feats.values()), 1)
        c, *_ = np.linalg.lstsq(X, e, rcond=None)
        res = e - X @ c
        print(f"MLP {i} layer {l}: MSE {np.mean(e**2):.3e}  bias^2 {np.mean(e)**2:.2e}  R2(features) {1 - np.mean(res**2)/np.mean(e**2):.3f}  "
              f"coefs {dict(zip(feats, np.round(c, 5)))}")
