"""Structure of the chain's final-layer residual (reads dump10.npz written by dump.py). Run with any python+numpy:
    python3 experiments/2026-09-27_residual/analyze.py
This is the exact analysis that was run inline on 2026-09-27; numbers are recorded in RESULTS.md."""
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
d = np.load(HERE / "dump10.npz")
pred, gt, var, mu, t, k4 = (d[k] for k in ("pred", "gt", "var", "mu_pre", "t", "K4d"))
r = (pred - gt)[:, -1, :]
s = r ** 2; tot = s.sum(); q = np.sort(s.ravel())[::-1]
print("== distribution of the per-neuron squared error at layer 15 ==")
for f in (0.01, 0.02, 0.05, 0.10, 0.25, 0.50):
    k = int(f * q.size); print(f"  worst {f*100:5.1f}% of neurons carry {100*q[:k].sum()/tot:5.1f}% of the MSE")
print(f"  kurtosis of r: {(s.mean()**-2 * (r**4).mean()):.1f}")
print(f"  rms r {np.sqrt(s.mean()):.3e}; rms per-MLP mean {np.sqrt((r.mean(1)**2).mean()):.3e}; "
      f"coherent share {100*(r.mean(1)**2).mean()/s.mean():.2f}%")
a = mu[:, -1, :] / np.sqrt(var[:, -1, :])
print("\n== error vs alpha (last layer) ==")
bins = [-99, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 99]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (a >= lo) & (a < hi)
    if m.sum() >= 10:
        print(f"  alpha [{lo},{hi}): {m.sum()//10} neurons/MLP  rms r {np.sqrt(s[m].mean()):.3e}  share {100*s[m].sum()/tot:5.1f}%")
sd = np.sqrt(var[:, -1, :])
feats = {"alpha": a, "sigma": sd, "var": var[:, -1, :], "t/sd^3": t[:, -1, :] / sd**3,
         "K4d/var^2": k4[:, -1, :] / var[:, -1, :]**2, "pred": pred[:, -1, :], "d_pred": pred[:, -1, :] - pred[:, -2, :]}
print("\n== corr(r, feature) ==")
for k, v in feats.items():
    print(f"  {k:10s} {np.corrcoef(v.ravel(), r.ravel())[0,1]:+.4f}")
print("\n== residual across layers ==")
rl = pred - gt
for l in (8, 10, 12, 13, 14, 15):
    print(f"  layer {l:2d}: rms {np.sqrt((rl[:,l,:]**2).mean()):.3e}  corr with layer-15 residual "
          f"{np.corrcoef(rl[:,l,:].ravel(), r.ravel())[0,1]:+.4f}")
