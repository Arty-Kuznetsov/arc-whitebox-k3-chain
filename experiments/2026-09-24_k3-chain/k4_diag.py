"""Which kappa4 model reproduces the Monte-Carlo kappa4 diagonal? Correlation / slope / rel-RMS per layer."""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
i = 0
mc = dict(np.load(HERE / "runs" / f"mc_oracle_{i}.npz"))
N = float(mc["N"])
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
variants = {"memo22": dict(k4="memo22"), "memo22+k3xedge": dict(k4="memo22", k4k3=1.0), "memo22+k3xedge+paths": dict(k4="memo22", k4k3=1.0, k4b=15, k4star=False), "radial": dict(k4=True),
            "memo22+births(all,paths+stars)": dict(k4="memo22", k4b=15),
            "memo22+births(all,paths)": dict(k4="memo22", k4b=15, k4star=False)}
res = {}
for name, kw in variants.items():
    _, info = run_chain(Ws, corr="split24rr", record=True, **kw)
    res[name] = [r["K4d"].numpy() for r in info]
print("MC noise (se/rms of k4) per layer:", " ".join(f"{np.sqrt(96/N)*np.mean(mc['var'][l]**2)/np.sqrt((mc['k4'][l]**2).mean()):.2f}" for l in range(2, 16)))
for name, ks in res.items():
    print(f"\n{name}:  l: corr slope(chain/MC) rel")
    print("  " + "  ".join(f"{l}:{np.corrcoef(ks[l], mc['k4'][l])[0,1]:.2f}/{float(ks[l]@mc['k4'][l]/(mc['k4'][l]@mc['k4'][l])):.2f}" for l in range(2, 16)))
k4 = mc["k4"]
print("\nMC kappa4 diag: mean / std / mean|.|/var^2 per layer:", " ".join(f"{l}:{k4[l].mean():.3f}/{k4[l].std():.3f}/{np.mean(np.abs(k4[l])/mc['var'][l]**2):.3f}" for l in range(1, 16)))
