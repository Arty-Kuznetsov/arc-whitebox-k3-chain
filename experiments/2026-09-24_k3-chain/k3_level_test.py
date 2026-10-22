"""Is the chain's kappa3 LEVEL deficit at depth (t slope ~0.985, D slope ~0.964 vs MC at layers 13-15, k3f config) worth
fixing? Rescale t and/or D per layer by the measured slopes (logs/chain_vs_mc_0.txt) and compare final MSE (MLP 0)."""
import re
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
rows = [l for l in open(HERE / "logs" / "chain_vs_mc_0.txt") if re.match(r"\s*\d+ \|", l)]
ts, Ds = np.ones(16), np.ones(16)
for r in rows:
    f = r.split("|"); l = int(f[0])
    if len(f) >= 4:
        ts[l] = float(f[2].split()[0]); Ds[l] = float(f[3].split()[0])
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
K3F = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2")
for name, o in {"k3f": None, "t scaled": {"t_scale": ts}, "D scaled": {"D_scale": Ds}, "t + D scaled": {"t_scale": ts, "D_scale": Ds}}.items():
    mu, _ = run_chain(Ws, oracle=o, **K3F)
    print(f"{name:14s} final MSE {float(((mu[-1].numpy() - gt[-1]) ** 2).mean()):.4e}", flush=True)
print("t slopes", np.round(ts, 4)); print("D slopes", np.round(Ds, 4))
