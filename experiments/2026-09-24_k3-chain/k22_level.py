"""Per-layer level of the chain's kappa4: pre-activation K22 pair-mean and K4 diagonal mean, for chain configs,
against the MC oracle (mc_oracle22_0 / mc_oracle31_0 pair-means, mc_oracle_0 diagonal). Hooks nonlin_slices to read
the K22 each layer receives and the post K22 it returns. Usage: k22_level.py <mlp> <cfg1> [<cfg2> ...], cfg = k=v,k=v"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1])
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
mcp = {}
for f in ("mc_oracle22_0.npz", "mc_oracle31_0.npz"):
    d = np.load(HERE / "runs" / f)
    for l in d["layers"]:
        K = d[f"k22_{int(l)}"].astype(np.float64); n = K.shape[0]
        mcp[int(l)] = K[~np.eye(n, dtype=bool)].mean()
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
def val(v):
    if v in ("True", "False"): return v == "True"
    try: return int(v)
    except ValueError: return v
orig = ch.nonlin_slices
for cfg in sys.argv[2:]:
    kw = dict(k4="memo22", corr="split24rr")
    if cfg != "base":
        kw.update({k: val(v) for k, v in (a.split("=") for a in cfg.split(","))})
    rec = []
    def hook(mu, C, t, D, K4d, K22, **k):
        o = orig(mu, C, t, D, K4d, K22, **k)
        n = mu.shape[0]; off = ~torch.eye(n, dtype=torch.bool)
        rec.append((float(K22[off].mean()) if K22 is not None else 0.0, float(K4d.mean()) if K4d is not None else 0.0,
                    float(o["K22"][off].mean()), float(o["K4"].mean())))
        return o
    ch.nonlin_slices = hook
    ch.run_chain([torch.tensor(w, dtype=torch.float64) for w in W], **kw)
    ch.nonlin_slices = orig
    print(f"--- {kw}")
    print(" l | pre K22 mean (chain / MC) | pre K4d mean (chain / MC) | post K22 mean | post K4 mean")
    for l, (k22, k4d, pk22, pk4) in enumerate(rec):
        r22 = f"{k22 / mcp[l]:.3f}" if l in mcp else "  -  "
        print(f"{l:2d} | {k22:.3e} ({r22}) | {k4d:.3e} ({k4d / mc['k4'][l].mean() if l > 1 else 0:.3f}) | {pk22:.3e} | {pk4:.3e}")
