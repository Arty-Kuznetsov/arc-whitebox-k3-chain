"""Per-layer MSE vs baked GT for a chain config. Usage: perlayer.py <mlp> [k=v ...] (default: k3f config)"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1])
def val(v):
    if v in ("True", "False"): return v == "True"
    if v == "None": return None
    try: return int(v)
    except ValueError:
        try: return float(v)
        except ValueError: return v
kw = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True)
extra = {k: val(v) for k, v in (a.split("=") for a in sys.argv[2:])}
if extra.pop("preset", None) == "k3j":
    kw.update(paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0)
kw.update(extra)
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[i]
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
mu, _ = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], **kw)
per = ((mu.numpy() - gt) ** 2).mean(1)
print(f"mlp {i} final {per[-1]:.4e} all {per.mean():.4e}\n  per-layer: " + " ".join(f"{x:.2e}" for x in per))
