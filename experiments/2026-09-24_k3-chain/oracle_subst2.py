"""oracle_subst.py on the k3e configuration (split24rr + memo22 levels + sigma^4 closure + c0d): how much is left if
the per-neuron pre-activation kappa4 / kappa3 diagonals (and the variance) were exact (MC, 1e7 samples)?"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
mc = dict(np.load(HERE / "runs" / f"mc_oracle_{i}.npz"))
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[i]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
L = len(Ws); allL = tuple(range(L)); deep = tuple(range(8, L))
K3E = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True)
arms = {
    "k3e chain": {},
    "k4 exact @ all": {"k4_layers": allL},
    "k3 exact @ all": {"k3_layers": allL},
    "k3+k4 exact @ all": {"k4_layers": allL, "k3_layers": allL},
    "k3+k4 exact @ layers 8-15": {"k4_layers": deep, "k3_layers": deep},
    "k3+k4+var exact @ all": {"k4_layers": allL, "k3_layers": allL, "var_layers": allL},
    "var exact @ all": {"var_layers": allL},
    "var exact @ last": {"var_layers": (L - 1,)},
    "var+k4 exact @ all": {"var_layers": allL, "k4_layers": allL},
    "var+k3 exact @ all": {"var_layers": allL, "k3_layers": allL},
    "k3+k4+var exact @ last only": {"k4_layers": (L - 1,), "k3_layers": (L - 1,), "var_layers": (L - 1,)},
}
import os
if os.environ.get("ONLY_NEW"):
    arms = {k: v for k, v in arms.items() if "var" in k and k != "k3+k4+var exact @ all"}
for name, o in arms.items():
    orc = dict(o, **{k: mc[k] for k in ("k3", "k4", "var")}) if o else None
    mu, _ = run_chain(Ws, oracle=orc, **K3E)
    print(f"{name:30s} final MSE vs baked GT = {float(((mu[-1].numpy() - gt[-1]) ** 2).mean()):.4e}", flush=True)
