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
L = len(Ws)
last, allL = (L - 1,), tuple(range(L))
arms = {
    "chain (split24rr + memo22)": {},
    "k4 exact @ last layer": {"k4_layers": last},
    "k4 exact @ all layers": {"k4_layers": allL},
    "k3 exact @ last layer": {"k3_layers": last},
    "var exact @ last layer": {"var_layers": last},
    "k3+k4 exact @ last": {"k3_layers": last, "k4_layers": last},
    "k3+k4+var exact @ last": {"k3_layers": last, "k4_layers": last, "var_layers": last},
    "k4 exact @ all, k3 exact @ all": {"k4_layers": allL, "k3_layers": allL},
}
for name, o in arms.items():
    orc = dict(o, **{k: mc[k] for k in ("k3", "k4", "var")}) if o else None
    mu, _ = run_chain(Ws, k4="memo22", corr="split24rr", oracle=orc)
    print(f"{name:34s} final MSE vs baked GT = {float(((mu[-1].numpy() - gt[-1]) ** 2).mean()):.4e}", flush=True)
print(f"(MC noise in the substituted k4 adds roughly (w41*se/24)^2 ~ few e-9)")
