"""Oracle levers on the k3f configuration (split24rr + closure + c0d + marg + k31 + feed P2), MLP 0: exact (MC) full
covariance, its off-diagonal only, the variance only, the kappa3 (2,1) slice D, and the diagonals, at all layers.
Needs runs/mc_oracle_0.npz and runs/mc_oracle_all_0.npz."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
mc = dict(np.load(HERE / "runs" / "mc_oracle_0.npz")); ma = np.load(HERE / "runs" / "mc_oracle_all_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
L = len(Ws); allL = tuple(range(L))
K3F = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, marg=True, k31pat=True, k4feed="P2")
base = {k: mc[k] for k in ("k3", "k4", "var")}
base["C"] = [ma[f"C{l}"] for l in range(L)]; base["D"] = [ma[f"D{l}"] for l in range(L)]
arms = {
    "k3f chain": None,
    "var exact": {"var_layers": allL},
    "C0 (off-diag) exact": {"C0_layers": allL},
    "C (full) exact": {"C_layers": allL},
    "D exact": {"D_layers": allL},
    "C + D exact": {"C_layers": allL, "D_layers": allL},
    "C + D + k3 + k4 exact": {"C_layers": allL, "D_layers": allL, "k3_layers": allL, "k4_layers": allL},
    "var + k3 + k4 exact": {"var_layers": allL, "k3_layers": allL, "k4_layers": allL},
}
for name, o in arms.items():
    orc = dict(o, **base) if o else None
    mu, _ = run_chain(Ws, oracle=orc, **K3F)
    print(f"{name:26s} final MSE vs baked GT = {float(((mu[-1].numpy() - gt[-1]) ** 2).mean()):.4e}", flush=True)
