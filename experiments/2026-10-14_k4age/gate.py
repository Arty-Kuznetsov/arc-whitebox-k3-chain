"""Gates for k4age.py: (1) no truncation reproduces mlp_kprop's k4S; (2) age >= depth reproduces it too;
(3) the closure block's diagonal equals 3 v var^2."""
import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from k4age import k4_run, weights, MLP, Kind, mlp_kprop, FactoredTensor4, _factored_get_dslice
n, depth = 48, 6
Ws = weights(n, depth, 3)
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
for l in range(depth): mlp.Ws[l].weight.data = Ws[l].T.clone()
out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True, up_to_layer=f"act{depth-1}")
ref = torch.stack([out[f"act{l}"][1].to_tensor() for l in range(depth)]).numpy()
full, cols, _ = k4_run(Ws, None, False)
print("gate 1 (stepped full == mlp_kprop k4S): max rel", float(np.abs(full - ref).max() / np.abs(ref).max()), "| cols", cols)
deep, cols2, _ = k4_run(Ws, depth + 5, False)
print("gate 2 (age >= depth == full): max abs", float(np.abs(deep - full).max()), "| cols", cols2)
a1, c1, _ = k4_run(Ws, 1, False); a1c, c1c, vf = k4_run(Ws, 1, True)
print("age 1 cols", c1, "| age 1 + closure cols", c1c, "| v fits", [round(v, 4) for _, v in vf])
C = torch.eye(n) + 0.1 * torch.ones(n, n); v = 0.37
Ac = torch.zeros(n, n, 1); Bc = torch.zeros(n, n, 1); Ac[:, :, 0] = C; Bc[:, :, 0] = 3 * v * C
d = _factored_get_dslice(Ac, Bc, (4,))
print("gate 3 (closure diagonal == 3 v var^2): max abs", float((d - 3 * v * torch.diagonal(C) ** 2).abs().max()))
