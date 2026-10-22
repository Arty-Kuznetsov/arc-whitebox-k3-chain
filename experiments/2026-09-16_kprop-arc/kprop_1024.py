"""Run ARC cumulant propagation at the real 1024x16 on baked-GT MLPs, compare to our 4.06e-6 baseline.
Usage: kprop_1024.py [K_mlps] [k_list e.g. 1,2,3]   (run in the ARC venv)."""
import sys
import time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind

BAKED = Path(__file__).resolve().parent / "baked"
gt = np.load(BAKED / "gt.npy")  # (K,16,1024) post-ReLU per-layer means (final = row 15)
K = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])
ks = [int(x) for x in sys.argv[2].split(",")] if len(sys.argv) > 2 else [1, 2, 3]


def kprop_means(W, k_max, factor):
    n = W.shape[1]
    depth = W.shape[0]
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    K_by = mlp_kprop(mlp, K_in, k_max=k_max, kind=Kind.SIMPLE, factor=factor,
                     output_all=True, up_to_layer=f"act{depth-1}")
    return torch.stack([K_by[f"act{l}"][1].to_tensor() for l in range(depth)]).numpy()  # (16,1024)


res = {k: {"final": [], "all": [], "t": 0.0} for k in ks}
for i in range(K):
    W = np.load(BAKED / f"weights_{i}.npy")
    g = gt[i]
    for k in ks:
        t0 = time.time()
        pred = kprop_means(W, k, factor=(k >= 3))
        res[k]["t"] += time.time() - t0
        res[k]["final"].append(float(((pred[-1] - g[-1]) ** 2).mean()))
        res[k]["all"].append(float(((pred - g) ** 2).mean()))
    print(f"  mlp {i}: " + "  ".join(f"k{k} final={res[k]['final'][-1]:.3e}" for k in ks), flush=True)

print(f"\n=== ARC cumulant propagation @ 1024x16 vs baked GT ({K} MLPs) ===")
print("baseline (our covariance+tetrachoric): final_layer_mse=4.06e-6, all_layers_mse=2.15e-6")
for k in ks:
    print(f"k={k}: final_layer_mse={np.mean(res[k]['final']):.4e}  "
          f"all_layers_mse={np.mean(res[k]['all']):.4e}  ({res[k]['t']/K:.1f}s/mlp)")
