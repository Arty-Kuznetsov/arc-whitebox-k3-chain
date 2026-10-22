"""Where does the faithful SIMPLE k=3 chain lose its accuracy? Inject exact (baked GT) post-activation means at
layers 0..u and let the chain run on from there; u = 14 isolates the final-layer readout error."""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
gt = np.load(BAKED / "gt.npy")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
k4 = sys.argv[2] if len(sys.argv) > 2 else True
if k4 == "True": k4 = True
res = {}
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    g = torch.tensor(gt[i], dtype=torch.float64)
    for u in (-1, 7, 11, 13, 14):
        mu, _ = run_chain(Ws, k4=k4, inject=g, inject_upto=u)
        res.setdefault(u, []).append(float(((mu[-1] - g[-1]) ** 2).mean()))
    print(f"  mlp {i} done", flush=True)
print(f"k4={k4}: final-layer MSE when exact means are injected at layers 0..u (mean of {N} MLPs)")
for u, v in res.items():
    print(f"  u={u:3d}: {np.mean(v):.4e}   ({'no injection' if u < 0 else f'error accumulated in layers {u+1}..15'})")
