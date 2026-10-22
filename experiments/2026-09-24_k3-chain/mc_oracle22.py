"""MC oracle (POST-activation variant with argv[4]=="post") for pair statistics: covariance C_ij and fourth-cumulant (2,2) slice k4[i,i,j,j] of pre-activations
at selected layers (centred with the exact means from mc_oracle_<i>.npz). -> runs/mc_oracle22_<i>.npz"""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i, N = int(sys.argv[1]), int(float(sys.argv[2]))
layers = [int(v) for v in sys.argv[3].split(",")]
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
post = len(sys.argv) > 4 and sys.argv[4] == 'post'
mu = torch.tensor(np.load(HERE / "runs" / f"mc_oracle_{i}.npz")["post_mean" if post else "mean"], dtype=torch.float32)
n = W[0].shape[0]
S2 = {l: torch.zeros(n, n, dtype=torch.float64) for l in layers}
S22 = {l: torch.zeros(n, n, dtype=torch.float64) for l in layers}
g = torch.Generator().manual_seed(777 + i)
chunk, done, t0 = 10000, 0, time.time()
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    for l in range(max(layers) + 1):
        z = x @ W[l]
        if l in S2:
            dz = (torch.relu(z) if post else z) - mu[l]
            S2[l] += (dz.T @ dz).double()
            q = dz * dz
            S22[l] += (q.T @ q).double()
        x = torch.relu(z)
    done += m
    if done % 1000000 < chunk:
        print(f"  {done/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)
out = {}
for l in layers:
    C = S2[l] / N
    E22 = S22[l] / N
    d = torch.diagonal(C)
    k22 = E22 - d[:, None] * d[None, :] - 2 * C * C
    out[f"C{l}"] = C.float().numpy(); out[f"k22_{l}"] = k22.float().numpy()
np.savez(HERE / "runs" / (f"mc_oracle22post_{i}.npz" if post else f"mc_oracle22_{i}.npz"), N=N, layers=np.array(layers), **out)
print("done")
