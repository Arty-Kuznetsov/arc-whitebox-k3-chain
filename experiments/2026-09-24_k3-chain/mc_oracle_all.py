"""MC oracle for the pair slices at EVERY layer: pre-activation covariance C (n x n) and kappa3[iij] (n x n), centred
with the exact means of runs/mc_oracle_<i>.npz. For oracle substitution of C / D in the chain (oracle_subst3.py).
Usage: mc_oracle_all.py <mlp> <N>  -> runs/mc_oracle_all_<mlp>.npz (float32 matrices)"""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i, N = int(sys.argv[1]), int(float(sys.argv[2]))
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
mu = torch.tensor(np.load(HERE / "runs" / f"mc_oracle_{i}.npz")["mean"], dtype=torch.float32)
L, n = len(W), W[0].shape[0]
S2 = [torch.zeros(n, n, dtype=torch.float64) for _ in range(L)]
S21 = [torch.zeros(n, n, dtype=torch.float64) for _ in range(L)]
g = torch.Generator().manual_seed(31337 + i)
chunk, done, t0 = 10000, 0, time.time()
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    for l in range(L):
        z = x @ W[l]
        dz = z - mu[l]
        S2[l] += (dz.T @ dz).double()
        S21[l] += ((dz * dz).T @ dz).double()     # E[dz_i^2 dz_j]
        x = torch.relu(z)
    done += m
    if done % 1000000 < chunk:
        print(f"  {done/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)
out = {"N": N}
for l in range(L):
    out[f"C{l}"] = (S2[l] / N).float().numpy()
    out[f"D{l}"] = (S21[l] / N).float().numpy()
np.savez(HERE / "runs" / f"mc_oracle_all_{i}.npz", **out)
print("done", f"{time.time()-t0:.0f}s")
