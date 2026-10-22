"""Monte-Carlo oracle for per-neuron pre-activation statistics of a baked MLP: mean, variance, third and fourth
cumulants at every layer, plus post-activation means. Used to find which quantity of the k=3 chain is off.
Usage: python mc_oracle.py <mlp_index> <N_samples> [seed]  -> runs/mc_oracle_<i>[_s<seed>].npz

With an explicit seed the result goes to its own file, so a second run of the same N is an
INDEPENDENT estimate of the same truth. The spread between two such files is the honest error bar
of every oracle-substitution arm — needed because a 6.6e-4 variance difference between two oracles
moves the chain's final MSE by 48% (experiments/2026-09-26_accuracy/RESULTS.md)."""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i, N = int(sys.argv[1]), int(float(sys.argv[2]))
seed = int(sys.argv[3]) if len(sys.argv) > 3 else None
out = f"mc_oracle_{i}.npz" if seed is None else f"mc_oracle_{i}_s{seed}.npz"
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
L, n = len(W), W[0].shape[0]
S = [torch.zeros(L, n, dtype=torch.float64) for _ in range(4)]   # sums of z, z^2, z^3, z^4 (shifted later)
Y = torch.zeros(L, n, dtype=torch.float64)
g = torch.Generator().manual_seed(12345 + i if seed is None else seed)
chunk, done, t0 = 20000, 0, time.time()
# two passes would be exact for central moments; with float64 accumulation of raw moments the cancellation is fine
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    for l in range(L):
        z = x @ W[l]
        zd = z.double()
        z2 = zd * zd
        S[0][l] += zd.sum(0); S[1][l] += z2.sum(0); S[2][l] += (z2 * zd).sum(0); S[3][l] += (z2 * z2).sum(0)
        x = torch.relu(z)
        Y[l] += x.double().sum(0)
    done += m
    if done % 1000000 < chunk:
        print(f"  {done/1e6:.1f}M samples, {time.time()-t0:.0f}s", flush=True)
m1, m2, m3, m4 = (s / N for s in S)
var = m2 - m1 ** 2
k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
np.savez(HERE / "runs" / out, mean=m1.numpy(), var=var.numpy(), k3=k3.numpy(), k4=k4.numpy(),
         post_mean=(Y / N).numpy(), N=N)
gt = np.load(BAKED / "gt.npy")[i]
print("post-mean MC vs baked GT (N=1e9): mse per layer", [f"{float(((Y[l]/N).numpy()-gt[l])**2).mean() if False else float((((Y[l]/N).numpy()-gt[l])**2).mean()):.1e}" for l in range(L)])
