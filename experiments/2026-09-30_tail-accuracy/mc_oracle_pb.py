"""Monte-Carlo oracle of the per-neuron DIAGONAL statistics (pre-activation mean, var, kappa3, kappa4 at every layer,
post-activation means) -- the estimator of experiments/2026-09-24_k3-chain/mc_oracle.py, with the thread count and the
output folder as arguments. Only the diagonals: NEGATIVE 50 showed that the pair slices (C, D) need >> 1e7 samples to be
a valid oracle, while the diagonals at 1e7 are.
v2 (after a container restart lost four runs at 8e6 of 1e7): power sums of z - c (c = the first chunk's float32 mean of
that layer) in float32 per chunk, accumulated in float64 across chunks (~2x less memory traffic than float64 powers;
accumulation error <1e-5 relative, far below the 1e7 sampling noise), and a checkpoint every 1e6 samples (sums, count,
RNG state) from which a restart resumes on the identical random stream.

    ../../.venv-torch/bin/python mc_oracle_pb.py <mlp> <N> <seed> [threads]  -> oracles/mc_<mlp>_s<seed>.npz
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i, N, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
L, n = len(W), W[0].shape[0]
ck = HERE / "oracles" / f"mc_{i}_s{seed}.ckpt.pt"
g = torch.Generator().manual_seed(seed)
if ck.exists():
    st = torch.load(ck)
    S, Y, c, done = st["S"], st["Y"], st["c"], st["done"]
    g.set_state(st["g"])
    print(f"  resumed at {done / 1e6:.1f}M", flush=True)
else:
    S, Y, c, done = torch.zeros(4, L, n, dtype=torch.float64), torch.zeros(L, n, dtype=torch.float64), None, 0
chunk, t0 = 20000, time.time()
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    if c is None:
        c = torch.zeros(L, n)
    for l in range(L):
        z = x @ W[l]
        if done == 0:
            c[l] = z.mean(0)
        zc = z - c[l]
        z2 = zc * zc
        S[0, l] += zc.sum(0).double()
        S[1, l] += z2.sum(0).double()
        S[2, l] += (z2 * zc).sum(0).double()
        S[3, l] += (z2 * z2).sum(0).double()
        x = torch.relu(z)
        Y[l] += x.sum(0).double()
    done += m
    if done % 1000000 < chunk:
        torch.save({"S": S, "Y": Y, "c": c, "done": done, "g": g.get_state()}, ck)
        print(f"  {done / 1e6:.1f}M samples, {time.time() - t0:.0f}s", flush=True)
m1, m2, m3, m4 = (s / done for s in S)
var = m2 - m1 ** 2
k3 = m3 - 3 * m2 * m1 + 2 * m1 ** 3
k4 = m4 - 4 * m3 * m1 - 3 * m2 ** 2 + 12 * m2 * m1 ** 2 - 6 * m1 ** 4
np.savez(HERE / "oracles" / f"mc_{i}_s{seed}.npz", mean=(m1 + c.double()).numpy(), var=var.numpy(), k3=k3.numpy(),
         k4=k4.numpy(), post_mean=(Y / done).numpy(), N=done, seed=seed)
gt = np.load(BAKED / "gt.npy")[i]
print("post-mean MC vs baked GT (1e9): mse per layer",
      [f"{float((((Y[l] / done).numpy() - gt[l]) ** 2).mean()):.1e}" for l in range(L)], f"[{time.time() - t0:.0f}s]")
