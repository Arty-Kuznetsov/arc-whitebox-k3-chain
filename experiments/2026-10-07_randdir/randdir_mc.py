"""Monte-Carlo side of the random-direction test of Propositions 1-2 (write-up §6.4; cloud #12 task C, never run).
At the target layers l the pre-activation of layer l is h_{l-1} W_l. We also read the SAME post-activations h_{l-1}
along a fixed random matrix V_l (i.i.d. N(0, 2/n), independent of the network) and accumulate per-column raw moments
of both: mean, variance, kappa3, kappa4 for the real columns and for the random directions, same samples, same N.
Two seeds give two independent estimates (their half-difference is the noise). V_l is saved so the chain reads the
same directions.  Usage: python randdir_mc.py <mlp> <N> <seed> [targets=4,8,11,14]  -> res/mc_<mlp>_s<seed>.npz"""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
m, N, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
targets = [int(x) for x in (sys.argv[4].split(",") if len(sys.argv) > 4 else "4,8,11,14".split(","))]
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{m}.npy")]
L, n = len(W), W[0].shape[0]
V = {}
for l in targets:   # the directions: fixed by (mlp, layer), identical for every seed and for the chain script
    gV = torch.Generator().manual_seed(777_000 + 1000 * m + l)
    V[l] = (torch.randn(n, n, generator=gV) * (2.0 / n) ** 0.5).float()
acc = {(l, kind): [torch.zeros(n, dtype=torch.float64) for _ in range(4)] for l in targets for kind in ("real", "rand")}
g = torch.Generator().manual_seed(seed)
chunk, done, t0 = 20000, 0, time.time()
def accum(a, z):
    zd = z.double()
    a[0] += zd.sum(0); z2 = zd * zd; a[1] += z2.sum(0); a[2] += (z2 * zd).sum(0); a[3] += (z2 * z2).sum(0)
while done < N:
    mm = min(chunk, N - done)
    h = torch.randn(mm, n, generator=g)
    for l in range(max(targets) + 1):
        z = h @ W[l]
        if l in V:
            accum(acc[(l, "real")], z); accum(acc[(l, "rand")], h @ V[l])
        h = torch.relu(z)
    done += mm
    if (done // chunk) % 50 == 0:
        print(f"  {done/N:5.1%}  {time.time()-t0:6.0f}s", flush=True)
out = {"N": N, "seed": seed, "targets": np.array(targets)}
for (l, kind), a in acc.items():
    s1, s2, s3, s4 = (x.numpy() / N for x in a)
    mu = s1; var = s2 - mu * mu
    k3 = s3 - 3 * mu * s2 + 2 * mu ** 3
    m4 = s4 - 4 * mu * s3 + 6 * mu * mu * s2 - 3 * mu ** 4
    k4 = m4 - 3 * var * var
    for name, val in (("mu", mu), ("var", var), ("k3", k3), ("k4", k4)):
        out[f"{name}_{kind}_{l}"] = val
for l in targets:
    out[f"V_{l}"] = V[l].numpy()
np.savez(HERE / "res" / f"mc_{m}_s{seed}.npz", **out)
print(f"saved res/mc_{m}_s{seed}.npz  N={N}  {time.time()-t0:.0f}s", flush=True)
