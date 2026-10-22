"""MC measurement of the kappa4 -> kappa3 feed that the next layer's kappa3 diagonal receives from layer l's ReLU:
    F_i = (3/2) sum_x W'_xi d2_x kappa4(z_x, z_x, u_i, u_i),   u_i = sum_y W'_yi d1_y dz_y,   W' = weights of layer l+1,
(the all-index version of the births Delta kappa3[xyz] = 1/2 d2_x d1_y d1_z kappa4[xxyz]), against the scale-mixture
closure kappa4(z_x,z_x,u,u) = v (var_x Var(u_i) + 2 Cov(z_x,u_i)^2). Pre-activations of layer l are centred with the
exact MC means. Usage: mc_feed.py <mlp> <N> <layers>  -> runs/mc_feed_<mlp>.npz"""
import sys, time, math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i, N = int(sys.argv[1]), int(float(sys.argv[2]))
layers = [int(v) for v in sys.argv[3].split(",")]
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
n = W[0].shape[0]
mu = torch.tensor(mc["mean"], dtype=torch.float32); var = torch.tensor(mc["var"], dtype=torch.float32)
s = var.sqrt(); a = mu / s
d1 = 0.5 * (1 + torch.erf(a / math.sqrt(2))); d2 = torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi) / s
P = {l: (W[l + 1] * d1[l][:, None]) for l in layers}                       # u = dz @ P  (P[y, i] = d1_y W'_yi)
acc = {l: dict(Szu2=torch.zeros(n, n, dtype=torch.float64), Szu=torch.zeros(n, n, dtype=torch.float64),
               Su2=torch.zeros(n, dtype=torch.float64), Sz2=torch.zeros(n, dtype=torch.float64),
               Szz=torch.zeros(n, n, dtype=torch.float64)) for l in layers}
g = torch.Generator().manual_seed(9090 + i)
chunk, done, t0 = 10000, 0, time.time()
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    for l in range(max(layers) + 1):
        z = x @ W[l]
        if l in acc:
            A = acc[l]
            dz = z - mu[l]
            u = dz @ P[l]
            dz2, u2 = dz * dz, u * u
            A["Szu2"] += (dz2.T @ u2).double(); A["Szu"] += (dz.T @ u).double()
            A["Su2"] += u2.sum(0).double(); A["Sz2"] += dz2.sum(0).double(); A["Szz"] += (dz.T @ dz).double()
        x = torch.relu(z)
    done += m
    if done % 1000000 < chunk:
        print(f"  {done/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)
out = {}
for l in layers:
    A = {k: v / N for k, v in acc[l].items()}
    k4 = A["Szu2"] - A["Sz2"][:, None] * A["Su2"][None, :] - 2 * A["Szu"] ** 2      # kappa4(z_x, z_x, u_i, u_i)
    C = A["Szz"]; vx = torch.diagonal(C)
    Wn = W[l + 1].double()
    F = 1.5 * ((Wn * d2[l][:, None].double()) * k4).sum(0)
    # scale-mixture closure with v from the MC (2,2) level of this layer is applied in the analysis script
    out[f"F{l}"] = F.numpy(); out[f"k4zu{l}"] = k4.float().numpy(); out[f"Szu{l}"] = A["Szu"].float().numpy()
    out[f"Su2{l}"] = A["Su2"].numpy(); out[f"C{l}"] = C.float().numpy()
np.savez(HERE / "runs" / f"mc_feed_{i}.npz", N=N, layers=np.array(layers), **out)
print("done", f"{time.time()-t0:.0f}s")
