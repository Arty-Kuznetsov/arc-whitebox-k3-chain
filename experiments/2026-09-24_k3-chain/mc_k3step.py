"""Where does the kappa3 error come from? One-step test of the ALL-DISTINCT kappa3 births/pass-through with exact
(MC) inputs. At layer l (pre z, post y = ReLU z) and next weights W' (layer l+1), the next pre-activation diagonal is
    t'_i = sum_a W'^3 k3y[aaa] + 3 sum_{a!=b} W'_ai^2 W'_bi k3y[aab] + sum_{abc distinct} W'W'W' k3y[abc].
Truth: t' from mc_oracle_0 (1e7). The repeated-index part R3 uses this run's MC post K3 / K21. The all-distinct part
T = t' - R3 is compared with SIMPLE's all-distinct terms from exact inputs:
  pass : sum_distinct (W'd1)^3 k3z[abc] = E[u^3] - repeated parts, u_i = sum_a W'_ai d1_a dz_a
  tree : 3 sum_a W'_ai d2_a [V_ai^2 - sum_b (C0_ab d1_b W'_bi)^2],  V = C0 diag(d1) W'
  feed : (3/2) v sum_x W'_xi d2_x var_x [w~^T C0 w~ - 2 w~_x (C0 w~)_x],  w~ = d1 o W'_:,i   (the P2 part)
         + v * tree (the P1 part, reported separately)
Usage: mc_k3step.py <mlp> <N> <layers>  (layers l with l+1 <= 15)"""
import sys, time, math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i_mlp, N = int(sys.argv[1]), int(float(sys.argv[2]))
layers = [int(v) for v in sys.argv[3].split(",")]
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i_mlp}.npy")]
mc = np.load(HERE / "runs" / f"mc_oracle_{i_mlp}.npz")
n = W[0].shape[0]
mu = torch.tensor(mc["mean"], dtype=torch.float32); var = torch.tensor(mc["var"], dtype=torch.float32)
pm = torch.tensor(mc["post_mean"], dtype=torch.float32)
s_ = var.sqrt(); a_ = mu / s_
d1 = 0.5 * (1 + torch.erf(a_ / math.sqrt(2))); d2 = torch.exp(-0.5 * a_ * a_) / math.sqrt(2 * math.pi) / s_
P = {l: W[l + 1] * d1[l][:, None] for l in layers}
Z = lambda *sh: torch.zeros(*sh, dtype=torch.float64)
acc = {l: dict(C=Z(n, n), Dz=Z(n, n), K21y=Z(n, n), K3y=Z(n), Vy=Z(n), u2=Z(n), u3=Z(n)) for l in layers}
g = torch.Generator().manual_seed(2718 + i_mlp)
chunk, done, t0 = 10000, 0, time.time()
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    for l in range(max(layers) + 1):
        z = x @ W[l]
        y = torch.relu(z)
        if l in acc:
            A = acc[l]
            dz = z - mu[l]; dy = y - pm[l]
            A["C"] += (dz.T @ dz).double(); A["Dz"] += ((dz * dz).T @ dz).double()
            dy2 = dy * dy
            A["K21y"] += (dy2.T @ dy).double(); A["K3y"] += (dy2 * dy).sum(0).double(); A["Vy"] += dy2.sum(0).double()
            u = dz @ P[l]
            A["u2"] += (u * u).sum(0).double(); A["u3"] += (u * u * u).sum(0).double()
        x = y
    done += m
    if done % 1000000 < chunk:
        print(f"  {done/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)
np.savez(HERE / "runs" / f"mc_k3step_{i_mlp}.npz", N=N, layers=np.array(layers),
         **{f"{k}{l}": (v / N).numpy() for l in layers for k, v in acc[l].items()})
print("done", f"{time.time()-t0:.0f}s")
