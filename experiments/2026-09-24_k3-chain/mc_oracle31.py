"""MC oracle for the one-step kappa4 test (FRONTIER A1): at selected layers of a baked MLP, the PRE-activation pair
slices C_ij, kappa3[iij], kappa4[iijj], kappa4[iiij] (the (3,1) slice SIMPLE does not carry), the POST-activation C,
kappa3[iij], kappa4[iijj], and the common-scale statistic q = (1/n) sum_i dz_i^2 / var_i (its mean, variance, kappa3).
All central moments use the exact means of runs/mc_oracle_<i>.npz (1e7 samples) as fixed centres.
Usage: python mc_oracle31.py <mlp_index> <N> <layers, e.g. 6,10,14>  -> runs/mc_oracle31_<i>.npz"""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
i, N = int(sys.argv[1]), int(float(sys.argv[2]))
layers = [int(v) for v in sys.argv[3].split(",")]
W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
mu = torch.tensor(mc["mean"], dtype=torch.float32)
mu_post = torch.tensor(mc["post_mean"], dtype=torch.float32)
var = torch.tensor(mc["var"], dtype=torch.float32)
n = W[0].shape[0]
Z = lambda: torch.zeros(n, n, dtype=torch.float64)
acc = {l: {k: Z() for k in ("S2", "S21", "S22", "S31", "P2", "P21", "P22")} for l in layers}
qs = {l: torch.zeros(3, dtype=torch.float64) for l in layers}
g = torch.Generator().manual_seed(4242 + i)
chunk, done, t0 = 10000, 0, time.time()
while done < N:
    m = min(chunk, N - done)
    x = torch.randn(m, n, generator=g)
    for l in range(max(layers) + 1):
        z = x @ W[l]
        y = torch.relu(z)
        if l in acc:
            a = acc[l]
            dz = z - mu[l]
            dz2 = dz * dz
            a["S2"] += (dz.T @ dz).double()
            a["S21"] += (dz2.T @ dz).double()          # E[dz_i^2 dz_j]
            a["S22"] += (dz2.T @ dz2).double()
            a["S31"] += ((dz2 * dz).T @ dz).double()   # E[dz_i^3 dz_j]
            q = (dz2 / var[l]).mean(1).double()
            qs[l] += torch.stack([q.sum(), (q * q).sum(), (q * q * q).sum()])
            dy = y - mu_post[l]
            dy2 = dy * dy
            a["P2"] += (dy.T @ dy).double()
            a["P21"] += (dy2.T @ dy).double()
            a["P22"] += (dy2.T @ dy2).double()
        x = y
    done += m
    if done % 500000 < chunk:
        print(f"  {done/1e6:.1f}M, {time.time()-t0:.0f}s", flush=True)
out = {"N": N, "layers": np.array(layers)}
for l in layers:
    a = {k: v / N for k, v in acc[l].items()}
    C = a["S2"]; d = torch.diagonal(C)
    out[f"C{l}"] = C.float().numpy()
    out[f"k21_{l}"] = a["S21"].float().numpy()
    out[f"k22_{l}"] = (a["S22"] - d[:, None] * d[None, :] - 2 * C * C).float().numpy()
    out[f"k31_{l}"] = (a["S31"] - 3 * d[:, None] * C).float().numpy()
    Cp = a["P2"]; dp = torch.diagonal(Cp)
    out[f"postC{l}"] = Cp.float().numpy()
    out[f"postk21_{l}"] = a["P21"].float().numpy()
    out[f"postk22_{l}"] = (a["P22"] - dp[:, None] * dp[None, :] - 2 * Cp * Cp).float().numpy()
    m1, m2, m3 = (qs[l] / N).tolist()
    out[f"q{l}"] = np.array([m1, m2 - m1 * m1, m3 - 3 * m2 * m1 + 2 * m1 ** 3])
    print(f"layer {l}: q mean {m1:.5f}  var {out[f'q{l}'][1]:.4e}  kappa3 {out[f'q{l}'][2]:.4e}")
np.savez(HERE / "runs" / f"mc_oracle31_{i}.npz", **out)
print("done", f"{time.time()-t0:.0f}s")
