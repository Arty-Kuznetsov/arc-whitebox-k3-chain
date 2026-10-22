"""Monte-Carlo oracle of the PRE-activation (2,1) slice of the third cumulant, D(l)_ij = kappa3[z_i, z_i, z_j], at two
layers, plus the post-activation covariance of the layer before each (so the step at l can be fed an exact C_pre(l) =
W_l^T C_post(l-1) W_l) and the post means. Question it serves: is D -- the slice the chain's kappa3 representation
delivers to the step -- what makes the covariance error the step gives birth to (cloud #11: born 22-30% of dC0 at
depth; with ALL inputs exact the step is accurate to ~0.1%)?

Estimator, per chunk of `chunk` samples, centred on a reference r (the exact pre-activation mean from the baked 1e9 post
means, r = W_l^T gt(l-1), so c = z - r is centred to ~1e-5): sums S1 = sum c, S2 = c^T c, S3 = (c o c)^T c, in float32
GEMM per chunk and float64 across chunks; the third central moment from the raw moments,
    kappa3[i,i,j] = m3_iij - 2 d_i m2_ij - d_j m2_ii + 2 d_i^2 d_j,   d = S1/N, m2 = S2/N, m3 = S3/N.
Checkpointed every 1e6 samples (sums, count, RNG state) so a restart resumes on the identical stream.

    python d_oracle.py <mlp> <N> <seed> [threads]   ->  oracles/d_<mlp>_s<seed>.npz (float32: D at LD, C_post at LC)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
LD = (9, 12)                      # pre-activation layers with D; the cloud's committed C_post oracles are at 8 and 11
LC = (8, 9, 11, 12)               # post-activation covariance layers (the step at 9 / 12 reads C_post(8) / C_post(11)
                                  # through C_pre and writes C_post(9) / C_post(12): the born error needs both)


def main():
    i, N, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
    torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 3)
    W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
    n = W[0].shape[0]
    gt = torch.tensor(np.load(BAKED / "gt.npy")[i], dtype=torch.float32)          # (L, n) baked 1e9 post means
    rpre = {l: gt[l - 1] @ W[l] for l in LD}                                       # exact pre-activation means
    rpost = {l: gt[l] for l in LC}
    out = HERE / "oracles"
    ck = out / f"d_{i}_s{seed}.ckpt.pt"
    g = torch.Generator().manual_seed(seed)
    if ck.exists():
        st = torch.load(ck); acc, done = st["acc"], st["done"]; g.set_state(st["g"])
        print(f"  resumed at {done / 1e6:.2f}M", flush=True)
    else:
        acc = {}
        for l in LD:
            acc[f"S1_{l}"] = torch.zeros(n, dtype=torch.float64)
            acc[f"S2_{l}"] = torch.zeros(n, n, dtype=torch.float64)
            acc[f"S3_{l}"] = torch.zeros(n, n, dtype=torch.float64)
            acc[f"S4_{l}"] = torch.zeros(n, n, dtype=torch.float64)      # (c o c)^T (c o c): the (2,2) fourth moment
        for l in LC:
            acc[f"P1_{l}"] = torch.zeros(n, dtype=torch.float64)
            acc[f"P2_{l}"] = torch.zeros(n, n, dtype=torch.float64)
        done = 0
    chunk, t0, done0, lmax = 20000, time.time(), done, max(max(LD), max(LC))
    while done < N:
        m = min(chunk, N - done)
        y = torch.randn(m, n, generator=g)
        for l in range(lmax + 1):
            z = y @ W[l]
            if l in LD:
                c = z - rpre[l]
                acc[f"S1_{l}"] += c.sum(0).double()
                acc[f"S2_{l}"] += torch.mm(c.T, c).double()
                cc = c * c
                acc[f"S3_{l}"] += torch.mm(cc.T, c).double()
                acc[f"S4_{l}"] += torch.mm(cc.T, cc).double()
            y = torch.relu(z)
            if l in rpost:
                yc = y - rpost[l]
                acc[f"P1_{l}"] += yc.sum(0).double()
                acc[f"P2_{l}"] += torch.mm(yc.T, yc).double()
        done += m
        if done % 1000000 < chunk or done >= N:
            torch.save({"acc": acc, "done": done, "g": g.get_state()}, ck)
            el = time.time() - t0
            print(f"  {done / 1e6:.2f}M samples, {el:.0f}s, eta {el / (done - done0) * (N - done):.0f}s", flush=True)
    res = {"N": done, "seed": seed, "layers": np.array(LD)}
    for l in LD:
        d = acc[f"S1_{l}"] / done; m2 = acc[f"S2_{l}"] / done; m3 = acc[f"S3_{l}"] / done
        m2d = torch.diagonal(m2)
        D = m3 - 2.0 * d[:, None] * m2 - d[None, :] * m2d[:, None] + 2.0 * (d * d)[:, None] * d[None, :]
        res[f"D_{l}"] = D.float().numpy()                       # D[i, j] = kappa3[i, i, j] of z(l)
        # K22[i, j] = kappa4[i, i, j, j]: central (2,2) moment from raw moments, minus the Gaussian (Isserlis) part
        m4 = acc[f"S4_{l}"] / done
        e22 = (m4 - 2.0 * d[None, :] * m3 - 2.0 * d[:, None] * m3.T + (d * d)[None, :] * m2d[:, None]
               + (d * d)[:, None] * m2d[None, :] + 4.0 * (d[:, None] * d[None, :]) * m2 - 3.0 * (d * d)[:, None] * (d * d)[None, :])
        u2 = m2d - d * d                                       # central second moments (variances)
        cij = m2 - d[:, None] * d[None, :]                     # central covariance
        res[f"K22_{l}"] = (e22 - u2[:, None] * u2[None, :] - 2.0 * cij * cij).float().numpy()
        res[f"mean_pre_{l}"] = (d + rpre[l].double()).float().numpy()
        res[f"var_pre_{l}"] = (m2d - d * d).float().numpy()
    for l in LC:
        dp = acc[f"P1_{l}"] / done
        Cp = acc[f"P2_{l}"] / done - dp[:, None] * dp[None, :]
        res[f"C_post_{l}"] = (0.5 * (Cp + Cp.T)).float().numpy()
        res[f"post_mean_{l}"] = (dp + rpost[l].double()).float().numpy()
    np.savez_compressed(out / f"d_{i}_s{seed}.npz", **res)
    print(f"done {done / 1e6:.1f}M in {time.time() - t0:.0f}s", flush=True)
    ck.unlink()


if __name__ == "__main__":
    main()
