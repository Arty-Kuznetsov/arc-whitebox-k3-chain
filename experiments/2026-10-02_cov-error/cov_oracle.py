"""Monte-Carlo oracle of the full POST-activation covariance C_exact(l) = Cov(y_l), y_l = ReLU(W_l^T y_{l-1}), of a
baked public MLP at layers 0..14 (the ones that feed a next layer), plus the post-activation means at every layer.
Cloud #11 (notes/CLOUD-BRIEF.md): the object nobody had measured is dC = C_chain - C_exact off the diagonal.

What the file gives, and the identities used downstream (same samples, so they are exact up to float rounding):
    mean_pre(l+1) = W_{l+1}^T post_mean(l),   var_pre(l+1) = diag(W_{l+1}^T C(l) W_{l+1})
so each covariance oracle is also an independent seed of the per-neuron variance oracle at layers 1..15.

Estimator: per chunk of `chunk` samples, y - ref (ref = the baked 1e9 post mean of that layer, so the second moment
needs no cancellation) is accumulated as (y - ref)^T (y - ref) in float32 GEMM per chunk and float64 across chunks;
C = S / N - d d^T with d = mean(y) - ref. Float32 per-chunk rounding is ~1e-6 of var, far below the 1e7 sampling
noise (~3e-4 of var per entry). Checkpoint every 1e6 samples (sums, count, RNG state): a container restart resumes on
the identical random stream.

    ../../.venv-torch/bin/python cov_oracle.py <mlp> <N> <seed> [threads]
      -> oracles/cov_<mlp>_s<seed>_C.npy      (L-1, n, n) float64, all covariance layers (local, gitignored, 126 MB)
         oracles/cov_<mlp>_s<seed>_meta.npz   post_mean (L, n), N, seed
         oracles/cov_<mlp>_s<seed>_L5-8-11-14.npz  float32, compressed: C at layers 5, 8, 11, 14 + post_mean
                                               (force-added to git for the Mac)
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
COMMIT_LAYERS = (5, 8, 11, 14)


def main():
    i, N, seed = int(sys.argv[1]), int(float(sys.argv[2])), int(sys.argv[3])
    torch.set_num_threads(int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    W = [torch.tensor(w, dtype=torch.float32) for w in np.load(BAKED / f"weights_{i}.npy")]
    L, n = len(W), W[0].shape[0]
    LC = L - 1                                        # covariance layers 0..L-2 (layer L-1 feeds nothing)
    ref = torch.tensor(np.load(BAKED / "gt.npy")[i], dtype=torch.float32)     # (L, n) baked 1e9 post means
    out_dir = HERE / "oracles"
    out_dir.mkdir(exist_ok=True)
    ck = out_dir / f"cov_{i}_s{seed}.ckpt.pt"
    g = torch.Generator().manual_seed(seed)
    if ck.exists():
        st = torch.load(ck)
        S, Y, done = st["S"], st["Y"], st["done"]
        g.set_state(st["g"])
        print(f"  resumed at {done / 1e6:.2f}M", flush=True)
    else:
        S = torch.zeros(LC, n, n, dtype=torch.float64)
        Y = torch.zeros(L, n, dtype=torch.float64)       # sums of y - ref
        done = 0
    chunk, t0, done0 = 20000, time.time(), done
    while done < N:
        m = min(chunk, N - done)
        x = torch.randn(m, n, generator=g)
        for l in range(L):
            x = torch.relu(x @ W[l])
            yc = x - ref[l]
            Y[l] += yc.sum(0).double()
            if l < LC:
                S[l] += torch.mm(yc.T, yc).double()
        done += m
        if done % 1000000 < chunk or done >= N:
            torch.save({"S": S, "Y": Y, "done": done, "g": g.get_state()}, ck)
            el = time.time() - t0
            print(f"  {done / 1e6:.2f}M samples, {el:.0f}s, eta {el / (done - done0) * (N - done):.0f}s", flush=True)
    d = Y / done                                       # mean(y) - ref
    post_mean = d + ref.double()
    C = S / done - d[:LC, :, None] * d[:LC, None, :]
    C = 0.5 * (C + C.transpose(1, 2))
    np.save(out_dir / f"cov_{i}_s{seed}_C.npy", C.numpy())
    np.savez(out_dir / f"cov_{i}_s{seed}_meta.npz", post_mean=post_mean.numpy(), N=done, seed=seed)
    np.savez_compressed(out_dir / f"cov_{i}_s{seed}_L5-8-11-14.npz",
                        C=C[list(COMMIT_LAYERS)].float().numpy(), layers=np.array(COMMIT_LAYERS),
                        post_mean=post_mean.float().numpy(), N=done, seed=seed)
    gt = np.load(BAKED / "gt.npy")[i]
    print("post-mean MC vs baked GT (1e9): mse per layer",
          [f"{float(((post_mean[l].numpy() - gt[l]) ** 2).mean()):.1e}" for l in range(L)],
          f"[{time.time() - t0:.0f}s]", flush=True)
    ck.unlink()


if __name__ == "__main__":
    main()
