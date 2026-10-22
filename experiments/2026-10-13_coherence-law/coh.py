"""When can a per-neuron learned correction of a moment-propagation estimator work? Data collection.
Write-up §6.4: the next layer's per-neuron errors are random projections of this layer's joint errors, so the part a
per-neuron model can learn is the COHERENT part (the level shared by all neurons). Prediction: that share grows with
L/n. mliston's learned corrector worked in Phase 1 (256 x 32, x0.81-0.83), every per-neuron corrector failed in
Phase 2 (1024 x 16). This script records, per network and layer, the chain's per-neuron statistics and their truth.

Estimator: the float64 research chain in config 'full' = ARC's SIMPLE k = 3 (reproduces mlp_kprop to 1e-15).
Truth, two modes:
  synthetic: random He-init networks (seeded), Monte-Carlo with two independent seeds (float32 GEMMs, float64 sums)
             -> per layer: pre-activation mean, variance, third cumulant, post-activation mean (seed A and seed B);
  baked:     the public 1024 x 16 networks of experiments/2026-09-16_kprop-arc/baked with their 1e9-sample post-means
             (the dataset's own ground truth; MC noise negligible), post-activation means only.
Output: res/n<n>_L<L>_<tag><k>.npz
usage:  python coh.py synth <n> <L> <first> <count> <N>      |      python coh.py baked <first> <count>
    ARC=$(cd ../../external/mlp_cumulant_propagation && pwd); env PYTHONPATH="$ARC/src:../2026-09-24_k3-chain" \\
        nice -n 10 uv run --project "$ARC" python coh.py synth 256 32 0 2 2e6
"""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)   # the chain builds its state in the default dtype
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
from chain import run_chain                                     # noqa: E402
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
CFG = {}                                                        # 'full' = ARC's SIMPLE k = 3


def chain_stats(Ws):
    rows, info = run_chain(Ws, record=True, **CFG)
    L = len(Ws)
    return dict(c_m=rows.numpy().astype(np.float64),
                c_mu=np.stack([info[l]["mu_pre"].numpy() for l in range(L)]).astype(np.float64),
                c_var=np.stack([info[l]["var"].numpy() for l in range(L)]).astype(np.float64),
                c_t=np.stack([info[l]["t"].numpy() for l in range(L)]).astype(np.float64))


def mc(Ws, seed, N, B=1 << 16):
    L, n = len(Ws), Ws[0].shape[0]
    W32 = [W.to(torch.float32) for W in Ws]
    S = torch.zeros(4, L, n, dtype=torch.float64)          # sums of z, z^2, z^3, relu(z)
    g = torch.Generator().manual_seed(seed)
    done = 0
    while done < N:
        b = min(B, N - done)
        y = torch.randn(b, n, generator=g, dtype=torch.float32)
        for l in range(L):
            z = y @ W32[l]; zd = z.double()
            S[0, l] += zd.sum(0); z2 = zd * zd; S[1, l] += z2.sum(0); S[2, l] += (z2 * zd).sum(0)
            y = torch.relu(z); S[3, l] += y.double().sum(0)
        done += b
    s1, s2, s3, sr = (x.numpy() / N for x in S)
    var = s2 - s1 * s1
    k3 = s3 - 3 * s1 * s2 + 2 * s1 ** 3
    return dict(mu=s1, var=var, k3=k3, m=sr)


def weights(n, L, k):
    g = torch.Generator().manual_seed(7_000_000 + 10_000 * n + 100 * L + k)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(L)]


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "synth":
        n, L, first, count, N = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(float(sys.argv[6]))
        for k in range(first, first + count):
            out = HERE / "res" / f"n{n}_L{L}_net{k}.npz"
            if out.exists(): print("exists", out.name, flush=True); continue
            t0 = time.time(); Ws = weights(n, L, k)
            c = chain_stats(Ws); t1 = time.time()
            A = mc(Ws, 11 + 1000 * k, N); B = mc(Ws, 12 + 1000 * k, N)
            np.savez(out, N=N, **c, **{f"a_{key}": v for key, v in A.items()}, **{f"b_{key}": v for key, v in B.items()})
            print(f"{out.name}: chain {t1-t0:.1f}s, MC 2x{N:.0e} {time.time()-t1:.0f}s", flush=True)
    elif mode == "baked":
        first, count = int(sys.argv[2]), int(sys.argv[3])
        gt = np.load(BAKED / "gt.npy")
        for k in range(first, first + count):
            out = HERE / "res" / f"n1024_L16_baked{k}.npz"
            if out.exists(): print("exists", out.name, flush=True); continue
            t0 = time.time(); Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{k}.npy")]
            c = chain_stats(Ws)
            np.savez(out, N=int(1e9), **c, gt_m=gt[k].astype(np.float64))
            print(f"{out.name}: chain {time.time()-t0:.1f}s", flush=True)
