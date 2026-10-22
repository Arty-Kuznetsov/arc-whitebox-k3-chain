"""What is the next order (K=4) worth, and in which pieces? Step 1: the cumulant-order ladder at small width, where
K=4 is affordable. ARC's reference (mlp_kprop) at k_max = 2, 3, 4 with kind SIMPLE and AUGMENT (factored for k >= 3)
against a Monte-Carlo ground truth, on random He-init ReLU MLPs of width n and depth L (the challenge's convention:
y_l = ReLU(W_l^T y_{l-1}), W ~ N(0, 2/n), x ~ N(0, I)). At 1024x16 the measured ladder is k1 3.3e-4, k2 4.5e-6,
k3 4.1e-8 (x74, x110 per order, the paper's (L/n)^K law with n/L = 64); the leaders sit at 1.1-1.6e-8, i.e. a small
piece of the next order. Here we measure how big the whole next order is at a comparable L/n, and later which pieces
of it carry the gain.
    ARC=$(cd ../../external/mlp_cumulant_propagation && pwd)
    env PYTHONPATH=$ARC/src nice -n 10 uv run --project $ARC python korder.py <n> <depth> <n_nets> <N_mc> [configs]
configs: comma list of k:kind, e.g. 2:S,3:S,3:A,4:S,4:A  (S = SIMPLE, A = AUGMENT)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_num_threads(3)
from mlp_kprop.mlp import MLP                                   # noqa: E402
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind            # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402  (chain.py on the path)
from chain import run_chain                                     # noqa: E402
OURS = {"full": dict(),                                         # = ARC's SIMPLE k=3 (the chain's own gate)
        "B9m2": dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),)),
        "B9m2nocut": dict(pb.BASE, tailr=(("marg2", "sat", None, -2.0, "gauss"),))}
n, depth, nets, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
cfgs = [((int(c.split(":")[0]), c.split(":")[1]) if ":" in c else (None, c))
        for c in (sys.argv[5] if len(sys.argv) > 5 else "2:S,3:S,3:A,4:S,4:A").split(",")]
KIND = {"S": Kind.SIMPLE, "A": Kind.AUGMENT}


def weights(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


def mc(Ws, seed, N, B=1 << 16):
    """Monte-Carlo post-ReLU means (all layers) and the last layer's per-neuron variance; float32 GEMMs, float64 sums.
    Cached in res/mc_n<n>_L<depth>_s<seed>_N<N>.npz."""
    f = HERE / "res" / f"mc_n{n}_L{depth}_s{seed}_N{N:.0e}.npz"
    if f.exists():
        d = np.load(f)
        return d["mu"], d["var"]
    g = torch.Generator().manual_seed(10_000 + seed)
    W32 = [W.to(torch.float32) for W in Ws]
    s1 = torch.zeros(depth, n, dtype=torch.float64)
    s2 = torch.zeros(n, dtype=torch.float64)
    done = 0
    while done < N:
        b = min(B, N - done)
        y = torch.randn(b, n, generator=g, dtype=torch.float32)
        for l in range(depth):
            y = torch.relu(y @ W32[l])
            s1[l] += y.sum(0, dtype=torch.float64)
        s2 += (y.double() ** 2).sum(0)
        done += b
    mu = s1 / N
    var = s2 / N - mu[-1] ** 2
    (HERE / "res").mkdir(exist_ok=True)
    np.savez(f, mu=mu.numpy(), var=var.numpy())
    return mu.numpy(), var.numpy()


def kprop(Ws, k, kind):
    torch.set_default_dtype(torch.float64)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    out = mlp_kprop(mlp, K_in, k_max=k, kind=KIND[kind], factor=(k >= 3), output_all=True, up_to_layer=f"act{depth - 1}")
    return torch.stack([out[f"act{l}"][1].to_tensor() for l in range(depth)]).numpy()


res = {"n": n, "depth": depth, "N_mc": N, "nets": []}
for s in range(nets):
    Ws = weights(s)
    t0 = time.time()
    mu, var = mc(Ws, s, N)
    row = {"seed": s, "mc_s": time.time() - t0, "var_last_mean": float(var.mean()),
           "mc_noise_mse": float(var.mean() / N)}
    print(f"net {s}: MC {N:.0e} in {row['mc_s']:.0f} s; final var {var.mean():.4f}; MC noise MSE {var.mean() / N:.2e}",
          flush=True)
    for k, kind in cfgs:
        t0 = time.time()
        try:
            if k is None:                           # our float64 research chain, config `kind`
                torch.set_default_dtype(torch.float64)
                m_, _ = run_chain([W.clone() for W in Ws], **OURS[kind])
                pred = np.stack([np.asarray(x, dtype=np.float64) for x in m_])
            else:
                pred = kprop(Ws, k, kind)
        except Exception as e:                      # a config the reference does not support at this size
            print(f"   {kind}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        dt = time.time() - t0
        fin = float(((pred[-1] - mu[-1]) ** 2).mean())
        alll = float(((pred - mu) ** 2).mean())
        row[f"k{k}{kind}" if k is not None else kind] = {"final": fin, "all": alll, "norm_final": fin / float(var.mean()), "s": dt}
        print(f"   {('k' + str(k) + kind) if k is not None else kind}: final MSE {fin:.3e} (normalised {fin / var.mean():.2e}), all-layer {alll:.3e}, {dt:.1f} s",
              flush=True)
    res["nets"].append(row)
    (HERE / "res").mkdir(exist_ok=True)
    json.dump(res, open(HERE / "res" / f"korder_n{n}_L{depth}.json", "w"), indent=1)
