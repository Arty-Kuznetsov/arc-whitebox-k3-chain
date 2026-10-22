"""Validate the ARC `mlp_kprop` adapter against Monte Carlo at small width, using EXACT
challenge conventions. If conventions are right, kprop final means track MC and improve with k.

Challenge: for each of `depth` layers, pre = x @ W_l with W_l ~ N(0, 2/n) (orientation (in,out)),
post = ReLU(pre); input ~ N(0,1); the score is post-ReLU means for every layer.
ARC MLP has NO ReLU after its last linear and its default He gives layer-0 var 1/n, so we:
  - build num_layers = depth+1 (⇒ `depth` ReLUs: act0..act{depth-1}),
  - inject W_l.T into Ws[0..depth-1]  (ARC Linear weight is (out,in) = W_challenge.T),
  - force init_kind='manual', w_var=2.0  ⇒ init_scale=[2]*(depth+1)  (avg metric = 2/n every layer),
  - read act{depth-1} as the final post-ReLU layer.
"""
import math
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind


def make_challenge_weights(n, depth, seed):
    g = torch.Generator().manual_seed(seed)
    scale = math.sqrt(2.0 / n)
    return [torch.randn(n, n, generator=g) * scale for _ in range(depth)]  # (in,out)


def mc_layer_means(Ws, N, seed):
    g = torch.Generator().manual_seed(seed)
    n = Ws[0].shape[0]
    x = torch.randn(N, n, generator=g)
    means = []
    for W in Ws:
        x = torch.clamp(x @ W, min=0.0)  # challenge: pre = x @ W, post = ReLU
        means.append(x.mean(0))
    return torch.stack(means)  # (depth, n)


def kprop_layer_means(Ws, k_max, kind=Kind.SIMPLE, use_avg_metric=True):
    n = Ws[0].shape[0]
    depth = len(Ws)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    assert mlp.init_scale == [2.0] * (depth + 1), mlp.init_scale
    assert not mlp.has_bias()
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.contiguous().clone()  # ARC (out,in) = W_challenge.T
    K_in = {1: torch.zeros(n), 2: torch.eye(n)}
    K_by = mlp_kprop(mlp, K_in, k_max=k_max, kind=kind, output_all=True,
                     use_avg_metric=use_avg_metric, up_to_layer=f"act{depth-1}")
    means = [K_by[f"act{l}"][1].to_tensor() for l in range(depth)]
    return torch.stack(means)  # (depth, n)


if __name__ == "__main__":
    n, depth, N = 64, 4, 2_000_000
    print(f"n={n} depth={depth} N_mc={N:,}")
    for seed in [0, 1]:
        Ws = make_challenge_weights(n, depth, seed)
        mc = mc_layer_means(Ws, N, seed + 100)
        h1 = mc_layer_means(Ws, N // 2, seed + 100)
        h2 = mc_layer_means(Ws, N // 2, seed + 500)
        nf = float(((h1[-1] - h2[-1]) ** 2).mean())  # ~4x the ref's own final-layer noise
        print(f"\nseed {seed}: MC final-mean ‖·‖={float(mc[-1].norm()):.4f}, "
              f"first3={[round(v,4) for v in mc[-1][:3].tolist()]}, noise_floor~{nf:.2e}")
        for k in [1, 2, 3]:
            km = kprop_layer_means(Ws, k)
            fin = float(((km[-1] - mc[-1]) ** 2).mean())
            allm = float(((km - mc) ** 2).mean())
            print(f"   kprop k={k}: ‖final‖={float(km[-1].norm()):.4f}  "
                  f"final_mse_vs_MC={fin:.3e}  all_layers_mse={allm:.3e}")
