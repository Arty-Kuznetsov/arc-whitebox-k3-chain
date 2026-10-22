"""REGENERATE the diagonal third cumulant from the covariance instead of transporting it.

diag_k3_probe.py killed the naive route: the memoryless diagonal transport carries only ~1.7%
of the true magnitude at depth, because kappa3 is made by OFF-diagonal cross terms.

But those cross terms are themselves a functional of the pre-activation mean and covariance,
which the k=2 chain already has. Wick/cluster expansion for functions of jointly Gaussian
variables, with derivative coefficients d_k = E[d^k ReLU(Z)] (matches the reference wick.py:
d_1 = Phi(alpha), d_2 = phi(alpha)/sigma) and pre-activation covariance C:

  kappa3_post[a,b,c] ~= d2_a d1_b d1_c C_ab C_ac + d1_a d2_b d1_c C_ab C_bc
                         + d1_a d1_b d2_c C_ac C_bc        (leading connected diagram, O(C^2))

Contracting with the next layer's weights u_i[a] = W[a,i] collapses to ONE matmul:

  V = (Wm * d1) @ C                       # Wm[i,a] = W[a,i]; V[i,c] = sum_a Wm[i,a] d1_a C_ac
  kappa3_pre[i,i,i] ~= 3 * sum_c Wm[i,c] * d2_c * V[i,c]^2

Cost: one (n,n)@(n,n) matmul per layer, ~2.1e9 FLOPs, against the k=2 chain's ~3.2e9/layer.
That keeps the whole estimator under the 10% multiplier floor, i.e. FREE on the metric.

This probe feeds the TRUE pre-activation state from the k=3 chain and scores the prediction
against the TRUE kappa3 diagonal one layer later. Compared against the memoryless baseline.

Run:  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
      env PYTHONPATH="$ARC/src" uv run --project "$ARC" python wick_regen_probe.py [n_mlps]
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.wick import relu_wick_coef

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])

SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)

# --- check our d1/d2 against the reference wick coefficients before using them ---
_mu = torch.tensor([0.0, 0.5, -0.7, 1.3]); _var = torch.tensor([1.0, 0.4, 2.0, 0.81])
_sig = _var.sqrt(); _al = _mu / _sig
d1_ref = relu_wick_coef(_mu, _var, k=1); d2_ref = relu_wick_coef(_mu, _var, k=2)
d1_ours = nPhi(_al); d2_ours = nphi(_al) / _sig
e1 = float((d1_ref - d1_ours).abs().max()); e2 = float((d2_ref - d2_ours).abs().max())
print(f"WICK CHECK  d1: max|ours - reference| = {e1:.2e}   d2: {e2:.2e}   "
      f"{'OK' if max(e1, e2) < 1e-12 else 'MISMATCH -- STOP'}\n")
if max(e1, e2) >= 1e-12:
    sys.exit(1)

stats = []
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE,
                   factor=True, output_all=True, up_to_layer=f"act{depth-1}")
    pre_mu = [Kb[f"pre{l}"][1].to_tensor().clone() for l in range(depth)]
    pre_C = [Kb[f"pre{l}"][2].to_tensor().clone() for l in range(depth)]
    pre_d = [Kb[f"pre{l}"][3].get_dslice((3,)).clone() if 3 in Kb[f"pre{l}"] else torch.zeros(n)
             for l in range(depth)]
    post_d = [Kb[f"act{l}"][3].get_dslice((3,)).clone() if 3 in Kb[f"act{l}"] else torch.zeros(n)
              for l in range(depth)]
    Wms = [mlp.Ws[l].weight.clone() for l in range(depth)]
    del Kb

    per_layer = []
    for l in range(depth - 1):
        sig = pre_C[l].diagonal().clamp_min(1e-12).sqrt()
        al = pre_mu[l] / sig
        d1, d2 = nPhi(al), nphi(al) / sig
        Wm = Wms[l + 1]                                   # Wm[i, a] = W_challenge[a, i]
        V = (Wm * d1) @ pre_C[l]                          # one n x n matmul
        wick = 3.0 * ((Wm * d2) * V * V).sum(dim=1)       # regenerated diagonal
        memo = (Wm ** 3) @ post_d[l]                      # the transported diagonal (baseline)
        true = pre_d[l + 1]
        tn = float((true ** 2).mean())
        row = []
        for name, p in (("wick", wick), ("memo", memo), ("both", wick + memo)):
            pn = float((p ** 2).mean())
            corr = float(torch.corrcoef(torch.stack([p, true]))[0, 1])
            rel = float(((p - true) ** 2).mean() / tn) ** 0.5
            row += [corr, rel, (pn / tn) ** 0.5]
        per_layer.append(row)
    stats.append(per_layer)
    print(f"  mlp {i} done", flush=True)

S = np.mean(stats, axis=0)
print(f"\n=== Regenerated vs transported diagonal kappa3 ({N_MLPS} MLPs) ===")
print("prediction of kappa3_pre[i,i,i] at layer l+1, scored against the k=3 chain's truth\n")
print(f"{'l+1':>4} | {'wick corr':>9} {'wick rel':>9} {'wick mag':>9} | "
      f"{'memo corr':>9} {'memo mag':>9} | {'both corr':>9} {'both rel':>9} {'both mag':>9}")
for l in range(depth - 1):
    wc, wr, wm, mc, mr, mm, bc, br, bm = S[l]
    print(f"{l+1:>4} | {wc:>9.3f} {wr:>9.3f} {wm:>9.3f} | {mc:>9.3f} {mm:>9.3f} | "
          f"{bc:>9.3f} {br:>9.3f} {bm:>9.3f}")
print(f"\nmeans: wick corr {S[:,0].mean():.3f} rel {S[:,1].mean():.3f} mag {S[:,2].mean():.3f}"
      f"  |  memo corr {S[:,3].mean():.3f} mag {S[:,5].mean():.3f}"
      f"  |  both corr {S[:,6].mean():.3f} rel {S[:,7].mean():.3f} mag {S[:,8].mean():.3f}")
print("\ncorr -> 1, rel -> 0, mag -> 1 means the diagonal can be REGENERATED from the covariance")
print("the k=2 chain already carries, and no cross-cumulant tensor needs to be transported.")
