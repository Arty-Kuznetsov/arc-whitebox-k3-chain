"""Per-layer localisation of the Gaussian-closure error at 1024x16.

Question nobody in the public corpus has answered at this shape: WHERE along depth is the
k=2 (Gaussian closure) error actually created, and is it a mean error or a covariance error?

Two oracle-injection sweeps, both ending in a plain k=2 continuation to the final layer:
  (A) GT-mean injection : run k=2; at layer l replace the post-ReLU MEAN by the baked N=1e9
      ground-truth mean, KEEP the k=2 covariance, continue k=2 to the end.
      -> isolates the error carried by the propagated MEAN alone.
  (B) k=3-state injection: run k=3 once; at layer l take its post-ReLU (mean, cov),
      continue k=2 to the end.
      -> isolates the error carried by the whole 2nd-order state (mean AND covariance).
  (C) covariance-only injection: at layer l keep k=2's OWN mean but take k=3's covariance,
      continue k=2 to the end.
      -> the decisive control: A and C together split the blame between mean and covariance.

Self-checks (printed first, must pass):
  no injection            -> full k=2   (expect 4.50e-6)
  (A) inject at layer 15  -> exactly GT (expect 0)
  (B) inject at layer 15  -> full k=3   (expect 3.71e-8 on MLP 0)
  (C) inject at layer 15  -> full k=2   (expect base exactly: only the mean is read back)

Run (light on the machine: 3 torch threads, one MLP at a time, k=3 state freed immediately):
  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
  env PYTHONPATH="$ARC/src" uv run --project "$ARC" python layer_blame.py [n_mlps]
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)  # be a good citizen: the laptop is shared with other work

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import (
    mlp_kprop, Kind, linear_kprop, relu_kprop, coerce_input,
)

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
gt = np.load(BAKED / "gt.npy")  # (K, 16, 1024) post-ReLU per-layer means
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 3, gt.shape[0])


def build(W):
    m = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
            nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        m.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    return m


def run_full(mlp, k_max):
    """Full chain, all post-ReLU states kept."""
    return mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=k_max,
                     kind=Kind.SIMPLE, factor=(k_max >= 3), output_all=True,
                     up_to_layer=f"act{depth-1}")


def continue_k2(mlp, mean, cov, from_layer):
    """Plain k=2 from an injected post-ReLU state; returns the final-layer mean."""
    K = coerce_input({1: mean, 2: cov}, k_max=2, kind=Kind.SIMPLE)
    for l in range(from_layer, depth):
        K = linear_kprop(K, mlp.Ws[l].weight, k_max=2, set_metric=mlp.init_scale[l])
        K = relu_kprop(K, k_max=2, kind=Kind.SIMPLE)
    return np.asarray(K[1].to_tensor())


mse = lambda pred, truth: float(((pred - truth) ** 2).mean())
A_all, B_all, C_all, base_all = [], [], [], []

for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    g = gt[i]
    mlp = build(W)

    # --- k=2 reference chain: keep (mean, cov) per layer, then free the tower ---
    Kb2 = run_full(mlp, 2)
    st2 = [(Kb2[f"act{l}"][1].to_tensor().clone(), Kb2[f"act{l}"][2].to_tensor().clone())
           for l in range(depth)]
    del Kb2

    # --- k=3 reference chain: same, then free immediately (it is the memory-heavy one) ---
    Kb3 = run_full(mlp, 3)
    st3 = [(Kb3[f"act{l}"][1].to_tensor().clone(), Kb3[f"act{l}"][2].to_tensor().clone())
           for l in range(depth)]
    del Kb3

    base = mse(continue_k2(mlp, torch.zeros(n), torch.eye(n), 0), g[-1])
    base_all.append(base)

    # (A) inject the GT mean at layer l, keep the k=2 covariance
    A = [mse(continue_k2(mlp, torch.tensor(g[l], dtype=torch.float64), st2[l][1], l + 1), g[-1])
         for l in range(depth)]
    # (B) inject the k=3 (mean, cov) at layer l
    B = [mse(continue_k2(mlp, st3[l][0], st3[l][1], l + 1), g[-1]) for l in range(depth)]
    # (C) keep k=2's own mean, take only k=3's covariance
    C = [mse(continue_k2(mlp, st2[l][0], st3[l][1], l + 1), g[-1]) for l in range(depth)]
    A_all.append(A); B_all.append(B); C_all.append(C)

    if i == 0:
        print("SELF-CHECKS (MLP 0)")
        print(f"  no injection            = {base:.4e}   (expect 4.50e-06 = full k=2)")
        print(f"  (A) inject GT at l=15   = {A[-1]:.4e}   (expect 0 exactly)")
        print(f"  (B) inject k=3 at l=15  = {B[-1]:.4e}   (expect 3.71e-08 = full k=3)")
        print(f"  (C) cov-only at l=15    = {C[-1]:.4e}   (expect {base:.4e} = base exactly)")
        print()
    print(f"  mlp {i}: base={base:.3e}  A[l=8]={A[8]:.3e}  B[l=8]={B[8]:.3e}", flush=True)
    del st2, st3, mlp

A = np.mean(A_all, axis=0); B = np.mean(B_all, axis=0); C = np.mean(C_all, axis=0)
base = float(np.mean(base_all))
print(f"\n=== Per-layer localisation of the k=2 closure error ({N_MLPS} MLPs, mean) ===")
print(f"baseline (no injection, full k=2) = {base:.4e}\n")
print("l  = layer where the oracle state is injected; the chain then runs plain k=2 to layer 15")
print("A = exact MEAN injected, cov left at k=2   C = cov replaced by k=3's, mean left at k=2")
print("B = both replaced by k=3's")
print(f"{'l':>2} | {'A: mse':>10} {'A/base':>7} {'dA':>10} | {'C: mse':>10} {'C/base':>7} | {'B: mse':>10} {'B/base':>7}")
for l in range(depth):
    dA = A[l] - (A[l - 1] if l else base)
    print(f"{l:>2} | {A[l]:>10.3e} {A[l]/base:>7.3f} {dA:>+10.2e} | "
          f"{C[l]:>10.3e} {C[l]/base:>7.3f} | {B[l]:>10.3e} {B[l]/base:>7.3f}")
print("\nReading: X[l]/base = fraction of the final error that survives when that quantity is fixed")
print("at layer l. dA = the marginal blame of layer l's mean. If C stays near 1.0 while A falls,")
print("the k=2 error is carried by the propagated MEAN and not by the covariance.")
np.savez(Path(__file__).resolve().parent / "layer_blame.npz",
         A=np.array(A_all), B=np.array(B_all), C=np.array(C_all), base=np.array(base_all))
print("saved per-MLP arrays to layer_blame.npz")
