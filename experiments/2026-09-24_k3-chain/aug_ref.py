# ARC reference mlp_kprop, k=3, factor=True: SIMPLE vs AUGMENT on baked MLPs (final + per-layer MSE vs N=1e9 GT).
# Usage: python aug_ref.py <kind:SIMPLE|AUGMENT> <mlp ids comma-separated>
import sys, time, torch, numpy as np
from pathlib import Path
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
gt = np.load(BAKED / "gt.npy"); n = 1024; depth = 16
mode = sys.argv[1]; ids = [int(x) for x in sys.argv[2].split(",")]
kind = Kind.AUGMENT if mode.startswith("AUGMENT") else getattr(Kind, mode)
if mode == "AUGMENT_G":
    # AUGMENT with the SIMPLE (1,1,1) births only: keep kappa4 = Sym(G x M) (dense G, r=1 harmonic projection, incl. the
    # pK111/pK211 -> K211 contributions) and the 211->111 feed, drop the extra path legs (E3, K22-path, C^2-path).
    import inspect, textwrap, mlp_kprop.factor_k3 as fk3
    src = textwrap.dedent(inspect.getsource(fk3.factored_nonlin_kprop_k3))
    old_leg = "legs = [(1, 1, WK_11, 1)]\n        if augment:"
    assert src.count(old_leg) == 1, "patch point not found"
    src = src.replace(old_leg, "legs = [(1, 1, WK_11, 1)]\n        if augment and False:")
    ns = dict(vars(fk3)); exec(compile(src, "factor_k3_patched", "exec"), ns)
    fk3.factored_nonlin_kprop_k3 = ns["factored_nonlin_kprop_k3"]
def mk(): return MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth+1, nonlin="relu", init_kind="manual", w_var=2.0)
fins = []
for i in ids:
    W = np.load(BAKED / f"weights_{i}.npy"); m = mk()
    for l in range(depth): m.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    t0 = time.time()
    Kb = mlp_kprop(m, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=kind, factor=True, output_all=True, up_to_layer=f"act{depth-1}", output_d_max=1)
    mu = np.stack([np.asarray(Kb[f"act{l}"][1].to_tensor()) for l in range(depth)])
    per = ((mu - gt[i]) ** 2).mean(axis=1); fins.append(per[-1])
    print(f"{mode} mlp {i}: final {per[-1]:.4e}  all-layers {per.mean():.4e}  ({time.time()-t0:.0f}s)", flush=True)
    print("  per-layer:", " ".join(f"{x:.2e}" for x in per), flush=True)
    del Kb, m
print(f"{mode} mean final over {ids}: {np.mean(fins):.4e}", flush=True)
