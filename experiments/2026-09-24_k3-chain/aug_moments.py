# Per-layer pre-activation moments (mean, var, kappa3 diag, kappa4 diag) of ARC's mlp_kprop (k=3, factor=True) for a
# baked MLP, captured by patching clone_tower (so no factored tensors are stored). Saves runs/aug_moments_<mode>_<i>.npz.
# Usage: python aug_moments.py <SIMPLE|AUGMENT|AUGMENT_G> <mlp>
import sys, time, torch, numpy as np
from pathlib import Path
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from mlp_kprop.mlp import MLP
import mlp_kprop.kprop_harmonic as kh
from mlp_kprop.harmonic import harmonic_diagslice
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
mode, i = sys.argv[1], int(sys.argv[2]); n, depth = 1024, 16
SAVE_G = len(sys.argv) > 3 and sys.argv[3] == "G"
kind = kh.Kind.AUGMENT if mode.startswith("AUGMENT") else getattr(kh.Kind, mode)
if mode == "AUGMENT_G":
    import inspect, textwrap, mlp_kprop.factor_k3 as fk3
    src = textwrap.dedent(inspect.getsource(fk3.factored_nonlin_kprop_k3))
    src = src.replace("legs = [(1, 1, WK_11, 1)]\n        if augment:", "legs = [(1, 1, WK_11, 1)]\n        if augment and False:")
    ns = dict(vars(fk3)); exec(compile(src, "factor_k3_patched", "exec"), ns); fk3.factored_nonlin_kprop_k3 = ns["factored_nonlin_kprop_k3"]
def stats(K, d_max=None):
    out = {"mean": K[1].to_tensor().clone(), "var": torch.diagonal(K[2].to_tensor()).clone()}
    out["k3"] = torch.zeros(n); out["k4"] = torch.zeros(n)
    if SAVE_G:
        out["k31"] = torch.zeros(n, n, dtype=torch.float32); out["k22"] = torch.zeros(n, n, dtype=torch.float32)
    if 3 in K:
        out["k3"] = K[3].get_dslice((3,)).clone() if hasattr(K[3], "get_dslice") else torch.diagonal(torch.diagonal(K[3].to_tensor())).clone()
    if 4 in K:
        out["k4"] = harmonic_diagslice(K[4], (4,)).clone()
        if SAVE_G:  # the pre-activation kappa4 (3,1) and (2,2) slices of Sym(G x M), float32
            out["k31"] = harmonic_diagslice(K[4], (3, 1)).to(torch.float32).clone()
            out["k22"] = harmonic_diagslice(K[4], (2, 2)).to(torch.float32).clone()
    return out
kh.clone_tower = stats
m = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
W = np.load(BAKED / f"weights_{i}.npy")
for l in range(depth): m.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
t0 = time.time()
Kb = kh.mlp_kprop(m, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=kind, factor=True, output_all=True, up_to_layer=f"act{depth-1}")
res = {}
for l in range(depth):
    s = Kb[f"pre{l}"]
    for k, v in s.items(): res[f"pre_{k}"] = res.get(f"pre_{k}", []) + [v.numpy()]
    res["post_mean"] = res.get("post_mean", []) + [Kb[f"act{l}"]["mean"].numpy()]
res = {k: np.stack(v) if len({x.shape for x in v}) == 1 else np.array(v, dtype=object) for k, v in res.items()}
np.savez(HERE / "runs" / f"aug_moments_{mode}_{i}{'_G' if SAVE_G else ''}.npz", **res)
gt = np.load(BAKED / "gt.npy")[i]
print(f"{mode} mlp {i}: final {((res['post_mean'][-1]-gt[-1])**2).mean():.4e} ({time.time()-t0:.0f}s); keys {list(res)}", flush=True)
