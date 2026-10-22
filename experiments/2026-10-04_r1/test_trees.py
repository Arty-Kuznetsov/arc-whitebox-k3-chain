"""Unit test of chain._k4_trees against ARC's factored k=4 step (split into per-type blocks by r1k4split): on a random
small MLP, take ARC k=4's pre-activation tower at layer m, run ARC's step, and compare its path / star / 3+2 blocks
with _k4_trees built from the same tower's C0, Wick gates and all-distinct kappa3 -- slice by slice."""
import sys
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
import r1k4split                                                           # noqa: E402
from mlp_kprop.mlp import MLP                                              # noqa: E402
from mlp_kprop.kprop_harmonic import Kind, coerce_input, linear_kprop, nonlin_kprop   # noqa: E402
from mlp_kprop.wick import WICK_COEF_D, relu_wick_coef                     # noqa: E402
from mlp_kprop.factor_k4 import _factored_get_dslice                       # noqa: E402
from r1lib import weights                                                  # noqa: E402
from chain import _k4_trees                                                # noqa: E402

NB = r1k4split.install()
n, depth, m = int(sys.argv[1]) if len(sys.argv) > 1 else 32, 4, 2
Ws = weights(n, depth, 7)
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
for l in range(depth):
    mlp.Ws[l].weight.data = Ws[l].T.clone()
K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE)
for l in range(m + 1):
    K = linear_kprop(K, mlp.Ws[l].weight, k_max=4, set_metric=mlp.init_scale[l], bias=mlp.Ws[l].bias)
    if l == m:
        break
    K = nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D["relu"], k_max=4, kind=Kind.SIMPLE, use_pK=True, factor=True)
mean, C = K[1].to_tensor(), K[2].to_tensor()
var = torch.diagonal(C)
K3 = K[3].to_tensor().clone()
idx = torch.arange(n)
K3[idx, idx, :] = 0; K3[idx, :, idx] = 0; K3[:, idx, idx] = 0
w = {k: relu_wick_coef(mean=mean, var=var, k=k, p=1) for k in (1, 2, 3)}
r_old = K[4]._factors[0].shape[2]
Kout = nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D["relu"], k_max=4, kind=Kind.SIMPLE, use_pK=True, factor=True)
A, B = Kout[4]._factors
mine = _k4_trees(C, w[1], w[2], w[3], K[3].to_tensor(), ("path", "star", "32"), zero_diag=False)
Am, Bm = mine._factors
ok = True
for ti, t in enumerate(("path", "star", "32")):
    cs = torch.arange(r_old + ti * n, r_old + (ti + 1) * n)
    cm = torch.arange(ti * n, (ti + 1) * n)
    for part in ((4,), (3, 1), (2, 2), (2, 1, 1)):
        ref = _factored_get_dslice(A[:, :, cs], B[:, :, cs], part)
        got = _factored_get_dslice(Am[:, :, cm], Bm[:, :, cm], part)
        err = float((got - ref).abs().max() / ref.abs().max().clamp_min(1e-300))
        ok &= err < 1e-12
        print(f"  {t:5s} {str(part):10s} max|ref| {float(ref.abs().max()):.3e}  rel err {err:.1e}")
print("PASS" if ok else "FAIL")
