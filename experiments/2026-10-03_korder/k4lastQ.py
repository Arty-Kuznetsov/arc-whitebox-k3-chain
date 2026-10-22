"""Does the kappa3 -> kappa4 transfer, inherited with the EXACT propagators, reproduce k=4's last-layer kappa4 diagonal?
In k=4's own world: T3_m(a,b,c) = 2 h_a d1_b d1_c k3pre_m(a,b,c) (h = var d1 d2 at layer m) is a (2,1,1)-class piece of
k4post(m); carried to the pre-activations of the last layer L by the gated propagator Q_m = D1_m W_{m+1} D1_{m+1} ...
W_L (first-order gating on every index), its contribution to K4d(L) is
    6 sum_abc Q_ai^2 Q_bi Q_ci T3_m(a,b,c)                       (the doubled index through Q o Q: ALL split paths)
against the W o W-per-layer path of the "doubled legs". Compare with k=4's K4d(L) after removing the (2,2)+diag classes
of its last step (the part the closure models).  args: n depth seed"""
import sys
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
g = torch.Generator().manual_seed(seed)
Ws = [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
for l in range(depth):
    mlp.Ws[l].weight.data = Ws[l].T.clone()
out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                up_to_layer=f"act{depth - 1}")
Phi = lambda x: 0.5 * (1.0 + torch.erf(x / 2 ** 0.5))
idx = torch.arange(n)
L = depth - 1
d1s, hs = {}, {}
for m in range(depth):
    pre = out[f"pre{m}"]
    mu = pre[1].to_tensor(); var = torch.clamp(torch.diagonal(pre[2].to_tensor()), min=1e-12)
    a = mu / torch.sqrt(var)
    d1s[m] = Phi(a); hs[m] = var * d1s[m] * torch.exp(-0.5 * a * a) / (2 * torch.pi) ** 0.5 / torch.sqrt(var)
truth = out[f"pre{L}"][4].get_dslice((4,))
# the closure-like part of the last step: diag + (2,2) classes of k4post(L-1)
P = out[f"act{L - 1}"][4]
d4 = P.get_dslice((4,)); s22 = P.get_dslice((2, 2)).clone(); s22[idx, idx] = 0
W = Ws[L]
biv = (W ** 4 * d4[:, None]).sum(0) + 3.0 * ((W ** 2) * (s22 @ (W ** 2))).sum(0)
rest = truth - biv
print(f"n={n} L={depth} seed={seed}: k=4's last-layer K4d = bivariate (2,2)+diag part + rest; corr(biv, truth) "
      f"{np.corrcoef(biv.numpy(), truth.numpy())[0, 1]:.3f}")
accQ = torch.zeros(n); accW = torch.zeros(n)
for m in range(1, L):
    k3 = out[f"pre{m}"][3].to_tensor()
    T3 = 2.0 * hs[m][:, None, None] * d1s[m][None, :, None] * d1s[m][None, None, :] * k3
    T3[idx, idx, :] = 0; T3[idx, :, idx] = 0; T3[:, idx, idx] = 0
    Q = d1s[m][:, None] * Ws[m + 1]                       # post m -> pre m+1
    Wsq = Q * Q                                            # the per-layer W o W path (with d1^2 gating)
    for k in range(m + 1, L):
        Q = Q @ (d1s[k][:, None] * Ws[k + 1])
        Wsq = Wsq @ ((d1s[k] ** 2)[:, None] * Ws[k + 1] ** 2)
    U = torch.einsum("abc,bi,ci->ai", T3, Q, Q)            # singles through Q
    cQ = 6.0 * (Q * Q * U).sum(0)                          # doubled index through Q o Q (all split paths)
    cW = 6.0 * (Wsq * U).sum(0)                            # doubled index through the product of W o W (zlegs' path)
    accQ += cQ; accW += cW
    print(f"  births at m={m}: corr(cQ, rest) {np.corrcoef(cQ.numpy(), rest.numpy())[0, 1]:+.3f}  "
          f"corr(cW, rest) {np.corrcoef(cW.numpy(), rest.numpy())[0, 1]:+.3f}  |cQ|/|rest| {float(cQ.norm() / rest.norm()):.3f}", flush=True)
for tag, acc in (("Q o Q, all m", accQ), ("W o W, all m", accW)):
    c = np.corrcoef(acc.numpy(), rest.numpy())[0, 1]
    sl = float((acc @ rest) / (acc @ acc))
    tot = biv + acc
    print(f"  {tag}: corr with rest {c:.3f}, slope {sl:.2f} | corr(biv + it, truth) {np.corrcoef(tot.numpy(), truth.numpy())[0, 1]:.3f}")
# --- how much of k=4's last-layer K4d do [the closure's var^2 pattern] and [the inherited transfer] explain together?
var_L = torch.diagonal(out[f"pre{L}"][2].to_tensor())
X1 = (var_L ** 2).numpy(); y = truth.numpy()
def r2(cols):
    F = np.stack(cols, 1); coef, *_ = np.linalg.lstsq(F, y, rcond=None); res = y - F @ coef
    return 1 - (res @ res) / ((y - y.mean()) @ (y - y.mean())), coef
for tag, cols in (("var^2 (closure pattern)", [X1, np.ones(n)]), ("var^2 + inherited T3 (QoQ)", [X1, accQ.numpy(), np.ones(n)]),
                  ("var^2 + inherited T3 (WoW)", [X1, accW.numpy(), np.ones(n)]), ("var^2 + biv + T3 (QoQ)", [X1, biv.numpy(), accQ.numpy(), np.ones(n)])):
    r, c = r2(cols)
    print(f"  R2 of k=4's K4d(L) on {tag:30s}: {r:.3f}   coef {np.array2string(c, precision=3)}")
