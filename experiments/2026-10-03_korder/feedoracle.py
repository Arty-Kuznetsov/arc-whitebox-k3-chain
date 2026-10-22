"""Is the feed's missing (2,1,1) content a lever? Our chain (B9 + marg2) with the kappa4 -> kappa3 feed fed by ARC k=4's
PRE-activation (2,1,1) slice instead of the closure's v var_a C_bc (chain.py `dense3`, a dense kappa3 tensor riding the
chain; small widths only). Gate: the dense path with S = the closure's own form must reproduce the base.
args: n depth nets N"""
import sys
from pathlib import Path
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402
n, depth, nets, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
idx = torch.arange(n)


def weights(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


res = {}
for s in range(nets):
    Ws = weights(s)
    mu_mc = np.load(HERE / "res" / f"mc_n{n}_L{depth}_s{s}_N{N:.0e}.npz")["mu"]
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                    up_to_layer=f"act{depth - 1}")
    S = {l: out[f"pre{l}"][4].get_dslice((2, 1, 1)).clone() for l in range(1, depth - 1)}
    arms = {"base": {}, "feed<-k4 slice": dict(dense3={"feedS": S, "mode": "replace"}),
            "feed<-k4 slice x0.5": dict(dense3={"feedS": {l: 0.5 * v for l, v in S.items()}, "mode": "replace"}),
            "feed<-k4, add": dict(dense3={"feedS": S, "mode": "add"})}
    for arm, kw in arms.items():
        m_, _ = run_chain([W.clone() for W in Ws], **dict(BASE, **kw))
        pred = np.stack([np.asarray(x, dtype=np.float64) for x in m_])
        res.setdefault(arm, []).append(float(((pred[-1] - mu_mc[-1]) ** 2).mean()))
    print(f"net {s}: " + "  ".join(f"{a} {v[-1]:.3e}" for a, v in res.items()), flush=True)
b = np.mean(res["base"])
for a, v in res.items():
    print(f"  {a:22s} {np.mean(v):.3e}  x{np.mean(v) / b:.3f}  per net " + " ".join(f"{x / y:.3f}" for x, y in zip(v, res["base"])))
