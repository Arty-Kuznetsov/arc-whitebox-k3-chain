"""Which statistics of the LAST layer matter for the final readout? Substitute ARC k=4's pre-activation var / t / K4d
(and combinations) at the last layer only (no downstream propagation, so no consistency issue).  args: n depth nets N"""
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
L = depth - 1
res = {}
for s in range(nets):
    g = torch.Generator().manual_seed(s)
    Ws = [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]
    mu_mc = np.load(HERE / "res" / f"mc_n{n}_L{depth}_s{s}_N{N:.0e}.npz")["mu"]
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                    up_to_layer=f"act{depth - 1}")
    T = out[f"pre{L}"]
    st = {"var": {L: torch.diagonal(T[2].to_tensor()).numpy()}, "k3": {L: T[3].get_dslice((3,)).numpy()},
          "k4": {L: T[4].get_dslice((4,)).numpy()}}
    arms = {"base": (), "var@L": ("var",), "t@L": ("k3",), "K4d@L": ("k4",), "t+K4d@L": ("k3", "k4"), "all@L": ("var", "k3", "k4")}
    for arm, keys in arms.items():
        orc = {}
        for k in keys:
            orc[k] = st[k]; orc[f"{k}_layers"] = {L}
        m_, _ = run_chain([W.clone() for W in Ws], oracle=(orc or None), **BASE)
        pred = np.stack([np.asarray(x, dtype=np.float64) for x in m_])
        res.setdefault(arm, []).append(float(((pred[-1] - mu_mc[-1]) ** 2).mean()))
b = np.mean(res["base"])
for a, v in res.items():
    print(f"  {a:10s} {np.mean(v):.3e}  x{np.mean(v) / b:.3f}  per net " + " ".join(f"{x / y:.3f}" for x, y in zip(v, res["base"])))
