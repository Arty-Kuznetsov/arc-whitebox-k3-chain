"""K4d of base / t3k4 exact / zlegs vs ARC k=4 on each arm's own trajectory: level error (mean diff / mean), corr, slope.
args: n depth seed"""
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
n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
g = torch.Generator().manual_seed(seed)
Ws = [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]
mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
for l in range(depth):
    mlp.Ws[l].weight.data = Ws[l].T.clone()
out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                up_to_layer=f"act{depth - 1}")
for arm, kw in (("base", {}), ("t3k4", dict(t3k4="exact")), ("zlegs", dict(zlegs=True)), ("znodg", dict(zlegs="nodiag"))):
    _, info = run_chain([W.clone() for W in Ws], record=True, **dict(BASE, **kw))
    line = f"{arm:6s}"
    for l in range(2, depth):
        x = info[l]["K4d"].numpy(); y = out[f"pre{l}"][4].get_dslice((4,)).numpy()
        line += f" | l{l} lvl {(x.mean() - y.mean()) / y.mean():+.3f} c {np.corrcoef(x, y)[0, 1]:.3f}"
    print(line)
