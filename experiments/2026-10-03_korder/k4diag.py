"""Our chain's kappa4 diagonal (and K22, var, t) against ARC k=4's, layer by layer, on the chain's own trajectory:
relative rms error, its coherent share (mean^2 / mean-square of the difference), z = mean/SE, correlation, slope.
If the kappa4 error is mostly LEVEL, a cheap fix exists; if it is pattern, it needs the tensor.  args: n depth nets"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP                                   # noqa: E402
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind            # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402
n, depth, nets = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
if len(sys.argv) > 4 and sys.argv[4] == "k4b":           # the chain's kappa4-birth readers at full depth, all tree parts
    BASE = dict(BASE, k4b=depth, k4bparts=("pd", "p22", "sd", "s22"))


def weights(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


for s in range(nets):
    Ws = weights(s)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                    up_to_layer=f"act{depth - 1}")
    _, info = run_chain([W.clone() for W in Ws], record=True, **BASE)
    print(f"net {s} (n={n}, L={depth}): ours vs k=4, per layer — rel.rms err | coherent share | z | corr | slope ours/k4")
    for l in range(1, depth):
        T = out[f"pre{l}"]
        ref = {"var": torch.diagonal(T[2].to_tensor()).numpy(), "t": T[3].get_dslice((3,)).numpy(),
               "K4d": T[4].get_dslice((4,)).numpy()}
        K22r = T[4].get_dslice((2, 2)).numpy()
        ours = {"var": info[l]["var"].numpy(), "t": info[l]["t"].numpy(), "K4d": info[l]["K4d"].numpy()}
        line = f"  l={l:2d}"
        for k in ("var", "t", "K4d"):
            x, y = ours[k], ref[k]
            e = x - y
            rel = np.sqrt((e @ e) / (y @ y))
            coh = e.mean() ** 2 / (e @ e / len(e))
            z = e.mean() / np.sqrt((e @ e) / len(e) / len(e))
            line += f" | {k} {rel:.3f} {coh:.2f} {z:+6.1f} {np.corrcoef(x, y)[0, 1]:.3f} {(x @ y) / (y @ y):.3f}"
        K22o = info[l].get("K22")
        if K22o is not None:
            M = ~np.eye(n, dtype=bool)
            x, y = K22o.numpy()[M], K22r[M]
            e = x - y
            line += f" | K22 {np.sqrt((e @ e) / (y @ y)):.3f} {e.mean() ** 2 / (e @ e / len(e)):.2f} corr {np.corrcoef(x, y)[0, 1]:.3f}"
        print(line, flush=True)
