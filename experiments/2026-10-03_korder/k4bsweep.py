"""How much of the next order's gain do the chain's existing kappa4-birth readers capture? `k4b` reads the all-distinct
kappa4 born at the last k4b ReLUs (tree diagrams: 12 paths + 4 stars) into the pre-activation K22 and diagonal through
the kappa3 legs; production uses k4b=1 with the star diagonal only. Sweep k4b and its parts, plus k4k3 / allages, at
small width against the cached Monte-Carlo means (korder.py).  args: n depth nets N"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402
n, depth, nets, N = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4]))
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
ALL = ("pd", "p22", "sd", "s22")
ARMS = {"base": {}}
for kb in (1, 2, 3, 4, 6, depth):
    ARMS[f"k4b{kb}_all"] = dict(k4b=kb, k4bparts=ALL)
ARMS["k4b2_sd"] = dict(k4b=2, k4bparts=("sd",))
ARMS["k4k3_0.5"] = dict(k4k3=0.5)
ARMS["allages"] = dict(e3age=None)
ARMS[f"k4b{depth}_all+k4k3+allages"] = dict(k4b=depth, k4bparts=ALL, k4k3=0.5, e3age=None)


def weights(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


res = {a: [] for a in ARMS}
for s in range(nets):
    Ws = weights(s)
    mu_mc = np.load(HERE / "res" / f"mc_n{n}_L{depth}_s{s}_N{N:.0e}.npz")["mu"]
    for arm, kw in ARMS.items():
        m_, _ = run_chain([W.clone() for W in Ws], **dict(BASE, **kw))
        pred = np.stack([np.asarray(x, dtype=np.float64) for x in m_])
        res[arm].append(float(((pred[-1] - mu_mc[-1]) ** 2).mean()))
b = np.mean(res["base"])
for arm in ARMS:
    print(f"  {arm:28s} {np.mean(res[arm]):.3e}  x{np.mean(res[arm]) / b:.3f}   per net " +
          " ".join(f"{v / w:.3f}" for v, w in zip(res[arm], res["base"])))
json.dump(res, open(HERE / "res" / f"k4bsweep_n{n}_L{depth}.json", "w"), indent=1)
