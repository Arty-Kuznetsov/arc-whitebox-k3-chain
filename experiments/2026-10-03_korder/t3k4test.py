"""The kappa3 -> kappa4 transfer (chain.py `t3k4`) at small width: final-layer MSE vs the cached MC means, and the kappa4
diagonal's agreement with ARC k=4 on the chain's own trajectory.  args: n depth nets N"""
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
ARMS = {"base": {}, "t3k4 exact": dict(t3k4="exact"), "zlegs": dict(zlegs=True), "zlegs x0.5": dict(zlegs=True, zmult=0.5),
        "zlegs x1.5": dict(zlegs=True, zmult=1.5), "znodiag": dict(zlegs="nodiag"), "znodiag x0.5": dict(zlegs="nodiag", zmult=0.5),
        "t3k4 last": dict(t3k4="exact", t3k4_last=True), "t3k4 last x2": dict(t3k4="exact", t3k4_last=True, t3k4mult=2.0),
        "t3k4 mf last": dict(t3k4="mf", t3k4_last=True), "feed off": dict(k4feed=None), "feed x0.5": dict(feedmult=0.5), "feed x1.5": dict(feedmult=1.5), "feed x2": dict(feedmult=2.0)}
if len(sys.argv) > 5:
    ARMS = {k: v for k, v in ARMS.items() if k in ("base",) + tuple(sys.argv[5].split(","))}


def weights(seed):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(n, n, generator=g, dtype=torch.float64) * (2.0 / n) ** 0.5 for _ in range(depth)]


res = {a: [] for a in ARMS}; corr = {a: [] for a in ARMS}; k4s = []
for s in range(nets):
    Ws = weights(s)
    mu_mc = np.load(HERE / "res" / f"mc_n{n}_L{depth}_s{s}_N{N:.0e}.npz")["mu"]
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE, factor=True, output_all=True,
                    up_to_layer=f"act{depth - 1}")
    k4s.append(float(((torch.stack([out[f"act{l}"][1].to_tensor() for l in range(depth)]).numpy()[-1] - mu_mc[-1]) ** 2).mean()))
    for arm, kw in ARMS.items():
        m_, info = run_chain([W.clone() for W in Ws], record=True, **dict(BASE, **kw))
        pred = np.stack([np.asarray(x, dtype=np.float64) for x in m_])
        res[arm].append(float(((pred[-1] - mu_mc[-1]) ** 2).mean()))
        cs = []
        for l in range(2, depth):
            x = info[l]["K4d"].numpy(); y = out[f"pre{l}"][4].get_dslice((4,)).numpy()
            cs.append(np.corrcoef(x, y)[0, 1])
        corr[arm].append(np.mean(cs))
print(f"n={n} L={depth}, {nets} nets: ARC k=4 {np.mean(k4s):.3e}")
b = np.mean(res["base"])
for arm in ARMS:
    print(f"  {arm:12s} final MSE {np.mean(res[arm]):.3e}  x{np.mean(res[arm]) / b:.3f}  (per net " +
          " ".join(f"{v / w:.3f}" for v, w in zip(res[arm], res["base"])) + f")  mean corr(K4d, k=4's) {np.mean(corr[arm]):.3f}")
