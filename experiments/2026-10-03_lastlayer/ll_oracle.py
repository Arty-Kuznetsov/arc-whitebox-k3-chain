"""At 1024x16: what are the LAST layer's per-neuron moments worth? Our float64 chain (research base B9 + marg2) with the
Monte-Carlo pre-activation var / kappa3 / kappa4 of layer 15 substituted at that layer only (no downstream, so no
consistency to break). Oracles: experiments/2026-09-24_k3-chain/oracles/mc_oracle_<m>.npz and _s777 (1e7 each);
P = the two seeds averaged, A / B = one each; Richardson 2P - (A+B)/2 removes the oracle's own noise to first order.
    PYTHONPATH=$ARC/src uv run --project $ARC python ll_oracle.py <mlp>"""
import sys
from pathlib import Path
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402
m = int(sys.argv[1])
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
oA, oB = (np.load(ORA / f) for f in (f"mc_oracle_{m}.npz", f"mc_oracle_{m}_s777.npz"))
gt = np.load(pb.BAKED / "gt.npy")[m]
Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{m}.npy")]
L = len(Ws) - 1


def run(orc):
    mu, _ = run_chain([W.clone() for W in Ws], oracle=orc, **BASE)
    return float(((np.asarray(mu[-1]) - gt[-1]) ** 2).mean())


base = run(None)
print(f"MLP {m}: base final MSE {base:.4e}")
src = {"P": {k: 0.5 * (oA[k] + oB[k]) for k in ("var", "k3", "k4")}, "A": {k: oA[k] for k in ("var", "k3", "k4")},
       "B": {k: oB[k] for k in ("var", "k3", "k4")}}
for arm, keys in (("var@L", ("var",)), ("t@L", ("k3",)), ("K4d@L", ("k4",)), ("t+K4d@L", ("k3", "k4")), ("all@L", ("var", "k3", "k4"))):
    r = {}
    for tag, S in src.items():
        orc = {}
        for k in keys:
            orc[k] = {L: S[k][L]}; orc[f"{k}_layers"] = {L}
        r[tag] = run(orc) / base
    rich = 2 * r["P"] - 0.5 * (r["A"] + r["B"])
    print(f"  {arm:8s} P x{r['P']:.4f}  A x{r['A']:.4f}  B x{r['B']:.4f}  Richardson x{rich:.4f}", flush=True)
