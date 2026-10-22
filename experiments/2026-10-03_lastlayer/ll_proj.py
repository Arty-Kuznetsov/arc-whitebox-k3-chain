"""In sample, MLPs 0 and 1 at 1024x16: the last layer's t and kappa4-diagonal errors (vs the 1e7 MC oracles, two seeds
averaged) projected on the neuron's own smooth features (moment_signal.py's 21) -- remove the PROJECTION at the last layer
only and see the final MSE (no downstream, so no consistency to break). Upper bound for a last-layer corrector.
    PYTHONPATH=$ARC/src uv run --project $ARC python ll_proj.py <mlp>"""
import sys
from math import erf
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
Wnp = np.load(pb.BAKED / f"weights_{m}.npy")
Ws = [torch.tensor(w, dtype=torch.float64) for w in Wnp]
L = len(Ws) - 1
mu, info = run_chain([W.clone() for W in Ws], record=True, **BASE)
base = float(((np.asarray(mu[-1]) - gt[-1]) ** 2).mean())
d = info[L]
mu_p = d["mu_pre"].numpy(); var = np.maximum(d["var"].numpy(), 1e-12); sd = np.sqrt(var); a = mu_p / sd
t = d["t"].numpy(); k4 = d["K4d"].numpy()
phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi); Ph = 0.5 * (1 + np.array([erf(z / np.sqrt(2)) for z in a]))
ac = np.clip(a, -8, 8)
wcol = (Wnp[L].astype(np.float64) ** 2).sum(0)
F = np.stack([np.ones_like(a), a, ac ** 2, ac ** 3, ac ** 4, phi, Ph, a * phi, (ac ** 2 - 1) * phi, sd, var,
              np.clip(t / (sd ** 3 + 1e-30), -50, 50), np.clip(k4 / (var * var + 1e-30), -50, 50), wcol,
              mu_p, t, k4, Ph * var, t * ac, k4 * ac], 1)
Fz = F.copy(); Fz[:, 1:] = (F[:, 1:] - F[:, 1:].mean(0)) / (F[:, 1:].std(0) + 1e-30)
print(f"MLP {m}: base {base:.4e}")
for ch, key, cur in (("t", "k3", t), ("K4d", "k4", k4)):
    tgtP = 0.5 * (oA[key][L] + oB[key][L])
    e = tgtP - cur
    beta, *_ = np.linalg.lstsq(Fz, e, rcond=None)
    fit = Fz @ beta
    r2 = 1 - ((e - fit) @ (e - fit)) / ((e - e.mean()) @ (e - e.mean()))
    for tag, new in (("exact (P)", tgtP), ("projection", cur + fit), ("half projection", cur + 0.5 * fit)):
        orc = {key: {L: new}, f"{key}_layers": {L}}
        mu2, _ = run_chain([W.clone() for W in Ws], oracle=orc, **BASE)
        print(f"  {ch}@L {tag:16s} x{float(((np.asarray(mu2[-1]) - gt[-1]) ** 2).mean()) / base:.4f}   (R2 of the projection {r2:.3f})", flush=True)
