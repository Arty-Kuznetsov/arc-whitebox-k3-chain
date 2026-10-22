"""Which networks are hard for us? The leaders' advantage grows on the networks where our error is largest
(INTEL-peers 5.2). On our 100 local networks (baked 1e9 ground truth, mini split): our production raw (k3z run) against
per-network features of the chain's own state (research base B9 + marg2, float64): per-layer fractions of dead
(alpha < -2.5), kink (|alpha| < 1) and saturated (alpha > 2.5) neurons, the kappa4 level v, ||t||/||sigma^3||,
||D||/||C0||, at depth. Spearman correlations with our raw.  args: n_nets"""
import json
import sys
from pathlib import Path
import numpy as np
import torch
torch.set_grad_enabled(False); torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                                       # noqa: E402
from chain import run_chain                                     # noqa: E402
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
runs = {p["mlp_index"]: p for p in json.load(open(HERE.parent / "2026-10-02_k3z/runs/k3z_n100.json"))["results"]["per_mlp"]}
mini = Path(HERE.parent / "2026-09-16_kprop-arc" / "baked")
out = []
for m in range(N):
    f = mini / f"weights_{m}.npy"
    if not f.exists():
        break
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(f)]
    _, info = run_chain([W.clone() for W in Ws], record=True, **BASE)
    feats = {}
    for l in (8, 12, 15):
        d = info[l]
        var = d["var"].numpy(); a = d["mu_pre"].numpy() / np.sqrt(np.maximum(var, 1e-30))
        feats[f"dead{l}"] = float((a < -2.5).mean()); feats[f"kink{l}"] = float((np.abs(a) < 1).mean())
        feats[f"sat{l}"] = float((a > 2.5).mean())
        k4 = d["K4d"].numpy(); feats[f"v{l}"] = float(k4.mean() / (3 * (var * var).mean()))
        t = d["t"].numpy(); feats[f"skew{l}"] = float(np.sqrt((t * t).mean() / (var ** 3).mean()))
    out.append(dict(m=m, raw=runs[m]["final_layer_mse"], **feats))
    print(m, f"{runs[m]['final_layer_mse']:.3e}", flush=True)
json.dump(out, open(HERE / "hardnets.json", "w"), indent=1)
from scipy.stats import spearmanr  # noqa: E402  (analysis only, research env)
raw = np.array([o["raw"] for o in out])
print(f"\nSpearman correlation with our production raw over {len(out)} networks:")
for k in [k for k in out[0] if k not in ("m", "raw")]:
    x = np.array([o[k] for o in out])
    print(f"  {k:8s} {spearmanr(x, raw)[0]:+.3f}   (mean {x.mean():.4g})")
