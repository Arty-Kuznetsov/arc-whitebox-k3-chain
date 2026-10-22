"""Record the torch reader's parts (pathdiag, star, 3+2) per (layer, generation) on baked MLP 0, prod-Dx variant."""
import sys, json
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery")); sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
import pb
from chain import run_chain
BASE = dict(pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), tailr=(("marg2", "sat", None, -2.0, "gauss"),))
BAND = dict(bandz=dict(a0=5, r=448, k=None, dtype=torch.float32, proj="qr", entry="pre", init="rr"))
rec = []
Y = dict(impl="chain", conv="arc", ages=2, types=("pathdiag", "star", "32"), resid=True, read=("K4d",), last32=False, dx="prod", rec=rec)
m = int(sys.argv[1]) if len(sys.argv) > 1 else 0
Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{m}.npy")]
mu, _ = run_chain([W.clone() for W in Ws], **dict(BASE, k4young=Y, **BAND))
gt = np.load(pb.BAKED / "gt.npy")
print("torch final MSE", float(((np.asarray(mu[-1]) - gt[m][-1]) ** 2).mean()))
out = {}
for (l, born, pd, st, t32) in rec:
    for name, v in (("pd", pd), ("st", st), ("32", t32)):
        if v is not None:
            out[f"l{l}_m{born}_{name}"] = v.numpy()
np.savez(HERE / f"torch_parts_m{m}.npz", **out)
print("saved", len(out), "arrays")
