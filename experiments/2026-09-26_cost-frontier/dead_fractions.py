"""Measured per-layer dead fractions of the k3l chain: fraction of neurons with alpha = mu_pre/sigma < thr.
Writes dead_fractions.json = {thr: [f_0 .. f_15]} (mean over the MLPs given), used by cost.py."""
import json, sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
from chain import run_chain
from sweep import K3L
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"

THR = [-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 3
acc = {f"{t:.2f}": np.zeros(16) for t in THR}
for i in range(N):
    W = np.load(BAKED / f"weights_{i}.npy")
    Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
    _, info = run_chain(Ws, record=True, **K3L)
    for l, d in enumerate(info):
        a = (d["mu_pre"] / d["var"].clamp_min(1e-30).sqrt()).numpy()
        for t in THR:
            acc[f"{t:.2f}"][l] += float((a < t).mean()) / N
    print(f"  mlp {i} done", flush=True)
out = {k: list(v) for k, v in acc.items()}
json.dump(out, open(HERE / "dead_fractions.json", "w"), indent=1)
for k, v in out.items():
    print(f"thr {k}: " + " ".join(f"{x:.3f}" for x in v))
