"""compare_oracle.py for any chain config: per layer, the chain's kappa3/kappa4 diagonals vs MC (corr, slope, rel), and
the readout MSE vs MC post-mean when the chain's own kappa4 (or kappa3) replaces MC's in an otherwise exact readout.
Usage: compare_oracle2.py <mlp> [k=v ...]  (e.g. c0d=22; values True/False/int/str)"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain, relu_wick
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1])
def val(v):
    if v in ("True", "False"): return v == "True"
    try: return int(v)
    except ValueError: return v
kw = dict(k4="memo22", corr="split24rr"); kw.update({k: val(v) for k, v in (a.split("=") for a in sys.argv[2:])})
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[i]
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record=True, **kw)
st = lambda a, b: (np.corrcoef(a, b)[0, 1], float(a @ b / (b @ b)))
print(f"MLP {i} {kw}: final MSE vs GT {float(((mu[-1].numpy() - gt[-1]) ** 2).mean()):.4e}")
print(" l | k3 corr slope | k4 corr slope | readout MSE vs MC post-mean: exact MC moments | chain k4 | chain k3 | chain own")
for l, r in enumerate(info):
    mv, vv, k3, k4 = (torch.tensor(mc[k][l]) for k in ("mean", "var", "k3", "k4"))
    ref = mc["post_mean"][l]
    w, _, _ = relu_wick(mv, vv)
    rd = lambda t3, t4: float((((w[(0, 1)] + w[(3, 1)] * t3 / 6 + w[(4, 1)] * t4 / 24).numpy() - ref) ** 2).mean())
    c3 = st(r["t"].numpy(), k3.numpy()) if l > 0 else (0, 0)
    c4 = st(r["K4d"].numpy(), k4.numpy()) if l > 1 else (0, 0)
    t_ch = r["t"] if l > 0 else torch.zeros_like(k3)
    k4_ch = r["K4d"] if l > 1 else torch.zeros_like(k4)
    own = float(((mu[l].numpy() - ref) ** 2).mean())
    print(f"{l:2d} | {c3[0]:.3f} {c3[1]:.3f} | {c4[0]:.3f} {c4[1]:.3f} | {rd(k3, k4):.3e} | {rd(k3, k4_ch):.3e} | {rd(t_ch, k4):.3e} | {own:.3e}")
