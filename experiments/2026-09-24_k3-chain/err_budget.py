"""Per-layer error budget of a chain config against the MC oracle (MLP 0, 1e7 samples): readout MSE vs MC post-mean
when ONE of the chain's pre-activation moments (mean, var, k3 diag, k4 diag) replaces MC's in the exact readout, plus
level/slope diagnostics. Usage: err_budget.py <mlp> [k=v ...]"""
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
kw = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True)
kw.update({k: val(v) for k, v in (a.split("=") for a in sys.argv[2:])})
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[i]
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record=True, **kw)
print(f"MLP {i}: final MSE vs GT {float(((mu[-1].numpy() - gt[-1]) ** 2).mean()):.4e}")
def rd(m, v, t3, t4):
    w, _, _ = relu_wick(m, v)
    return w[(0, 1)] + w[(3, 1)] * t3 / 6 + w[(4, 1)] * t4 / 24
print(" l | own     | exactMC | ch.mean | ch.var  | ch.k3   | ch.k4   | var ratio slope rel | k3 lvl | k4 lvl")
for l, r in enumerate(info):
    mv, vv, k3, k4 = (torch.tensor(mc[k][l]) for k in ("mean", "var", "k3", "k4"))
    ref = torch.tensor(mc["post_mean"][l])
    e = lambda x: float(((x - ref) ** 2).mean())
    cm, cv = r["mu_pre"], r["var"]
    ct = r["t"] if l > 0 else torch.zeros_like(k3)
    c4 = r["K4d"] if l > 1 else torch.zeros_like(k4)
    rel = float(((cv - vv) ** 2).mean().sqrt() / vv.mean())
    print(f"{l:2d} | {e(mu[l]):.2e} | {e(rd(mv, vv, k3, k4)):.2e} | {e(rd(cm, vv, k3, k4)):.2e} | {e(rd(mv, cv, k3, k4)):.2e} | "
          f"{e(rd(mv, vv, ct, k4)):.2e} | {e(rd(mv, vv, k3, c4)):.2e} | {float(cv.mean()/vv.mean()):.5f} {float(cv @ vv / (vv @ vv)):.5f} {rel:.2e} | "
          f"{float(ct.abs().mean() / k3.abs().mean()) if l else 0:.3f} | {float(c4.mean() / k4.mean()) if l > 1 else 0:.3f}")
