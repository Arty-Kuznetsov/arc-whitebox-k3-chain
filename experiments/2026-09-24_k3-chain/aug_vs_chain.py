"""Per-layer moments of ARC AUGMENT (runs/aug_moments_AUGMENT_<i>.npz) and of our k3f chain against the MC oracle
(runs/mc_oracle_<i>.npz): var rel. error, kappa3/kappa4 diag corr+level, and the readout injection (MSE vs MC post-mean
when one moment of the method replaces MC's in the exact readout). Usage: aug_vs_chain.py <mlp> [mode]"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain, relu_wick
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "AUGMENT"
extra = dict(a.split("=") for a in sys.argv[3:])
if extra.pop("preset", None) == "k3i":
    extra.update(paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="none", k4b=1, k4bparts=("sd",), pairx=("k22",))
A = np.load(HERE / "runs" / f"aug_moments_{mode}_{i}.npz", allow_pickle=True)
mc = np.load(HERE / "runs" / f"mc_oracle_{i}.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[i]
kw = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True)
kw.update(extra)
mu, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record=True, **kw)
T = lambda x: torch.tensor(np.asarray(x, dtype=np.float64))
def rd(m, v, t3, t4):
    w, _, _ = relu_wick(T(m), T(v)); return (w[(0, 1)] + w[(3, 1)] * T(t3) / 6 + w[(4, 1)] * T(t4) / 24).numpy()
print(f"MLP {i}: final vs GT  chain {((mu[-1].numpy()-gt[-1])**2).mean():.4e}  {mode} {((A['post_mean'][-1]-gt[-1])**2).mean():.4e}")
print(" l | var relerr ch/aug     | k3 corr ch/aug  lvl ch/aug   | k4 corr ch/aug  lvl ch/aug   | inj var ch/aug      | inj k3 ch/aug       | inj k4 ch/aug")
for l in range(len(info)):
    mv, vv, k3, k4, ref = (mc[k][l] for k in ("mean", "var", "k3", "k4", "post_mean"))
    e = lambda x: float(((x - ref) ** 2).mean())
    cv, av = info[l]["var"].numpy(), A["pre_var"][l]
    ct, at = info[l]["t"].numpy(), A["pre_k3"][l]
    c4, a4 = info[l]["K4d"].numpy(), A["pre_k4"][l]
    rel = lambda x: np.sqrt(((x - vv) ** 2).mean()) / vv.mean()
    cc = lambda x, y: np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else 0.0
    lv = lambda x, y: float(np.abs(x).mean() / np.abs(y).mean()) if np.abs(y).mean() > 0 else 0.0
    print(f"{l:2d} | {rel(cv):.2e} {rel(av):.2e} | {cc(ct,k3):.3f} {cc(at,k3):.3f} {lv(ct,k3):.3f} {lv(at,k3):.3f} | "
          f"{cc(c4,k4):.3f} {cc(a4,k4):.3f} {lv(c4,k4):.3f} {lv(a4,k4):.3f} | {e(rd(mv,cv,k3,k4)):.2e} {e(rd(mv,av,k3,k4)):.2e} | "
          f"{e(rd(mv,vv,ct,k4)):.2e} {e(rd(mv,vv,at,k4)):.2e} | {e(rd(mv,vv,k3,c4)):.2e} {e(rd(mv,vv,k3,a4)):.2e}")
