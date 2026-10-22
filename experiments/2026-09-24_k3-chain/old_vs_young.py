"""At each layer of the faithful chain, split kappa3 slices into young (age <= y) and old sources and measure
how well the old part is predicted by the young part (per-neuron regression) — can old content be regenerated?"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
W = np.load(BAKED / "weights_0.npy")
Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
Y = [int(v) for v in sys.argv[1:]] or [2, 4]
orig = chain.read_slices
rec = []
def spy(sources, n, d21_ages=None):
    t, D = orig(sources, n, d21_ages)
    S = len(sources)
    out = {"S": S}
    for y in Y:
        if S > y:
            ty, Dy = orig(sources[-y:], n)
            out[y] = (ty, Dy, t - ty, D - Dy)
    rec.append(out)
    return t, D
chain.read_slices = spy
chain.run_chain(Ws, k4="memo22")
def stats(a, b):
    a, b = a.flatten(), b.flatten()
    c = float(torch.corrcoef(torch.stack([a, b]))[0, 1])
    beta = float((a @ b) / (a @ a))
    resid = float(((b - beta * a) ** 2).sum() / (b ** 2).sum())
    return c, beta, resid
print("layer S | y: corr(t_young,t_old) beta resid_frac | corr(D_young,D_old) beta resid_frac | |t_old|/|t|  |D_old|/|D|")
for l, r in enumerate(rec):
    for y in Y:
        if y in r:
            ty, Dy, to, Do = r[y]
            ct = stats(ty, to); cd = stats(Dy, Do)
            fr_t = float(to.norm() / (ty + to).norm()); fr_d = float(Do.norm() / (Dy + Do).norm())
            print(f"{l+1:3d} {r['S']:2d} | y={y}: t {ct[0]:.3f} {ct[1]:.3f} {ct[2]:.3f} | D {cd[0]:.3f} {cd[1]:.3f} {cd[2]:.3f} | {fr_t:.3f} {fr_d:.3f}")
