"""kappa4 pair slices at layers 6/10/14 (MLP 0): MC oracle (runs/mc_oracle31_0.npz, 4e6 samples) vs ARC AUGMENT's
Sym(G x M) (runs/aug_moments_AUGMENT_0_G.npz) vs our scale-mixture closure (chain.py, k3i config).
For K31 (kappa4[iiij]) and K22 (kappa4[iijj]): corr over off-diagonal pairs, slope (MC on model), level ratio."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle31_0.npz")
A = np.load(HERE / "runs" / "aug_moments_AUGMENT_0_G.npz", allow_pickle=True)
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
kw = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2", marg=True, k31pat=True,
          paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3)
_, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], record="full", **kw)
n = W.shape[1]; off = ~np.eye(n, dtype=bool)
def st(model, ref):
    x, y = model[off].astype(np.float64), ref[off].astype(np.float64)
    return np.corrcoef(x, y)[0, 1], float(x @ y / (x @ x)), float(np.abs(x).mean() / np.abs(y).mean())
print("layer | slice | model    | corr   slope(MC on model) level(model/MC)")
for l in (6, 10, 14):
    for sl, key_mc, key_a, key_c in (("K31", f"k31_{l}", "pre_k31", "K31"), ("K22", f"k22_{l}", "pre_k22", "K22")):
        ref = mc[key_mc]
        for name, m in (("AUGMENT", A[key_a][l]), ("closure", info[l][key_c].numpy() if info[l][key_c] is not None else None)):
            if m is None: continue
            c, s, lv = st(np.asarray(m), ref)
            print(f"{l:5d} | {sl}  | {name:8s} | {c:.3f}  {s:.3f}  {lv:.3f}")
    # the closure's C-proportional K31 vs the MC covariance itself (how much of MC K31 is ~C)
    C = mc[f"C{l}"]; c, s, lv = st(C, mc[f"k31_{l}"]); print(f"{l:5d} | K31  | MC C     | {c:.3f}")
