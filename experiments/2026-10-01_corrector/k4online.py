"""ONLINE test of the kappa4 pattern corrector -- the step that killed the corrector in NEGATIVE 70.

Offline (k4loo.py), a pooled ridge on 38 own+pair features fitted on 9 MLPs and applied to the 10th from the RECORDED
trajectory gives held-out raw x0.9739 (s = 0.5, 10/10 better). Here the same model is applied ONLINE through chain.py's
`mcorr`: features from the LIVE state at layers 8-15, so a correction at layer l moves layer l+1's state before
its own correction is computed. If the feedback is mild, online ~ offline.
"""
import sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
import k4split as K                                  # noqa: E402
import k4loo as LOO                                  # noqa: E402  (NB: importing runs its collect+LOO -- see guard)

LAM, S = 1e-3, float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
HELD = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "0,2,5,9").split(",")]
HI = int(sys.argv[3]) if len(sys.argv) > 3 else 16   # corrected layers: 8 .. HI-1 (production cannot do 15)


def online_mcorr(model, Ws, lo=8):
    b, b0, mu, sd = model
    def f(l, st):
        if l < lo or l >= HI or st["K4d"] is None:
            return None
        n = st["n"]
        d = dict(mu_pre=st["mu_pre"], var=st["var"], t=st["t"], K4d=st["K4d"], C=st["C"], D=st["D"], K22=st["K22"])
        F = np.concatenate([K.own(d, Ws, l, n), K.pair(d), np.full((n, 1), l / 16.0)], 1)
        F = np.where(np.isfinite(F), F, 0.0)
        k4c = st["K4d"].numpy().astype(np.float64)
        pe = (((F - mu) / sd) @ b + b0) * (np.abs(k4c).mean() + 1e-30)
        return {"k4": torch.tensor(S * pe, dtype=torch.float64)}
    return f


ratios = []
for h in HELD:
    train = {(i, l): LOO.DATA[i][1][l] for i in LOO.avail if i != h for l in LOO.DATA[i][1]}
    model = LOO.fit(train, LOO.ALL, LAM)
    base_h = LOO.DATA[h][0]
    Ws = np.load(K.pb.BAKED / f"weights_{h}.npy")
    pred, _ = K.run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], mcorr=online_mcorr(model, Ws), **K.B9)
    r = float(((pred[-1] - torch.tensor(K._GT[h], dtype=torch.float64)[-1]) ** 2).mean())
    ratios.append(r / base_h)
    print(f"  held-out MLP {h}: ONLINE raw x{r/base_h:.4f}   (s={S})", flush=True)
ratios = np.array(ratios)
print(f"ONLINE s={S} layers 8..{HI-1}: mean x{ratios.mean():.4f}  median x{np.median(ratios):.4f}  better {int((ratios<1).sum())}/{len(ratios)}")
