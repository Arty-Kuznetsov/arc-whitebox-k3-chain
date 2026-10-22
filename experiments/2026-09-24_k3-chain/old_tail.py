"""Can the kappa3 diagonal carried by OLD sources be replaced by a cheap closure? (FRONTIER B1, with the amplitude-mode
idea: positive homogeneity of ReLU nets -> a common amplitude z = a u gives kappa3_i = 1.5 v mu_i var_i and
kappa3[iij] = (v/2)(2 mu_i C_ij + mu_j var_i), v = the scale-mixture level of the kappa4 closure.)
Runs the chain (k4pat + k22pat + c0d, corr='freefull') on a baked MLP, splits t = kappa3[iii] and D = kappa3[iij] by
source age at every layer, and fits the old part (age > K) by per-layer least squares on cheap features.
Usage: old_tail.py <mlp> <K1,K2,...>"""
import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
i = int(sys.argv[1]); Ks = [int(v) for v in sys.argv[2].split(",")]
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / f"weights_{i}.npy")
rec = []
orig_read, orig_nl = ch.read_slices, ch.nonlin_slices
cur = {}
def read_hook(sources, n, d21_ages=None):
    t, D = orig_read(sources, n, d21_ages)
    cur["t"], cur["D"] = t, D
    cur["parts"] = {K: orig_read(sources[:-K], n) if len(sources) > K else (torch.zeros(n), torch.zeros(n, n)) for K in Ks}
    return t, D
def nl_hook(mu, C, t, D, K4d, K22, **kw):
    if "parts" in cur:
        v = float(K4d.mean() / (3.0 * (torch.diagonal(C) ** 2).mean())) if K4d is not None and float(K4d.abs().sum()) > 0 else 0.0
        rec.append(dict(mu=mu.clone(), C=C.clone(), t=cur["t"], D=cur["D"], parts=cur["parts"], v=v))
        cur.clear()
    return orig_nl(mu, C, t, D, K4d, K22, **kw)
ch.read_slices, ch.nonlin_slices = read_hook, nl_hook
ch.run_chain([torch.tensor(w, dtype=torch.float64) for w in W], k4="memo22", corr="freefull", k4pat=True, k22pat=True,
             c0d=True, k4lvl=True)
ch.read_slices, ch.nonlin_slices = orig_read, orig_nl

def fit(y, X):
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return 1 - (r @ r) / (y @ y), coef
print("per layer: |t_old|/|t| ; R^2 of t_old on features (no intercept): amp=mu*var | amp+s3 (var^1.5) | +mu^3 | young t |"
      " all ; amp slope / (1.5 v) ;  D_old: |D_old|/|D|, R^2 on (v/2)(2 mu_i C_ij + mu_j var_i)")
for K in Ks:
    print(f"--- K = {K} (old = sources older than {K} layers)")
    for l, r in enumerate(rec):
        if l <= K + 1 or l % 2 == 0 and l < 12:
            continue
        mu, C, t = r["mu"].numpy(), r["C"].numpy(), r["t"].numpy()
        var = np.diag(C); t_old, D_old = (x.numpy() for x in r["parts"][K])
        t_young = t - t_old
        amp = mu * var
        feats = {"amp": [amp], "amp+s3": [amp, var ** 1.5], "+mu3": [amp, var ** 1.5, mu ** 3], "young": [t_young],
                 "all": [amp, var ** 1.5, mu ** 3, t_young, np.ones_like(mu)]}
        out = [f"{fit(t_old, np.stack(f, 1))[0]:.3f}" for f in feats.values()]
        slope = fit(t_old, amp[:, None])[1][0] / (1.5 * r["v"]) if r["v"] else float("nan")
        n = len(mu); off = ~np.eye(n, dtype=bool)
        Damp = 2 * mu[:, None] * (C - np.diag(var)) + var[:, None] * mu[None, :]
        rD, cD = fit(D_old[off], Damp[off][:, None])
        print(f"l{l:2d}: |t_old|/|t| {np.linalg.norm(t_old)/np.linalg.norm(t):.2f} | R2 " + " ".join(out)
              + f" | slope/(1.5v) {slope:.2f} | D: {np.linalg.norm(D_old)/np.linalg.norm(r['D'].numpy()):.2f} R2 {rD:.3f} "
              f"coef/(v/2) {cD[0] / (0.5 * r['v']) if r['v'] else float('nan'):.2f}")
