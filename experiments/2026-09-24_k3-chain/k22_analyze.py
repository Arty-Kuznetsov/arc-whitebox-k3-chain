"""Structure of the true (2,2) slice k4[i,i,j,j] (MC) vs the chain's K22 and vs simple structural models."""
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
o = np.load(HERE / "runs" / "mc_oracle22_0.npz")
mc = np.load(HERE / "runs" / "mc_oracle_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
_, info = run_chain(Ws, k4="memo22", corr="split24rr", record=True, k4k3=1.0)
# the chain records K4d only; recompute its pre-activation K22 by re-running with a hook
import chain as ch
store = {}
orig = ch.nonlin_slices
def hook(mu, C, t, D, K4d, K22, **kw):
    store.setdefault("K22", []).append(None if K22 is None else K22.clone()); store.setdefault("C", []).append(C.clone())
    return orig(mu, C, t, D, K4d, K22, **kw)
ch.nonlin_slices = hook
run_chain(Ws, k4="memo22", corr="split24rr")
ch.nonlin_slices = orig
N = float(o["N"])
off = lambda M: M[~np.eye(M.shape[0], dtype=bool)]
print(f"N={N:.0e}; per-pair MC noise ~ sqrt(E[dz^4 dz^4]/N) ~ 3*var^2/sqrt(N)")
for l in [int(v) for v in o["layers"]]:
    k22 = o[f"k22_{l}"].astype(np.float64); Cmc = o[f"C{l}"].astype(np.float64)
    d = np.diag(Cmc)
    y = off(k22)
    c = off(Cmc)
    rho = off(Cmc / np.sqrt(np.outer(d, d)))
    chK22 = store["K22"][l]
    x = off(chK22.numpy()) if chK22 is not None else np.zeros_like(y)
    noise = 3 * np.mean(d) ** 2 / np.sqrt(N)
    def fit(X):  # least squares y ~ X (columns), returns coefs, R2
        X = np.column_stack(X); b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b
        return b, 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
    ones = np.ones_like(y)
    b1, r1 = fit([ones])
    b2, r2 = fit([ones, c])
    b3, r3 = fit([ones, c, c * c])
    b4, r4 = fit([x])
    b5, r5 = fit([ones, x])
    print(f"layer {l}: MC k22 mean {y.mean():.2e} std {y.std():.2e} (noise/pair {noise:.1e}) | chain K22 mean {x.mean():.2e} "
          f"| R2: const {r1:.3f}, const+C {r2:.3f} (lam={b2[1]:.3f}), +C^2 {r3:.3f}, chain {r4:.3f} (slope {b4[0]:.2f}), const+chain {r5:.3f}")
    print(f"          MC k4 diag mean {mc['k4'][l].mean():.2e}; 3*mean(k22)*? ratio diag/(k22 mean) = {mc['k4'][l].mean()/y.mean():.2f}; mean rho {rho.mean():.3f}")
