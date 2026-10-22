"""Task 2 of CLOUD-BRIEF #10: the OUT-OF-SAMPLE payoff of a learned moment corrector, measured through the chain.

Fit a per-layer linear model of the chain's moment error on MLP A, apply it ONLINE on MLP B (chain.py `mcorr`, a
function of B's own live state -- never a factor measured on B's trajectory, NEGATIVE 57), and measure B's
final-layer MSE against the baked 1e9 GT.  Then swap A and B.

  python xmlp.py <A> <B> [channels] [shrink...]      e.g. python xmlp.py 0 1 k3,k4 0.25,0.5,0.75,1.0
"""
import sys
from pathlib import Path
import numpy as np, torch
from math import erf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                            # noqa: E402
from chain import run_chain                          # noqa: E402

torch.set_grad_enabled(False); torch.set_num_threads(3)
A = int(sys.argv[1]); B = int(sys.argv[2])
CHANS = (sys.argv[3] if len(sys.argv) > 3 else "var,k3,k4").split(",")
SHRINK = [float(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "0.25,0.5,0.75,1.0").split(",")]
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
KEY = {"var": ("var", "var"), "k3": ("t", "k3"), "k4": ("K4d", "k4")}


def weights(i):
    return np.load(pb.BAKED / f"weights_{i}.npy")


_GT = None


def gt(i):
    """baked 1e9 ground truth, shape (10, 16, 1024) -- the last layer is what the score uses."""
    global _GT
    if _GT is None:
        _GT = np.load(pb.BAKED / "gt.npy")
    return _GT[i]


def oracle_pair(i):
    out = []
    for f in (f"mc_oracle_{i}.npz", f"mc_oracle_{i}_s777.npz"):
        if not (ORA / f).exists():
            raise SystemExit(f"missing oracle {f} -- bake it: python mc_oracle.py {i} 1e7 [777]")
        d = np.load(ORA / f)
        out.append({k: d[k] for k in ("mean", "var", "k3", "k4")})
    return out


def F_of(state_np, Ws, l, L):
    mu, var, t, k4 = state_np
    var = np.maximum(var, 1e-12); sd = np.sqrt(var); a = mu / sd
    phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    Ph = 0.5 * (1.0 + np.array([erf(z / np.sqrt(2)) for z in a]))
    n = len(mu)
    w_next = Ws[l + 1] if l + 1 < len(Ws) else Ws[l]
    u = (w_next.astype(np.float64) ** 2).sum(1)
    wcol = (Ws[l].astype(np.float64) ** 2).sum(0)
    ac = np.clip(a, -8, 8)
    return np.stack([np.ones(n), a, ac * ac, ac ** 3, ac ** 4, phi, Ph, a * phi, (ac * ac - 1) * phi, sd, var,
                     np.clip(t / (sd ** 3 + 1e-30), -50.0, 50.0),        # skew: clipped -- dead neurons sit at the 1e-12
                     np.clip(k4 / (var * var + 1e-30), -50.0, 50.0),   # variance clamp and overflow float64
                     u, wcol,
                     np.argsort(np.argsort(-a)) / n, mu, t, k4, Ph * var, t * ac, k4 * ac], 1)


def fit_on(i):
    """per-layer OLS coefficients of the moment error on the chain's own features, averaged over the two oracles."""
    Ws = weights(i)
    _, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], record=True, **B9)
    L = len(info)
    oracs = oracle_pair(i)
    coef = {c: {} for c in CHANS}
    for l in range(1, L):
        st = tuple(info[l][k].numpy().astype(np.float64) for k in ("mu_pre", "var", "t", "K4d"))
        F = F_of(st, Ws, l, L)
        keep = (F.std(0) > 0) & np.isfinite(F).all(0)
        keep[0] = False                                  # the intercept is handled by centring, not by a column
        mu_f = F.mean(0); sd_f = F.std(0); sd_f[sd_f == 0] = 1.0
        # STANDARDISE on the SOURCE MLP and carry (mu_f, sd_f) to the target: lstsq's rank truncation is
        # scale-dependent, so an unstandardised fit produces coefficients that extrapolate wildly off-distribution.
        Fz = ((F - mu_f) / sd_f)[:, keep]
        for c in CHANS:
            ck, ok = KEY[c]
            x = info[l][ck].numpy().astype(np.float64)
            y = 0.5 * ((x - oracs[0][ok][l]) + (x - oracs[1][ok][l]))   # average the two oracles: half the noise
            y0 = y.mean()
            beta, *_ = np.linalg.lstsq(Fz, y - y0, rcond=1e-8)
            full = np.zeros(F.shape[1]); full[keep] = beta
            coef[c][l] = (full, y0, mu_f, sd_f)
    return coef, L


def make_mcorr(coef, Ws, L, s):
    def f(l, st):
        if l == 0 or not any(l in coef[c] for c in CHANS):
            return None
        state_np = (st["mu_pre"].numpy().astype(np.float64),
                    torch.clamp(st["var"], min=1e-12).numpy().astype(np.float64),
                    st["t"].numpy().astype(np.float64),
                    (st["K4d"].numpy().astype(np.float64) if st["K4d"] is not None
                     else np.zeros(st["n"])))
        F = F_of(state_np, Ws, l, L)
        F = np.where(np.isfinite(F), F, 0.0)
        out = {}
        for c in CHANS:
            if l in coef[c]:
                beta, y0, mu_f, sd_f = coef[c][l]
                out[c] = torch.tensor(s * (((F - mu_f) / sd_f) @ beta + y0), dtype=torch.float64)
        return out
    return f


def raw_of(i, mcorr=None):
    Ws = weights(i)
    pred, _ = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], record=True, mcorr=mcorr, **B9)
    g = torch.tensor(gt(i), dtype=torch.float64)
    return float(((pred[-1] - g[-1]) ** 2).mean())


print(f"base B9, channels {CHANS}")
for (src, dst) in ((A, B), (B, A)):
    coef, L = fit_on(src)
    base = raw_of(dst)
    print(f"\n--- fit on MLP {src}  ->  apply on MLP {dst}   (base raw {base:.6e})")
    for s in SHRINK:
        r = raw_of(dst, make_mcorr(coef, weights(dst), L, s))
        print(f"    shrink {s:4.2f}:  raw {r:.6e}   x{r/base:.4f}")
    # in-sample reference, to show the generalisation gap
    base_s = raw_of(src)
    for s in (1.0,):
        r = raw_of(src, make_mcorr(coef, weights(src), L, s))
        print(f"    [in sample, MLP {src}] shrink {s:4.2f}: raw {r:.6e} x{r/base_s:.4f}  (base {base_s:.6e})")
