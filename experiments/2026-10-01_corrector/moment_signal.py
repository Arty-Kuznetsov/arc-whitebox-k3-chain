"""Task 1 of CLOUD-BRIEF #10, run on the Mac: is the chain's per-neuron moment error predictable from the chain's
own cheap state?  Uses the two independent 1e7 MC oracles of MLP 0 committed in
experiments/2026-09-24_k3-chain/oracles/ -- no baking, no training, ~2 minutes.

  python signal.py <mlp_index> [seedA seedB]

For every layer and every channel (var, kappa3 = t, kappa4 = K4d) it reports:
  * R2 of ridge on the chain's own per-neuron features,
  * the TWO-SEED NOISE FLOOR: the same regression run against oracle A vs oracle B, and corr(eA, eB) -- the part of
    the "error" that is just oracle noise cannot be predicted by anything, so corr(eA, eB) caps any achievable R2,
  * the same restricted to the kink core (|alpha| < 1), which is where NEGATIVE 68 says the error is born.
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
i = int(sys.argv[1]) if len(sys.argv) > 1 else 0
sA = sys.argv[2] if len(sys.argv) > 2 else ""
sB = sys.argv[3] if len(sys.argv) > 3 else "777"
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"


def oracle(idx, seed):
    f = ORA / (f"mc_oracle_{idx}.npz" if seed == "" else f"mc_oracle_{idx}_s{seed}.npz")
    d = np.load(f)
    return {k: d[k] for k in ("mean", "var", "k3", "k4")}, int(d["N"])


B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
Ws = np.load(pb.BAKED / f"weights_{i}.npy")
Wt = [torch.tensor(w, dtype=torch.float64) for w in Ws]
pred, info = run_chain(Wt, record=True, **B9)
L, n = len(info), info[0]["mu_pre"].shape[0]
print(f"MLP {i}: chain B9 recorded, {L} layers x {n} neurons")

oa, Na = oracle(i, sA)
ob, Nb = oracle(i, sB)
print(f"oracles: A N={Na:.0e} seed '{sA or 'default'}'   B N={Nb:.0e} seed '{sB}'")

# alignment check: the chain's pre-activation mean against the oracle's
mu_c = np.stack([info[l]["mu_pre"].numpy() for l in range(L)])
print(f"alignment: corr(chain mu_pre, oracle mean) per layer = "
      f"{[round(float(np.corrcoef(mu_c[l], oa['mean'][l])[0,1]), 4) for l in (0, 5, 10, 15)]}  (layers 0,5,10,15)")

CH = {"var": ("var", "var"), "kappa3": ("t", "k3"), "kappa4": ("K4d", "k4")}


FEATNAMES = ["a", "a2", "a3", "a4", "phi", "Phi", "a*phi", "(a2-1)*phi", "sd", "var", "skew", "exkurt",
             "u_next", "wcol", "rank", "mu", "t", "K4d", "d1*var", "t*a", "K4d*a"]


def feats(l):
    """Per-neuron features the chain already has at layer l. Layer-level aggregates are CONSTANT within a layer and
    carry no information for a within-layer regression (they made the design matrix singular in the first version);
    they are added back in the pooled fit below."""
    d = info[l]
    mu = d["mu_pre"].numpy().astype(np.float64)
    var = np.maximum(d["var"].numpy().astype(np.float64), 1e-12)
    sd = np.sqrt(var); a = mu / sd
    t = d["t"].numpy().astype(np.float64); k4 = d["K4d"].numpy().astype(np.float64)
    phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    Ph = 0.5 * (1.0 + np.array([erf(z / np.sqrt(2)) for z in a]))
    w_next = Ws[l + 1] if l + 1 < len(Ws) else Ws[l]
    u = (w_next.astype(np.float64) ** 2).sum(1)
    wcol = (Ws[l].astype(np.float64) ** 2).sum(0)
    ac = np.clip(a, -8, 8)
    F = np.stack([a, ac * ac, ac ** 3, ac ** 4, phi, Ph, a * phi, (ac * ac - 1) * phi, sd, var,
                  np.clip(t / (sd ** 3 + 1e-30), -50.0, 50.0),        # skew: clipped -- dead neurons sit at the 1e-12
                     np.clip(k4 / (var * var + 1e-30), -50.0, 50.0),   # variance clamp and overflow float64
                     u, wcol,
                  np.argsort(np.argsort(-a)) / n, mu, t, k4, Ph * var, t * ac, k4 * ac], 1)
    return F, a


def r2(F, y):
    """R^2 of an ordinary least-squares fit with an intercept, after dropping degenerate columns.
    In-sample, so it is an upper bound for this feature set; the cross-MLP number is task 2."""
    y = y - y.mean()
    if not np.isfinite(y).all() or y.std() == 0:
        return float("nan")
    keep = (F.std(0) > 0) & np.isfinite(F).all(0)
    if keep.sum() == 0:
        return float("nan")
    Fz = F[:, keep]
    Fz = (Fz - Fz.mean(0)) / Fz.std(0)
    try:
        beta, *_ = np.linalg.lstsq(Fz, y, rcond=1e-10)
    except np.linalg.LinAlgError:
        return float("nan")
    res = y - Fz @ beta
    den = float(y @ y)
    return 1.0 - float(res @ res) / (den if den > 0 else 1e-300)


print()
hdr = f"{'layer':>5s} " + " ".join(f"{c:>26s}" for c in CH)
print(hdr); print(f"{'':>5s} " + " ".join(f"{'R2A   R2B  corr(eA,eB) rel':>26s}" for _ in CH))
rows = {c: [] for c in CH}
for l in range(1, L):   # layer 0 is degenerate: mu_pre = 0 for every neuron and there are no sources yet
    F, a = feats(l)
    line = f"{l:5d} "
    for c, (ck, ok) in CH.items():
        x = info[l][ck].numpy().astype(np.float64)
        ea = x - oa[ok][l]; eb = x - ob[ok][l]
        cc = float(np.corrcoef(ea, eb)[0, 1])
        ra, rb = r2(F, ea - ea.mean()), r2(F, eb - eb.mean())
        rel = float(np.sqrt(np.mean(ea ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-300))
        rows[c].append((ra, rb, cc, rel))
        line += f" {ra:6.3f}{rb:6.3f}{cc:8.3f}{rel:7.3f}"
    print(line)
print()
for c in CH:
    A = np.array(rows[c])
    print(f"{c:8s} mean over layers 1-15: R2 {A[:,0].mean():.3f}/{A[:,1].mean():.3f}   "
          f"corr(eA,eB) {A[:,2].mean():.3f} (the ceiling on R2)   relative error of the channel {A[:,3].mean():.3f}")

print()
print("=== restricted to the kink core |alpha| < 1 (NEGATIVE 68: where the error is born) ===")
print(f"{'layer':>5s} {'n_core':>7s} " + " ".join(f"{c:>20s}" for c in CH))
for l in range(1, L):
    F, a = feats(l)
    m = np.abs(a) < 1.0
    if m.sum() < 60:
        continue
    line = f"{l:5d} {int(m.sum()):7d} "
    for c, (ck, ok) in CH.items():
        x = info[l][ck].numpy().astype(np.float64)
        ea = (x - oa[ok][l])[m]; eb = (x - ob[ok][l])[m]
        cc = float(np.corrcoef(ea, eb)[0, 1])
        ra = r2(F[m], ea - ea.mean())
        line += f" {ra:8.3f}{cc:8.3f}    "
    print(line)
print("(columns: R2 vs oracle A, then corr(eA,eB) = the noise ceiling)")
