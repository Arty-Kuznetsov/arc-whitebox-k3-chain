"""Does the DAMAGING part of the moment error become visible once the features include the neuron's ENVIRONMENT?

NEGATIVE 70 was measured with 20 features that are all functions of the neuron's OWN moments. Its split showed the
damaging error lives in the complement of that span. But the chain already materialises C0, D and K22 every layer, so
ROW REDUCTIONS of them are O(n^2) -- free against the n^3 products -- and they describe the neuron's environment.
This re-runs the §70 split with those added.

  python pairfeat.py <mlp>     -> R^2 per channel with own-only vs own+pair features, and the projection split
                                  (remove Pe / remove (I-P)e) under the enlarged span.
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
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
_GT = np.load(pb.BAKED / "gt.npy")
Ws = np.load(pb.BAKED / f"weights_{i}.npy")
pred, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], record="full", **B9)
L, n = len(info), info[0]["mu_pre"].shape[0]
oa = np.load(ORA / f"mc_oracle_{i}.npz"); ob = np.load(ORA / f"mc_oracle_{i}_s777.npz")
print(f"MLP {i}: B9 recorded with pair slices, {L} layers x {n}")
CH = {"var": ("var", "var"), "k3": ("t", "k3"), "k4": ("K4d", "k4")}


def own(l):
    d = info[l]
    mu = d["mu_pre"].numpy().astype(np.float64)
    var = np.maximum(d["var"].numpy().astype(np.float64), 1e-12); sd = np.sqrt(var); a = mu / sd
    t = d["t"].numpy().astype(np.float64); k4 = d["K4d"].numpy().astype(np.float64)
    phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
    Ph = 0.5 * (1.0 + np.array([erf(z / np.sqrt(2)) for z in a]))
    w_next = Ws[l + 1] if l + 1 < len(Ws) else Ws[l]
    u = (w_next.astype(np.float64) ** 2).sum(1); wcol = (Ws[l].astype(np.float64) ** 2).sum(0)
    ac = np.clip(a, -8, 8)
    return np.stack([a, ac*ac, ac**3, ac**4, phi, Ph, a*phi, (ac*ac-1)*phi, sd, var,
                     np.clip(t/(sd**3+1e-30), -50, 50), np.clip(k4/(var*var+1e-30), -50, 50),
                     u, wcol, np.argsort(np.argsort(-a))/n, mu, t, k4, Ph*var, t*ac, k4*ac], 1)


def pair(l):
    """ROW REDUCTIONS of the matrices the chain already has: all O(n^2), i.e. free against n^3."""
    d = info[l]
    C = d["C"].numpy().astype(np.float64) if "C" in d else None
    D = d["D"].numpy().astype(np.float64)
    K22 = d["K22"].numpy().astype(np.float64) if d.get("K22") is not None else None
    var = np.maximum(d["var"].numpy().astype(np.float64), 1e-12)
    W = Ws[l].astype(np.float64)
    cols = []
    if C is not None:
        C0 = C - np.diag(np.diag(C))
        cols += [np.abs(C0).sum(1), (C0*C0).sum(1), (C0**3).sum(1), C0.sum(1),
                 (C0*C0).sum(1)/ (var + 1e-30), np.abs(C0).max(1),
                 (C0*C0*C0*C0).sum(1), (np.abs(C0)**1.5).sum(1)]
        wsq = (W*W)
        cols += [wsq.T @ np.diag(C).astype(np.float64) if False else (wsq.sum(0)*0.0)]  # placeholder removed below
        cols = cols[:-1]
    cols += [np.abs(D).sum(1), (D*D).sum(1), D.sum(1), np.abs(D).sum(0), (D*D).sum(0), D.sum(0)]
    if K22 is not None:
        cols += [np.abs(K22).sum(1), (K22*K22).sum(1), K22.sum(1)]
    return np.stack(cols, 1)


def r2(F, y):
    y = y - y.mean()
    if not np.isfinite(y).all() or y.std() == 0:
        return float("nan")
    keep = (F.std(0) > 0) & np.isfinite(F).all(0)
    if keep.sum() == 0:
        return float("nan")
    Fz = F[:, keep]; Fz = (Fz - Fz.mean(0)) / Fz.std(0)
    try:
        b, *_ = np.linalg.lstsq(Fz, y, rcond=1e-8)
    except np.linalg.LinAlgError:
        return float("nan")
    r = y - Fz @ b
    return 1.0 - float(r @ r) / float(y @ y)


print(f"\n{'layer':>5s} {'np':>3s} " + "  ".join(f"{c:>22s}" for c in CH))
print(f"{'':>5s} {'':>3s} " + "  ".join(f"{'own   own+pair   ceiling':>22s}" for _ in CH))
for l in range(1, L):
    Fo, Fp = own(l), pair(l)
    Fb = np.concatenate([Fo, Fp], 1)
    line = f"{l:5d} {Fp.shape[1]:3d} "
    for c, (ck, ok) in CH.items():
        x = info[l][ck].numpy().astype(np.float64)
        ea = x - oa[ok][l]; eb = x - ob[ok][l]
        cc = float(np.corrcoef(ea, eb)[0, 1])
        line += f"  {r2(Fo, ea):6.3f} {r2(Fb, ea):8.3f} {cc:9.3f}"
    print(line)
