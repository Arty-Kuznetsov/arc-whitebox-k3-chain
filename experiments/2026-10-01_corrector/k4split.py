"""Rule 3 applied to the one free axis left: is a better kappa4 PATTERN worth anything?

A new ansatz for K4d is, by construction, a smooth function of the chain's state. So before deriving one, split the
exact kappa4 error  e = K4d_chain - k4_oracle  into its projection Pe onto the chain's own features and the rest,
and substitute each part ALONE, for kappa4 ONLY (var and kappa3 stay the chain's), offline through `oracle`:
  * remove all of e     -> the kappa4 channel's ceiling (today's gate: x0.753 / x0.809)
  * remove Pe           -> the best ANY smooth pattern on these features could ever do
  * remove (I-P)e       -> the part no pattern can reach
Two feature sets: OWN (21 functions of the neuron's own moments) and OWN+PAIR (+17 O(n^2) row reductions of C0, D,
K22 -- free, since the chain materialises them every layer). If 'remove Pe' buys nothing with both, the pattern
axis is closed in one run, with no ansatz derived.
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
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
_GT = np.load(pb.BAKED / "gt.npy")


def run(i, oracle=None, record=False):
    Ws = np.load(pb.BAKED / f"weights_{i}.npy")
    pred, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], record=record, oracle=oracle, **B9)
    return float(((pred[-1] - torch.tensor(_GT[i], dtype=torch.float64)[-1]) ** 2).mean()), info


def own(d, Ws, l, n):
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


def pair(d):
    C = d["C"].numpy().astype(np.float64); C0 = C - np.diag(np.diag(C))
    D = d["D"].numpy().astype(np.float64)
    K22 = d["K22"].numpy().astype(np.float64) if d.get("K22") is not None else np.zeros_like(C0)
    var = np.maximum(d["var"].numpy().astype(np.float64), 1e-12)
    return np.stack([np.abs(C0).sum(1), (C0*C0).sum(1), (C0**3).sum(1), C0.sum(1), (C0*C0).sum(1)/var,
                     np.abs(C0).max(1), (C0**4).sum(1), (np.abs(C0)**1.5).sum(1),
                     np.abs(D).sum(1), (D*D).sum(1), D.sum(1), np.abs(D).sum(0), (D*D).sum(0), D.sum(0),
                     np.abs(K22).sum(1), (K22*K22).sum(1), K22.sum(1)], 1)


def proj(F, e):
    keep = (F.std(0) > 0) & np.isfinite(F).all(0)
    Fz = F[:, keep]; Fz = (Fz - Fz.mean(0)) / Fz.std(0)
    Q, _ = np.linalg.qr(np.column_stack([np.ones(len(e)), Fz]))
    return Q @ (Q.T @ e)


if __name__ == "__main__":
  for i in (0, 1):
      Ws = np.load(pb.BAKED / f"weights_{i}.npy")
      base, info = run(i, record="full")
      L, n = len(info), info[0]["mu_pre"].shape[0]
      o4 = np.load(ORA / f"mc_oracle_{i}.npz")["k4"]
      arrs = {k: {} for k in ("all", "Pown", "Rown", "Pboth", "Rboth")}
      r2 = {"own": [], "both": []}
      for l in range(1, L):
          d = info[l]
          k4c = d["K4d"].numpy().astype(np.float64)
          e = k4c - o4[l]
          Fo = own(d, Ws, l, n); Fb = np.concatenate([Fo, pair(d)], 1)
          po, pbth = proj(Fo, e), proj(Fb, e)
          arrs["all"][l] = k4c - e
          arrs["Pown"][l] = k4c - po;   arrs["Rown"][l] = k4c - (e - po)
          arrs["Pboth"][l] = k4c - pbth; arrs["Rboth"][l] = k4c - (e - pbth)
          if l >= 8:
              r2["own"].append(1 - ((e - po) ** 2).sum() / ((e - e.mean()) ** 2).sum())
              r2["both"].append(1 - ((e - pbth) ** 2).sum() / ((e - e.mean()) ** 2).sum())
      print(f"\n=== MLP {i}, base B9 {base:.6e}   (in-sample R2 of the kappa4 error, layers 8-15: "
            f"own {np.mean(r2['own']):.3f}, own+pair {np.mean(r2['both']):.3f})")
      for k, lab in (("all", "remove ALL of e (kappa4 ceiling)"), ("Pown", "remove P_own e   (best own-pattern)"),
                     ("Rown", "remove (I-P_own) e"), ("Pboth", "remove P_both e  (best own+pair pattern)"),
                     ("Rboth", "remove (I-P_both) e")):
          orc = {"k4": arrs[k], "k4_layers": set(arrs[k])}
          r, _ = run(i, orc)
          print(f"  {lab:42s} x{r/base:.4f}")
