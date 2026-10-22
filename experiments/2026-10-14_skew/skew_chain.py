"""A Gaussian closure that carries the skewness as a function of the mean (the 'k = 2.5' closure).
Law (skew_law.py, write-up §6.2/§6.4): a neuron's pre-activation kappa3 is t_i ~ a(l) + c(l) mu_i, R^2 0.8-0.9 at depth,
c(l) ~ 13.8/n at layer 1, the same for every network of a shape (CV <= 3% at 1024 x 16). The same argument (the partial
trace dominates, v || m) gives the (2,1) slice D_ij = kappa3[i,i,j] ~ t_j / 3 for i != j: rows identical, O(n).
Arms, all on our float64 chain in config 'full' (= ARC's SIMPLE k = 3, which keeps kappa4 as one radial scalar), the
kappa3 the step reads replaced at EVERY layer through the online hook mcorr (sources are still carried but unused):
  full        : the chain's own t and D (k = 3)
  zero        : t = 0, D = 0 (k = 3 without kappa3, the Gaussian-closure reference at equal kappa4 treatment)
  clos_own    : t = a + c mu, D_ij = t_j / 3, (a, c) per layer fitted on THIS network's Monte Carlo (form upper bound)
  clos_univ   : the same with (a, c) per layer averaged over OTHER networks of the shape (deployable: offline constants)
  clos_univ_t : clos_univ on the diagonal only, D = 0 (what D is worth)
  tlaw_Down   : t = a + c mu (universal), the chain's own D   } which of the two approximations
  town_Dr1    : the chain's own t, D_ij = t_j / 3            } loses the gain
  ptrace      : (2026-10-14 evening, ../2026-10-14_vm Prop. 5) the chain's kappa3 projected on its partial trace:
                v = t + D.sum(0) (v_a = sum_b kappa3(a, b, b), D_ij = kappa3(i, i, j)), then t = 3v/n, D_ab = v_b/n
                (kappa3 ~ Sym(I x v)/n: the Isserlis pairing part only, no fitted constant)
Also recorded for 'full': corr(D_ij, t_j/3) over i != j (is the chain's own D the rank-one form?).
Truth: 1024 x 16 public networks 0-9 against the 1e9 ground truth (constants from the Monte-Carlo moment oracles of
networks 10-29); synthetic shapes against the two-seed MC of ../2026-10-13_coherence-law (constants leave one out).
usage: python skew_chain.py baked | synth <n> <L>
"""
import sys, glob, json, numpy as np, torch
from pathlib import Path
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
from chain import run_chain                                     # noqa: E402
CL = HERE.parent / "2026-10-13_coherence-law"; sys.path.insert(0, str(CL))
from coh import weights, BAKED                                  # noqa: E402


def fit_ac(mu, k3):
    """per layer least squares k3 ~ a + c mu; mu, k3: (L, n)"""
    L = mu.shape[0]; out = np.zeros((L, 2))
    for l in range(L):
        X = np.stack([np.ones_like(mu[l]), mu[l]], 1); out[l] = np.linalg.lstsq(X, k3[l], rcond=None)[0]
    return out


def run(Ws, arm, ac=None):
    rec = []
    def hook(l, st):
        t, D, mu = st["t"], st["D"], st["mu_pre"]
        if arm == "full":
            if l >= 2:
                n = len(t); off = ~torch.eye(n, dtype=torch.bool); T3 = (t[None, :] / 3.0).expand(n, n)
                rec.append(float(torch.corrcoef(torch.stack([D[off], T3[off]]))[0, 1]))
            return {}
        if arm == "zero":
            return {"k3": t, "D": D}
        n = len(t)
        if arm == "ptrace":
            v = t + D.sum(0); tn = 3.0 * v / n
            Dn = (v[None, :] / n).expand(n, n).clone(); Dn.fill_diagonal_(0.0)
            return {"k3": t - tn, "D": D - Dn}
        if arm == "town_Dr1":
            tn = t
        else:
            a, c = float(ac[l, 0]), float(ac[l, 1]); tn = a + c * mu
        dd = {"k3": t - tn}
        if arm in ("clos_own", "clos_univ", "town_Dr1"):
            Dn = (tn[None, :] / 3.0).expand(n, n).clone(); Dn.fill_diagonal_(0.0); dd["D"] = D - Dn
        elif arm == "clos_univ_t":
            dd["D"] = D
        return dd
    rows, _ = run_chain([W.clone() for W in Ws], mcorr=hook)
    return rows.numpy().astype(np.float64), rec


def main():
    res = {"arms": {}, "corr_D": []}
    arms = ["full", "zero", "clos_own", "clos_univ", "clos_univ_t", "tlaw_Down", "town_Dr1", "ptrace"]
    if len(sys.argv) > 4 and sys.argv[-1] == "--only-ptrace": arms = ["full", "zero", "clos_univ", "ptrace"]
    if sys.argv[1] == "baked":
        tag = "n1024_L16"
        mo = lambda k: np.load(HERE.parent / "2026-09-24_k3-chain" / "runs" / f"mc_oracle_{k}.npz")
        univ = np.mean([fit_ac(mo(k)["mean"], mo(k)["k3"]) for k in range(10, 30)], 0)
        gt = np.load(BAKED / "gt.npy")
        cases = []
        for k in range(10):
            Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{k}.npy")]
            own = fit_ac(mo(k)["mean"], mo(k)["k3"])
            cases.append((k, Ws, own, univ, lambda pred, k=k: float(np.mean((pred[-1] - gt[k][-1]) ** 2))))
    else:
        n, L = int(sys.argv[2]), int(sys.argv[3]); tag = f"n{n}_L{L}"
        files = sorted(glob.glob(str(CL / "res" / f"{tag}_net*.npz")))
        D = [np.load(f) for f in files]
        fits = [fit_ac(0.5 * (d["a_mu"] + d["b_mu"]), 0.5 * (d["a_k3"] + d["b_k3"])) for d in D]
        cases = []
        for j, (f, d) in enumerate(zip(files, D)):
            k = int(Path(f).stem.split("net")[1])
            univ = np.mean([fits[i] for i in range(len(D)) if i != j], 0)
            mse = lambda pred, d=d: float(np.mean((pred[-1] - d["a_m"][-1]) * (pred[-1] - d["b_m"][-1])))
            cases.append((k, weights(n, L, k), fits[j], univ, mse))
    for arm in arms: res["arms"][arm] = []
    finals = {}
    for k, Ws, own, univ, mse in cases:
        line = f"{tag} net {k}:"
        for arm in arms:
            ac = own if arm == "clos_own" else univ
            pred, rec = run(Ws, arm, ac)
            m = mse(pred); res["arms"][arm].append(m); line += f" {arm} {m:.3e}"
            finals[f"{arm}_{k}"] = pred[-1]
            if rec: res["corr_D"].append(float(np.mean(rec)))
        print(line, flush=True)
    full, zero = np.array(res["arms"]["full"]), np.array(res["arms"]["zero"])
    print(f"== {tag}: corr(D_ij, t_j/3) over i != j, chain 'full', mean over layers >= 2: {np.mean(res['corr_D']):.3f}")
    for arm in arms:
        a = np.array(res["arms"][arm]); kept = (zero - a) / (zero - full)
        print(f"   {arm:<12} final MSE {np.mean(a):.3e} | x zero {np.mean(a / zero):.3f} | share of k=3's kappa3 gain kept "
              f"{100*np.mean(kept):6.1f}% (min {100*np.min(kept):6.1f}%, max {100*np.max(kept):6.1f}%)")
    suf = "_ptrace" if arms == ["full", "zero", "clos_univ", "ptrace"] else ""
    json.dump(res, open(HERE / "res" / f"skew_chain{suf}_{tag}.json", "w"), indent=1)
    np.savez(HERE / "res" / f"skew_chain{suf}_{tag}_final.npz", **finals)


if __name__ == "__main__":
    main()
