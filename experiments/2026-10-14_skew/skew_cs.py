"""Does the skew-law closure remove exactly the LEARNABLE part of a Gaussian closure's error?
The coherence law (../2026-10-13_coherence-law): a per-neuron model of the neuron's own state learns ~65% of the final
error of ARC's k = 2 at every shape, and the mechanism named there is the skew slaved to the mean. skew_chain.py: the
closure (t = a + c mu, D_ij = t_j / 3, universal constants) removes ~65% of the error of the kappa3-free chain. If the two
are the same thing, the closure's residual should have (almost) no learnable part left.
This adds the closure arm 'clos_univ' of skew_chain.py as estimator 'cs' (cs_m, cs_mu, cs_var, cs_t) to the files of
the coherence law, and the 'zero' arm as 'cz', so `coh_analyze.py --est cs|cz` measures their coherent and learnable shares
with the same truth and the same features as for 'c' (k = 3) and 'c2' (ARC k = 2).
usage: python skew_cs.py <glob tag, e.g. n256_L16_net | n1024_L16_baked>"""
import sys, glob, numpy as np, torch
from pathlib import Path
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
from chain import run_chain                                     # noqa: E402
CL = HERE.parent / "2026-10-13_coherence-law"; sys.path.insert(0, str(CL))
from coh import weights, BAKED                                  # noqa: E402
from skew_chain import fit_ac                                   # noqa: E402


def stats(Ws, ac, arm):
    def hook(l, st):
        t, D, mu = st["t"], st["D"], st["mu_pre"]
        if arm == "cz":
            return {"k3": t, "D": D}
        n = len(t); tn = float(ac[l, 0]) + float(ac[l, 1]) * mu
        Dn = (tn[None, :] / 3.0).expand(n, n).clone(); Dn.fill_diagonal_(0.0)
        return {"k3": t - tn, "D": D - Dn}
    rows, info = run_chain([W.clone() for W in Ws], record=True, mcorr=hook)
    L = len(Ws); g = lambda k: np.stack([info[l][k].numpy() for l in range(L)]).astype(np.float64)
    return {f"{arm}_m": rows.numpy().astype(np.float64), f"{arm}_mu": g("mu_pre"), f"{arm}_var": g("var"), f"{arm}_t": g("t")}


tag = sys.argv[1]
files = sorted(glob.glob(str(CL / "res" / f"{tag}*.npz")))
if "baked" in tag:
    mo = lambda k: np.load(HERE.parent / "2026-09-24_k3-chain" / "runs" / f"mc_oracle_{k}.npz")
    univ_all = np.mean([fit_ac(mo(k)["mean"], mo(k)["k3"]) for k in range(10, 30)], 0)
    fits = None
else:
    DD = [np.load(f) for f in files]
    fits = [fit_ac(0.5 * (d["a_mu"] + d["b_mu"]), 0.5 * (d["a_k3"] + d["b_k3"])) for d in DD]
for j, f in enumerate(files):
    d = dict(np.load(f)); name = Path(f).stem
    if "cs_m" in d and "cz_m" in d: print("has cs/cz:", name, flush=True); continue
    if "baked" in name:
        k = int(name.split("baked")[1]); Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{k}.npy")]
        ac = univ_all
    else:
        n_, L_, k = name.split("_"); Ws = weights(int(n_[1:]), int(L_[1:]), int(k[3:]))
        ac = np.mean([fits[i] for i in range(len(files)) if i != j], 0)
    d.update(stats(Ws, ac, "cs")); d.update(stats(Ws, ac, "cz"))
    np.savez(f, **d)
    ref = d["gt_m"] if "gt_m" in d else 0.5 * (d["a_m"] + d["b_m"])
    print(f"cs/cz added: {name}; final MSE cs {np.mean((d['cs_m'][-1] - ref[-1]) ** 2):.2e} "
          f"cz {np.mean((d['cz_m'][-1] - ref[-1]) ** 2):.2e} c {np.mean((d['c_m'][-1] - ref[-1]) ** 2):.2e}", flush=True)
