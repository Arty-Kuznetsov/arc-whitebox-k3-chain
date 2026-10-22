"""Is a neuron's pre-activation skewness slaved to its mean? (write-up §6.4: true kappa3_i ~ function of mu_i, corr
0.87-0.93 at the last layer). Per layer and network: kappa3_i ~ a(l) + c(l) mu_i, noise-corrected R^2 from two
independent Monte-Carlo seeds (energies as cross products), and how much c(l) varies across networks (if it is the same
for every network of a shape, a k = 2 estimator can carry kappa3 as c(l) mu with offline constants).
Data: ../2026-10-13_coherence-law/res (synthetic shapes, two seeds) and ../2026-09-24_k3-chain/oracles (public
1024 x 16 networks 0 and 1, two seeds at 1e7)."""
import glob, json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sets = {}
for tag in ["n256_L8", "n256_L16", "n256_L32", "n512_L16"]:
    nets = []
    for f in sorted(glob.glob(str(HERE.parent / "2026-10-13_coherence-law" / "res" / f"{tag}_net*.npz"))):
        d = np.load(f); nets.append((0.5 * (d["a_mu"] + d["b_mu"]), d["a_k3"], d["b_k3"], 0.5 * (d["a_var"] + d["b_var"])))
    sets[tag] = nets
O = HERE.parent / "2026-09-24_k3-chain" / "oracles"
nets = []
for k in (0, 1):
    a, b = np.load(O / f"mc_oracle_{k}.npz"), np.load(O / f"mc_oracle_{k}_s777.npz")
    nets.append((0.5 * (a["mean"] + b["mean"]), a["k3"], b["k3"], 0.5 * (a["var"] + b["var"])))
sets["n1024_L16 (public)"] = nets
out = {}
for tag, nets in sets.items():
    L = nets[0][0].shape[0]
    R2 = np.zeros((len(nets), L)); C = np.zeros((len(nets), L)); A = np.zeros((len(nets), L)); R2s = np.zeros((len(nets), L))
    for j, (mu, kA, kB, var) in enumerate(nets):
        for l in range(L):
            X = np.stack([np.ones_like(mu[l]), mu[l]], 1)
            b = np.linalg.lstsq(X, 0.5 * (kA[l] + kB[l]), rcond=None)[0]; p = X @ b
            S = np.mean((kA[l] - kA[l].mean()) * (kB[l] - kB[l].mean()))          # signal variance, unbiased
            R = np.mean((kA[l] - p) * (kB[l] - p)) - (np.mean(kA[l]) - p.mean()) * (np.mean(kB[l]) - p.mean())
            R2[j, l] = 1 - R / S if S > 0 else np.nan
            A[j, l], C[j, l] = b
            # the same with the standardised skewness gamma = k3/sigma^3 against alpha = mu/sigma
            s = np.sqrt(var[l]); g = 0.5 * (kA[l] + kB[l]) / s ** 3; al = mu[l] / s
            Xs = np.stack([np.ones_like(al), al], 1); bs = np.linalg.lstsq(Xs, g, rcond=None)[0]
            gA, gB = kA[l] / s ** 3, kB[l] / s ** 3; ps = Xs @ bs
            Ss = np.mean((gA - gA.mean()) * (gB - gB.mean())); Rs = np.mean((gA - ps) * (gB - ps)) - (gA.mean() - ps.mean()) * (gB.mean() - ps.mean())
            R2s[j, l] = 1 - Rs / Ss if Ss > 0 else np.nan
    cv = np.std(C, 0) / np.abs(np.mean(C, 0))
    out[tag] = dict(R2_by_layer=np.nanmedian(R2, 0).tolist(), R2_gamma_alpha=np.nanmedian(R2s, 0).tolist(),
                    c_mean=np.mean(C, 0).tolist(), c_cv=cv.tolist(), a_mean=np.mean(A, 0).tolist(), nets=len(nets))
    sel = list(range(1, L, max(1, L // 6))) + [L - 1]
    print(f"== {tag} ({len(nets)} nets)")
    print("   layer        " + " ".join(f"{l:>6d}" for l in sel))
    print("   R2(k3 ~ mu)  " + " ".join(f"{np.nanmedian(R2[:, l]):6.2f}" for l in sel))
    print("   R2(g ~ a)    " + " ".join(f"{np.nanmedian(R2s[:, l]):6.2f}" for l in sel))
    print("   c(l) mean    " + " ".join(f"{np.mean(C[:, l]):6.3f}" for l in sel))
    print("   c(l) CV      " + " ".join(f"{cv[l]:6.3f}" for l in sel))
json.dump(out, open(HERE / "res" / "skew_law.json", "w"), indent=1)
