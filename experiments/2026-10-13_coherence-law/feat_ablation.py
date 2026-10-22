"""Which per-neuron features carry the learnable share of k = 2's final error? Leave one network out, as coh_analyze.py.
Base quantities from the estimator's own last-layer state: alpha = mu/sigma, log sigma; plus the kappa3 readout pattern
(Stein: d E[ReLU] / d kappa3 = alpha phi(alpha) / (6 sigma^2)) and the kurtosis pattern."""
import sys, glob, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
est = sys.argv[1]; tags = sys.argv[2:]
def ridge(X, y, lam=1e-6):
    m, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - m) / sd]); b = np.linalg.solve(Z.T @ Z + lam * len(X) * np.eye(Z.shape[1]), Z.T @ y)
    return lambda Xn: np.hstack([np.ones((len(Xn), 1)), (Xn - m) / sd]) @ b
for tag in tags:
    nets = []
    for f in sorted(glob.glob(str(HERE / "res" / f"{tag}*.npz"))):
        d = dict(np.load(f)); l = d["c_m"].shape[0] - 1
        mu, var = d[f"{est}_mu"][l], np.maximum(d[f"{est}_var"][l], 1e-30); s = np.sqrt(var); a = mu / s
        phi = np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi)
        F = {"alpha": a, "alpha2": a * a, "alpha3": a ** 3, "logs": np.log(s), "logs2": np.log(s) ** 2, "a_logs": a * np.log(s),
             "k3pat": a * phi / s, "k4pat": s * phi * (a * a - 1), "phi": phi, "s": s}
        if "gt_m" in d: eA = eB = d[f"{est}_m"][l] - d["gt_m"][l]
        else: eA = d[f"{est}_m"][l] - d["a_m"][l]; eB = d[f"{est}_m"][l] - d["b_m"][l]
        nets.append((F, eA, eB))
    sets = {"intercept": [], "alpha": ["alpha"], "alpha poly": ["alpha", "alpha2", "alpha3"], "log sigma": ["logs", "logs2"],
            "s, phi": ["s", "phi"], "k3 pattern": ["k3pat"], "k4 pattern": ["k4pat"],
            "alpha poly + log sigma": ["alpha", "alpha2", "alpha3", "logs", "logs2", "a_logs"],
            "all": ["alpha", "alpha2", "alpha3", "logs", "logs2", "a_logs", "k3pat", "k4pat", "phi", "s"]}
    print(f"== {est} {tag} ({len(nets)} nets): learnable share of the final error, leave one out")
    for name, keys in sets.items():
        lf = []
        for k in range(len(nets)):
            tr = [j for j in range(len(nets)) if j != k]
            X = lambda j: (np.stack([nets[j][0][q] for q in keys], 1) if keys else np.zeros((len(nets[j][1]), 0)))
            f = ridge(np.vstack([X(j) for j in tr]), np.concatenate([nets[j][1] for j in tr]))
            a_, b_ = nets[k][1], nets[k][2]; p = f(X(k))
            lf.append(1 - np.mean((a_ - p) * (b_ - p)) / np.mean(a_ * b_))
        print(f"   {name:<24} {100*np.mean(lf):6.1f}%  (min {100*np.min(lf):5.1f}%, max {100*np.max(lf):5.1f}%)")
