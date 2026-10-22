"""Analysis for coh.py: per shape, the coherent share of each order's per-neuron error and the share of the final
post-mean error a per-neuron model learns out of sample (leave one network out).
Noise: Monte-Carlo noise is correlated ACROSS neurons (shared samples), so mean(nu^2)/n underestimates the noise in a
mean over neurons. Every energy below is a cross product of the two independent seeds, unbiased for the signal:
  level^2 ~ mean(eA) * mean(eB),  energy ~ mean(eA * eB),  residual ~ mean((eA - p) * (eB - p)) for p independent of both.
Baked networks (1e9-sample truth) use eA = eB = e.
usage: python coh_analyze.py [--est c|c2] <glob tag, e.g. n256_L32_net> [...]   -> res/summary_<est>_<tag>.json + table
   --est c  = our chain in config full = ARC SIMPLE k = 3 (default); --est c2 = ARC SIMPLE k = 2 (coh_k2.py);
   --est cs / cz = the skew-law closure / kappa3 = 0 arms of ../2026-10-14_skew (skew_cs.py)"""
import sys, json, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent


EST = "c"


def load(tag):
    files = sorted(glob.glob(str(HERE / "res" / f"{tag}*.npz")))
    nets = []
    for f in files:
        d = dict(np.load(f))
        if "gt_m" in d:
            if f"{EST}_m" not in d: continue
            e = {"m": d[f"{EST}_m"] - d["gt_m"]}; nets.append(dict(d=d, eA=e, eB=e, synth=False)); continue
        if f"{EST}_m" not in d: continue
        c = lambda k: d[f"{EST}_{k}"]
        eA = {"m": c("m") - d["a_m"], "mu": c("mu") - d["a_mu"], "var": c("var") - d["a_var"], "k3": c("t") - d["a_k3"]}
        eB = {"m": c("m") - d["b_m"], "mu": c("mu") - d["b_mu"], "var": c("var") - d["b_var"], "k3": c("t") - d["b_k3"]}
        nets.append(dict(d=d, eA=eA, eB=eB, synth=True))
    return files, nets


def coh_share(a, b):
    """coherent share of a per-neuron error vector, from two independent noisy copies a, b of it"""
    E = float(np.mean(a * b))
    return float(np.mean(a) * np.mean(b)) / E if E > 0 else float("nan"), E


def feats(d, l):
    mu, var, t = d[f"{EST}_mu"][l], np.maximum(d[f"{EST}_var"][l], 1e-30), d[f"{EST}_t"][l]
    s = np.sqrt(var); a = mu / s; ls = np.log(s); g = t / s ** 3
    return np.stack([a, a * a, a ** 3, ls, ls * ls, a * ls, g, a * g], 1)


def ridge(X, y, lam=1e-6):
    m, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - m) / sd])
    beta = np.linalg.solve(Z.T @ Z + lam * len(X) * np.eye(Z.shape[1]), Z.T @ y)
    return lambda Xn: np.hstack([np.ones((len(Xn), 1)), (Xn - m) / sd]) @ beta


def analyze(tag):
    files, nets = load(tag)
    if not nets: print("no files for", tag); return None
    L, n = nets[0]["d"]["c_m"].shape
    out = {"est": EST, "tag": tag, "n": n, "L": L, "L_over_n": L / n, "nets": len(nets), "synth": nets[0]["synth"]}
    # 1. coherent share per order and layer (mean over networks)
    orders = ((["m", "mu", "var", "k3"] if EST in ("c", "cs") else ["m", "mu", "var"]) if nets[0]["synth"] else ["m"])
    for o in orders:
        cs = np.array([[coh_share(N_["eA"][o][l], N_["eB"][o][l])[0] for l in range(L)] for N_ in nets])
        out[f"cs_{o}_by_layer"] = np.nanmean(cs, 0).tolist()
        out[f"cs_{o}_last"] = [float(x) for x in cs[:, -1]]
    # 2. learnability of the final post-mean error, leave one network out
    l = L - 1
    X = [feats(N_["d"], l) for N_ in nets]
    res = {"loo": [], "level_oracle": [], "level_const": [], "in_sample": []}
    for k in range(len(nets)):
        tr = [j for j in range(len(nets)) if j != k]
        if not tr: break
        f = ridge(np.vstack([X[j] for j in tr]), np.concatenate([nets[j]["eA"]["m"][l] for j in tr]))
        a, b = nets[k]["eA"]["m"][l], nets[k]["eB"]["m"][l]
        S = float(np.mean(a * b)); p = f(X[k])
        res["loo"].append(1 - float(np.mean((a - p) * (b - p))) / S)
        res["level_oracle"].append(float(np.mean(a) * np.mean(b)) / S)
        c = float(np.mean([np.mean(nets[j]["eA"]["m"][l]) for j in tr]))
        res["level_const"].append(1 - float(np.mean((a - c) * (b - c))) / S)
        g = ridge(X[k], a); q = g(X[k])
        res["in_sample"].append(1 - float(np.mean((a - q) * (b - q))) / S)
    # the mechanism test: a scale-mixture kurtosis level v shared by all neurons shifts E[ReLU(z)] by
    # (kappa4/24) * sigma * phi(alpha) * (alpha^2 - 1) / sigma^4 * sigma^4 ... = c * sigma * phi(alpha) * (alpha^2 - 1);
    # one coefficient c fitted on the other networks (leave one out), no intercept
    res["k4pattern"] = []
    for k in range(len(nets)):
        tr = [j for j in range(len(nets)) if j != k]
        if not tr: break
        def pat(d):
            mu, var = d[f"{EST}_mu"][l], np.maximum(d[f"{EST}_var"][l], 1e-30); sg = np.sqrt(var); a = mu / sg
            return sg * np.exp(-0.5 * a * a) / np.sqrt(2 * np.pi) * (a * a - 1)
        xs = np.concatenate([pat(nets[j]["d"]) for j in tr]); ys = np.concatenate([nets[j]["eA"]["m"][l] for j in tr])
        cf = float(xs @ ys / (xs @ xs))
        a, b = nets[k]["eA"]["m"][l], nets[k]["eB"]["m"][l]; q = cf * pat(nets[k]["d"])
        res["k4pattern"].append(1 - float(np.mean((a - q) * (b - q))) / float(np.mean(a * b)))
    for key, v in res.items():
        out[f"lf_{key}"] = v; out[f"lf_{key}_mean"] = float(np.mean(v)) if v else float("nan")
    out["final_mse"] = [float(np.mean(N_["eA"]["m"][l] * N_["eB"]["m"][l])) for N_ in nets]
    json.dump(out, open(HERE / "res" / f"summary_{EST}_{tag.rstrip('_')}.json", "w"), indent=1)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--est"]: EST = args[1]; args = args[2:]
    rows = [analyze(t) for t in args]
    print("estimator:", {"c": "ARC SIMPLE k=3 (our chain, config full)", "c2": "ARC SIMPLE k=2",
                         "cs": "k=3 chain, kappa3 replaced by the skew law (../2026-10-14_skew)",
                         "cz": "k=3 chain, kappa3 = 0 (../2026-10-14_skew)"}[EST])
    print(f"{'shape':>10} {'L/n':>7} {'nets':>4} | {'final MSE':>9} | {'CS final m':>10} | {'LF loo':>7} {'level*':>7} {'const':>7} {'k4pat':>7} {'in-sample':>9} | {'CS mu':>6} {'CS var':>6} {'CS k3':>6}")
    for r in rows:
        if not r: continue
        cs_m = np.mean(r["cs_m_last"])
        extra = ""
        if r["synth"]:
            extra = f"{np.mean(r['cs_mu_last']):6.3f} {np.mean(r['cs_var_last']):6.3f} " + (f"{np.mean(r['cs_k3_last']):6.3f}" if "cs_k3_last" in r else "   n/a")
        print(f"{r['n']:>4}x{r['L']:<5} {r['L_over_n']:7.4f} {r['nets']:>4} | {np.mean(r['final_mse']):9.2e} | {cs_m:10.3f} | "
              f"{r['lf_loo_mean']:7.3f} {r['lf_level_oracle_mean']:7.3f} {r['lf_level_const_mean']:7.3f} {r['lf_k4pattern_mean']:7.3f} {r['lf_in_sample_mean']:9.3f} | {extra}")
