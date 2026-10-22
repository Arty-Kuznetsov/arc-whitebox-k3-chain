"""Analysis of vm_mc.py outputs (RESULTS.md §2 predictions P1-P4).
usage: python vm_analyze.py <file stem> [...]   e.g. vm_0_gauss_s1 vm_0_sphere_s1"""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
Phi = np.vectorize(lambda a: 0.5 * (1 + math.erf(a / math.sqrt(2))))
phi = lambda a: np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def slope(y, x):
    X = np.stack([np.ones_like(x), x], 1); return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def analyze(stem):
    d = np.load(HERE / "res" / f"{stem}.npz"); net = stem.split("_")[1].split("-relu")[0]
    for a in ("-leaky", "-gelu", "-silu"):
        net = net.replace(a, "")
    if net.startswith("s"):
        import torch, sys as _s
        _s.path.insert(0, str(HERE.parent / "2026-10-13_coherence-law")); torch.set_default_dtype(torch.float64)
        from coh import weights
        shp, k = net[1:].split("-"); n_, L_ = shp.split("x"); W = [w.numpy() for w in weights(int(n_), int(L_), int(k))]
    else:
        W = np.load(BAKED / f"weights_{net}.npy", mmap_mode="r")
    L, n = d["mu"].shape; q = float(d["q"]) if "q" in d.files else 2.0
    rows = []
    for l in range(L):
        mu, var, k3, m, v, vpre = (d[k][l] for k in ("mu", "var", "k3", "m", "v", "vpre"))
        s = np.sqrt(var); a = mu / s
        f1, f2 = mu * Phi(a), s * phi(a)
        F = np.stack([f1, f2], 1); lam, *_ = np.linalg.lstsq(F, v, rcond=None); p = F @ lam
        r = dict(layer=l, c_meas_n=slope(k3, mu) * n if l > 0 else 0.0,
                 c_pred_next_n=3.0 * q * float(v @ m) / float(m @ m),
                 cos_v_m=cos(v, m), cos_vpre_mu=cos(vpre, mu) if l > 0 else float("nan"),
                 lam1=float(lam[0]), lam2=float(lam[1]), R2_two=1 - float(((v - p) ** 2).sum() / ((v - v.mean()) ** 2).sum()),
                 cos_f1_m=cos(f1, m), cos_Cm_m=cos(d["Cmhat"][l], m),
                 rayleigh=float(d["Cmhat"][l] @ (m / np.linalg.norm(m))) / float(d["varh"][l].mean()),
                 alpha_sd=float(a.std()), sig2w=float((var * mu * (mu - mu.mean())).sum() / ((mu - mu.mean()) ** 2).sum()))
        # amplitude along the mean: beta = (h - m).mhat; <v, mhat> = E[beta Q] exactly, Q = beta^2 + ||y_perp||^2
        Eb2, Eb3, EbQ = (float(d["amp"][l, k]) for k in (1, 2, 3))
        nm = float(np.linalg.norm(m)); trC = float(d["varh"][l].sum()); trCp = trC - Eb2
        eps = Eb2 / nm ** 2
        r.update(eps_n=eps * n, amp_skew=Eb3 / Eb2 ** 1.5, vm_check=float(v @ (m / nm)) / EbQ,
                 share_b3=Eb3 / EbQ, R_sm=(EbQ - Eb3) / (2 * eps * nm * trCp),
                 cos_v_Cm=cos(v, d["Cmhat"][l]), amp_resid=float(np.linalg.norm(v - EbQ / Eb2 * d["Cmhat"][l]) / np.linalg.norm(v)),
                 cn_scale=12 * eps * trCp, cn_b3=6 * Eb3 / nm, cn_rest=6 * (EbQ - Eb3 - 2 * eps * nm * trCp) / nm)
        if l + 1 < L:
            Wn = np.asarray(W[l + 1], dtype=np.float64) * (q / 2.0) ** 0.5
            r["cos_Wv_vpre_next"] = cos(Wn.T @ v, d["vpre"][l + 1])
            r["cos_Wv_mu_next"] = cos(Wn.T @ v, d["mu"][l + 1])
        rows.append(r)
    for l in range(L - 1):
        rows[l + 1]["c_pred_n"] = rows[l]["c_pred_next_n"]
    return rows


if __name__ == "__main__":
    res = {}
    for stem in sys.argv[1:]:
        rows = analyze(stem); res[stem] = rows
        print(f"== {stem}")
        print(" l | c_meas*n c_pred*n | cos(v,m) cos(vpre,mu) cos(Wv,vpre') | lam1  lam2  R2(2f) cos(f1,m) | cos(Cm,m) rayl | sd(a) sig2w")
        for r in rows:
            print(f"{r['layer']:2d} | {r['c_meas_n']:8.2f} {r.get('c_pred_n', float('nan')):8.2f} | {r['cos_v_m']:8.3f} {r['cos_vpre_mu']:12.3f} "
                  f"{r.get('cos_Wv_vpre_next', float('nan')):13.3f} | {r['lam1']:5.2f} {r['lam2']:5.2f} {r['R2_two']:6.3f} {r['cos_f1_m']:9.3f} | "
                  f"{r['cos_Cm_m']:9.3f} {r['rayleigh']:5.1f} | {r['alpha_sd']:5.2f} {r['sig2w']:5.3f}")
    for stem, rows in res.items():
        print(f"== {stem}: the amplitude mode along the mean")
        print(" l | eps*n  skew(amp) | cos(v,Cm) |v-kCm|/|v| | share E[b^3] | R_sm  | c_pred*n = 12 eps trCperp + 6E[b^3]/|m| + rest")
        for r in rows:
            print(f"{r['layer']:2d} | {r['eps_n']:6.3f} {r['amp_skew']:6.2f} | {r['cos_v_Cm']:9.3f} {r['amp_resid']:10.3f} | {r['share_b3']:11.3f} | {r['R_sm']:5.2f} | "
                  f"{r['c_pred_next_n']:6.2f} = {r['cn_scale']:6.2f} + {r['cn_b3']:6.2f} + {r['cn_rest']:6.2f}")
    json.dump(res, open(HERE / "res" / ("analysis_" + "_".join(s.replace("vm_", "") for s in sys.argv[1:]) + ".json"), "w"), indent=1)
