"""Task 3 of cloud #11: the STRUCTURE of the off-diagonal post-covariance error dC (run only if task 2 passed).
Everything on the chain's own B9 trajectory; dC = C_chain - C_or, offdiag = zero diagonal; P = the two-seed average,
noise matrix N = (C_A - C_B) / 2 (zero mean, the pooled estimate's noise variance).

    state M        the chain's pre-activation state at the analysis layers LS (record="full") -> res3/state_<M>.npz,
                   plus the decomposition of the chain's own K11 into the terms of its pair expansion (self-check:
                   their sum reproduces offdiag(C_chain) to float rounding)
    spec M         (a) eigen-spectrum of offdiag(dC_P) vs the noise N at LS; best-rank-r energy shares; seed-to-seed
                   reproducibility of the top eigenvectors              -> res3/spec_<M>.json
    rank M SET R VAR   (a) payoff: at the layers of SET the LIVE off-diagonal correction offdiag(C_or - C_live) is
                   replaced by its best rank-R part (eigh, top |lambda|; diagonal re-zeroed) -> res3/rank_*.json
    sparse M SET Q SEL (b) payoff of the top-Q fraction of off-diagonal entries by |offdiag(C_SEL - C_live)|, the
                   values taken from the OTHER seed (cross-fitting: no selection-on-noise bias) -> res3/sparse_*.json
    who M          (b) who the top pairs are: kink core x kink core, |C0| and |D| quantiles, by enrichment
    feat           (c) entrywise linear model of offdiag(dC_P) on matrices the chain holds; R^2 in sample (per MLP)
                   and out of sample (fit on MLP 0, apply to MLP 1), against the two-seed noise ceiling
    gain M         the PER-NEURON GAIN model dC0 ~ Delta K + K Delta, Delta = diag(delta), K = offdiag(C_chain): what a
                   per-neuron error of the pair coefficient (d1_j, from the neuron's own moment errors) does to the
                   pairs. n parameters, solved exactly: (diag(rowsum K o K) + K o K) delta = rowsum(K o dC0).
                   Frobenius share explained, per layer, with the noise floor          -> res3/gain_<M>.json
    gainarm M SET VAR  payoff of the gain-model part of the LIVE off-diagonal correction (oracle delta per layer)
    born M         BORN vs CARRIED: for each layer l, the chain's own step at l run on the EXACT C_pre(l) =
                   W_l^T C_or(l-1) W_l (everything else the chain's own: mu, kappa3 legs/D/t, kappa4 levels; the
                   patterns rebuilt from the exact C_pre, as the chain does) -> born(l) = offdiag(C'(l) - C_or(l)).
                   Input and target from the SAME seed (A, then B), so the linear part of the sampling noise cancels;
                   the A/B difference gives what is left. Compared with the total offdiag(dC(l))  -> res3/born_<M>.json
"""
import json
import sys

import numpy as np
import torch

import ce
import t2
from chain import relu_wick, run_chain

OUT = ce.HERE / "res3"
OUT.mkdir(exist_ok=True)
LS = (5, 8, 11, 14)
L = ce.L


def offd(X):
    X = np.array(X, dtype=np.float64, copy=True)
    np.fill_diagonal(X, 0.0)
    return X


# ----------------------------------------------------------------------------------------------- the chain's state
def state(m):
    f = OUT / f"state_{m}.npz"
    if f.exists():
        return dict(np.load(f))
    Ws, _ = ce.ta.mlp(m)
    _, info = run_chain(Ws, record="full", **ce.B9)
    keep = {}
    for l in LS:
        x = info[l]
        for k in ("t", "D", "mu_pre", "var", "K4d"):
            keep[f"{k}_{l}"] = x[k].numpy()
        for k in ("C", "K22", "K31"):
            keep[f"{k}_{l}"] = x[k].double().numpy()          # recorded in float32 by chain.py
    np.savez(f, **keep)
    return dict(np.load(f))


def coefs(mu, var, t, K4d):
    """B9's pair coefficients coef(k, 1) (marg=True: marginal Edgeworth correction for k <= 2), k = 1..4."""
    w, _, _ = relu_wick(torch.tensor(mu), torch.tensor(var))
    tt, kk = torch.tensor(t), torch.tensor(K4d)
    c = {}
    for k in range(1, 5):
        v = w[(k, 1)]
        if k <= 2:
            v = v + w[(k + 3, 1)] * tt / 6 + w[(k + 4, 1)] * kk / 24
        c[k] = v.numpy()
    return c


def terms(st, l, Cpre):
    """The terms of the chain's Q11 (B9: marg, c0d, K31, pairx k22, tet 3) at layer l, plus candidate extra terms.
    Cpre: the float64 pre-activation covariance (recomputed from the recorded post C, exact)."""
    mu, var, t, K4d = (st[f"{k}_{l}"] for k in ("mu_pre", "var", "t", "K4d"))
    D, K22, K31 = st[f"D_{l}"], st[f"K22_{l}"], st[f"K31_{l}"]
    C0 = offd(Cpre)
    c = coefs(mu, var, t, K4d)
    a = {k: c[k][:, None] for k in c}
    b = {k: c[k][None, :] for k in c}
    T = {
        "C0": a[1] * b[1] * C0,
        "D": 0.5 * (a[1] * b[2] * D.T + a[2] * b[1] * D),
        "C0sq": 0.5 * a[2] * b[2] * C0 * C0,
        "K22": 0.25 * a[2] * b[2] * K22,
        "K31": (1.0 / 6.0) * (a[3] * b[1] * K31 + a[1] * b[3] * K31.T),
        "C0D": 0.5 * (a[2] * b[3] * D.T + a[3] * b[2] * D) * C0,
        "C0K22": 0.25 * a[3] * b[3] * C0 * K22,
        "C0cu": (1.0 / 6.0) * a[3] * b[3] * C0 ** 3,
    }
    X = {  # candidates the expansion does not have (next Hermite order, cloud #10's next-order diagrams)
        "C0^4": (1.0 / 24.0) * a[4] * b[4] * C0 ** 4,
        "C0K31": (1.0 / 6.0) * (a[4] * b[2] * C0 * K31 + a[2] * b[4] * C0 * K31.T),
        "D^2": 0.125 * (a[4] * b[2] * D * D + a[2] * b[4] * D.T * D.T),
        "DDt": 0.25 * a[3] * b[3] * D * D.T,
        "d1d1": a[1] * b[1] * np.ones_like(C0),
        "vv": np.sqrt(var)[:, None] * np.sqrt(var)[None, :],
    }
    for M in list(T.values()) + list(X.values()):
        np.fill_diagonal(M, 0.0)
    return T, X


def pre_cov(m, l, Cch=None):
    Wn, _ = ce.wn(m)
    if Cch is None:
        Cch, _ = t2.rec(m)
    return Wn[l].T @ np.asarray(Cch[l - 1]) @ Wn[l]


def check_terms(m):
    st = state(m)
    Cch, _ = t2.rec(m)
    for l in LS:
        T, _ = terms(st, l, pre_cov(m, l, Cch))
        s = sum(T.values())
        ref = offd(Cch[l])
        print(f"MLP {m} L{l}: sum of Q11 terms vs offdiag(C_chain): max rel {np.abs(s - ref).max() / np.abs(ref).max():.1e}"
              + "".join(f"  {k} {np.linalg.norm(v) / np.linalg.norm(ref):.2e}" for k, v in T.items()), flush=True)


# ------------------------------------------------------------------------------------------------------ (a) spectrum
def spec(m):
    Cch, r_ = t2.rec(m)
    sA, sB = ce.SEEDS
    CA, CB = ce.cov(m, sA), ce.cov(m, sB)
    out = {}
    for l in range(L - 1):
        a, b = np.asarray(CA[l]), np.asarray(CB[l])
        dA, dB = offd(Cch[l] - a), offd(Cch[l] - b)
        dP, N = 0.5 * (dA + dB), 0.5 * offd(a - b)
        lp, Vp = np.linalg.eigh(dP)
        ln = np.linalg.eigvalsh(N)
        la, Va = np.linalg.eigh(dA)
        lb, Vb = np.linalg.eigh(dB)
        op = np.argsort(-np.abs(lp))
        on = np.sort(np.abs(ln))[::-1]
        oa, ob = np.argsort(-np.abs(la)), np.argsort(-np.abs(lb))
        EP, EN = float((lp ** 2).sum()), float((ln ** 2).sum())
        share = {r: float((lp[op[:r]] ** 2).sum() / EP) for r in (1, 4, 16, 64, 256)}
        share_n = {r: float((on[:r] ** 2).sum() / EN) for r in (1, 4, 16, 64, 256)}
        # reproducibility of the top eigenvectors between the two independent seeds
        rep = [float(abs(Va[:, oa[k]] @ Vb[:, ob[k]])) for k in range(8)]
        # do the top directions of dC0_P align with anything the chain holds? (random overlap ~ 1/sqrt(n) = 0.03)
        Cc = np.asarray(Cch[l])
        pm = r_["post_mean"][l] / np.linalg.norm(r_["post_mean"][l])
        lc, Vc = np.linalg.eigh(offd(Cc))
        Vc5 = Vc[:, np.argsort(-np.abs(lc))[:5]]
        one = np.ones(Cc.shape[0]) / np.sqrt(Cc.shape[0])
        sdv = np.sqrt(np.diag(Cc))
        sdv = sdv / np.linalg.norm(sdv)
        align = [dict(mean=float(abs(Vp[:, op[k]] @ pm)), ones=float(abs(Vp[:, op[k]] @ one)),
                      sd=float(abs(Vp[:, op[k]] @ sdv)), topC=float(np.abs(Vc5.T @ Vp[:, op[k]]).max()))
                 for k in range(3)]
        out[l] = dict(E_P=EP, E_noise=EN, S=EP - EN, share_P=share, share_noise=share_n,
                      top_P=[float(lp[op[k]]) for k in range(10)], top_noise=[float(on[k]) for k in range(10)],
                      rep_top8=rep, align_top3=align)
        print(f"MLP {m} L{l:2d}: |dC0_P|^2 {EP:.3e} noise {EN:.3e} (signal {100 * (EP - EN) / EP:5.1f}%) | top |lam| P "
              f"{abs(lp[op[0]]):.2e} {abs(lp[op[1]]):.2e} {abs(lp[op[2]]):.2e} noise {on[0]:.2e} | rank r share P "
              + " ".join(f"{r}:{share[r]:.3f}" for r in (1, 4, 16, 64)) + " | noise "
              + " ".join(f"{r}:{share_n[r]:.3f}" for r in (1, 4, 16, 64)) + " | seed rep top3 "
              + " ".join(f"{x:.2f}" for x in rep[:3]) + " | top1 vs mean/ones/sd/topC "
              + "/".join(f"{align[0][k]:.2f}" for k in ("mean", "ones", "sd", "topC")), flush=True)
    (OUT / f"spec_{m}.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------------------- payoff arms
def _mats(m, layers):
    sA, sB = ce.SEEDS
    CA, CB = ce.cov(m, sA), ce.cov(m, sB)
    return {l: (torch.tensor(np.asarray(CA[l])), torch.tensor(np.asarray(CB[l]))) for l in layers}


def _run(m, hook):
    Ws, g = ce.ta.mlp(m)
    mu, _ = run_chain(Ws, postc=hook, **ce.B9)
    return float(((mu[-1] - g) ** 2).mean())


def rank_arm(m, set_, r, var):
    f = OUT / f"rank_{m}_{set_}_{r}_{var}.json"
    if f.exists():
        return json.loads(f.read_text())
    mats = _mats(m, t2.SETS[set_])

    def hook(l, C):
        if l not in mats:
            return C
        A, B = mats[l]
        X = {"A": A, "B": B, "P": 0.5 * (A + B)}[var] - C
        X = X - torch.diag(torch.diagonal(X))
        lam, V = torch.linalg.eigh(X)
        idx = torch.argsort(lam.abs(), descending=True)[:r]
        Xr = (V[:, idx] * lam[idx]) @ V[:, idx].T
        Xr = Xr - torch.diag(torch.diagonal(Xr))
        return C + 0.5 * (Xr + Xr.T)

    mse = _run(m, hook)
    res = dict(mse=mse, x=mse / t2.b9_ref(m), mlp=m, set=set_, r=r, var=var)
    f.write_text(json.dumps(res))
    print(f"rank MLP {m} {set_} r={r} {var}: x{res['x']:.4f}", flush=True)
    return res


def sparse_arm(m, set_, q, sel):
    f = OUT / f"sparse_{m}_{set_}_{q}_{sel}.json"
    if f.exists():
        return json.loads(f.read_text())
    mats = _mats(m, t2.SETS[set_])

    def hook(l, C):
        if l not in mats:
            return C
        A, B = mats[l]
        S_, V_ = (A, B) if sel == "A" else (B, A)
        Xs = (S_ - C).abs()
        Xs.fill_diagonal_(0.0)
        iu = torch.triu_indices(C.shape[0], C.shape[0], 1)
        vals = Xs[iu[0], iu[1]]
        k = max(1, int(q * vals.numel()))
        thr = torch.topk(vals, k).values[-1]
        mask = (Xs >= thr) & (Xs > 0)
        X = (V_ - C) * mask
        return C + 0.5 * (X + X.T)

    mse = _run(m, hook)
    res = dict(mse=mse, x=mse / t2.b9_ref(m), mlp=m, set=set_, q=q, sel=sel)
    f.write_text(json.dumps(res))
    print(f"sparse MLP {m} {set_} q={q} sel={sel}: x{res['x']:.4f}", flush=True)
    return res


# -------------------------------------------------------------------------------------------------- (b) who they are
def who(m, qs=(0.01, 0.05, 0.20)):
    Cch, _ = t2.rec(m)
    st = state(m)
    sA, sB = ce.SEEDS
    CA, CB = ce.cov(m, sA), ce.cov(m, sB)
    out = {}
    for l in LS:
        dA, dB = offd(Cch[l] - np.asarray(CA[l])), offd(Cch[l] - np.asarray(CB[l]))
        alpha = st[f"mu_pre_{l}"] / np.sqrt(st[f"var_{l}"])
        core = np.abs(alpha) < 1.0
        iu = np.triu_indices(dA.shape[0], 1)
        cc = (core[:, None] & core[None, :])[iu]
        C0 = np.abs(offd(pre_cov(m, l, Cch)))[iu]
        Dm = np.abs(st[f"D_{l}"])
        Ds = (Dm + Dm.T)[iu]
        rows = {}
        for sel, oth in ((dA, dB), (dB, dA)):
            v = np.abs(sel[iu])
            o = oth[iu]
            for q in qs:
                k = int(q * v.size)
                top = np.argpartition(-v, k)[:k]
                r = rows.setdefault(q, dict(cc=[], c0q=[], dq=[], energy_other=[]))
                r["cc"].append(float(cc[top].mean() / cc.mean()))
                r["c0q"].append(float((C0[top][:, None] > np.quantile(C0, 0.9)).mean() / 0.1))
                r["dq"].append(float((Ds[top][:, None] > np.quantile(Ds, 0.9)).mean() / 0.1))
                r["energy_other"].append(float((o[top] ** 2).sum() / (o ** 2).sum()))
        out[l] = {str(q): {k: float(np.mean(x)) for k, x in r.items()} for q, r in rows.items()}
        out[l]["core_frac"] = float(core.mean())
        print(f"MLP {m} L{l}: core {core.mean():.2f}; " + " | ".join(
            f"top {100 * q:.0f}%: core x core x{out[l][str(q)]['cc']:.2f}, |C0| top-decile x{out[l][str(q)]['c0q']:.2f}, "
            f"|D| top-decile x{out[l][str(q)]['dq']:.2f}, other-seed energy {out[l][str(q)]['energy_other']:.3f}"
            for q in qs), flush=True)
    (OUT / f"who_{m}.json").write_text(json.dumps(out, indent=1))


# ----------------------------------------------------------------------------------------- (c) entrywise prediction
def design(m, l):
    Cch, _ = t2.rec(m)
    st = state(m)
    T, X = terms(st, l, pre_cov(m, l, Cch))
    feats = dict(T)
    feats.update(X)
    feats["Cpost"] = offd(Cch[l])
    feats["Cpost^2"] = offd(np.asarray(Cch[l]) ** 2)
    # gain-type features: K o (x_j + x_k) for per-neuron quantities x the chain holds (a per-neuron pair-coefficient
    # error that is a linear function of the neuron's own state -- the production form of the gain model)
    K = feats["Cpost"]
    mu, var, t, K4d = (st[f"{k}_{l}"] for k in ("mu_pre", "var", "t", "K4d"))
    sd_ = np.sqrt(var)
    for name, x in (("alpha", mu / sd_), ("logvar", np.log(var)), ("skew", t / sd_ ** 3), ("kurt", K4d / var ** 2),
                    ("|alpha|", np.abs(mu / sd_))):
        feats["K(" + name + ")"] = K * (x[:, None] + x[None, :])
    sA, sB = ce.SEEDS
    a, b = np.asarray(ce.cov(m, sA)[l]), np.asarray(ce.cov(m, sB)[l])
    dA, dB = offd(Cch[l] - a), offd(Cch[l] - b)
    iu = np.triu_indices(a.shape[0], 1)
    F = np.column_stack([feats[k][iu] for k in feats])
    return list(feats), F, dA[iu], dB[iu]


def r2(y, yhat):
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def feat():
    out = {}
    for l in LS:
        names, F0, a0, b0 = design(0, l)
        _, F1, a1, b1 = design(1, l)
        res = {}
        for m, F, a, b in ((0, F0, a0, b0), (1, F1, a1, b1)):
            y = 0.5 * (a + b)
            rel = float(np.corrcoef(a, b)[0, 1])                 # split-half reliability of dC entries
            ceil = 2 * rel / (1 + rel)                           # Spearman-Brown: ceiling of R^2 against the average
            sd = F.std(0)
            Fs = np.column_stack([np.ones(len(y)), F / np.where(sd > 0, sd, 1)])
            beta, *_ = np.linalg.lstsq(Fs, y, rcond=None)
            # per-feature single correlations
            single = {n: float(np.corrcoef(F[:, i], y)[0, 1]) if sd[i] > 0 else 0.0 for i, n in enumerate(names)}
            # cross-seed honesty: fit on seed A's dC, score on seed B's
            bA, *_ = np.linalg.lstsq(Fs, a, rcond=None)
            res[m] = dict(r2_in=r2(y, Fs @ beta), r2_cross_seed=r2(b, Fs @ bA), reliability=rel, ceiling=ceil,
                          single_corr=single, beta=[float(x) for x in beta], sd=[float(x) for x in sd])
        # out of sample: the MLP-0 model (on raw features) applied to MLP 1
        y0, y1 = 0.5 * (a0 + b0), 0.5 * (a1 + b1)
        G0 = np.column_stack([np.ones(len(y0)), F0])
        G1 = np.column_stack([np.ones(len(y1)), F1])
        g, *_ = np.linalg.lstsq(G0, y0, rcond=None)
        res["oos_0to1"] = r2(y1, G1 @ g)
        out[l] = dict(names=names, **{str(k): v for k, v in res.items()})
        print(f"L{l}: R^2 in-sample MLP0 {res[0]['r2_in']:.3f} (cross-seed {res[0]['r2_cross_seed']:.3f}, ceiling "
              f"{res[0]['ceiling']:.3f}) MLP1 {res[1]['r2_in']:.3f} (ceiling {res[1]['ceiling']:.3f}) | out of sample "
              f"0->1 {res['oos_0to1']:.3f} | best single corr MLP0: "
              + ", ".join(f"{n} {v:+.2f}" for n, v in sorted(res[0]["single_corr"].items(), key=lambda x: -abs(x[1]))[:4]),
              flush=True)
    (OUT / "feat.json").write_text(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------ the per-neuron gain model of dC0
def gain_fit(K, dC0):
    """delta minimising sum_{j != k} (dC0_jk - K_jk (delta_j + delta_k))^2 (K, dC0 symmetric, zero diagonal)."""
    K2 = K * K
    A = np.diag(K2.sum(1)) + K2
    return np.linalg.solve(A, (K * dC0).sum(1))


def gain(m):
    Cch, _ = t2.rec(m)
    sA, sB = ce.SEEDS
    CA, CB = ce.cov(m, sA), ce.cov(m, sB)
    out = {}
    for l in range(L - 1):
        a, b = np.asarray(CA[l]), np.asarray(CB[l])
        K = offd(Cch[l])
        dA, dB = offd(Cch[l] - a), offd(Cch[l] - b)
        dP, N = 0.5 * (dA + dB), 0.5 * offd(a - b)
        res = {}
        for tag, X in (("P", dP), ("noise", N), ("A", dA), ("B", dB)):
            dl = gain_fit(K, X)
            G = K * (dl[:, None] + dl[None, :])
            res[tag] = dict(E=float((X * X).sum()), E_expl=float((X * X).sum() - ((X - G) ** 2).sum()),
                            delta_rms=float(np.sqrt((dl ** 2).mean())))
        # cross-seed honesty: delta fitted on A, scored on B's independent noise
        dl = gain_fit(K, dA)
        G = K * (dl[:, None] + dl[None, :])
        res["A_on_B"] = float(((dB * dB).sum() - ((dB - G) ** 2).sum()) / max((dB * dB).sum(), 1e-300))
        S = res["P"]["E"] - res["noise"]["E"]
        res["signal_share"] = float((res["P"]["E_expl"] - res["noise"]["E_expl"]) / S) if S > 0 else None
        out[l] = res
        print(f"MLP {m} L{l:2d}: |dC0_P|^2 {res['P']['E']:.3e} (noise {res['noise']['E']:.3e}) | gain model explains "
              f"{res['P']['E_expl'] / max(res['P']['E'], 1e-300):.3f} of P, "
              f"{res['noise']['E_expl'] / max(res['noise']['E'], 1e-300):.3f} of noise -> "
              f"of the signal {res['signal_share'] if res['signal_share'] is None else round(res['signal_share'], 3)} | "
              f"fit on A scored on B {res['A_on_B']:.3f} | rms delta {res['P']['delta_rms']:.1e}", flush=True)
    (OUT / f"gain_{m}.json").write_text(json.dumps(out, indent=1))


def gain_arm(m, set_, var):
    f = OUT / f"gainarm_{m}_{set_}_{var}.json"
    if f.exists():
        return json.loads(f.read_text())
    mats = _mats(m, t2.SETS[set_])

    def hook(l, C):
        if l not in mats:
            return C
        A, B = mats[l]
        X = ({"A": A, "B": B, "P": 0.5 * (A + B)}[var] - C).numpy()
        np.fill_diagonal(X, 0.0)
        K = C.numpy().copy()
        np.fill_diagonal(K, 0.0)
        dl = gain_fit(K, X)
        G = torch.tensor(K * (dl[:, None] + dl[None, :]))
        return C + G

    mse = _run(m, hook)
    res = dict(mse=mse, x=mse / t2.b9_ref(m), mlp=m, set=set_, var=var)
    f.write_text(json.dumps(res))
    print(f"gainarm MLP {m} {set_} {var}: x{res['x']:.4f}", flush=True)
    return res


# ------------------------------------------------------------------------------------- born at the step vs carried
def born_run(m, l, seed):
    """offdiag of the chain's post C at layer l when its C_pre(l) is the exact one of `seed` (layer l-1 oracle)."""
    Ws, _ = ce.ta.mlp(m)
    Cin = torch.tensor(np.asarray(ce.cov(m, seed)[l - 1]))
    out = {}

    def pre(k, Cp):
        return Ws[k].T @ Cin @ Ws[k] if k == l else Cp

    def post(k, C):
        if k == l:
            out["C"] = C.clone()
            raise StopIteration
        return C

    try:
        run_chain(Ws, prec=pre, postc=post, **ce.B9)
    except StopIteration:
        pass
    return out["C"].numpy()


def born(m):
    f = OUT / f"born_{m}.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    Cch, _ = t2.rec(m)
    sA, sB = ce.SEEDS
    for l in range(1, L - 1):
        if str(l) in res:
            continue
        a, b = np.asarray(ce.cov(m, sA)[l]), np.asarray(ce.cov(m, sB)[l])
        bA = offd(born_run(m, l, sA) - a)
        bB = offd(born_run(m, l, sB) - b)
        tA, tB = offd(Cch[l] - a), offd(Cch[l] - b)
        bP, tP = 0.5 * (bA + bB), 0.5 * (tA + tB)
        nb, nt = float(((bA - bB) ** 2).sum() / 4), float(((tA - tB) ** 2).sum() / 4)
        Sb, St = float((bP * bP).sum()) - nb, float((tP * tP).sum()) - nt
        # the carried part = total - born; its signal energy and the cross term
        cP = tP - bP
        nc = float((((tA - bA) - (tB - bB)) ** 2).sum() / 4)
        Sc = float((cP * cP).sum()) - nc
        res[str(l)] = dict(S_total=St, S_born=Sb, S_carried=Sc, N_total=nt, N_born=nb, N_carried=nc,
                           corr_born_total=float((bP * tP).sum() / np.sqrt((bP * bP).sum() * (tP * tP).sum())))
        print(f"MLP {m} L{l:2d}: signal |dC0|^2 total {St:.3e} | born {Sb:.3e} ({100 * Sb / St:5.1f}%) carried {Sc:.3e} "
              f"({100 * Sc / St:5.1f}%) | noise total {nt:.2e} born {nb:.2e}", flush=True)
        f.write_text(json.dumps(res, indent=1))


def feat_born(m):
    """Entrywise linear model of the BORN part (local step error, same-seed differenced) on the local design features;
    in sample, R^2 against the A/B split-half ceiling.                                     -> res3/featborn_<M>.json"""
    out = {}
    sA, sB = ce.SEEDS
    for l in LS:
        names, F, _, _ = design(m, l)
        iu = np.triu_indices(F.shape[0] and int(round((1 + np.sqrt(1 + 8 * F.shape[0])) / 2)), 1)
        bA = offd(born_run(m, l, sA) - np.asarray(ce.cov(m, sA)[l]))[iu]
        bB = offd(born_run(m, l, sB) - np.asarray(ce.cov(m, sB)[l]))[iu]
        y = 0.5 * (bA + bB)
        rel = float(np.corrcoef(bA, bB)[0, 1])
        sd = F.std(0)
        Fs = np.column_stack([np.ones(len(y)), F / np.where(sd > 0, sd, 1)])
        beta, *_ = np.linalg.lstsq(Fs, y, rcond=None)
        bAf, *_ = np.linalg.lstsq(Fs, bA, rcond=None)
        single = {n: float(np.corrcoef(F[:, i], y)[0, 1]) if sd[i] > 0 else 0.0 for i, n in enumerate(names)}
        out[l] = dict(r2_in=r2(y, Fs @ beta), r2_cross_seed=r2(bB, Fs @ bAf), reliability=rel,
                      ceiling=2 * rel / (1 + rel), single_corr=single)
        print(f"MLP {m} L{l}: BORN dC0: R^2 in {out[l]['r2_in']:.3f} (A->B {out[l]['r2_cross_seed']:.3f}, ceiling "
              f"{out[l]['ceiling']:.3f}) | best single: " + ", ".join(
                  f"{n} {v:+.2f}" for n, v in sorted(single.items(), key=lambda x: -abs(x[1]))[:4]), flush=True)
    (OUT / f"featborn_{m}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "state":
        state(int(sys.argv[2]))
        check_terms(int(sys.argv[2]))
    elif mode == "spec":
        spec(int(sys.argv[2]))
    elif mode == "rank":
        rank_arm(int(sys.argv[2]), sys.argv[3], int(sys.argv[4]), sys.argv[5])
    elif mode == "sparse":
        sparse_arm(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), sys.argv[5])
    elif mode == "who":
        who(int(sys.argv[2]))
    elif mode == "feat":
        feat()
    elif mode == "featborn":
        feat_born(int(sys.argv[2]))
    elif mode == "born":
        born(int(sys.argv[2]))
    elif mode == "gain":
        gain(int(sys.argv[2]))
    elif mode == "gainarm":
        gain_arm(int(sys.argv[2]), sys.argv[3], sys.argv[4])
