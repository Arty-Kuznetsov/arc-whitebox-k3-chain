"""Task 2 of cloud #11: is the OFF-DIAGONAL post-covariance error dC = C_chain - C_exact worth anything?

    rec   M         the chain's own post covariance on B9 (postc recording hook, bitwise B9), layers 0..14
                    -> res2/base_C_<M>.npy (local), res2/base_rec_<M>.npz (var, mu_pre per layer)
    split M         (a) the next layer's variance error e_i(l+1) = sum_jk W_ji W_ki dC_jk(l), split into the DIAGONAL
                    part sum_j W_ji^2 dC_jj and the OFF-DIAGONAL part, per layer, all neurons and the kink core
                    (|alpha_chain(l+1)| < 1); oracle noise subtracted with the two seeds -> res2/split_<M>.json
    arm M SET PART VAR
                    (b) the chain payoff: at the layers of SET ("s4" = 5, 8, 11, 14 as in the brief; "all" = 0..14) the
                    post covariance C (the only thing feeding the next layer's C_pre = W^T C W) gets
                        PART "all": C := C_or           "diag": diag(C) := diag(C_or)     "off": offdiag(C) := offdiag(C_or)
                    absolute values, on the LIVE trajectory (so it stays self-consistent); VAR = the oracle:
                        "A", "B" = one 1e7 seed; "P" = their average (2e7);
                        "N" = NOISE CONTROL: C += PART of (C_A - C_B) / 2 -- zero mean, the noise variance of "P" --
                              its raw ratio is what the pooled oracle's own sampling noise costs the chain.
                    PART "diagr" / "allr": as diag / all, but the diagonal keeps the chain's own LEVEL (mean).
                    -> res2/arm_<M>_<SET>_<PART>_<VAR>.json (mse, x vs B9)

Oracles: ce.cov(m, seed), seeds ce.SEEDS = (2101, 2102), from cov_oracle.py (1e7 each)."""
import json
import sys

import numpy as np
import torch

import ce
from chain import run_chain

OUT = ce.HERE / "res2"
OUT.mkdir(exist_ok=True)
L = ce.L
SETS = {"s4": (5, 8, 11, 14), "all": tuple(range(L - 1)), "deep": tuple(range(8, L - 1)),
        **{f"l{l}": (l,) for l in range(L - 1)}}  # single layers: where the payoff is born


def b9_ref(m):
    return json.loads((ce.ta.RES / "B9" / f"m{m}.json").read_text())["mse"]


def rec(m):
    fC, fr = OUT / f"base_C_{m}.npy", OUT / f"base_rec_{m}.npz"
    if fC.exists() and fr.exists():
        return np.load(fC, mmap_mode="r"), dict(np.load(fr))
    Ws, g = ce.ta.mlp(m)
    Cs = {}

    def hook(l, C):
        Cs[l] = C.clone()
        return C

    mu, info = run_chain(Ws, record=True, postc=hook, **ce.B9)
    mse = float(((mu[-1] - g) ** 2).mean())
    np.save(fC, np.stack([Cs[l].numpy() for l in range(L - 1)]))
    np.savez(fr, var=np.stack([x["var"].numpy() for x in info]), mu_pre=np.stack([x["mu_pre"].numpy() for x in info]),
             post_mean=mu.numpy(), mse=mse)
    print(f"MLP {m}: recorded; mse {mse:.6e} (B9 res {b9_ref(m):.6e})", flush=True)
    return np.load(fC, mmap_mode="r"), dict(np.load(fr))


def parts(W, dC):
    """(e_diag, e_off) of the next-layer variance error for one post-covariance error matrix dC (float64)."""
    dd = np.diagonal(dC).copy()
    e_diag = (W * W).T @ dd
    e_all = np.einsum("ji,ji->i", W, dC @ W)        # diag(W^T dC W)
    return e_diag, e_all - e_diag


def split(m):
    Cch, r = rec(m)
    Wn, _ = ce.wn(m)
    sA, sB = ce.SEEDS
    CA, CB = ce.cov(m, sA), ce.cov(m, sB)
    # 4-seed total variance error (Mac's two seeds + the two covariance seeds), for the cross-check of e_total
    vs = ce.var_seeds(m, Wn)
    layers = []
    for l in range(L - 1):
        W = Wn[l + 1]
        Cc = np.asarray(Cch[l])
        eA = parts(W, Cc - np.asarray(CA[l]))
        eB = parts(W, Cc - np.asarray(CB[l]))
        alpha = r["mu_pre"][l + 1] / np.sqrt(r["var"][l + 1])
        core = np.abs(alpha) < 1.0
        v = r["var"][l + 1]
        row = dict(l=l, n_core=int(core.sum()))
        for tag, mask in (("all", np.ones_like(core)), ("core", core)):
            d_P = 0.5 * (eA[0] + eB[0])[mask]
            o_P = 0.5 * (eA[1] + eB[1])[mask]
            d_D = (eA[0] - eB[0])[mask]
            o_D = (eA[1] - eB[1])[mask]
            t_P, t_D = d_P + o_P, d_D + o_D
            E = lambda a, b: float(a @ b)
            res = dict(
                E_diag=E(d_P, d_P), E_off=E(o_P, o_P), E_tot=E(t_P, t_P), X=E(d_P, o_P),
                N_diag=E(d_D, d_D) / 4, N_off=E(o_D, o_D) / 4, N_tot=E(t_D, t_D) / 4, NX=E(d_D, o_D) / 4,
                V=E(v[mask], v[mask]))
            res["S_diag"] = res["E_diag"] - res["N_diag"]
            res["S_off"] = res["E_off"] - res["N_off"]
            res["S_tot"] = res["E_tot"] - res["N_tot"]
            res["S_X"] = res["X"] - res["NX"]
            # the same total from the 4-seed variance oracle
            e4 = np.array([r["var"][l + 1] - s[l + 1] for s in vs])[:, mask]
            res["E_tot4"] = float((e4.mean(0) ** 2).sum())
            res["N_tot4"] = float(e4.var(0, ddof=1).sum() / e4.shape[0])
            res["S_tot4"] = res["E_tot4"] - res["N_tot4"]
            row[tag] = res
        layers.append(row)
        a, c = row["all"], row["core"]
        print(f" L{l:2d}->{l + 1:2d} rel.rms e {np.sqrt(a['E_tot'] / a['V']):.2e} | signal: diag {a['S_diag']:.2e} "
              f"off {a['S_off']:.2e} 2x {2 * a['S_X']:+.2e} tot {a['S_tot']:.2e} (4-seed {a['S_tot4']:.2e}) | "
              f"noise/E tot {a['N_tot'] / a['E_tot']:.2f} | core: diag {c['S_diag']:.2e} off {c['S_off']:.2e} "
              f"2x {2 * c['S_X']:+.2e} tot {c['S_tot']:.2e}", flush=True)
    out = dict(mlp=m, layers=layers)
    for tag in ("all", "core"):
        tot = {k: sum(x[tag][k] for x in layers) for k in layers[0][tag]}
        out["sum_" + tag] = tot
        print(f"MLP {m} [{tag}] sum over layers: signal diag {tot['S_diag']:.3e} off {tot['S_off']:.3e} "
              f"cross 2x {2 * tot['S_X']:+.3e} total {tot['S_tot']:.3e} (4-seed {tot['S_tot4']:.3e}); "
              f"noise of total {tot['N_tot']:.3e} = {100 * tot['N_tot'] / tot['E_tot']:.1f}% of |e|^2", flush=True)
    (OUT / f"split_{m}.json").write_text(json.dumps(out, indent=1))
    return out


def oracle_mats(m, layers, var):
    """A / B: one seed; P: their average; N: (A - B)/2 (added, not substituted: the noise control);
    R0: the chain's own base-trajectory C (substituted: a reset to base, the self-check of R, ~x1);
    R: base C + (A - B)/2 (substituted: the noise control with the RESET dynamics of a substitution arm)."""
    sA, sB = ce.SEEDS
    CA = ce.cov(m, sA) if var not in ("B", "R0") else None
    CB = ce.cov(m, sB) if var not in ("A", "R0") else None
    Cb = rec(m)[0] if var in ("R", "R0") else None
    out = {}
    for l in layers:
        if var == "A":
            x = np.asarray(CA[l])
        elif var == "B":
            x = np.asarray(CB[l])
        elif var == "R0":
            x = np.asarray(Cb[l])
        elif var == "R":
            x = np.asarray(Cb[l]) + 0.5 * (np.asarray(CA[l]) - np.asarray(CB[l]))
        else:
            a, b = np.asarray(CA[l]), np.asarray(CB[l])
            x = 0.5 * (a + b) if var == "P" else 0.5 * (a - b)
        out[l] = torch.tensor(x, dtype=torch.float64)
    return out


def make_postc(mats, part, mode):
    """part: all / diag / off, and diagr / allr = the same with the CHAIN'S OWN LEVEL of the diagonal kept (the exact
    diagonal shifted by the mean of (chain - exact): only its rough part is substituted)."""
    def f(l, C):
        X = mats.get(l)
        if X is None:
            return C
        dX, dC = torch.diag(torch.diagonal(X)), torch.diag(torch.diagonal(C))
        if mode == "sub":
            if part in ("diagr", "allr"):
                xd = torch.diagonal(X) + (torch.diagonal(C) - torch.diagonal(X)).mean()
                return (C - dC if part == "diagr" else X - dX) + torch.diag(xd)
            return {"all": X.clone(), "diag": C - dC + dX, "off": X - dX + dC}[part]
        if part in ("diagr", "allr"):                                      # mode "add" (the noise control)
            xd = torch.diagonal(X) - torch.diagonal(X).mean()
            return (C if part == "diagr" else C + X - dX) + torch.diag(xd)
        return {"all": C + X, "diag": C + dX, "off": C + X - dX}[part]     # mode "add"
    return f


def arm(m, set_, part, var):
    f = OUT / f"arm_{m}_{set_}_{part}_{var}.json"
    if f.exists():
        return json.loads(f.read_text())
    mats = oracle_mats(m, SETS[set_], var)
    Ws, g = ce.ta.mlp(m)
    mu, _ = run_chain(Ws, postc=make_postc(mats, part, "add" if var == "N" else "sub"), **ce.B9)
    mse = float(((mu[-1] - g) ** 2).mean())
    res = dict(mse=mse, x=mse / b9_ref(m), mlp=m, set=set_, part=part, var=var)
    f.write_text(json.dumps(res))
    print(f"arm MLP {m} {set_:3s} {part:4s} {var}: mse {mse:.4e}  x{res['x']:.4f}", flush=True)
    return res


def arm_ch(m, set_, part, var, ch):
    """Channel split of an arm: the substituted post C' at layer l reaches layer l+1 ONLY through
    ch = "var": the diagonal of C_pre(l+1) (the next variances; the pre off-diagonal stays the chain's own), or
    ch = "pair": the off-diagonal of C_pre(l+1) (the next variances stay the chain's own).
    The postc hook keeps C' aside and returns the chain's own C; prec at l+1 mixes W^T C W with W^T C' W."""
    f = OUT / f"armch_{m}_{set_}_{part}_{var}_{ch}.json"
    if f.exists():
        return json.loads(f.read_text())
    mats = oracle_mats(m, SETS[set_], var)
    sub = make_postc(mats, part, "add" if var == "N" else "sub")
    Ws, g = ce.ta.mlp(m)
    held = {}

    def post(l, C):
        if l in mats:
            held[l + 1] = sub(l, C)
        return C

    def pre(l, Cp):
        Cn = held.pop(l, None)
        if Cn is None:
            return Cp
        Cpn = Ws[l].T @ Cn @ Ws[l]
        if ch == "both":  # self-check: must reproduce the plain arm
            return Cpn
        if ch == "var":
            return Cp - torch.diag(torch.diagonal(Cp)) + torch.diag(torch.diagonal(Cpn))
        return Cpn - torch.diag(torch.diagonal(Cpn)) + torch.diag(torch.diagonal(Cp))

    mu, _ = run_chain(Ws, postc=post, prec=pre, **ce.B9)
    mse = float(((mu[-1] - g) ** 2).mean())
    res = dict(mse=mse, x=mse / b9_ref(m), mlp=m, set=set_, part=part, var=var, ch=ch)
    f.write_text(json.dumps(res))
    print(f"armch MLP {m} {set_:3s} {part:4s} {var} via {ch}: mse {mse:.4e}  x{res['x']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "rec":
        rec(int(sys.argv[2]))
    elif mode == "split":
        split(int(sys.argv[2]))
    elif mode == "arm":
        arm(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5])
    elif mode == "armch":
        arm_ch(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
