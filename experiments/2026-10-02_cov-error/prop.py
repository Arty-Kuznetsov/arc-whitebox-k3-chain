"""The PROPORTIONAL part of the off-diagonal post-covariance error, dC0(l) ~ beta(l) C0_chain(l), and what removing it
buys. W_{l+1} is independent of dC(l) (both C_chain(l) and C_exact(l) depend on W_0..W_l only), so across neurons i
    e_off,i = w_i^T dC0 w_i,  v_off,i = w_i^T C0 w_i   (w_i = column i of W_{l+1}) have Cov = (8/n^2) <dC0, C0>_F,
and the regression coefficient of the variance error e(l+1) on v_off(l+1) estimates beta(l) = <dC0, C0>_F / |C0|_F^2
from per-neuron VARIANCE oracles alone (v_diag = sum_j W_ji^2 C_jj enters as a nuisance regressor).

    fit SRC      beta(l), l = 0..14, per MLP, from the variance oracles of SRC: "mac" (the Mac's two 1e7 seeds) or
                 "cov" (the matrix projection <dC0_P, C0>/|C0|^2 straight from the covariance oracles)
                 -> res2/prop_beta_<SRC>.json
    arm M SRC FROM [SCHED]   online arm on MLP M: at every layer l = 0..14, C <- C - s(l) offdiag(C) on the LIVE
                 trajectory; FROM = 0 / 1 (the MLP whose beta is used: in sample or transferred) or "avg";
                 SCHED: "full" s = beta (the accumulated error, applied at every layer: double-counts, NEGATIVE 57),
                 "inc" s = beta(l) - beta(l-1) (the increment), "half" = inc / 2, "neg" = -inc (sign control),
                 "last" s = beta(14) at l = 14 only        -> res2/prop_<M>_<SRC>_<FROM>[_<SCHED>].json"""
import json
import sys

import numpy as np
import torch

import ce
import t2
from chain import run_chain

OUT = ce.HERE / "res2"
L = ce.L


def fit(src):
    out = {}
    for m in (0, 1):
        Wn, _ = ce.wn(m)
        Cch, r = t2.rec(m)
        if src == "mac":
            vs = [np.load(f)["var"] for f in ce.MAC[m]]
        beta, share = [], []
        for l in range(L - 1):
            C = np.asarray(Cch[l])
            C0 = C - np.diag(np.diag(C))
            if src == "mac":
                W = Wn[l + 1]
                vd = (W * W).T @ np.diag(C)
                vo = np.einsum("ji,ji->i", W, C0 @ W)
                e = [r["var"][l + 1] - v[l + 1] for v in vs]
                eP, dn = 0.5 * (e[0] + e[1]), e[0] - e[1]
                X = np.column_stack([np.ones_like(vd), vd, vo])
                b, *_ = np.linalg.lstsq(X, eP, rcond=None)
                beta.append(float(b[2]))
                S = float(eP @ eP - dn @ dn / 4)
                bo, *_ = np.linalg.lstsq(np.column_stack([np.ones_like(vo), vo]), eP, rcond=None)
                ro = eP - np.column_stack([np.ones_like(vo), vo]) @ bo
                share.append(float((eP @ eP - ro @ ro) / S) if S > 0 else None)
            else:
                sA, sB = ce.SEEDS
                dP = C - 0.5 * (np.asarray(ce.cov(m, sA)[l]) + np.asarray(ce.cov(m, sB)[l]))
                np.fill_diagonal(dP, 0.0)
                beta.append(float((dP * C0).sum() / (C0 * C0).sum()))
                share.append(float((dP * C0).sum() ** 2 / ((C0 * C0).sum() * (dP * dP).sum())))
        out[m] = dict(beta=beta, share=share)
        print(f"MLP {m} [{src}] beta(l): " + " ".join(f"{b:+.1e}" for b in beta), flush=True)
        print(f"   explained share: " + " ".join("—" if s is None else f"{s:.2f}" for s in share), flush=True)
    (OUT / f"prop_beta_{src}.json").write_text(json.dumps(out, indent=1))
    return out


def arm(m, src, frm, sched="full"):
    f = OUT / (f"prop_{m}_{src}_{frm}.json" if sched == "full" else f"prop_{m}_{src}_{frm}_{sched}.json")
    if f.exists():
        return json.loads(f.read_text())
    B = json.loads((OUT / f"prop_beta_{src}.json").read_text())
    beta = (np.array(B["0"]["beta"]) + np.array(B["1"]["beta"])) / 2 if frm == "avg" else np.array(B[str(frm)]["beta"])
    inc = np.diff(np.concatenate([[0.0], beta]))
    beta = {"full": beta, "inc": inc, "half": 0.5 * inc, "neg": -inc,
            "last": np.where(np.arange(len(beta)) == len(beta) - 1, beta, 0.0)}[sched]

    def hook(l, C):
        C0 = C - torch.diag(torch.diagonal(C))
        return C - float(beta[l]) * C0

    Ws, g = ce.ta.mlp(m)
    mu, _ = run_chain(Ws, postc=hook, **ce.B9)
    mse = float(((mu[-1] - g) ** 2).mean())
    res = dict(mse=mse, x=mse / t2.b9_ref(m), mlp=m, src=src, frm=frm, sched=sched)
    f.write_text(json.dumps(res))
    print(f"prop MLP {m} beta[{src}] from {frm} sched {sched}: x{res['x']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    if sys.argv[1] == "fit":
        fit(sys.argv[2])
    else:
        arm(int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else "full")
