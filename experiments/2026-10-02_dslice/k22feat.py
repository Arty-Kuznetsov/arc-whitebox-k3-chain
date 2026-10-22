"""How predictable is the TRUE K22 = kappa4[i,i,j,j] from pair features the chain holds at the same layer? (O(n^2), so
any closure built from them is free.) The closure uses two (var_i var_j, C0_ij^2) and reaches corr 0.62; the oracle's
pattern projected on them has R2 0.36-0.41. Pair features at layer l are projections through the SAME W_l as K22's
rough part, so unlike per-neuron features they can carry it. Ridge on the alive block, layers 9 and 12:
in sample (fit and score on one MLP) and out of sample (fit on the other MLP, same layer), target = oracle K22 (P).
    PYTHONPATH=$ARC/src:. uv run --project $ARC python k22feat.py <fit_mlp> [<test_mlp>]
"""
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dslice as S                                    # noqa: E402


def feats(info, l):
    var = info[l]["var"].numpy().astype(np.float64)
    mu = info[l]["mu_pre"].numpy().astype(np.float64)
    sd = np.sqrt(var); a = mu / sd
    C = info[l]["C"].numpy().astype(np.float64); C0 = C - np.diag(np.diag(C))
    D = info[l]["D"].numpy().astype(np.float64)
    t = info[l]["t"].numpy().astype(np.float64)
    K4 = info[l]["K4d"].numpy().astype(np.float64)
    M = np.outer(a >= -2.5, a >= -2.5); np.fill_diagonal(M, False)
    vv = np.outer(var, var); rho = C0 / np.sqrt(vv)
    cols = {
        "var_i*var_j": vv, "C0^2": C0 * C0, "C0": C0, "|C0|": np.abs(C0), "rho^2*vv": rho * rho * vv,
        "D_ij": D, "D_ji": D.T, "D_ij^2": D * D, "D_ji^2": D.T * D.T, "D_ij*D_ji": D * D.T,
        "C0*D_ij": C0 * D, "C0*D_ji": C0 * D.T, "C0^3": C0 ** 3, "C0^4/vv": C0 ** 4 / vv,
        "t_i*t_j": np.outer(t, t), "t_i*C0": t[:, None] * C0, "t_j*C0": t[None, :] * C0,
        "sqrt(K4_i K4_j)": np.sqrt(np.abs(np.outer(K4, K4))), "K4_i*var_j+K4_j*var_i": np.outer(K4, var) + np.outer(var, K4),
        "vv*a_i": vv * a[:, None], "vv*a_j": vv * a[None, :], "vv*a_i*a_j": vv * np.outer(a, a),
        "C0^2*a_i*a_j": C0 * C0 * np.outer(a, a),
    }
    names = list(cols)
    X = np.stack([cols[k][M] for k in names], 1)
    return X, M, names


def target(m, l):
    A, B = (np.load(S.HERE / "oracles" / f"d_{m}_s{s}.npz") for s in S.SEEDS)
    KA, KB = A[f"K22_{l}"].astype(np.float64), B[f"K22_{l}"].astype(np.float64)
    return 0.5 * (KA + KB), KA, KB


def fit(X, y, lam=1e-6):
    mu, sdv = X.mean(0), X.std(0); sdv[sdv == 0] = 1
    Z = (X - mu) / sdv
    b = np.linalg.solve(Z.T @ Z + lam * len(y) * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return lambda Xn: ((Xn - mu) / sdv) @ b + y.mean()


def r2(y, p):
    return 1 - float(((y - p) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())


def main(fm, tm=None):
    data = {}
    for m in {fm, tm} - {None}:
        Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
        _, info = S.run(Ws, m, record="full")
        data[m] = info
    for l in S.LD:
        X, M, names = feats(data[fm], l)
        KP, KA, KB = target(fm, l)
        y = KP[M]
        ceil = 1 - float(((KA[M] - KB[M]) ** 2).mean() / 4) / float(np.var(y))   # noise ceiling of the 2-seed mean
        chain = data[fm][l]["K22"].numpy().astype(np.float64)[M]
        print(f"layer {l}, fit on MLP {fm}: noise ceiling R2 {ceil:.3f} | the chain's closure as a predictor: corr^2 "
              f"{np.corrcoef(chain, y)[0, 1] ** 2:.3f}", flush=True)
        for label, idx in (("ansatz (vv, C0^2)", [0, 1]), ("all features", list(range(len(names))))):
            f = fit(X[:, idx], y)
            msg = f"   {label:20s} in sample R2 {r2(y, f(X[:, idx])):.3f}"
            if tm is not None:
                Xt, Mt, _ = feats(data[tm], l)
                yt = target(tm, l)[0][Mt]
                msg += f" | fit {fm} -> test {tm}: R2 {r2(yt, f(Xt[:, idx])):.3f}"
            print(msg, flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else None)
