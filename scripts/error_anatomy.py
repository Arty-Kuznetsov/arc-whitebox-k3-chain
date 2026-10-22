"""Error anatomy of any ARC White-Box estimator, from its outputs alone (write-up §6.4).

Runs an estimator.py (a whestbench BaseEstimator) on the public 1024 x 16 networks that have the dataset's own
1e9-sample ground truth of the post-activation means at every layer (scripts/fetch_baked.py puts them in
experiments/2026-09-16_kprop-arc/baked/), and reports where its error is and how much of it a per-neuron correction
could ever remove:
  1. the final-layer MSE and its COHERENT share (the level shared by all neurons): the most a per-neuron level
     correction can buy, since the rest is independent across neurons (Proposition 1);
  2. the coherent share at every layer;
  3. the parity check of Proposition 2 at order one: the pre-activation mean is linear, so its truth is W^T (true means
     of the previous layer) and the estimator's own is W^T (its means of the previous layer); the error is a random
     projection with no coherent level, z = mean / SE ~ N(0, 1) at every layer if the estimator propagates its means
     through the true weights;
  4. the share of the final error a per-neuron model of the estimator's own outputs learns out of sample (ridge on
     smooth features of the neuron's predicted mean, its implied pre-activation mean and its weight-column norm;
     leave one network out).
Nothing here sees the estimator's internals, so (4) is a lower bound for a corrector with internal state; (1) is the
upper bound for any per-neuron level correction.
usage (whest env):  uv run --project external/whest-starterkit python scripts/error_anatomy.py src/estimator_k3z.py [nets=0-9]
"""
import sys, json, time
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from whestbench import SetupContext
from whestbench.domain import MLP
from whestbench.loader import load_estimator_from_path

ROOT = Path(__file__).resolve().parent.parent
BAKED = ROOT / "experiments" / "2026-09-16_kprop-arc" / "baked"


def predict_all(est, W):
    L, n = W.shape[0], W.shape[1]
    mlp = MLP(width=n, depth=L, weights=[fnp.asarray(np.ascontiguousarray(W[i]).astype(np.float32)) for i in range(L)], seed=0)
    with flops.BudgetContext(flop_budget=2 ** 41, quiet=True) as ctx:
        out = est.predict(mlp, 2 ** 41)
        arr = np.asarray(out.tolist(), dtype=np.float64)
        used = ctx.flops_used
    return arr, used


def ridge_fit(X, y, lam=1e-6):
    m, sd = X.mean(0), X.std(0) + 1e-12
    Z = np.hstack([np.ones((len(X), 1)), (X - m) / sd])
    b = np.linalg.solve(Z.T @ Z + lam * len(X) * np.eye(Z.shape[1]), Z.T @ y)
    return lambda Xn: np.hstack([np.ones((len(Xn), 1)), (Xn - m) / sd]) @ b


def main():
    est_path = sys.argv[1]
    lo, hi = (sys.argv[2].split("-") if len(sys.argv) > 2 else ("0", "9"))
    nets = list(range(int(lo), int(hi) + 1))
    gt_all = np.load(BAKED / "gt.npy")
    est, _ = load_estimator_from_path(Path(est_path))
    est.setup(SetupContext(width=1024, depth=16, flop_budget=2 ** 41, api_version="2.0", seed=0))
    per = []
    for k in nets:
        W = np.load(BAKED / f"weights_{k}.npy").astype(np.float64)
        t0 = time.time(); P, used = predict_all(est, W); gt = gt_all[k].astype(np.float64)
        L = P.shape[0]
        e = P - gt
        cs = [float(np.mean(e[l]) ** 2 / np.mean(e[l] ** 2)) for l in range(L)]
        z_mu = [float(np.mean(W[l].T @ e[l - 1]) / np.sqrt(np.mean((W[l].T @ e[l - 1]) ** 2) / W.shape[1])) for l in range(1, L)]
        mu_hat = W[L - 1].T @ P[L - 2]
        X = np.stack([P[L - 1], np.sqrt(np.maximum(P[L - 1], 0)), mu_hat, mu_hat * mu_hat, np.linalg.norm(W[L - 1], axis=0),
                      P[L - 1] * mu_hat], 1)
        per.append(dict(net=k, flops=used, final_mse=float(np.mean(e[-1] ** 2)), cs=cs, z_mu=z_mu, X=X, e=e[-1]))
        print(f"net {k}: final MSE {per[-1]['final_mse']:.3e}, coherent share {100*cs[-1]:.2f}%, util {100*used/2**41:.2f}%, "
              f"{time.time()-t0:.0f}s", flush=True)
    lf = []
    for i, r in enumerate(per):
        tr = [q for j, q in enumerate(per) if j != i]
        if not tr: break
        f = ridge_fit(np.vstack([q["X"] for q in tr]), np.concatenate([q["e"] for q in tr]))
        lf.append(1 - float(np.mean((r["e"] - f(r["X"])) ** 2) / np.mean(r["e"] ** 2)))
    z = np.array([r["z_mu"] for r in per])
    out = dict(estimator=est_path, nets=nets, final_mse=float(np.mean([r["final_mse"] for r in per])),
               coherent_share_final=float(np.mean([r["cs"][-1] for r in per])),
               coherent_share_by_layer=np.mean([r["cs"] for r in per], 0).tolist(),
               parity_z_mean_sq=float(np.mean(z ** 2)), parity_z_max=float(np.max(np.abs(z))),
               learnable_share_loo=float(np.mean(lf)) if lf else None, learnable_share_loo_per_net=lf)
    print(json.dumps({k: v for k, v in out.items() if k != "coherent_share_by_layer"}, indent=1))
    print("coherent share by layer:", [round(x, 4) for x in out["coherent_share_by_layer"]])
    name = Path(est_path).stem
    (ROOT / "experiments" / "2026-10-13_coherence-law" / "res").mkdir(parents=True, exist_ok=True)
    json.dump(out, open(ROOT / "experiments" / "2026-10-13_coherence-law" / "res" / f"anatomy_{name}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
