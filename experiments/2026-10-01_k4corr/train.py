"""Train the kappa4 corrector on the PRODUCTION trajectory, then validate it THROUGH the production estimator.

  python train.py fit   -> lambda by leave-one-MLP-out R2 in feature space; the final model on all 10 -> model.npy
  python train.py loo   -> for every held-out MLP h: the model fitted on the other 9, embedded in estimator_k3v,
                           run through flopscope on MLP h; raw vs k3v with the corrector off (= k3t, bitwise)
"""
import sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REC = np.load(HERE / "recorded.npy", allow_pickle=True).item()
for _extra in sorted(HERE.glob("recorded_*.npy")):
    REC.update(np.load(_extra, allow_pickle=True).item())
MLPS = sorted(REC)


def rows(i):
    for (l, F, k4, ora) in REC[i]:
        sc = np.abs(k4).mean() + 1e-30
        yield F, (k4 - ora) / sc


def fit(train_ids, lam):
    F = np.concatenate([F for i in train_ids for F, _ in rows(i)])
    y = np.concatenate([y for i in train_ids for _, y in rows(i)])
    mu, sd = F.mean(0), F.std(0); sd[sd == 0] = 1.0
    Fz = (F - mu) / sd
    b = np.linalg.solve(Fz.T @ Fz + lam * len(y) * np.eye(Fz.shape[1]), Fz.T @ (y - y.mean()))
    return b, y.mean(), mu, sd


def r2_loo(lam):
    num = den = 0.0
    for h in MLPS:
        b, b0, mu, sd = fit([i for i in MLPS if i != h], lam)
        for F, y in rows(h):
            p = ((F - mu) / sd) @ b + b0
            num += ((y - p) ** 2).sum(); den += ((y - y.mean()) ** 2).sum()
    return 1 - num / den


if sys.argv[1] == "fit":
    sc = {lam: r2_loo(lam) for lam in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)}
    print("held-out R2 (production trajectory, layers 8-14):", "  ".join(f"{k:g}:{v:.3f}" for k, v in sc.items()))
    lam = max(sc, key=sc.get)
    b, b0, mu, sd = fit(MLPS, lam)
    np.save(HERE / (sys.argv[2] if len(sys.argv) > 2 else "model.npy"), np.array({"lam": lam, "b": b, "b0": b0, "mu": mu, "sd": sd}, dtype=object),
            allow_pickle=True)
    print(f"final model on all {len(MLPS)} MLPs at lambda {lam:g}: |b| max {np.abs(b).max():.4f}, b0 {b0:+.5f}")
else:
    sys.path.insert(0, str(ROOT / "src"))
    import whestbench as wb
    import flopscope as flops
    import flopscope.numpy as fnp
    flops.configure(symmetry_warnings=False)
    import estimator_k3v as V
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
    S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    gt = np.load(ROOT / "experiments/2026-09-16_kprop-arc/baked/gt.npy")
    ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
    if not hasattr(ds, "column_names"):
        ds = ds["mini"]
    mlps = {i: m for i, m in enumerate(wb.iter_mlps(ds)) if i < 10}

    def raw_of(i, model):
        V._K4C = None if model is None else tuple(fnp.asarray(np.asarray(x, dtype=np.float32)) for x in model)
        est = V.Estimator(k4corr_s=S)
        with flops.BudgetContext(flop_budget=2 ** 50, wall_time_limit_s=3600.0, quiet=True) as ctx:
            out = np.asarray(est.predict(mlps[i], 2 ** 41), dtype=np.float64)
        return float(((out[-1] - gt[i][-1]) ** 2).mean()), ctx.flops_used

    ratios = []
    for h in MLPS:
        model = fit([i for i in MLPS if i != h], lam)
        r0, f0 = raw_of(h, None)
        r1, f1 = raw_of(h, (model[0], np.float32(model[1]), model[2], model[3]))
        ratios.append(r1 / r0)
        print(f"  held-out MLP {h}: PRODUCTION raw {r0:.4e} -> {r1:.4e}  x{r1/r0:.4f}   FLOPs x{f1/f0:.5f}", flush=True)
    ratios = np.array(ratios)
    print(f"PRODUCTION leave-one-out (lam {lam:g}, s {S}): mean x{ratios.mean():.4f}  median x{np.median(ratios):.4f}"
          f"  better {int((ratios < 1).sum())}/{len(ratios)}")
