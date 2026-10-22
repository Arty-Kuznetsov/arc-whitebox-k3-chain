"""Task 1 of Prompt A: regime attribution of B9's final-layer error. Regimes by the chain's own alpha = mu/sigma of the
layer: dead alpha < -2.5, kink |alpha| <= 2.5, sat alpha > 2.5. Substitutions are ABSOLUTE oracle values (never factors
measured on the base trajectory, NEGATIVE 57), for one regime at a time:
    split             where the error LANDS: B9's final MSE split by the output neuron's regime (+ regime fractions)
    inject R [mlps]   exact post-activation means (baked GT, 1e9 samples) of regime R at layers 0..L-2
    mc M S CH R       MC oracle (oracles/mc_M_sS.npz, S = a seed or "pool" = both seeds averaged) of channel CH
                      (var = pre-activation variance, t = kappa3 diagonal, k4 = kappa4 diagonal, joint = all three)
                      for regime R at layers 1..L-1 (layer 0 is Gaussian and exact in the chain: no oracle there)
Pair slices (C off-diagonal, D, i.e. the pair-covariance and kappa3-birth channels) have no valid oracle at 1e7 samples
(NEGATIVE 50): they are tested as term restrictions in task 2 instead. Output: res1/<name>.json"""
import json
import sys

import numpy as np
import torch

import ta
from chain import run_chain

REG = {"dead": (-1e9, -2.5), "kink": (-2.5, 2.5), "sat": (2.5, 1e9), "all": (-1e9, 1e9),
       "shdead": (-2.5, -1.0), "core": (-1.0, 1.0), "shsat": (1.0, 2.5)}  # the kink split: shoulders vs core
OUT = ta.HERE / "res1"
OUT.mkdir(exist_ok=True)
mode = sys.argv[1]


def mc_oracle(m, seed, ch, R):
    files = [ta.HERE / "oracles" / f"mc_{m}_s{s}.npz" for s in ((1001, 1002) if seed == "pool" else (int(seed),))]
    z = [np.load(f) for f in files]
    get = lambda k: [np.mean([zz[k][l] for zz in z], axis=0) for l in range(z[0][k].shape[0])]
    lay = range(1, z[0]["var"].shape[0])
    return {"var": get("var"), "k3": get("k3"), "k4": get("k4"), "regime": REG[R],
            "var_layers": lay if ch in ("var", "joint") else (), "k3_layers": lay if ch in ("t", "joint") else (),
            "k4_layers": lay if ch in ("k4", "joint") else ()}


if mode == "split":
    out = {}
    for m in ta.SCREEN:
        Ws, g = ta.mlp(m)
        mu, info = run_chain(Ws, record=True, **ta.B9)
        n = g.shape[0]
        a = info[-1]["mu_pre"] / info[-1]["var"].sqrt()
        e = (mu[-1] - g) ** 2
        masks = {"dead": a < -2.5, "kink": a.abs() <= 2.5, "sat": a > 2.5}
        fr = [[float((x["mu_pre"] / x["var"].sqrt() < -2.5).double().mean()),
               float((x["mu_pre"] / x["var"].sqrt() > 2.5).double().mean())] for x in info]
        out[m] = {"mse": float(e.mean()), "share": {R: float(e[k].sum() / e.sum()) for R, k in masks.items()},
                  "count": {R: int(k.sum()) for R, k in masks.items()}, "frac_dead_sat_per_layer": fr}
        print(m, out[m]["mse"], out[m]["share"], out[m]["count"], flush=True)
    (OUT / "split.json").write_text(json.dumps(out, indent=1))
elif mode == "inject":
    R = sys.argv[2]
    mlps = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else ta.SCREEN
    f = OUT / f"inject_{R}.json"
    out = json.loads(f.read_text()) if f.exists() else {}
    for m in mlps:
        Ws, g = ta.mlp(m)
        gt = ta._GT[ta.pb.BAKED][m]
        inj = [torch.tensor(gt[l]) for l in range(len(Ws))]
        mu, _ = run_chain(Ws, inject=inj, inject_upto=len(Ws) - 2, inject_regime=REG[R], **ta.B9)
        out = json.loads(f.read_text()) if f.exists() else {}
        out[str(m)] = float(((mu[-1] - g) ** 2).mean())
        f.write_text(json.dumps(out, indent=1))
        print(R, m, out[str(m)], flush=True)
elif mode == "mc":
    m, seed, ch, R = int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5]
    Ws, g = ta.mlp(m)
    mu, _ = run_chain(Ws, oracle=mc_oracle(m, seed, ch, R), **ta.B9)
    r = float(((mu[-1] - g) ** 2).mean())
    (OUT / f"mc_{m}_{seed}_{ch}_{R}.json").write_text(json.dumps({"mse": r}))
    print(m, seed, ch, R, r, flush=True)
