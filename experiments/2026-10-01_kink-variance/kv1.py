"""Task 1 of cloud #10: LEVEL vs PATTERN of the kink-core variance error.
    decomp       per layer 1..15, core = |alpha_chain| < 1 on B9: e = var_chain - var_exact (pooled 4e7 oracle);
                 (a) level c(l) = <e,v>/<v,v> (v = var_exact); (b) linear pattern of the rest on standardized cheap
                 features (var, mu, alpha, t, K4d, incoming and outgoing weight norms, alpha-rank) + intercept;
                 (c) residual. Energy shares of ||e||^2 and of the oracle-noise-corrected ||e||^2 - noise.
    arm M NAME   chain arm on MLP M: the core variance at layers 1..15 replaced by ABSOLUTE values
                 full = var_exact; level = var_chain_base - c(l) var_exact; lp = var_chain_base - (level + pattern);
                 kinkfull = var_exact on the whole kink |alpha| <= 2.5 (#9A's arm, now with the 4e7 oracle).
Outputs res1/decomp.json, res1/arm_<M>_<NAME>.json."""
import json
import sys

import numpy as np
import torch

import kv
import ta
from chain import run_chain

OUT = kv.HERE / "res1"
L = 16


def base_record(m):
    f = OUT / f"base_rec_{m}.npz"
    if not f.exists():
        Ws, _ = ta.mlp(m)
        _, info = run_chain(Ws, record=True, **kv.B9)
        np.savez(f, **{k: np.stack([x[k].numpy() for x in info]) for k in ("var", "mu_pre", "t", "K4d")})
    return dict(np.load(f))


def decompose(m):
    rec, orc = base_record(m), kv.oracle(m)
    seeds = [kv.oracle(m, i)["var"] for i in range(4)]
    Wn = [w.numpy() for w in ta.mlp(m)[0]]
    layers, parts = [], {}
    for l in range(1, L):
        v_ch, v_ex = rec["var"][l], orc["var"][l]
        a = rec["mu_pre"][l] / np.sqrt(v_ch)
        core = np.abs(a) < 1.0
        e, v = (v_ch - v_ex)[core], v_ex[core]
        noise = float((np.array([s[l][core] for s in seeds]).var(0, ddof=1) / 4.0).sum())
        c = float(e @ v / (v @ v))
        lev = c * v
        r = e - lev
        uin = (Wn[l] ** 2).sum(0)[core]
        uout = (Wn[l + 1] ** 2).sum(1)[core] if l + 1 < L else np.zeros(int(core.sum()))
        rank = (np.argsort(np.argsort(a)) / a.size)[core]
        F = np.column_stack([v_ch[core], rec["mu_pre"][l][core], a[core], rec["t"][l][core], rec["K4d"][l][core],
                             uin, uout, rank])
        sd = F.std(0)
        F = (F - F.mean(0)) / np.where(sd > 0, sd, 1.0)
        F = np.column_stack([np.ones(F.shape[0]), F])
        coef, *_ = np.linalg.lstsq(F, r, rcond=None)
        proj = F @ coef
        E = float(e @ e)
        layers.append(dict(l=l, n=int(core.sum()), E=E, noise=noise, level=float(lev @ lev) / E,
                           pattern=float(proj @ proj) / E, resid=float((r - proj) @ (r - proj)) / E, c=c,
                           rel_err=float(np.sqrt(E / (v @ v)))))
        parts[l] = (core, lev, proj)
    return layers, parts, rec, orc


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "decomp":
        out = {}
        for m in (0, 1):
            layers, _, _, _ = decompose(m)
            E = sum(x["E"] for x in layers)
            N = sum(x["noise"] for x in layers)
            lv = sum(x["level"] * x["E"] for x in layers)
            pt = sum(x["pattern"] * x["E"] for x in layers)
            out[m] = dict(layers=layers, E=E, noise=N, level_share=lv / E, pattern_share=pt / E,
                          level_share_signal=lv / (E - N), pattern_share_signal=pt / (E - N))
            print(f"MLP {m}: energy |e|^2 {E:.3e} (oracle noise {N:.3e} = {100 * N / E:.1f}%); level {100 * lv / E:.1f}% "
                  f"pattern {100 * pt / E:.1f}% of |e|^2 -> of the signal: level {100 * lv / (E - N):.1f}% "
                  f"pattern {100 * pt / (E - N):.1f}%")
            for x in layers[::2]:
                print(f"   L{x['l']:2d} n={x['n']:4d} rel.err {x['rel_err']:.2e} c {x['c']:+.2e} level {x['level']:.3f} "
                      f"pattern {x['pattern']:.3f} resid {x['resid']:.3f} noise/E {x['noise'] / x['E']:.3f}")
        (OUT / "decomp.json").write_text(json.dumps(out, indent=1))
    elif mode == "arm":
        m, name = int(sys.argv[2]), sys.argv[3]
        layers, parts, rec, orc = decompose(m)
        var = [rec["var"][l].copy() for l in range(L)]
        for l in range(1, L):
            core, lev, proj = parts[l]
            if name in ("full", "kinkfull"):
                var[l] = orc["var"][l].copy()
            elif name == "level":
                var[l][core] = rec["var"][l][core] - lev
            elif name == "lp":
                var[l][core] = rec["var"][l][core] - lev - proj
        rg = (-2.5, 2.5 + 1e-9) if name == "kinkfull" else (-1.0, 1.0)
        Ws, g = ta.mlp(m)
        mu, _ = run_chain(Ws, oracle={"var": var, "var_layers": range(1, L), "regime": rg}, **kv.B9)
        r = float(((mu[-1] - g) ** 2).mean())
        (OUT / f"arm_{m}_{name}.json").write_text(json.dumps({"mse": r, "x": r / ta.per_mlp("B9")[m]}))
        print(m, name, r, r / ta.per_mlp("B9")[m])
