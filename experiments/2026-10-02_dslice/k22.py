"""The kappa4 (2,2) slice K22 = kappa4[i,i,j,j]: the closure's ANSATZ K22 = v (var_i var_j + 2 C0_ij^2) against the
Monte-Carlo oracle (d_oracle.py, layers 9 and 12, two seeds). Unlike D, K22 is not carried -- it is rebuilt every layer
from one level and a pattern, so its error could be STRUCTURED (a wrong pattern) and hence correctable in O(n^2).
  1. accuracy on the alive block (noise-subtracted): relative error, slope (level), correlation (pattern);
  2. payoff, layers 9 + 12 (P, A, B, Richardson): the exact K22; the exact PATTERN at the chain's level (the oracle
     rescaled to the chain's sum over the block); the chain's pattern at the exact level;
  3. how much of the oracle's pattern the ansatz's two terms explain (least squares on the block, per layer).
    PYTHONPATH=$ARC/src:. uv run --project $ARC python k22.py <mlp>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dslice as S                                    # noqa: E402


def main(m):
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
    A, B = (np.load(S.HERE / "oracles" / f"d_{m}_s{s}.npz") for s in S.SEEDS)
    base, info = S.run(Ws, m, record="full")
    out = {"mlp": m, "base": base}
    print(f"=== MLP {m}: base {base:.6e}", flush=True)
    blk = {}
    for l in S.LD:
        a = info[l]["mu_pre"].numpy() / np.sqrt(info[l]["var"].numpy())
        M = np.outer(a >= -2.5, a >= -2.5)
        np.fill_diagonal(M, False)
        blk[l] = M
        Kc = info[l]["K22"].numpy().astype(np.float64)
        KA, KB = A[f"K22_{l}"].astype(np.float64), B[f"K22_{l}"].astype(np.float64)
        KP = 0.5 * (KA + KB)
        x, y = KP[M], Kc[M]
        noise = float(((KA[M] - KB[M]) ** 2).sum()) / 4.0
        err = float(((y - x) ** 2).sum()) - noise
        sig = float((x ** 2).sum()) - noise
        # the ansatz's two terms fitted to the oracle (pattern capacity): K22 ~ c1 var_i var_j + c2 C0_ij^2
        var = info[l]["var"].numpy(); C0 = info[l]["C"].numpy().astype(np.float64)
        F = np.stack([np.outer(var, var)[M], (C0 * C0)[M]], 1)
        coef, *_ = np.linalg.lstsq(F, x, rcond=None)
        res = x - F @ coef
        r2 = 1 - float(res @ res) / float(((x - x.mean()) ** 2).sum())
        out[f"acc_{l}"] = dict(rel_err=float(np.sqrt(max(err, 0) / max(sig, 1e-300))), noise_rel=float(np.sqrt(noise / max(sig, 1e-300))),
                              corr=float(np.corrcoef(x, y)[0, 1]), slope=float((x @ y) / (x @ x)),
                              ansatz_r2=r2, ansatz_coef=[float(c) for c in coef],
                              chain_v_terms=float((y @ F[:, 0]) / (F[:, 0] @ F[:, 0])))
        o = out[f"acc_{l}"]
        print(f"  K22 at l={l}: rel error {o['rel_err']:.3f} (noise {o['noise_rel']:.3f}), corr {o['corr']:.4f}, "
              f"slope chain/oracle {o['slope']:.3f}; ansatz terms fitted to the oracle: R2 {r2:.3f} "
              f"(coef var*var {coef[0]:.3e}, C0^2 {coef[1]:.3e})", flush=True)

    def subs(kind, src):
        K = {}
        for l in S.LD:
            Kc = info[l]["K22"].numpy().astype(np.float64)
            Ko = src(l)
            M = blk[l]
            if kind == "exact":
                Kn = Kc.copy(); Kn[M] = Ko[M]
            elif kind == "pattern":       # the oracle's pattern at the chain's level (sum over the block)
                Kn = Kc.copy(); Kn[M] = Ko[M] * (Kc[M].sum() / Ko[M].sum())
            elif kind == "level":         # the chain's pattern at the oracle's level
                Kn = Kc * (Ko[M].sum() / Kc[M].sum())
            K[l] = Kn
        return K
    srcs = {"P": lambda l: 0.5 * (A[f"K22_{l}"].astype(np.float64) + B[f"K22_{l}"].astype(np.float64)),
            "A": lambda l: A[f"K22_{l}"].astype(np.float64), "B": lambda l: B[f"K22_{l}"].astype(np.float64)}
    pay = {}
    for kind in ("exact", "pattern", "level"):
        r = {}
        for tag, f in srcs.items():
            rr, _ = S.run(Ws, m, oracle={"K22": subs(kind, f), "K22_layers": set(S.LD)})
            r[tag] = rr / base
        r["richardson"] = 2 * r["P"] - 0.5 * (r["A"] + r["B"])
        pay[kind] = r
        print(f"  PAYOFF {kind:8s} K22 at 9+12: P x{r['P']:.4f}  A x{r['A']:.4f}  B x{r['B']:.4f}  "
              f"Richardson x{r['richardson']:.4f}", flush=True)
    out["payoff"] = pay
    json.dump(out, open(S.HERE / "res" / f"k22_{m}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
