"""Which PART of the exact D carries dslice.py's payoff (MLP 0: exact D at layers 9+12 = raw x0.889, and D makes 73-78%
of the born covariance error)? The chain's D differs from the oracle's in three disjoint places:
  block -- rows of neurons that are alive and not saturated (alpha in [-2.5, 2.5]) x columns of alive neurons
           (alpha >= -2.5): the entries the chain computes; there its error is ~7% (quick_d2.py);
  dead  -- rows and columns of dead neurons (alpha < -2.5), zeroed by the matched cut;
  sat   -- rows of saturated neurons (alpha > 2.5), zeroed by E2 AFTER the substitution, so they act only through D_full,
           i.e. through E2's restoration of the kappa4 level.
Arms (layers 9 and 12 together, oracle = two-seed mean P; A and B for the spread): each part exact alone, the block
exact + the rest the chain's, and the accuracy of D on the block (noise-subtracted). Born decomposition for the block.
    PYTHONPATH=$ARC/src:. uv run --project $ARC python dslice2.py <mlp>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dslice as S                                    # noqa: E402


def masks(info, l):
    a = info[l]["mu_pre"].numpy() / np.sqrt(info[l]["var"].numpy())
    alive, sat = a >= -2.5, a > 2.5
    block = np.outer(alive & ~sat, alive)
    dead = ~np.outer(alive, alive)
    satr = np.outer(sat, alive)
    for M in (block, dead, satr):
        np.fill_diagonal(M, False)
    return block, dead, satr


def main(m):
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
    O = {s: np.load(S.HERE / "oracles" / f"d_{m}_s{s}.npz") for s in S.SEEDS}
    A, B = (O[s] for s in S.SEEDS)
    base, info = S.run(Ws, m, record="full")
    out = {"mlp": m, "base": base}
    print(f"=== MLP {m}: base {base:.6e}", flush=True)
    MS = {l: masks(info, l) for l in S.LD}
    for l in S.LD:
        blk = MS[l][0]
        Dc = info[l]["D"].numpy()
        DA, DB = A[f"D_{l}"].astype(np.float64), B[f"D_{l}"].astype(np.float64)
        DP = 0.5 * (DA + DB)
        x, y = DP[blk], Dc[blk]
        noise = float(((DA[blk] - DB[blk]) ** 2).sum()) / 4.0
        err = float(((y - x) ** 2).sum()) - noise
        sig = float((x ** 2).sum()) - noise
        out[f"block_{l}"] = dict(rel_err=float(np.sqrt(max(err, 0) / sig)), corr=float(np.corrcoef(x, y)[0, 1]),
                                 slope=float((x @ y) / (x @ x)), noise_rel=float(np.sqrt(noise / sig)))
        print(f"  D block at l={l}: rel error {out[f'block_{l}']['rel_err']:.3f} (noise-subtracted; noise "
              f"{out[f'block_{l}']['noise_rel']:.3f}), corr {out[f'block_{l}']['corr']:.4f}, slope {out[f'block_{l}']['slope']:.3f}",
              flush=True)

    def sub(parts, src):
        """D_sub at each layer: the oracle on the masked parts, the chain's own D elsewhere."""
        res = {}
        for l in S.LD:
            Dc = info[l]["D"].numpy().copy()
            Do = src(l)
            M = np.zeros_like(Dc, dtype=bool)
            for k, nm in enumerate(("block", "dead", "sat")):
                if nm in parts:
                    M |= MS[l][k]
            Dc[M] = Do[M]
            res[l] = Dc
        return res

    srcs = {"P": lambda l: 0.5 * (A[f"D_{l}"].astype(np.float64) + B[f"D_{l}"].astype(np.float64)),
            "A": lambda l: A[f"D_{l}"].astype(np.float64), "B": lambda l: B[f"D_{l}"].astype(np.float64)}
    pay = {}
    for lab, parts in (("block", ("block",)), ("dead", ("dead",)), ("sat", ("sat",)),
                       ("block+dead", ("block", "dead")), ("all", ("block", "dead", "sat"))):
        r = {}
        for tag, f in srcs.items():
            rr, _ = S.run(Ws, m, oracle={"D": sub(parts, f), "D_layers": set(S.LD)})
            r[tag] = rr / base
        r["richardson"] = 2 * r["P"] - 0.5 * (r["A"] + r["B"])
        pay[lab] = r
        print(f"  PAYOFF exact D on {lab:10s}: P x{r['P']:.4f}  A x{r['A']:.4f}  B x{r['B']:.4f}  "
              f"Richardson x{r['richardson']:.4f}", flush=True)
    out["payoff"] = pay
    json.dump(out, open(S.HERE / "res" / f"dslice2_{m}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
