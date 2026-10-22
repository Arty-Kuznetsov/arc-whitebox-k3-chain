"""Where does the chain's D error come from? D at layers 9 and 12 against the two-seed oracle, for variants of the kappa3
representation made more exact one at a time, cost ignored (torch, float64):
  base B9; corr="full" (the exact correction leg instead of the rank-24 thin leg); e3age=16 (path births carried at
  every age, "allages"); pathD="full"; k4b=5 (kappa4 births to depth 5); all of them together.
Reports D's relative error (noise-subtracted), its slope against the oracle, and each variant's raw on that MLP.
    PYTHONPATH=$ARC/src:. uv run --project $ARC python dvar.py <mlp>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dslice as S                                    # noqa: E402
from dslice2 import masks                             # noqa: E402

VARIANTS = [("base", {}), ("corr_full", dict(corr="full")), ("allages", dict(e3age=16)),
            ("pathD_full", dict(pathD="full")), ("k4b5", dict(k4b=5)), ("nocuts", dict(deadcut=None, colcut=None)),
            ("band448", dict(ssp=(5, 448, 0, 0))),
            ("all", dict(corr="full", e3age=16, pathD="full", k4b=5))]


def main(m):
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
    A, B = (np.load(S.HERE / "oracles" / f"d_{m}_s{s}.npz") for s in S.SEEDS)
    out = {}
    _, info0 = S.run_chain(Ws, record="full", **S.B9)
    for lab, kw in VARIANTS:
        pred, info = S.run_chain(Ws, record="full", **dict(S.B9, **kw))
        r = {"raw": S.raw(pred, m)}
        for l in S.LD:
            blk = masks(info0, l)[0]                       # the base chain's computed block, the same for every variant
            Dc = info[l]["D"].numpy()
            DA, DB = A[f"D_{l}"].astype(np.float64), B[f"D_{l}"].astype(np.float64)
            DP = 0.5 * (DA + DB)
            noise = float(((DA[blk] - DB[blk]) ** 2).sum()) / 4.0
            err = float(((Dc[blk] - DP[blk]) ** 2).sum()) - noise
            sig = float((DP[blk] ** 2).sum()) - noise
            x, y = DP[blk], Dc[blk]
            r[f"rel_err_{l}"] = float(np.sqrt(max(err, 0.0) / sig))
            r[f"slope_{l}"] = float((x @ y) / (x @ x))
        out[lab] = r
        print(f"  MLP {m} {lab:11s} raw {r['raw']:.4e}  D rel err l9 {r['rel_err_9']:.3f} (slope {r['slope_9']:.3f})  "
              f"l12 {r['rel_err_12']:.3f} (slope {r['slope_12']:.3f})", flush=True)
    json.dump(out, open(S.HERE / "res" / f"dvar_{m}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
