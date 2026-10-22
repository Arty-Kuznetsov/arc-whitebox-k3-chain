"""Harness of cloud #10 (the kink-core variance hunt): the tail-accuracy harness (`ta.py`: per-(arm, MLP) result files,
lossless early stop, MLPs 10-41 = fresh) with results in this folder, plus the oracle loader. Oracles per MLP 0 / 1:
four independent 1e7 MC runs (tail-accuracy seeds 1001 / 1002 + the Mac's default / 777 in ../2026-09-24_k3-chain/
oracles); 'pool' = their average (4e7)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-30_tail-accuracy"))
import ta  # noqa: E402

ta.RES = HERE / "res"
B9, FULL = ta.B9, ta.FULL
OR = {m: [HERE.parent / "2026-09-30_tail-accuracy" / "oracles" / f"mc_{m}_s{s}.npz" for s in (1001, 1002)]
      + [HERE.parent / "2026-09-24_k3-chain" / "oracles" / f"mc_oracle_{m}{x}.npz" for x in ("", "_s777")] for m in (0, 1)}


def oracle(m, which="pool"):
    """which: 'pool' (all four averaged) or an index 0..3."""
    zs = [np.load(f) for f in OR[m]]
    if which != "pool":
        zs = [zs[int(which)]]
    return {k: np.mean([z[k] for z in zs], axis=0) for k in ("mean", "var", "k3", "k4", "post_mean")}
