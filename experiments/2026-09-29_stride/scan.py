"""The strided old tier (the 2026-09-29 completeness critic's gap 1): does the transport have to match the readout?

Every source's legs are moved one layer at a time only because every source is READ at every layer. But all W are
known in advance and the chain computes the gates itself, so the OLD tier (age >= a0) can be read exactly only at
checkpoint layers and JUMPED between them through one shared operator J = M_{c+k}...M_{c+1} (k-1 products per jump,
shared by every old source), i.e. one product per leg per jump instead of per layer. In between, the old tier's
(t, D) come from a cheap rule -- here the exact repeated-index push (state-closure section 1), which drops only the
all-distinct term -- and its error is RESET at every checkpoint, so it lives at most k-1 steps. That is why the
tolerance on the all-distinct fraction becomes 5-10x softer than for an unreset tail (MST, cloud Prompt A).
Critic's projection on the k3l atlas: (a0,k) = (4,2) x0.79, (4,3) x0.72, (3,3) x0.65 on utilisation.

Two production meanings, two modes:
  mst   - old legs not materialised between checkpoints: t_old AND D_old from the rule.
  Donly - old legs still move every layer (t_old exact and O(n^2)); only the old tier's D PAIRINGS are skipped
          between checkpoints, D_old from the rule. Independent of how well t is predicted.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-29_stride/scan.py
Self-check: stride=(a0, 1) makes every layer a checkpoint and must reproduce the base (summation order only).
Kill (critic): the best rule at (4,2) worse than x1.05 raw. Pass: <= x1.03.
"""
import json, time
from pathlib import Path
import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
OUT = Path(__file__).resolve().parent
K3M = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-2.5, k31mult=0.65)
ARMS = [("k3m base (2.3462e-08)", {}),
        ("SELF-CHECK stride (4,1)", dict(stride=(4, 1, "mst"))),
        ("mst   (4,2)", dict(stride=(4, 2, "mst"))),
        ("Donly (4,2)", dict(stride=(4, 2, "Donly"))),
        ("mst   (4,3)", dict(stride=(4, 3, "mst"))),
        ("Donly (4,3)", dict(stride=(4, 3, "Donly"))),
        ("mst   (3,2)", dict(stride=(3, 2, "mst"))),
        ("mst   (3,3)", dict(stride=(3, 3, "mst"))),
        ("Donly (3,3)", dict(stride=(3, 3, "Donly"))),
        ("mst   (6,2)", dict(stride=(6, 2, "mst")))]
gt = np.load(BAKED / "gt.npy")
data = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
         torch.tensor(gt[m][-1])) for m in range(len(gt))]
out, base, bper = {}, None, None
for tag, kw in ARMS:
    cfg = dict(K3M); cfg.update(kw)
    t0 = time.time()
    try:
        per = [float(((run_chain(Ws, **cfg)[0][-1] - g) ** 2).mean()) for Ws, g in data]
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  {tag:26s} FAILED {type(e).__name__}: {str(e)[:90]}", flush=True); continue
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": r / base, "better": b, "per_mlp": per}
    print(f"  {tag:26s} raw {r:.4e}  raw x{r/base:.4f}  better {b}/10  [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "scan_n10.json").write_text(json.dumps(out, indent=1))
