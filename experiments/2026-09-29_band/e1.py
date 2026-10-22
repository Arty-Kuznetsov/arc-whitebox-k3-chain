"""E1 of the 2026-09-29 ideation synthesis: is a TRANSPORTED band basis (no Gram, no Rayleigh-Ritz refit, no eigh)
accurate enough at r = 384-448 on the current k3m base? 10 baked MLPs. No new code: `ssp=(a0, r, q, ov)` with
q = 0 means "project only" (the basis is transported like a leg), and `shared_sub=(a0, r)` is the ideal refitted band.

Why it decides the port: if the transported basis survives at 384-448, the right production layout for OLD legs is
coefficient form (fixed Z on a transported Q) and the dead-column port should be built around it; if not, the wide
dense (n x sum w) layout is the one to build. The synthesis's skeptics measured (3 independent toys, one on real
weights) that a transported rank-r basis captures what an ideal rank-0.8r one does -- no spectral gap near 320 --
so (6,320,0,0) is expected at +7-8%: that arm is the control that the toys were right.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" ../2026-09-29_band/e1.py
Kill (from the synthesis): transported 384 worse than +3.0% AND 448 worse than +2.0%.
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
        ("IDEAL refitted band 6:320", dict(shared_sub=(6, 320))),
        ("transported 6:320 (control, expect +7-8%)", dict(ssp=(6, 320, 0, 0))),
        ("transported 6:384", dict(ssp=(6, 384, 0, 0))),
        ("transported 6:448", dict(ssp=(6, 448, 0, 0))),
        ("transported 5:448", dict(ssp=(5, 448, 0, 0)))]
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
        print(f"  {tag:44s} FAILED {type(e).__name__}: {str(e)[:80]}", flush=True); continue
    r = float(np.mean(per))
    if base is None:
        base, bper = r, per
    b = sum(1 for a, c in zip(per, bper) if a < c)
    out[tag] = {"kw": {k: str(v) for k, v in kw.items()}, "raw": r, "x": r / base, "better": b, "per_mlp": per}
    print(f"  {tag:44s} raw {r:.4e}  raw x{r/base:.4f}  better {b}/10  [{time.time()-t0:.0f}s]", flush=True)
    (OUT / "e1_n10.json").write_text(json.dumps(out, indent=1))
