"""Common harness of the port battery (experiments/2026-09-29_port-battery, cloud session #8).

Base = the shipped k3m configuration in torch (experiments/2026-09-29_colcut/pairs.py BASE + deadcut=-2.5),
10 baked MLPs = 2.3462e-08 (gate). Every arm is a dict of run_chain switches on top of it; results go to JSON with
per-MLP numbers so ratios and better-counts can be recomputed. Threads: PB_THREADS (default 3, as ablate.py; the
gates are run at 3; changing it only moves the last bits through the reduction order).

    cd experiments/2026-09-29_port-battery
    PB_THREADS=3 ../../.venv-torch/bin/python <script>.py
"""
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(int(os.environ.get("PB_THREADS", "3")))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
from chain import run_chain  # noqa: E402

BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
BASE = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
            marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
            k4bparts=("sd",), pairx=("k22",), tet=3, k31mult=0.65)
INF = 1e9
K3M = dict(BASE, deadcut=-2.5)
K3M_RAW = 2.3462e-08            # gate (Mac and every cloud run so far)
GRADED_K3M = 5.3906e-9          # submission 332403
UTIL_K3M = 0.24835              # its graded utilisation


def matched(t, **kw):
    """The matched dead cut: row cut (deadcut) and dead-side birth-column cut at the same threshold t."""
    return dict(BASE, deadcut=t, colcut=(t, INF), **kw)


_DATA = None


def data(n=10):
    global _DATA
    if _DATA is None:
        gt = np.load(BAKED / "gt.npy")
        _DATA = [([torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{m}.npy")],
                  torch.tensor(gt[m][-1])) for m in range(len(gt))]
    return _DATA[:n]


def run(cfg, n=10, logs=None):
    """Final-layer MSE per MLP. cfg is deep-copied per MLP (bandz carries a per-run log); if `logs` is a list,
    the bandz log of every MLP is appended to it."""
    per = []
    for Ws, g in data(n):
        c = copy.deepcopy(cfg)
        mu, _ = run_chain(Ws, **c)
        per.append(float(((mu[-1] - g) ** 2).mean()))
        if logs is not None and c.get("bandz") is not None:
            logs.append(c["bandz"].get("log", []))
        del mu
    return float(np.mean(per)), per


def jsonable(v):
    if isinstance(v, dict):
        return {str(k): jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [jsonable(x) for x in v]
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def run_arms(arms, out_path, ref_per=None, n=10, keep_logs=False, tag_width=46):
    """arms: list of (tag, cfg). ref_per: per-MLP reference for x and better-count (default: the first arm)."""
    out_path = Path(out_path)
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for tag, cfg in arms:
        if tag in out:
            print(f"  {tag:{tag_width}s} (cached) raw {out[tag]['raw']:.4e}", flush=True)
            if ref_per is None:
                ref_per = out[tag]["per_mlp"]
            continue
        t0 = time.time()
        logs = [] if keep_logs else None
        try:
            r, per = run(cfg, n=n, logs=logs)
        except Exception as e:  # an arm that breaks numerically is a result, not a crash of the battery
            print(f"  {tag:{tag_width}s} FAILED {type(e).__name__}: {str(e)[:100]}", flush=True)
            out[tag] = {"failed": f"{type(e).__name__}: {str(e)[:300]}", "cfg": jsonable(cfg)}
            out_path.write_text(json.dumps(out, indent=1))
            continue
        if ref_per is None:
            ref_per = per
        ref = float(np.mean(ref_per))
        b = sum(1 for a, c in zip(per, ref_per) if a < c)
        out[tag] = {"cfg": jsonable(cfg), "raw": r, "x": r / ref, "better": b, "per_mlp": per,
                    "secs": time.time() - t0}
        if keep_logs:
            out[tag]["band_log"] = logs
        print(f"  {tag:{tag_width}s} raw {r:.4e}  x{r / ref:.4f}  better {b}/{len(per)}  "
              f"[{time.time() - t0:.0f}s]", flush=True)
        out_path.write_text(json.dumps(out, indent=1))
    return out
