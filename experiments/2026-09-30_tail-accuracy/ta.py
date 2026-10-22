"""Harness of the tail-regime hunt (cloud session, Prompt A of notes/CLOUD-BRIEF.md). Final-layer MSE of a chain config
per MLP, one result file per (arm, MLP) under res/<tag>/m<m>.json, so an arm can be split over workers, resumed after
a kill, and screened with LOSSLESS early stopping: the kill rule (raw worse than the base, or better on fewer than 7/10)
is certain once 4 of the 10 screening MLPs are worse, so the arm stops there (res/<tag>/KILLED).

MLPs 0-9 = the screening set (experiments/2026-09-16_kprop-arc/baked, the same 10 as every earlier screen);
MLPs 10-41 = the confirmation set (baked42/, scripts/fetch_baked.py --n-mlps 42, never used for selection).
Threads: PB_THREADS (default 1: four workers on four cores)."""
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
os.environ.setdefault("PB_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb  # noqa: E402  (sets the thread count and puts chain.py on the path)
import t4_stack  # noqa: E402
from chain import run_chain  # noqa: E402

torch.set_num_threads(int(os.environ["PB_THREADS"]))
RES = HERE / "res"
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")          # screening base: k3m + matched cut -2.5 + E2 (sep)
FULL, FULL_U = t4_stack.stack_cfg(["e2_2.5_sep", "band_5_448_pre_rr", "e3_-3.0_b"], t=-2.5)  # #8 t4_final25a
BASES = {"B9": B9, "FULL": FULL, "K3M": pb.K3M}
SCREEN, CONF = list(range(10)), list(range(10, 42))
_GT = {}


def mlp(m):
    folder = pb.BAKED if m < 10 else HERE / "baked42"
    if folder not in _GT:
        _GT[folder] = np.load(folder / "gt.npy")
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(folder / f"weights_{m}.npy")]
    return Ws, torch.tensor(_GT[folder][m][-1])


def per_mlp(tag):
    d = RES / tag
    return {int(f.stem[1:]): json.loads(f.read_text())["mse"] for f in d.glob("m*.json")} if d.exists() else {}


def run_arm(tag, cfg, mlps, ref=None, early=True):
    """ref: {m: base mse}; early stop at 4 worse (screening only)."""
    d = RES / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "cfg.json").write_text(json.dumps(pb.jsonable(cfg)))
    for m in mlps:
        if (d / "KILLED").exists():
            return
        f = d / f"m{m}.json"
        if f.exists():
            continue
        t0 = time.time()
        Ws, g = mlp(m)
        try:
            mu, _ = run_chain(Ws, **copy.deepcopy(cfg))
        except Exception as e:  # a numerically broken arm is a result
            (d / "FAILED").write_text(f"{type(e).__name__}: {str(e)[:300]}")
            return
        mse = float(((mu[-1] - g) ** 2).mean())
        f.write_text(json.dumps({"mse": mse, "secs": time.time() - t0}))
        if ref is not None and early:
            per = per_mlp(tag)
            worse = sum(1 for k, v in per.items() if k in ref and v >= ref[k])
            if worse >= 4:
                (d / "KILLED").write_text(f"{worse} of {len(per)} worse")
                return


def summary(tag, ref_tag="B9", mlps=SCREEN):
    per, ref = per_mlp(tag), per_mlp(ref_tag)
    ms = [m for m in mlps if m in per and m in ref]
    if not ms:
        return None
    x = float(np.mean([per[m] for m in ms]) / np.mean([ref[m] for m in ms]))
    better = sum(1 for m in ms if per[m] < ref[m])
    st = "KILLED" if (RES / tag / "KILLED").exists() else "FAILED" if (RES / tag / "FAILED").exists() else \
        ("done" if len(ms) == len(mlps) else "partial")
    return dict(x=x, better=better, n=len(ms), status=st, raw=float(np.mean([per[m] for m in ms])))
