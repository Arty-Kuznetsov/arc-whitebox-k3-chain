"""STEP 4: Stage A at the a0 that actually reaches the floor (1, 2, 3) -- never measured in 2026-09-26_old-tail,
which started at a0=4 where the prize is only 3.0e-9.

At a0=1 the "tail" is EVERY source, so a merge error there is an error in the whole kappa3 state; the error
budget can be much tighter than the 3-5% measured at a0=4.  `exact` is the plumbing self-check (must be x1.000).

    ../../.venv-torch/bin/python stage_a_low.py exact_a1 nTD_1em2_a1 --n 10
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)
BASE10 = 2.3135e-08
ARMS = {"base": {}}
for a0 in (1, 2, 3):
    ARMS[f"exact_a{a0}"] = dict(tailx=dict(a0=a0, mode="exact"))
    for eps, tag in ((1e-3, "1em3"), (1e-2, "1em2"), (1e-1, "1em1")):
        ARMS[f"sTD_{tag}_a{a0}"] = dict(tailx=dict(a0=a0, mode="scaleTD", eps=eps))
        for sd in range(3):
            ARMS[f"nTD_{tag}_s{sd}_a{a0}"] = dict(tailx=dict(a0=a0, mode="noiseTD", eps=eps, seed=sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", default="logs/stage_a_low.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    from chain import run_chain
    BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    gt = np.load(BAKED / "gt.npy")
    for name in a.arms:
        kw = ARMS[name]
        mses, t0 = [], time.time()
        for i in range(a.n):
            W = np.load(BAKED / f"weights_{i}.npy")
            Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
            ck = dict(K3L)
            for k, v in kw.items():
                ck[k] = dict(v) if isinstance(v, dict) else v      # fresh tailx per MLP (it carries a counter)
            mu, _ = run_chain(Ws, **ck)
            mses.append(float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean()))
            del Ws, mu, W
        raw = float(np.mean(mses))
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(dict(arm=name, n=a.n, raw=raw, x=raw / BASE10, per_mlp=mses,
                                     secs=round(time.time() - t0, 1))) + "\n")
        print(f"{name:22s} raw {raw:.4e}  x{raw/BASE10:7.3f}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
