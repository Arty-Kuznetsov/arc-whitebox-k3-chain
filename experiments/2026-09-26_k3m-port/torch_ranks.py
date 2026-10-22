"""Accuracy of the truncated path leg at the ranks that can actually pay in production.

The cost-frontier sweep measured pathrank 256 (and 128/384 on n=4 only).  The production bill of a
rank-rho path leg is ~3.2-5.0 * rho/n units against 0.607 for the full Strassen transport, so only
rho <~ 190 can pay at all; this script measures the torch accuracy inside that window on 10 MLPs.

    ../../.venv-torch/bin/python torch_ranks.py 192 128 64 --n 10

Reads chain.py's run_chain; edits nothing.  Gate: rank None must reproduce A_cut25 = 2.3462e-08.
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ranks", nargs="+")                 # integers, or "none"
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--deadcut", type=float, default=-2.5)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="logs/torch_ranks.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    from chain import run_chain
    BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    gt = np.load(BAKED / "gt.npy")
    for r in a.ranks:
        rank = None if r.lower() == "none" else int(r)
        ck = dict(K3L); ck["deadcut"] = a.deadcut; ck["pathrank"] = rank
        mses, t0 = [], time.time()
        for i in range(a.n):
            W = np.load(BAKED / f"weights_{i}.npy")
            Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
            mu, _ = run_chain(Ws, **ck)
            mses.append(float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean()))
            del Ws, mu, W
        raw = float(np.mean(mses))
        rec = dict(pathrank=rank, deadcut=a.deadcut, n=a.n, threads=a.threads, raw=raw,
                   per_mlp=mses, secs=round(time.time() - t0, 1))
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"pathrank {str(rank):>5s}  deadcut {a.deadcut}  raw {raw:.4e}  ({rec['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
