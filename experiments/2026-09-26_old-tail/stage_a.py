"""Stage A of the old-tail question: how ACCURATE does the old tail have to be?

Before building any merge, measure what breaks when the contribution of sources of age >= a0 is degraded.
Uses chain.py's `tailx` switch (added by this experiment; with it off `ablate.py k4memo 3` still prints
3.7533e-08 and `k3l_t 10` still 2.3135e-08).

    ../../.venv-torch/bin/python stage_a.py <arm> [<arm> ...] --n 10 --threads 3

Arms are named in ARMS below.  Every record goes to logs/stage_a.jsonl with its x-factor against the
k3l_t base 2.3135e-08 (10 MLPs).
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))

# the k3l/k3m torch configuration -- the base of every number in this folder
K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)
BASE10 = 2.3135e-08          # ablate.py k3l_t 10, the standing gate
ARMS = {}


def add(name, **kw):
    ARMS[name] = kw


add("base")                                                   # self-check: must print 2.3135e-08
for a0 in (4, 5, 6):
    add(f"S0_exact_a{a0}", tailx=dict(a0=a0, mode="exact"))     # plumbing self-check: must equal base
    add(f"A1_dropD_a{a0}", tailx=dict(a0=a0, mode="dropD"))     # tail feeds t only
    add(f"A2_dropT_a{a0}", tailx=dict(a0=a0, mode="dropT"))     # tail feeds D only
    add(f"A4_diagD_a{a0}", tailx=dict(a0=a0, mode="diagD"))     # degenerate -> must equal A1
    add(f"X_d21_a{a0}", d21_ages=a0 - 1)                        # cross-check of A1 against the old switch
    for r in (64, 128, 256):
        add(f"A3_rankD{r}_a{a0}", tailx=dict(a0=a0, mode="rankD", rank=r))
    for eps in (1e-3, 1e-2, 1e-1):
        tag = {1e-3: "1em3", 1e-2: "1em2", 1e-1: "1em1"}[eps]
        add(f"A5c_scaleT_{tag}_a{a0}", tailx=dict(a0=a0, mode="scaleT", eps=eps))       # coherent, t only
        add(f"A5c_scaleTD_{tag}_a{a0}", tailx=dict(a0=a0, mode="scaleTD", eps=eps))     # coherent, t AND D
        for sd in range(10):
            add(f"A5i_noiseT_{tag}_s{sd}_a{a0}", tailx=dict(a0=a0, mode="noiseT", eps=eps, seed=sd))
            add(f"A5i_noiseTD_{tag}_s{sd}_a{a0}", tailx=dict(a0=a0, mode="noiseTD", eps=eps, seed=sd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="logs/stage_a.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    from chain import run_chain
    BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    gt = np.load(BAKED / "gt.npy")
    names = []
    for pat in a.arms:
        names += ([k for k in ARMS if k.startswith(pat[:-1])] if pat.endswith("*") else [pat])
    for name in names:
        kw = ARMS[name]
        mses, t0 = [], time.time()
        for i in range(a.n):
            W = np.load(BAKED / f"weights_{i}.npy")
            Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
            ck = dict(K3L)
            for k, v in kw.items():                    # a FRESH tailx dict per MLP: it carries a layer counter
                ck[k] = dict(v) if isinstance(v, dict) else v
            mu, _ = run_chain(Ws, **ck)
            mses.append(float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean()))
            del Ws, mu, W
        raw = float(np.mean(mses))
        rec = dict(arm=name, n=a.n, threads=a.threads, raw=raw, x=raw / BASE10, per_mlp=mses,
                   secs=round(time.time() - t0, 1), kw=repr(kw))
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{name:26s} raw {raw:.4e}  x{raw/BASE10:7.3f}  ({rec['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
