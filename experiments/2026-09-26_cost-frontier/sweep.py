"""Cost-frontier sweep: run named configurations of chain.py's run_chain on the baked MLPs and append the
final-layer MSE to results.jsonl, together with the modelled utilisation from cost.py.

    python sweep.py <cfg1> [cfg2 ...] --n 5 --threads 2

Configurations live in CONFIGS below (this file only READS chain.py / ablate.py; nothing there is edited).
Every entry is (run_chain kwargs, cost.units kwargs).  The self-check config `k3l` must give 2.3135e-08 on
10 MLPs -- that is the standing gate of this sweep.
"""
import argparse, json, time, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-24_k3-chain"))
sys.path.insert(0, str(HERE))
import cost

K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True, k4feed="P2",
           marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1, pathD="merge", k4b=1,
           k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0, k31mult=0.65)


def cfg(**kw):
    d = dict(K3L); d.update(kw); return d


# ---- the grid.  value = (run_chain kwargs, cost-model kwargs) -----------------------------------------
CONFIGS = {
    "k3l":            (cfg(), dict()),                       # self-check: 2.3135e-08 on 10 MLPs

    # A. deadcut threshold -- the only cost lever with a production port (k3j) and a <1% model error
    "A_cut25":        (cfg(deadcut=-2.5), dict(deadcut=-2.5)),
    "A_cut20":        (cfg(deadcut=-2.0), dict(deadcut=-2.0)),
    "A_cut15":        (cfg(deadcut=-1.5), dict(deadcut=-1.5)),
    "A_cut10":        (cfg(deadcut=-1.0), dict(deadcut=-1.0)),

    # C. source window, with and without the amplitude-mode tail closure
    "C_w12":          (cfg(window=12), dict(window=12)),
    "C_w10":          (cfg(window=10), dict(window=10)),
    "C_w8":           (cfg(window=8),  dict(window=8)),
    "C_w12amp":       (cfg(window=12, tail="amp"), dict(window=12)),
    "C_w10amp":       (cfg(window=10, tail="amp"), dict(window=10)),
    "C_w8amp":        (cfg(window=8,  tail="amp"), dict(window=8)),

    # D. the path leg: one full-width transport per layer
    "D_nopath":       (cfg(paths=None, e3age=None, pathD="full"), dict(paths=False)),
    "D_prank256":     (cfg(pathrank=256), dict(pathrank=256)),

    # B. shared subspace -- kept as evidence; the model (calibrated on k3k) says the entry update eats it
    "B_6_320":        (cfg(shared_sub=(6, 320)), dict(bands=((6, 320),))),
    "B_8_256":        (cfg(shared_sub=(8, 256)), dict(bands=((8, 256),))),

    # E. per-source low rank of the tree leg B only (the hub leg A is the sensitive one, NEGATIVE 22)
    "E_srB6_320":     (cfg(srank={6: 320, "legs": ("B",)}), dict()),
}

def add(name, chain_kw, cost_kw):
    CONFIGS[name] = (cfg(**chain_kw), cost_kw)

# X. combinations -- single levers are nearly exhausted, so these carry the frontier
for _t in (-2.5, -2.0, -1.5, -1.0):
    _tag = f"c{int(-_t*10):02d}"
    for _w in (12, 10, 8):
        add(f"X_{_tag}_w{_w}", dict(deadcut=_t, window=_w), dict(deadcut=_t, window=_w))
        add(f"X_{_tag}_w{_w}amp", dict(deadcut=_t, window=_w, tail="amp"), dict(deadcut=_t, window=_w))
add("X_c20_np", dict(deadcut=-2.0, paths=None, e3age=None, pathD="full"), dict(deadcut=-2.0, paths=False))
add("X_c20_w12_p256", dict(deadcut=-2.0, window=12, pathrank=256),
    dict(deadcut=-2.0, window=12, pathrank=256))
add("X_c15_w10_p256", dict(deadcut=-1.5, window=10, pathrank=256),
    dict(deadcut=-1.5, window=10, pathrank=256))
add("X_c10_w8_np", dict(deadcut=-1.0, window=8, paths=None, e3age=None, pathD="full"),
    dict(deadcut=-1.0, window=8, paths=False))
for _t in (-2.75, -2.25, -1.75, -1.25, -0.75, -0.5):
    _tag = f"c{int(-_t*100):03d}"
    add(f"A_{_tag}", dict(deadcut=_t), dict(deadcut=_t))
add("X_c10_np", dict(deadcut=-1.0, paths=None, e3age=None, pathD="full"), dict(deadcut=-1.0, paths=False))
add("X_c15_np", dict(deadcut=-1.5, paths=None, e3age=None, pathD="full"), dict(deadcut=-1.5, paths=False))
add("X_c15_p256", dict(deadcut=-1.5, pathrank=256), dict(deadcut=-1.5, pathrank=256))
add("D_prank128", dict(pathrank=128), dict(pathrank=128))
add("D_prank384", dict(pathrank=384), dict(pathrank=384))
for _t, _tag in ((-2.5, "c250"), (-2.25, "c225"), (-2.75, "c275")):
    for _p in (128, 256, 384):
        add(f"X_{_tag}_p{_p}", dict(deadcut=_t, pathrank=_p), dict(deadcut=_t, pathrank=_p))
add("X_c250_np", dict(deadcut=-2.5, paths=None, e3age=None, pathD="full"), dict(deadcut=-2.5, paths=False))
add("F_best", dict(deadcut=-2.5, pathrank=256, srank={6: 320, "legs": ("B",)}),
    dict(deadcut=-2.5, pathrank=256))
add("F_best_noB", dict(deadcut=-2.5, pathrank=256), dict(deadcut=-2.5, pathrank=256))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--out", default="results.jsonl")
    a = ap.parse_args()
    torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
    torch.set_num_threads(a.threads)
    from chain import run_chain
    BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"
    gt = np.load(BAKED / "gt.npy")
    Wcache = {}
    for name in a.names:
        ck, mk = CONFIGS[name]
        u = cost.util(**mk)
        mses, t0 = [], time.time()
        for i in range(a.n):
            if i not in Wcache:
                Wcache[i] = np.load(BAKED / f"weights_{i}.npy")
            W = Wcache[i]
            Ws = [torch.tensor(W[l], dtype=torch.float64) for l in range(W.shape[0])]
            mu, _ = run_chain(Ws, **ck)
            mses.append(float(((mu[-1].numpy() - gt[i][-1]) ** 2).mean()))
            del Ws, mu
        raw = float(np.mean(mses))
        rec = dict(name=name, n=a.n, threads=a.threads, raw=raw, util=u,
                   adjusted=raw * max(0.1, u), per_mlp=mses, secs=round(time.time() - t0, 1),
                   cost_kw=repr(mk))
        with open(HERE / a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"{name:20s} raw {raw:.4e}  util {100*u:6.2f}%  adjusted {rec['adjusted']:.4e}"
              f"  ({rec['secs']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
