"""R1, part 3: WHICH young kappa4 births must a carrier hold?  r1age.py found that kappa4 born in the last two
layers, exact, plus the closure (at its level) for the older kappa4 gives the whole kappa4-world gain on net 0.
Here k=4's tree block is split by graph type (r1k4split.py: per ReLU [path | star | 3+2 | correction]) and the
young (age <= 2) part handed to the chain is restricted to a subset of the types; everything not carried is given
to the closure (its least-squares fit on the scale-mixture pattern, as r1age.py). Channels K4+K31+S (the kappa4
world), layers 1..L-2.
  path  12 d1_i C0_ij d2_j C0_jk d2_k C0_kl d1_l        (two hubs of one generation, Gaussian)
  star   4 d1_i C0_ij d3_j C0_jk d1_k C0_jl d1_l        (one hub, Gaussian)
  3+2   12 d1_i C0_ij d2_j k3pre(j,k,l) d1_k d1_l       (kappa3 handed on through an edge: couples the new
                                                          generation to every older kappa3 source -- the costly one)
  corr  the exact repeated slices minus those of (gated old + trees): T3 = 2 var_a d1_a d2_a d1_b d1_c k3(a,b,c),
        Gaussian repeated-index births, exact-minus-gated pass-through of old repeated slices
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r1types.py <n> <depth> <seed> [age]
Output: res/types_n<n>_L<depth>_s<seed>_a<age>.json
"""
import json
import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("R1_THREADS", "3")))
import r1k4split                                                                         # noqa: E402
from mlp_kprop.factor_k4 import _factored_get_dslice                                    # noqa: E402
from r1lib import HERE, Timer, chain_kwargs, final_mse, k4_stats, mc, weights           # noqa: E402
from r1age_lib import lsfit, offd_mask, patterns, zrep3                                 # noqa: E402

NB = r1k4split.install()
TYPES = ("path", "star", "32", "corr")
n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
AGE = int(sys.argv[4]) if len(sys.argv) > 4 else 2
N = int(1e7)
layers = set(range(1, depth - 1))
offd = offd_mask(n)
KEEPS = {"all": TYPES, "-32": ("path", "star", "corr"), "-path": ("star", "32", "corr"), "-star": ("path", "32", "corr"),
         "-corr": ("path", "star", "32"), "path+corr": ("path", "corr"), "star+corr": ("star", "corr"),
         "32+corr": ("32", "corr"), "corr": ("corr",), "trees": ("path", "star", "32")}
subs = {k: {"k4": {}, "K31": {}, "S": {}} for k in KEEPS}
desc = {}


def keep(l, K):
    if l == 0 or 4 not in K:
        return
    A, B = K[4]._factors
    r = A.shape[2]
    assert r == NB * n * l, (r, n, l)
    P = patterns(K[2].to_tensor())
    blk = [(l - b // NB, TYPES[b % NB]) for b in range(NB * l)]          # (age, type) of each n-column block

    def sl(pred, part):
        bs = [b for b in range(NB * l) if pred(*blk[b])]
        if not bs:
            return None
        cs = torch.cat([torch.arange(b * n, (b + 1) * n) for b in bs])
        X = _factored_get_dslice(A[:, :, cs], B[:, :, cs], part)
        return X * offd if part in ((3, 1), (2, 2)) else X

    parts = {"K4d": (4,), "K31": (3, 1), "S": (2, 1, 1)}
    row = {}
    for k, p in parts.items():
        X, Pk = sl(lambda a, t: True, p), P[k]
        R = X - lsfit(X, Pk) * Pk
        row[k] = {}
        for t in TYPES:   # each young type's share of the closure residual (beyond its own closure fit)
            Y = sl(lambda a, t_, t=t: a <= AGE and t_ == t, p)
            if Y is not None:
                Ry = Y - lsfit(Y, Pk) * Pk
                row[k][t] = {"share": float((Ry * Ry).sum() / (R * R).sum()),
                             "corr": float((Ry * R).sum() / torch.sqrt((Ry * Ry).sum() * (R * R).sum()))}
        if l in layers:
            key = {"K4d": "k4", "K31": "K31", "S": "S"}[k]
            for name, kept in KEEPS.items():
                Y = sl(lambda a, t, kept=kept: a <= AGE and t in kept, p)
                Y = torch.zeros_like(X) if Y is None else Y
                sub = Y + lsfit(X - Y, Pk) * Pk
                subs[name][key][l] = sub.to(torch.float32) if k == "S" else sub.numpy().copy()
    desc[l] = row
    print(f"  l={l}: " + "  ".join(f"{k} " + " ".join(f"{t} {v['share']:.2f}/{v['corr']:+.2f}" for t, v in row[k].items())
                                   for k in row), flush=True)


Ws = weights(n, depth, seed)
mu_mc, var_mc = mc(Ws, seed, N)
noise = float(var_mc.mean() / N)
T = Timer()
st, mu4 = k4_stats(Ws, s211_layers=layers, keep=keep)
k4S = float(((mu4[-1] - mu_mc[-1]) ** 2).mean())
print(f"seed {seed}: k=4 (split blocks) + type split in {T():.0f} s; k4S {k4S:.4e} (must equal the unsplit run)", flush=True)
res = {"n": n, "depth": depth, "seed": seed, "age": AGE, "mc_noise": noise, "k4S": k4S,
       "desc": {str(l): v for l, v in desc.items()}, "arms": {}}


def arm(tag, subset, override=None):
    st2 = st if override is None else dict(st, k4=override["k4"], K31=override["K31"])
    kw = chain_kwargs(subset, st2, layers, feedS=(override["S"] if override is not None else None))
    e = final_mse(Ws, kw, mu_mc)
    res["arms"][tag] = e
    b = res["arms"].get("base", e)
    print(f"  {tag:24s} {e:.4e}  x{(e - noise) / (b - noise):.3f}", flush=True)


arm("base", set())
arm("k4w", {"K4", "K31", "S"})
for name in KEEPS:
    arm(f"young{AGE}:{name}", {"K4", "K31", "S"}, subs[name])
(HERE / "res").mkdir(exist_ok=True)
(HERE / "res" / f"types_n{n}_L{depth}_s{seed}_a{AGE}.json").write_text(json.dumps(res, indent=1))
