"""R1 -> R2 bridge: the young-kappa4 carrier in the form the chain would actually run it.  r1age.py / r1types.py
handed the chain k=4's young kappa4 exact plus the OLD part's closure fit (so the total level was k=4's). A carrier
inside the chain keeps the chain's OWN closure (level and pattern, tuned) and adds only what the young objects hold
BEYOND the closure's pattern:  X = X_chain + (Y - lsfit(Y, P) P),  Y = k=4's young (age <= AGE) part of the slice,
P = the scale-mixture pattern (r1age_lib.patterns). K4d and K31 through the additive oracle keys (chain.py k4add,
K31add), S through the feed's dense3 "add" mode. Arms by the young types carried (path, star, 3+2, corr; r1k4split).
Reference arms: base, k4w (full substitution), emu (r1types' young exact + old closure fit), add-full (the whole
k=4 residual beyond its closure fit, all ages).
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r1res.py <n> <depth> <seed> [age]
Output: res/res_n<n>_L<depth>_s<seed>_a<age>.json
"""
import json
import os
import sys

import torch

torch.set_num_threads(int(os.environ.get("R1_THREADS", "3")))
import r1k4split                                                                         # noqa: E402
from mlp_kprop.factor_k4 import _factored_get_dslice                                    # noqa: E402
from r1lib import BASE, HERE, Timer, chain_kwargs, final_mse, k4_stats, mc, weights     # noqa: E402
from r1age_lib import lsfit, offd_mask, patterns                                        # noqa: E402

NB = r1k4split.install()
TYPES = ("path", "star", "32", "corr")
n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
AGE = int(sys.argv[4]) if len(sys.argv) > 4 else 2
NOS = len(sys.argv) > 5 and sys.argv[5] == "noS"      # K4 + K31 only (no feed slice)
N = int(1e7)
layers = set(range(1, depth - 1))
offd = offd_mask(n)
KEEPS = {"all": TYPES, "trees": ("path", "star", "32"), "-32": ("path", "star", "corr"), "-path": ("star", "32", "corr"),
         "-star": ("path", "32", "corr"), "32": ("32",), "32+corr": ("32", "corr"), "path+32": ("path", "32")}
emu = {k: {"k4": {}, "K31": {}, "S": {}} for k in ("all", "trees")}
add = {k: {"k4": {}, "K31": {}, "S": {}} for k in list(KEEPS) + ["full"]}
KEY = {"K4d": "k4", "K31": "K31", "S": "S"}


def keep(l, K):
    if l not in layers or 4 not in K:
        return
    A, B = K[4]._factors
    assert A.shape[2] == NB * n * l
    P = patterns(K[2].to_tensor())
    blk = [(l - b // NB, TYPES[b % NB]) for b in range(NB * l)]

    def sl(pred, part):
        bs = [b for b in range(NB * l) if pred(*blk[b])]
        if not bs:
            return None
        cs = torch.cat([torch.arange(b * n, (b + 1) * n) for b in bs])
        X = _factored_get_dslice(A[:, :, cs], B[:, :, cs], part)
        return X * offd if part == (3, 1) else X

    for k, p in {"K4d": (4,), "K31": (3, 1), "S": (2, 1, 1)}.items():
        X, Pk = sl(lambda a, t: True, p), P[k]
        cast = (lambda Z: Z.to(torch.float32)) if k == "S" else (lambda Z: Z.numpy().copy())
        add["full"][KEY[k]][l] = cast(X - lsfit(X, Pk) * Pk)
        for name, kept in KEEPS.items():
            Y = sl(lambda a, t, kept=kept: a <= AGE and t in kept, p)
            Y = torch.zeros_like(X) if Y is None else Y
            add[name][KEY[k]][l] = cast(Y - lsfit(Y, Pk) * Pk)
            if name in emu:
                emu[name][KEY[k]][l] = cast(Y + lsfit(X - Y, Pk) * Pk)
    print(f"  l={l} done", flush=True)


Ws = weights(n, depth, seed)
mu_mc, var_mc = mc(Ws, seed, N)
noise = float(var_mc.mean() / N)
T = Timer()
st, mu4 = k4_stats(Ws, s211_layers=layers, keep=keep)
print(f"seed {seed}: k=4 split in {T():.0f} s", flush=True)
res = {"n": n, "depth": depth, "seed": seed, "age": AGE, "mc_noise": noise,
       "k4S": float(((mu4[-1] - mu_mc[-1]) ** 2).mean()), "arms": {}}


def run(tag, kw):
    e = final_mse(Ws, kw, mu_mc)
    res["arms"][tag] = e
    b = res["arms"].get("base", e)
    print(f"  {tag:20s} {e:.4e}  x{(e - noise) / (b - noise):.3f}", flush=True)


run("base", dict(BASE))
CH = {"K4", "K31"} if NOS else {"K4", "K31", "S"}
run("k4w", chain_kwargs(CH, st, layers))
for name, ov in emu.items():
    run(f"emu:{name}", chain_kwargs(CH, dict(st, k4=ov["k4"], K31=ov["K31"]), layers, feedS=ov["S"]))
for name, d in add.items():
    kw = dict(BASE, oracle={"k4add": d["k4"], "k4add_layers": set(d["k4"]), "K31add": d["K31"],
                            "K31add_layers": set(d["K31"])})
    if not NOS:
        kw["dense3"] = {"feedS": d["S"], "mode": "add"}
    run(f"add:{name}", kw)
(HERE / "res").mkdir(exist_ok=True)
(HERE / "res" / f"res_n{n}_L{depth}_s{seed}_a{AGE}{'_noS' if NOS else ''}.json").write_text(json.dumps(res, indent=1))
