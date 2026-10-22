"""R1, part 2: how far back does the next order's kappa4 reach?  ARC's factored k=4 keeps kappa4 as
Sym(sum_r A_r (x) B_r) and every ReLU appends two blocks of n columns: the TREES born there (paths, stars, the 3+2
kappa3-edge transfer) and the CORRECTION that makes the repeated slices exact (the T3 transfer kappa3 -> (2,1,1), the
exact-minus-gated pass-through of the old repeated slices, ...). Columns keep their place under the linear step and
the gating, so a block's age at the pre-activation of layer l is l - m (m = its birth layer).
Per pre-activation layer l, for each slice the chain reads -- K4d = k4[iiii], K31 = k4[iiij], K22 = k4[iijj],
S = k4[aabc] (the feed) -- this script
  (a) splits k=4's value into age groups and asks how much of what the scale-mixture closure misses
      (residual after the least-squares closure fit) sits in the young blocks;
  (b) substitutes into our chain (layers 1..L-2, as r1fact.py with SKIP=1) the kappa4 world K4+K31+S in the form a
      carrier of the YOUNG kappa4 would deliver: the young blocks exact + the old blocks' least-squares fit on the
      closure's pattern (the closure carrying the old part at its right level), for ages a = 1, 2, 3; also with t, D.
Closure patterns (the scale mixture k4 = v Sym(C (x) C)): K4d ~ 3 var^2, K31 ~ 3 var_i C0_ij,
K22 ~ var_i var_j + 2 C0_ij^2, S ~ var_a C0_bc + 2 C0_ab C0_ac.
    env PYTHONPATH=$ARC/src nice -n 2 uv run --project $ARC python r1age.py <n> <depth> <seed>
Output: res/age_n<n>_L<depth>_s<seed>.json
"""
import json
import os
import sys

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("R1_THREADS", "3")))
from mlp_kprop.factor_k4 import _factored_get_dslice                                    # noqa: E402
from r1lib import HERE, Timer, chain_kwargs, final_mse, k4_stats, mc, weights           # noqa: E402

n, depth, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
N = int(1e7)
layers = set(range(1, depth - 1))      # SKIP = 1
AGES = (1, 2, 3)
idx = torch.arange(n)
offd = ~torch.eye(n, dtype=torch.bool)


def zrep3(S):
    S = S.clone()
    S[idx, idx, :] = 0
    S[idx, :, idx] = 0
    S[:, idx, idx] = 0
    return S


def patterns(C):
    var = torch.diagonal(C)
    C0 = C - torch.diag(var)
    pS = zrep3(var[:, None, None] * C0[None, :, :] + 2.0 * C0[:, :, None] * C0[:, None, :])
    return {"K4d": 3.0 * var * var, "K31": 3.0 * var[:, None] * C0, "K22": (torch.outer(var, var) + 2 * C0 * C0) * offd,
            "S": pS}


def lsfit(X, P):
    return float((X * P).sum() / (P * P).sum())


desc = {}            # (a) per layer, per slice: shares of the closure residual by age group
subs = {a: {"k4": {}, "K31": {}, "S": {}} for a in ("cl",) + AGES}


def keep(l, K):
    if l == 0 or 4 not in K:
        return
    A, B = K[4]._factors
    r = A.shape[2]
    assert r == 2 * n * l, (r, n, l)
    C = K[2].to_tensor()
    P = patterns(C)
    age_of_block = [l - b // 2 for b in range(2 * l)]          # block b = (birth m = b // 2, tree if b even)

    def cols(pred):
        return torch.cat([torch.arange(b * n, (b + 1) * n) for b in range(2 * l) if pred(b)]) \
            if any(pred(b) for b in range(2 * l)) else None

    def sl(cs, part):
        if cs is None:
            return None
        X = _factored_get_dslice(A[:, :, cs], B[:, :, cs], part)
        return X if part != (2, 1, 1) else zrep3(X)

    parts = {"K4d": (4,), "K31": (3, 1), "K22": (2, 2), "S": (2, 1, 1)}
    tot = {k: sl(torch.arange(r), p) for k, p in parts.items()}
    tot["K31"] = tot["K31"] * offd
    tot["K22"] = tot["K22"] * offd
    row = {}
    for k in parts:
        X, Pk = tot[k], P[k]
        al = lsfit(X, Pk)
        R = X - al * Pk
        row[k] = {"closure_R2": 1.0 - float((R * R).sum() / ((X - X.mean()) ** 2).sum()), "groups": {}}
        for a in AGES:
            Y = sl(cols(lambda b: age_of_block[b] <= a), parts[k])
            if k in ("K31", "K22"):
                Y = Y * offd
            Ry = Y - lsfit(Y, Pk) * Pk                              # the young part beyond its own closure fit
            Ro = R - Ry                                             # = the old part beyond its fit
            row[k]["groups"][a] = {"young_share": float((Ry * Ry).sum() / (R * R).sum()),
                                   "old_share": float((Ro * Ro).sum() / (R * R).sum()),
                                   "corr_young_R": float((Ry * R).sum() / torch.sqrt((Ry * Ry).sum() * (R * R).sum()))}
            if l in layers and k in ("K4d", "K31", "S"):
                sub = Y + lsfit(X - Y, Pk) * Pk                     # young exact + old as its closure fit
                key = {"K4d": "k4", "K31": "K31", "S": "S"}[k]
                subs[a][key][l] = sub.to(torch.float32) if k == "S" else sub.numpy().copy()
        if l in layers and k in ("K4d", "K31", "S"):
            key = {"K4d": "k4", "K31": "K31", "S": "S"}[k]
            subs["cl"][key][l] = (al * Pk).to(torch.float32) if k == "S" else (al * Pk).numpy().copy()
        # tree vs correction blocks (all ages): share of the residual
        for typ, par in (("tree", 0), ("corr", 1)):
            Y = sl(cols(lambda b: b % 2 == par), parts[k])
            if k in ("K31", "K22"):
                Y = Y * offd
            Ry = Y - lsfit(Y, Pk) * Pk
            row[k][typ + "_share"] = float((Ry * Ry).sum() / (R * R).sum())
    desc[l] = row
    print(f"  l={l}: " + "  ".join(
        f"{k} cl.R2 {row[k]['closure_R2']:.2f} young1/2/3 " + "/".join(
            f"{row[k]['groups'][a]['young_share']:.2f}" for a in AGES) for k in parts), flush=True)


Ws = weights(n, depth, seed)
mu_mc, var_mc = mc(Ws, seed, N)
noise = float(var_mc.mean() / N)
T = Timer()
st, mu4 = k4_stats(Ws, s211_layers=layers, keep=keep)
print(f"seed {seed}: k=4 + age split in {T():.0f} s", flush=True)
res = {"n": n, "depth": depth, "seed": seed, "mc_noise": noise, "k4S": float(((mu4[-1] - mu_mc[-1]) ** 2).mean()),
       "desc": {str(l): v for l, v in desc.items()}, "arms": {}}


def arm(tag, subset, override=None):
    st2 = dict(st)
    if override is not None:
        st2 = dict(st, k4=override["k4"], K31=override["K31"])
    kw = chain_kwargs(subset, st2, layers, feedS=(override["S"] if override is not None else None))
    e = final_mse(Ws, kw, mu_mc)
    res["arms"][tag] = e
    b = res["arms"].get("base", e)
    print(f"  {tag:28s} {e:.4e}  x{(e - noise) / (b - noise):.3f}", flush=True)


arm("base", set())
for fam, extra in (("", set()), ("tD+", {"t", "D"})):
    arm(fam + "k4w", {"K4", "K31", "S"} | extra)
    arm(fam + "k4w-closurefit", {"K4", "K31", "S"} | extra, subs["cl"])
    for a in AGES:
        arm(fam + f"k4w-young{a}", {"K4", "K31", "S"} | extra, subs[a])
(HERE / "res").mkdir(exist_ok=True)
(HERE / "res" / f"age_n{n}_L{depth}_s{seed}.json").write_text(json.dumps(res, indent=1))
