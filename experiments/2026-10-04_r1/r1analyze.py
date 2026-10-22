"""Analysis of the R1 factorial (r1fact.py): per width, the ratio of every subset of the eight channels to our chain,
x = sum_nets (MSE_arm - noise) / sum_nets (MSE_base - noise)   (noise = the MC noise var/N, subtracted),
plus per-net ratios. Kill rule R1 (notes/PLAN-2026-10-04.md): some PROPER subset must reach within x1.2 of the full
joint gain, i.e. x(subset) <= 1.2 x(all eight). Prints: the full joint, the best subset of every size, every minimal
subset that passes, and a log-linear fit of main effects + pairwise interactions.
    python r1analyze.py [n ...]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CH = ("V", "C0", "t", "D", "K4", "K22", "K31", "S")


def name(s):
    return "+".join(c for c in CH if c in s) or "base"


def load(n, skip=1):
    rows = []
    for f in sorted((HERE / "res").glob(f"fact_n{n}_L8_s*_skip{skip}.json")):
        r = json.loads(f.read_text())
        if len(r["arms"]) == 256:
            rows.append(r)
    return rows


def table(rows):
    """{subset name: (pooled x, [per-net x])}"""
    out = {}
    for r in rows:
        nz = r["mc_noise"]
        b = r["arms"]["base"] - nz
        for a, e in r["arms"].items():
            out.setdefault(a, [[], []])
            out[a][0].append(e - nz)
            out[a][1].append((e - nz) / b)
    base_sum = sum(out["base"][0])
    return {a: (sum(v[0]) / base_sum, v[1]) for a, v in out.items()}


def main(n):
    rows = load(n)
    if not rows:
        print(f"n={n}: no complete seeds yet")
        return
    T = table(rows)
    k4 = sum(r["k4S"] - r["mc_noise"] for r in rows) / sum(r["arms"]["base"] - r["mc_noise"] for r in rows)
    full = T[name(CH)][0]
    print(f"\n===== n={n}, L=8, {len(rows)} nets (seeds {[r['seed'] for r in rows]}), upstream layers {rows[0]['layers']}")
    print(f"k=4 SIMPLE itself: x{k4:.3f} of our chain   |   all eight upstream: x{full:.3f}   |   pass line x{1.2 * full:.3f}")
    print("\nsingle channels (pooled x, per-net):")
    for c in CH:
        x, per = T[c]
        print(f"  {c:4s} x{x:.3f}   " + " ".join(f"{p:.3f}" for p in per))
    print("\nleave-one-out from all eight:")
    for c in CH:
        s = name(set(CH) - {c})
        x, per = T[s]
        print(f"  -{c:4s} x{x:.3f}  (x{x / full:.3f} of the full joint)   " + " ".join(f"{p:.3f}" for p in per))
    print("\nbest subset of each size:")
    for k in range(1, 9):
        best = sorted((T[name(c)][0], name(c)) for c in itertools.combinations(CH, k))[:3]
        print(f"  {k}: " + " | ".join(f"{nm} x{x:.3f}" for x, nm in best))
    passing = [set(c) for k in range(1, 8) for c in itertools.combinations(CH, k) if T[name(c)][0] <= 1.2 * full]
    minimal = [s for s in passing if not any(o < s for o in passing)]
    print(f"\nminimal passing subsets (x <= {1.2 * full:.3f}): {len(minimal)}")
    for s in sorted(minimal, key=lambda s: (len(s), T[name(s)][0])):
        x, per = T[name(s)]
        print(f"  {name(s):30s} x{x:.3f}   per-net max {max(per):.3f}   " + " ".join(f"{p:.3f}" for p in per))
    # log-linear model with pairwise interactions on the pooled ratio
    subs = [set(c) for k in range(0, 9) for c in itertools.combinations(CH, k)]
    y = np.log([T[name(s)][0] for s in subs])
    pairs = list(itertools.combinations(range(8), 2))
    X = np.array([[1.0] + [float(c in s) for c in CH] + [float(CH[i] in s and CH[j] in s) for i, j in pairs]
                  for s in subs])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    print(f"\nlog-linear fit, main + pairwise (R^2 {1 - resid.var() / y.var():.3f}; residual sd {resid.std():.3f}):")
    print("  main effects (x per channel, others absent): " + "  ".join(f"{c} x{np.exp(beta[1 + i]):.3f}"
                                                                     for i, c in enumerate(CH)))
    top = sorted(((beta[9 + k], pairs[k]) for k in range(len(pairs))), key=lambda t: t[0])
    print("  strongest synergies (x on top of the two mains): " + "  ".join(
        f"{CH[i]}*{CH[j]} x{np.exp(b):.3f}" for b, (i, j) in top[:6]))
    print("  strongest antagonisms: " + "  ".join(f"{CH[i]}*{CH[j]} x{np.exp(b):.3f}" for b, (i, j) in top[-4:]))


for n in ([int(a) for a in sys.argv[1:]] or [128, 256]):
    main(n)
