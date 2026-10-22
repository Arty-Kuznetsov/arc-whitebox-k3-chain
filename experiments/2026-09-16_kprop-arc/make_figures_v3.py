"""Figures for write-up v3 (the method story), drawn only from logged outputs.

Inputs: logs/*.txt in this folder, the 100-MLP grader-format reports in ../2026-09-23_*/ .
Outputs: ../../writeup/figures/f{1..5}_*.png|.pdf and DATA-v3.md (the table twin of every figure).
The v2 figures (fig1_ladder, fig2_cost, fig3_rank, fig5_adaptive) are still made by make_figures.py
and are reused in the appendix.

Run from experiments/2026-09-16_kprop-arc/ inside the ARC venv (matplotlib lives there):
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src" uv run --project "$ARC" python make_figures_v3.py
"""
import json
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
EXP = HERE.parent
OUT = HERE.parent.parent / "writeup" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
NUM = r"([-+]?[0-9.]+(?:e[+-]?\d+)?)"

# palette: dataviz reference instance (light surface), validated with validate_palette.js --mode light
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue, steps 250..700
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 7.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
    "lines.linewidth": 2, "lines.markersize": 5.5,
})
MK = dict(marker="o", markeredgecolor=SURF, markeredgewidth=1.3)
SMK = dict(marker="o", ms=3.6, markeredgecolor=SURF, markeredgewidth=0.8)


def read(name):
    return (LOGS / name).read_text()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def report(path):
    r = json.loads(Path(path).read_text())["results"]
    return dict(adj=r["adjusted_final_layer_score"], final=r["final_layer_mse"], all=r["all_layers_mse"],
                util=r["mean_compute_utilization"] * 100, per_layer=r["per_layer_mse"],
                failed=r["n_failed_mlps"], n=len(r["per_mlp"]))


md = ["# Figure data (v3) — table twins of the write-up figures\n\n",
      "Parsed by `experiments/2026-09-16_kprop-arc/make_figures_v3.py` from `logs/*.txt` and the 100-MLP reports. "
      "Baked N=1e9 ground truth, width 1024, depth 16, B = 2**41. MLP counts are stated per figure.\n"]

# ---------------- F1: the ingredient map + where the k=2 error lives ----------------
t = read("state_split3.txt")
arms = {}
for mean, cov, edge, mse in re.findall(rf"^\s*(\S+)\s+(\S+)\s+(True|False|k=3) \|\s+{NUM} \|", t, re.M):
    arms[(mean, cov, edge)] = float(mse)
ours = float(re.search(rf"^\s*4\s+t1 \|\s+{NUM}", read("offdiag_sweep2.txt"), re.M).group(1))
k2_3 = arms[("none", "none", "False")]
bars = [  # label, value, category
    ("k=2 chain, Gaussian readout", k2_3, "deployable"),
    ("upstream mean ← k=3 chain", arms[("k3", "none", "False")], "oracle"),
    ("upstream mean ← ground truth", arms[("gt", "none", "False")], "oracle"),
    ("mean ← ground truth, + our readout", arms[("gt", "none", "True")], "oracle"),
    ("mean, cov ← k=3 chain", arms[("k3", "k3", "False")], "oracle"),
    ("mean, cov ← k=3 chain, + our readout", arms[("k3", "k3", "True")], "oracle"),
    ("mean ← GT, cov ← k=3, + our readout", arms[("gt", "k3", "True")], "oracle"),
    ("reference k=3 chain (148% of B)", arms[("--", "--", "k=3")], "reference"),
    ("our method (15% of B)", ours, "deployable"),
]
col = {"deployable": C1, "oracle": C3, "reference": C2}
lab = {"deployable": "deployable (computed by us end to end)", "oracle": "oracle-injected at layers 0–14",
       "reference": "ARC reference k=3 (over budget)"}

tb = read("layer_blame.txt")
blame = [tuple(float(x) for x in m) for m in re.findall(
    rf"^\s*(\d+) \|\s+{NUM}\s+{NUM}\s+{NUM} \|\s+{NUM}\s+{NUM} \|\s+{NUM}\s+{NUM}\s*$", tb, re.M)]
bl = np.array(blame)  # l, A, A/base, dA, C, C/base, B, B/base

fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.3), gridspec_kw=dict(width_ratios=[1.35, 1]))
y = np.arange(len(bars))[::-1]
for yi, (name, v, cat) in zip(y, bars):
    a1.barh(yi, v - 1e-8, height=0.62, color=col[cat], left=1e-8)
    a1.text(v * 1.12, yi, f"{v:.2e}  ({k2_3 / v:.0f}×)" if cat != "deployable" or v != k2_3 else f"{v:.2e}  (1×)",
            va="center", fontsize=7, color=INK)
a1.set_xscale("log"); a1.set_xlim(1e-8, 2.5e-5)
a1.set_yticks(y); a1.set_yticklabels([b[0] for b in bars], fontsize=7.2)
a1.tick_params(axis="y", length=0); a1.grid(axis="y", visible=False)
a1.set_xlabel("final-layer MSE (log; 3 MLPs)")
a1.set_title("What each ingredient of the state is worth", loc="left")
a1.legend(handles=[Patch(color=col[c], label=lab[c]) for c in ("deployable", "oracle", "reference")],
           loc="upper center", bbox_to_anchor=(0.35, -0.2), ncol=2, fontsize=6.6)

L = bl[:, 0]
for key, idx, c, name in (("A", 2, C1, "mean fixed"), ("C", 5, C2, "covariance fixed"), ("B", 7, C3, "both fixed")):
    a2.plot(L, bl[:, idx], color=c, label=name, **SMK)
a2.axhline(1.0, color=AXIS, lw=1)
a2.set_xlabel("injection layer l")
a2.set_ylabel("fraction of k=2 final error remaining")
a2.set_title("Where the k=2 error lives", loc="left")
a2.set_ylim(-0.03, 1.25); a2.set_xticks(range(0, 16, 3))
a2.text(8.5, 1.19, "covariance fixed alone: up to 16% worse", fontsize=6.8, color=INK2, ha="center")
a2.legend(loc="lower left")
save(fig, "f1_map")

md.append("\n## F1 — ingredient map (`state_split3.txt`, 3 MLPs) and per-layer blame (`layer_blame.txt`, 3 MLPs)\n\n")
md.append("| arm | final_layer_mse | vs k=2 |\n|---|---|---|\n")
for name, v, cat in bars:
    md.append(f"| {name} [{cat}] | {v:.4e} | {k2_3 / v:.1f}× |\n")
md.append("\n| l | mean fixed: remaining | cov fixed: remaining | both fixed: remaining |\n|---|---|---|---|\n")
for r in bl:
    md.append(f"| {int(r[0])} | {r[2]:.3f} | {r[5]:.3f} | {r[7]:.3f} |\n")

# ---------------- F2: the readout, and regeneration vs transport ----------------
tm = read("edgeworth_mc_check.txt")
mc = [tuple(float(x) for x in m) for m in re.findall(
    rf"^\s*{NUM}\s+{NUM} \|\s+{NUM}\s+{NUM}\s+{NUM} \|", tm.split("STEP 2")[1], re.M)]
mc = np.array(mc)  # a, skew, errG, errE, removed
te = read("edgeworth_probe.txt")
ep = np.array([tuple(float(x) for x in m) for m in re.findall(
    rf"^\s*(\d+) \|\s+{NUM}\s+{NUM}\s+{NUM} \|\s+{NUM} \|\s+{NUM}\s+{NUM}\s*$", te, re.M)])
tw = read("wick_regen_probe.txt")
wr = np.array([tuple(float(x) for x in m) for m in re.findall(
    rf"^\s*(\d+) \|\s+{NUM}\s+{NUM}\s+{NUM} \|\s+{NUM}\s+{NUM} \|\s+{NUM}\s+{NUM}\s+{NUM}\s*$", tw, re.M)])

fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(7.6, 2.75))
skews = sorted(set(mc[:, 1]))
for c, s in zip(RAMP[1:], skews):
    sub = mc[(mc[:, 1] == s) & (mc[:, 0] != 0)]
    b1.plot(sub[:, 0], sub[:, 4] * 100, color=c, label=f"skew {s:g}", **MK)
b1.set_xlabel("a = μ/σ  (a = 0: term is 0)"); b1.set_ylabel("closure error removed (%)")
b1.set_title("(a) Edgeworth term, exact test", loc="left")
b1.set_ylim(80, 101); b1.legend(loc="lower center", ncol=2, fontsize=6.6)

b2.plot(ep[:, 0], ep[:, 1], color=C2, label="Gaussian readout", **SMK)
b2.plot(ep[:, 0], ep[:, 2], color=C1, label="+ Edgeworth term", **SMK)
b2.plot(ep[:, 0], ep[:, 4], color=MUTED, lw=1.4, ls="--", label="k=3 chain's own")
b2.set_yscale("log"); b2.set_xlabel("layer"); b2.set_ylabel("one-step mean MSE (log)")
b2.set_title("(b) readout on the k=3 state", loc="left"); b2.legend(loc="lower right", fontsize=6.6)
b2.set_xticks(range(0, 16, 3))

b3.plot(wr[:, 0], wr[:, 1], color=C1, label="regenerated: corr.", **SMK)
b3.plot(wr[:, 0], wr[:, 3], color=C1, ls="--", lw=1.5, label="regenerated: magnitude")
b3.plot(wr[:, 0], wr[:, 4], color=C2, label="transported: corr.", **SMK)
b3.plot(wr[:, 0], wr[:, 5], color=C2, ls="--", lw=1.5, label="transported: magnitude")
b3.set_xlabel("layer"); b3.set_ylim(0, 1.05); b3.set_xticks(range(1, 16, 2)); b3.set_xlim(0.5, 22.5)
for yv, txt, c in ((wr[-1, 1], "regenerated\ncorrelation", C1), (wr[-1, 4], "transported\ncorrelation", C2),
                   (wr[-1, 3] + 0.03, "regenerated\nmagnitude", C1), (wr[-1, 5] - 0.03, "transported\nmagnitude", C2)):
    b3.text(15.6, yv, txt, fontsize=6.2, color=INK, va="center")
b3.set_title("(c) diagonal κ₃: two routes", loc="left")
b3.set_ylabel("vs the reference chain's κ₃ diagonal")
save(fig, "f2_readout")

md.append("\n## F2a — Edgeworth term vs the exact Gamma-family ReLU mean (`edgeworth_mc_check.txt`)\n\n")
md.append("| a | skew | Gaussian-closure error | Edgeworth error | removed |\n|---|---|---|---|---|\n")
for r in mc:
    md.append(f"| {r[0]:g} | {r[1]:g} | {r[2]:.3e} | {r[3]:.3e} | {r[4]:.3f} |\n")
md.append("\n## F2b — one-step readout on the k=3 chain's state (`edgeworth_probe.txt`, 2 MLPs)\n\n")
md.append("| layer | Gaussian | Edgeworth | ratio | k=3 chain | mean |skew| |\n|---|---|---|---|---|---|\n")
for r in ep:
    md.append(f"| {int(r[0])} | {r[1]:.3e} | {r[2]:.3e} | {r[3]:.2f} | {r[4]:.3e} | {r[6]:.4f} |\n")
md.append("\n## F2c — predicting κ₃[i,i,i] one layer on (`wick_regen_probe.txt`, 2 MLPs)\n\n")
md.append("| layer | regenerated corr | regenerated magnitude | transported corr | transported magnitude |\n|---|---|---|---|---|\n")
for r in wr:
    md.append(f"| {int(r[0])} | {r[1]:.3f} | {r[3]:.3f} | {r[4]:.3f} | {r[5]:.3f} |\n")

# ---------------- F3: the source window ----------------
ms = read("multisource.txt")
msr = [(int(j), float(m), float(p), float(a)) for j, m, p, a in re.findall(
    rf"^\s*(\d+) \|\s+{NUM}\s+[0-9.]+x \|\s+{NUM}%\s+[0-9.]+\s+{NUM}\s*$", ms, re.M)]
k2_2 = [m for j, m, _, _ in msr if j == 0][0]
o2 = read("offdiag_sweep2.txt")
osr = [(int(j), terms, float(m), float(p), float(a)) for j, terms, m, p, a in re.findall(
    rf"^\s*(\d+)\s+(t1|off) \|\s+{NUM}\s+[0-9.]+x \|\s+{NUM}%\s+[0-9.]+\s+{NUM}\s*$", o2, re.M)]
t1 = sorted([r for r in osr if r[1] == "t1"])
off = sorted([r for r in osr if r[1] == "off"])

fig, (c1, c2) = plt.subplots(1, 2, figsize=(7.2, 2.85))
js = [j for j, *_ in msr if j > 0]
c1.plot(js, [k2_2 / m for j, m, _, _ in msr if j > 0], color=C1, label="mean correction only (2 MLPs)", **MK)
c1.plot([r[0] for r in t1], [k2_3 / r[2] for r in t1], color=C2,
        label="+ off-diagonal covariance (3 MLPs)", **MK)
c1.axvline(4, color=AXIS, lw=1); c1.text(4.2, 9.5, "shipped J = 4", fontsize=6.8, color=INK2)
c1.set_xlabel("J = number of carried sources"); c1.set_ylabel("final-layer MSE gain over k=2")
c1.set_ylim(0, 10); c1.set_title("(a) accuracy saturates at J = 4–5", loc="left"); c1.legend(loc="center right", bbox_to_anchor=(1.0, 0.62))

c2.plot([r[0] for r in off], [r[4] for r in off], color=C1, label="mean correction only", **MK)
c2.plot([r[0] for r in t1], [r[4] for r in t1], color=C2, label="+ off-diagonal covariance", **MK)
for r in t1:
    c2.annotate(f"{r[3]:.1f}%", (r[0], r[4]), xytext=(0, 6), textcoords="offset points", fontsize=6.5,
                color=INK2, ha="center")
c2.set_yscale("log"); c2.yaxis.set_minor_formatter(NullFormatter())
c2.set_yticks([8e-8, 1e-7, 1.25e-7, 1.5e-7, 2e-7]); c2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.3g}"))
c2.set_xlabel("J  (labels: % of budget, Strassen level 1)"); c2.set_ylabel("adjusted score (log, 3 MLPs)")
c2.set_title("(b) cost-aware optimum", loc="left"); c2.legend(loc="upper left")
save(fig, "f3_sources")

md.append("\n## F3 — source window (`multisource.txt`, 2 MLPs, no Strassen; `offdiag_sweep2.txt`, 3 MLPs, Strassen level 1)\n\n")
md.append("| J | mean-only: final MSE | gain vs k=2 | % of B | adjusted |\n|---|---|---|---|---|\n")
for j, m, p, a in msr:
    md.append(f"| {j} | {m:.4e} | {k2_2 / m:.2f}× | {p:.2f}% | {a:.3e} |\n")
md.append("\n| J | terms | final MSE | gain vs k=2 | % of B | adjusted |\n|---|---|---|---|---|---|\n")
for j, terms, m, p, a in off + t1:
    md.append(f"| {j} | {terms} | {m:.4e} | {k2_3 / m:.2f}× | {p:.2f}% | {a:.3e} |\n")

# ---------------- F4: the diagonal / off-diagonal split, and the birth scale ----------------
vd = read("vardiag.txt")
vdr = {(d, o): float(m) for d, o, m in re.findall(rf"^\s*(True|False)\s+(True|False) \|\s+{NUM}", vd, re.M)}
none = vdr[("False", "False")]
o1 = read("offdiag_sweep.txt")
t2 = float(re.search(rf"^\s*4\s+t2 \|\s+{NUM}", o1, re.M).group(1))
t1v = float(re.search(rf"^\s*4\s+t1 \|\s+{NUM}", o1, re.M).group(1))
split = [
    ("diagonal only", none / vdr[("True", "False")]),
    ("diagonal + off-diagonal", none / vdr[("True", "True")]),
    ("off-diagonal, leg 2 only", none / t2),
    ("off-diagonal, both legs", none / vdr[("False", "True")]),
    ("off-diagonal, leg 1 only (shipped)", none / t1v),
]
bs = read("birth_scale.txt")
bsr = np.array([(float(s), float(m)) for s, m in re.findall(rf"^\s*{NUM} \|\s+{NUM}\s+[0-9.]+x\s*$", bs, re.M)])
base = bsr[bsr[:, 0] == 1.0][0, 1]

fig, (d1, d2) = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(width_ratios=[1.3, 1]))
yy = np.arange(len(split))[::-1]
for yi, (name, g) in zip(yy, split):
    d1.barh(yi, np.log2(g), height=0.6, color=C1 if g >= 1 else C2)
for yi, (name, g) in zip(yy, split):
    x = np.log2(g)
    d1.text(x + (0.08 if x >= 0 else -0.08), yi, f"{g:.2f}×", va="center", ha="left" if x >= 0 else "right",
            fontsize=7, color=INK)
d1.axvline(0, color=INK2, lw=1)
d1.set_yticks(yy); d1.set_yticklabels([s[0] for s in split], fontsize=7.2)
d1.tick_params(axis="y", length=0); d1.grid(axis="y", visible=False)
d1.set_xlim(-2.9, 1.7); d1.set_xticks(np.log2([0.25, 0.5, 1, 2]))
d1.set_xticklabels(["0.25×", "0.5×", "1×", "2×"])
d1.set_xlabel("gain vs no covariance correction (log₂; J = 4, 3 MLPs)")
d1.set_title("(a) same correction, opposite effects", loc="left")

d2.plot(bsr[:, 0], base / bsr[:, 1], color=C1, **MK)
d2.axvline(1.0, color=AXIS, lw=1); d2.text(0.994, 0.36, "derived\nvalue", fontsize=6.8, color=INK2, ha="right")
d2.axvline(1 / 0.918, color=AXIS, lw=1, ls="--")
d2.text(1 / 0.918 + 0.006, 0.36, "rescaled to the true\nlayer-1 magnitude", fontsize=6.8, color=INK2)
d2.set_xlabel("factor on the born source"); d2.set_ylabel("gain vs derived value")
d2.set_ylim(0.3, 1.08); d2.set_title("(b) no constant to fit", loc="left")
save(fig, "f4_split")

md.append("\n## F4a — covariance-correction split at J = 4 (`vardiag.txt`, `offdiag_sweep.txt`, 3 MLPs; no-correction MSE "
          f"{none:.4e})\n\n| applied | gain |\n|---|---|\n")
for name, g in split:
    md.append(f"| {name} | {g:.2f}× |\n")
md.append("\n## F4b — birth-scale sweep (`birth_scale.txt`, 3 MLPs)\n\n| factor | final MSE | vs 1.00 |\n|---|---|---|\n")
for s, m in bsr:
    md.append(f"| {s:.3f} | {m:.4e} | {base / m:.2f}× |\n")

# ---------------- F5: the progression on 100 MLPs ----------------
stages = [
    ("k=2 + tetrachoric", EXP / "2026-09-23_writeup-ablations" / "report_k2.json", None),
    ("+ Edgeworth mean, J = 1", EXP / "2026-09-23_edgeworth" / "report.json", 2.2219498214326448e-07),
    ("J = 3 sources", EXP / "2026-09-23_multisource" / "report.json", 1.1359980192082731e-07),
    ("J = 4, Strassen", EXP / "2026-09-23_multisource" / "report_J4S1.json", 9.901564582002446e-08),
    ("+ off-diagonal covariance", EXP / "2026-09-23_multisource" / "report_covcorr.json", 7.070972551635606e-08),
]
rows = [(n, report(p), g) for n, p, g in stages if Path(p).exists()]
if not Path(stages[0][1]).exists():  # fall back to the banked September number (no per-layer data)
    rows.insert(0, ("k=2 + tetrachoric", dict(adj=4.058e-7, final=4.058e-6, util=2.355, per_layer=None, all=2.150e-6,
                                              failed=0, n=100), None))

fig, (e1, e2) = plt.subplots(1, 2, figsize=(7.4, 3.0))
e1.axvspan(0, 10, color=GRID, alpha=0.55, lw=0)
e1.text(0.4, 5.6e-7, "≤ 10% of B: multiplier floored at 0.1", fontsize=6.6, color=INK2, va="top")
xs = [r[1]["util"] for r in rows]; ys = [r[1]["adj"] for r in rows]
e1.plot(xs, ys, color=C1, label="local, 100 public MLPs", **MK)
gx = [r[1]["util"] for r in rows if r[2]]; gy = [r[2] for r in rows if r[2]]
e1.plot(gx, gy, ls="", marker="D", ms=5.5, color=SURF, markeredgecolor=C2, markeredgewidth=1.6,
        label="AIcrowd grader, 50 hidden MLPs")
for (n, r, g), x, yv in zip(rows, xs, ys):
    e1.annotate(n, (x, yv), xytext=(5, 5), textcoords="offset points", fontsize=6.5, color=INK)
e1.axhline(5.08e-8, color=AXIS, lw=1, ls="--")
e1.text(0.4, 5.08e-8 * 0.83, "public rank 100 (2026-09-23)", fontsize=6.6, color=INK2, va="top")
e1.set_yscale("log"); e1.set_xlim(0, 17); e1.set_ylim(4e-8, 6e-7)
e1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}%"))
e1.yaxis.set_minor_formatter(NullFormatter())
e1.set_yticks([5e-8, 1e-7, 2e-7, 4e-7]); e1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
e1.set_xlabel("FLOPs used (% of budget B)"); e1.set_ylabel("adjusted score (log, lower is better)")
e1.set_title("(a) every stage, scored", loc="left"); e1.legend(loc="lower left", bbox_to_anchor=(0.0, 0.1), fontsize=6.6)

pl = [(n, r) for n, r, _ in rows if r["per_layer"] is not None]
for c, (n, r) in zip(RAMP[-len(pl):], pl):
    e2.plot(range(16), r["per_layer"], color=c, label=n, marker="o", ms=3.5, markeredgecolor=SURF,
            markeredgewidth=0.8, lw=1.6)
e2.set_yscale("log"); e2.set_xlabel("layer"); e2.set_ylabel("per-layer MSE (log, 100 MLPs)")
e2.set_xticks(range(0, 16, 3)); e2.set_title("(b) the gain grows with depth", loc="left")
e2.legend(loc="lower right", fontsize=6.3)
save(fig, "f5_progress")

md.append("\n## F5 — progression on the 100-MLP public `mini` split (grader-format reports)\n\n")
md.append("| stage | adjusted | final MSE | all-layers MSE | % of B | failed | grader (50 hidden MLPs) |\n"
          "|---|---|---|---|---|---|---|\n")
for n, r, g in rows:
    md.append(f"| {n} | {r['adj']:.4e} | {r['final']:.4e} | {r['all']:.4e} | {r['util']:.2f}% | "
              f"{r['failed']}/{r['n']} | {g if g else '—'} |\n")
if pl:
    md.append("\n| layer | " + " | ".join(n for n, _ in pl) + " |\n|---|" + "---|" * len(pl) + "\n")
    for l in range(16):
        md.append(f"| {l} | " + " | ".join(f"{r['per_layer'][l]:.3e}" for _, r in pl) + " |\n")

(OUT / "DATA-v3.md").write_text("".join(md))
print("wrote", OUT / "DATA-v3.md")
