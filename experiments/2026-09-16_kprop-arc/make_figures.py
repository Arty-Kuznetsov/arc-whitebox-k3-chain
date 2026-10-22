"""Generate the write-up figures from the logged experiment outputs (logs/*.txt).

Run from experiments/2026-09-16_kprop-arc/ inside the ARC venv (matplotlib lives there):
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src" uv run --project "$ARC" python make_figures.py
Outputs: ../../writeup/figures/fig{1..5}_*.png|.pdf and DATA.md (the table twin of every figure).
"""
import re
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
LOGS = HERE / "logs"
OUT = HERE.parent.parent / "writeup" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
B = 2 ** 41
NUM = r"([0-9.]+(?:e[+-]?\d+)?)"

# palette: dataviz reference instance (light surface)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#86b6ef", "#2a78d6", "#104281"]  # ordinal blue ramp (steps 250 / 450 / 650)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
    "lines.linewidth": 2, "lines.markersize": 6,
})
MK = dict(marker="o", markeredgecolor=SURF, markeredgewidth=1.5)


def read(name):
    return (LOGS / name).read_text()


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def pct(x, _):
    return f"{x:g}%"


data_md = ["# Figure data — table twins of the write-up figures\n\n",
           "Parsed from `experiments/2026-09-16_kprop-arc/logs/*.txt` by `make_figures.py` "
           "(re-run of 2026-09-22; baked N=1e9 GT, width 1024, depth 16, B = 2**41).\n"]

# ---------------- Fig 1: accuracy–cost ladder ----------------
t = read("kprop_1024.txt")
ladder = {int(k): (float(f), float(a)) for k, f, a in
          re.findall(rf"k=(\d): final_layer_mse={NUM}\s+all_layers_mse={NUM}", t)}
per_mlp = {1: [], 2: [], 3: []}
for line in t.splitlines():
    m = re.match(r"\s*mlp \d+:(.*)", line)
    if m:
        for k, v in re.findall(rf"k(\d) final={NUM}", m.group(1)):
            per_mlp[int(k)].append(float(v))
fb = read("flop_breakdown.txt")
flops = {int(k): float(v) for k, v in re.findall(rf"=== k=(\d) total={NUM}", fb)}
try:
    flops[1] = float(re.search(rf"k=1 total={NUM}", read("flop_k1.txt")).group(1))
except Exception:
    flops[1] = 8.66e7  # flopscope mean-propagation count from the starter-kit docs
LB_TOP = (1.64e-8, 1.91e-8)  # raw final MSE of leaderboard #1..#10 on 2026-09-22
OUR_K2 = (5.179e10 / B * 100, 3.850e-6)  # our flopscope k=2 + tetrachoric on the baked GT (mini, 100 MLPs; report k1_mini.json)

fig, ax = plt.subplots(figsize=(6.4, 3.7))
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(100, 1e4, color=GRID, alpha=0.6, lw=0)
ax.axvline(10, color=AXIS, lw=1); ax.axvline(100, color=AXIS, lw=1)
ax.axhspan(*LB_TOP, color=C2, alpha=0.35, lw=0)
xs = [flops[k] / B * 100 for k in (1, 2, 3)]
ys = [ladder[k][0] for k in (1, 2, 3)]
for k, x in zip((1, 2, 3), xs):
    ax.plot([x] * len(per_mlp[k]), per_mlp[k], ls="", marker="o", ms=4.5, color=C1, alpha=0.5, markeredgewidth=0, zorder=2)
ax.plot(xs, ys, color=C1, **MK)
for k, x, y in zip((1, 2, 3), xs, ys):
    ax.annotate(f"k={k}\n{y:.1e}", (x, y), xytext=(8, 4), textcoords="offset points", fontsize=8, color=INK)
ax.plot(*OUR_K2, ls="", marker="D", ms=6, color=SURF, markeredgecolor=C1, markeredgewidth=1.8)
ax.annotate("our fnp k=2 + tetrachoric", OUR_K2, xytext=(-70, -34), textcoords="offset points", fontsize=7.5,
            color=INK2, ha="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=6))
ax.text(10, 2.2e-3, " 10% (multiplier floor)", fontsize=7.5, color=INK2, va="top")
ax.text(100, 9e-4, " 100% (budget B)", fontsize=7.5, color=INK2, va="top")
ax.text(0.0045, LB_TOP[1] * 1.3, "leaderboard #1–#10, raw final MSE (2026-09-22)", fontsize=7.5, color=INK2)
ax.set_xlim(0.003, 500); ax.set_ylim(5e-9, 3e-3)
ax.xaxis.set_major_formatter(FuncFormatter(pct))
ax.set_xlabel("FLOPs per MLP, % of budget B = 2⁴¹ (log)")
ax.set_ylabel("final_layer_mse vs N=1e9 ground truth (log)")
ax.set_title("Cumulant propagation at 1024×16: accuracy vs cost by order k (5 MLPs, mean + per-MLP dots)")
save(fig, "fig1_ladder")
data_md.append("\n## Fig 1 — accuracy/cost ladder\n\n| k | final_layer_mse (mean of 5 MLPs) | all_layers_mse | FLOPs (ARC counter) | % of B | adjusted |\n|---|---|---|---|---|---|\n")
for k in (1, 2, 3):
    f = flops[k]; mult = max(0.1, f / B)
    adj = f"{ladder[k][0] * mult:.2e}" if f <= B else "— (over budget)"
    data_md.append(f"| {k} | {ladder[k][0]:.3e} | {ladder[k][1]:.3e} | {f:.3e} | {f / B * 100:.3f}% | {adj} |\n")
data_md.append(f"| our fnp k=2 (mini, 100 MLPs) | {OUR_K2[1]:.3e} | 2.118e-06 | 5.179e+10 | {OUR_K2[0]:.3f}% | {OUR_K2[1] * 0.1:.2e} |\n")
data_md.append("\nPer-MLP final_layer_mse: " + "; ".join(
    f"k={k}: " + ", ".join(f"{v:.2e}" for v in per_mlp[k]) for k in (1, 2, 3)) + "\n")

# ---------------- Fig 2: k=3 cost — depth scaling + breakdown ----------------
kc = read("kprop_cost.txt")
depths = [(int(d), float(v)) for d, v in re.findall(rf"depth=\s*(\d+):\s*{NUM}", kc)]
bd = re.findall(r"^\s+(.+?)\s+([0-9.]+e[+-]\d+)\s+([0-9.]+)%\s*$", kc.split("=== depth scaling")[0], flags=re.M)
bd = [(n.strip(), float(v), float(p)) for n, v, p in bd]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw=dict(width_ratios=[1.05, 1]))
d = np.array([x for x, _ in depths]); v = np.array([y for _, y in depths])
a1.set_xscale("log", base=2); a1.set_yscale("log"); a1.xaxis.set_minor_formatter(NullFormatter())
a1.axhline(B, color=AXIS, lw=1); a1.text(2, B * 0.62, "budget B = 2⁴¹", fontsize=7.5, color=INK2, va="top")
a1.plot(d, v[-1] * (d / d[-1]) ** 2, color=MUTED, lw=1.2, ls="--", label="∝ L² reference")
a1.plot(d, v, color=C1, label="k=3 total FLOPs (measured)", **MK)
for x, y in zip(d, v):
    a1.annotate(f"{y / B * 100:.0f}% of B", (x, y), xytext=(7, -11), textcoords="offset points", fontsize=7.5, color=INK2)
a1.set_xticks(d); a1.set_xticklabels([str(x) for x in d])
a1.set_xlabel("depth L (width n = 1024)"); a1.set_ylabel("FLOPs per MLP (log)")
a1.set_title("k=3 cost grows as O(L²)"); a1.legend(loc="lower right")
top = bd[:4]; other = sum(p for _, _, p in bd[4:])
names = [n for n, _, _ in top] + ["all other regions"]; pcts = [p for _, _, p in top] + [other]
y = np.arange(len(names))[::-1]
a2.barh(y, pcts, height=0.55, color=C1)
for yi, p in zip(y, pcts):
    a2.text(p + 1, yi, f"{p:.1f}%", va="center", fontsize=8, color=INK)
a2.set_yticks(y); a2.set_yticklabels(names, fontsize=8); a2.set_xlim(0, 60)
a2.xaxis.set_major_formatter(FuncFormatter(pct)); a2.grid(axis="y", visible=False)
a2.set_title("Where the k=3 FLOPs go (depth 16)")
save(fig, "fig2_cost")
data_md.append("\n## Fig 2 — k=3 cost\n\n| depth L | k=3 FLOPs | % of B | ratio per depth doubling |\n|---|---|---|---|\n")
for i, (dd, vv) in enumerate(depths):
    r = f"×{vv / depths[i - 1][1]:.2f}" if i else "—"
    data_md.append(f"| {dd} | {vv:.3e} | {vv / B * 100:.1f}% | {r} |\n")
data_md.append("\n| region (depth 16, k=3) | FLOPs | share |\n|---|---|---|\n" + "".join(f"| `{n}` | {vv:.3e} | {p:.1f}% |\n" for n, vv, p in bd))

# ---------------- Fig 3: κ3 is full-rank; rank-capping destroys accuracy ----------------
rd = read("rank_diag.txt")
rows = re.findall(r"^\s*(\d+) \| FactoredT\s+\|\s*(\d+) \|\s*(\d+),\s*(\d+)\s+\|\s*([0-9.]+) / ([0-9.]+) / ([0-9.]+)", rd, flags=re.M)
L = np.array([int(r[0]) for r in rows]); R = np.array([int(r[1]) for r in rows])
r3 = np.array([int(r[2]) for r in rows]); r6 = np.array([int(r[3]) for r in rows])
e8, e32, e128 = (np.array([float(r[i]) for r in rows]) for i in (4, 5, 6))
rc = read("rcap_test.txt")
full = float(re.search(rf"full k=3 \(no cap\): final_mse={NUM}", rc).group(1))
caps = [(int(a), float(b)) for a, b in re.findall(rf"R_max=\s*(\d+): final_mse={NUM}", rc)]
k2_ref = ladder[2][0]
fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(11.2, 3.4))
a1.set_yscale("log")
a1.plot(L, R, color=C1, label="stored rank R = 2n·(l+1)", **MK)
a1.plot(L, r6, color=C2, label="effective rank (σ > 1e-6·σ_max)", **MK)
a1.plot(L, r3, color=C3, label="effective rank (σ > 1e-3·σ_max)", **MK)
a1.axhline(1024, color=AXIS, lw=1); a1.text(0, 1024 * 0.72, "n = 1024", fontsize=7.5, color=INK2)
a1.set_ylim(250, 8e4); a1.set_xlabel("layer"); a1.set_ylabel("rank of the κ₃ factor (log)")
a1.set_title("κ₃ factor rank per layer (MLP 0)"); a1.legend(loc="center right")
a2.plot(L, e8, color=SEQ[0], label="top 8 singular values", **MK)
a2.plot(L, e32, color=SEQ[1], label="top 32", **MK)
a2.plot(L, e128, color=SEQ[2], label="top 128", **MK)
a2.set_ylim(0, 1.02); a2.set_xlabel("layer"); a2.set_ylabel("fraction of Σσ² captured")
a2.set_title("Energy in the leading singular values"); a2.legend(loc="lower right")
a3.set_xscale("log", base=2); a3.set_yscale("log"); a3.xaxis.set_minor_formatter(NullFormatter())
a3.axhline(k2_ref, color=AXIS, lw=1); a3.text(caps[0][0], k2_ref * 0.72, "full k=2", fontsize=7.5, color=INK2, va="top")
a3.set_ylim(full / 2.5, k2_ref * 2.2)
a3.axhline(full, color=AXIS, lw=1); a3.text(caps[0][0], full * 1.35, "full k=3 (no cap)", fontsize=7.5, color=INK2)
a3.plot([c for c, _ in caps], [m for _, m in caps], color=C1, **MK)
a3.set_xticks([c for c, _ in caps]); a3.set_xticklabels([str(c) for c, _ in caps], fontsize=7.5)
a3.set_xlabel("rank cap R_max (top columns by norm)"); a3.set_ylabel("final_layer_mse (log)")
a3.set_title("Capping the rank destroys the k=3 gain (MLP 0)")
save(fig, "fig3_rank")
data_md.append("\n## Fig 3 — κ₃ rank and rank-capping (MLP 0)\n\n| layer | R | eff-rank (>1e-3) | eff-rank (>1e-6) | energy top-8 | top-32 | top-128 |\n|---|---|---|---|---|---|---|\n")
for r in rows:
    data_md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |\n")
data_md.append(f"\n| R_max | final_layer_mse | vs full k=3 ({full:.3e}) |\n|---|---|---|\n" +
               "".join(f"| {c} | {m:.3e} | ×{m / full:.1f} |\n" for c, m in caps))

# ---------------- Fig 4: activation regimes ----------------
kd = read("kink_diag.txt")
kr = re.findall(r"^\s*(\d+) \|\s*([0-9.]+)\s+\|\s*([0-9.]+) \|\s*([0-9.]+) \|\s*([0-9.]+)\s+\|\s*([0-9.]+)", kd, flags=re.M)
Lk = np.array([int(r[0]) for r in kr])
f1, f2, f3, dead, on = (np.array([float(r[i]) for r in kr]) for i in (1, 2, 3, 4, 5))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
a1.stackplot(Lk, dead, f3, on, labels=["dead (α < −3)", "kink (|α| < 3)", "on (α > 3)"], colors=[C2, C1, C3], alpha=0.85)
a1.set_ylim(0, 1); a1.set_xlim(0, 15); a1.set_xlabel("layer"); a1.set_ylabel("fraction of neurons")
a1.set_title("Regime by α = μ_pre / σ_pre (MLP 0)"); a1.legend(loc="lower left")
a2.plot(Lk, f1, color=SEQ[0], label="|α| < 1", **MK)
a2.plot(Lk, f2, color=SEQ[1], label="|α| < 2", **MK)
a2.plot(Lk, f3, color=SEQ[2], label="|α| < 3", **MK)
a2.axhline(f2.mean(), color=AXIS, lw=1)
a2.text(0.2, f2.mean() + 0.025, f"mean over layers, |α| < 2: {f2.mean():.2f}", fontsize=7.5, color=INK2)
a2.set_ylim(0, 1.02); a2.set_xlabel("layer"); a2.set_ylabel("fraction of neurons")
a2.set_title("Neurons near the ReLU elbow"); a2.legend(loc="lower left")
save(fig, "fig4_kink")
data_md.append("\n## Fig 4 — activation regimes per layer (MLP 0)\n\n| layer | |α|<1 | |α|<2 | |α|<3 | dead (α<−3) | on (α>3) |\n|---|---|---|---|---|---|\n")
for r in kr:
    data_md.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} |\n")
data_md.append(f"\nMean fraction |α|<2 across layers: {f2.mean():.3f}\n")

# ---------------- Fig 5: layer-adaptive k ----------------
def rows_adaptive(name, key):
    tt = read(name)
    return [(int(a), float(b), float(c), float(dd)) for a, b, c, dd in
            re.findall(rf"{key}=\s*(\d+)\s*\|\s*{NUM}\s*\|\s*([0-9.]+)%\s*\|\s*[0-9.]+\s*\|\s*{NUM}", tt)]


fw = rows_adaptive("layer_adaptive.txt", "m")
rv = rows_adaptive("reverse_adaptive.txt", "j")
k3_ref = float(re.search(rf"full k=3 final_mse={NUM}", read("layer_adaptive.txt")).group(1))
k2_ref1 = float(re.search(rf"full k=2={NUM}", read("layer_adaptive.txt")).group(1))
fig, axes = plt.subplots(2, 2, figsize=(8.6, 5.4), sharex="col")
for col, (rws, key, title) in enumerate([(fw, "m", "forward: k=3 for the first m layers, then k=2"),
                                          (rv, "j", "reverse: k=2 first, then k=3 for the last j layers")]):
    x = [r[0] for r in rws]; mse = [r[1] for r in rws]; bud = [r[2] for r in rws]
    a = axes[0, col]; a.set_yscale("log")
    a.axhline(k2_ref1, color=AXIS, lw=1); a.axhline(k3_ref, color=AXIS, lw=1)
    a.text(0.2, k2_ref1 * 1.4, "full k=2", fontsize=7.5, color=INK2); a.text(0.2, k3_ref * 1.4, "full k=3", fontsize=7.5, color=INK2)
    a.plot(x, mse, color=C1, **MK); a.set_ylabel("final_layer_mse (log)"); a.set_title(title, fontsize=9)
    a.set_ylim(k3_ref / 2.5, k2_ref1 * 3)
    b = axes[1, col]; b.set_yscale("log")
    b.axhspan(100, 1e3, color=GRID, alpha=0.6, lw=0); b.axhline(10, color=AXIS, lw=1); b.axhline(100, color=AXIS, lw=1)
    b.text(0.2, 10 * 1.3, "10% multiplier floor", fontsize=7.5, color=INK2); b.text(0.2, 100 * 1.3, "budget B", fontsize=7.5, color=INK2)
    b.plot(x, bud, color=C2, **MK); b.set_ylim(1, 300); b.yaxis.set_major_formatter(FuncFormatter(pct))
    b.set_ylabel("FLOPs, % of B (log)"); b.set_xlabel(f"{key} = number of k=3 layers (of 16)"); b.set_xticks(x)
save(fig, "fig5_adaptive")
for name, rws, key in (("forward", fw, "m"), ("reverse", rv, "j")):
    data_md.append(f"\n## Fig 5 — layer-adaptive k, {name} (MLP 0)\n\n| {key} | final_layer_mse | % of B | adjusted | vs full k=2 |\n|---|---|---|---|---|\n")
    for a_, m_, p_, adj_ in rws:
        data_md.append(f"| {a_} | {m_:.3e} | {p_:.1f}% | {adj_:.2e}{' (over budget)' if p_ > 100 else ''} | ×{k2_ref1 / m_:.1f} |\n")
data_md.append(f"\nSelf-check endpoints: full k=2 = {k2_ref1:.3e}, full k=3 = {k3_ref:.3e}.\n")

(OUT / "DATA.md").write_text("".join(data_md))
print("wrote", OUT / "DATA.md")
