"""Figures added in write-up v4.1 (run: uvx --with matplotlib python make_v4_1_figures.py, from writeup/).
  fig_v4_regimes.png   -- §2(b): oracle substitution of exact post-activation means by activation regime (10 networks)
  fig_v4_ablation.png  -- §5.2: the production ablation, one switch off at a time (32 networks), adjusted x and raw x
  fig_v4_ladder.png    -- §7: the cumulant ladder at small width against MC truth, and the young-tree carrier per network
  fig_v4_tree.png      -- §7: the three young kappa4 trees (path, star, 3+2)
Numbers are the ones printed in the write-up's tables; sources: experiments/2026-09-30_tail-accuracy (regimes, cloud #9A),
experiments/2026-10-03_evidence (ablation), experiments/2026-10-03_korder + 2026-10-04_r3 (ladder, carrier)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, GREY, GREEN = "#1f4e79", "#c55a11", "#7f7f7f", "#2e7d32"

# ---- 1. regimes (§2b)
labels = ["dead\n(α < −2.5)", "saturated", "kink\n(all)", "kink core\n(|α| < 1)", "saturating\nshoulder", "dead\nshoulder", "all\nneurons"]
vals = [0.995, 0.49, 0.21, 0.34, 0.52, 0.93, 0.21]
cols = [GREY, GREY, BLUE, BLUE, BLUE, BLUE, ORANGE]
fig, ax = plt.subplots(figsize=(7.2, 2.6))
ax.bar(range(len(vals)), vals, color=cols, width=0.65)
ax.axhline(1.0, color="k", lw=0.6, ls=":")
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"×{v:.2f}" if v < 0.99 else f"×{v:.3f}", ha="center", fontsize=8)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel("final-layer MSE ×\n(exact means substituted)"); ax.set_ylim(0, 1.12)
ax.set_title("Where the error is born: substituting exact post-activation means by regime (10 networks)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_regimes.png", dpi=200); plt.close(fig)

# ---- 2. ablation (§5.2), adjusted × and raw × when the switch is OFF, networks 0–31
abl = [("marginal Edgeworth pair coefficients", 1.558, 1.558), ("matched dead-neuron cuts", 1.433, 1.237),
       ("edge × κ₃ diagram C0·D", 1.303, 1.304), ("closure's κ₄ (3,1) slice K31", 1.166, 1.167),
       ("E2: saturated rows of D out, κ₄ level restored", 1.137, 1.065), ("path births (first readout)", 1.099, 1.145),
       ("riding band (old sources dense)", 1.098, 0.973), ("κ₄ → κ₃ feed", 1.093, 1.096),
       ("K31 at the closure's level (×1)", 1.071, 1.072), ("Strassen level 5", 1.052, 1.000),
       ("dead-side birth-column cut alone", 1.051, 1.004), ("marg2", 1.020, 1.020),
       ("learned κ₄-diagonal corrector", 1.013, 1.014), ("symmetric C_pre + triangular R⁻¹", 1.013, 1.000),
       ("star-diagonal κ₄ birth", 1.012, 1.012), ("C0³ and C0∘K22 pair diagrams", 1.007, 1.008)]
abl = abl[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.6))
y = range(len(abl))
ax.barh(y, [a[1] for a in abl], color=BLUE, height=0.55, label="adjusted score × (switch off)")
ax.plot([a[2] for a in abl], list(y), "o", color=ORANGE, ms=4.5, label="raw MSE × (switch off)")
ax.axvline(1.0, color="k", lw=0.6)
ax.set_yticks(list(y)); ax.set_yticklabels([a[0] for a in abl], fontsize=8)
ax.set_xlim(0.95, 1.62); ax.set_xlabel("× of the full estimator when the component is off (networks 0–31)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("What each component is worth on the final stack", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_ablation.png", dpi=200); plt.close(fig)

# ---- 3. ladder + carrier (§7)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.1), gridspec_kw={"width_ratios": [1.25, 1]})
groups = [("n = 128, L = 8", {"k = 2": 1.4e-4, "k = 3": 9.4e-6, "k = 3 aug.": 4.5e-6, "ours": 3.18e-6, "k = 4": 1.10e-6}),
          ("n = 256, L = 8", {"k = 3": 8.0e-7, "k = 3 aug.": 3.58e-7, "ours": 3.55e-7, "k = 4": 4.7e-8}),
          ("n = 1024, L = 16", {"k = 1": 3.3e-4, "k = 2": 4.5e-6, "k = 3": 4.1e-8, "ours": 2.03e-8})]
order = ["k = 1", "k = 2", "k = 3", "k = 3 aug.", "ours", "k = 4"]
colors = {"k = 1": GREY, "k = 2": GREY, "k = 3": GREY, "k = 3 aug.": "#a6a6a6", "ours": BLUE, "k = 4": GREEN}
x0 = 0; ticks = []; tlabels = []
for name, d in groups:
    for k in order:
        if k in d:
            a1.bar(x0, d[k], color=colors[k], width=0.8)
            a1.text(x0, d[k] * 1.25, k, rotation=90, ha="center", va="bottom", fontsize=6.5)
            x0 += 1
    ticks.append(x0 - 0.5 - len(d) / 2); tlabels.append(name); x0 += 1.2
a1.set_yscale("log"); a1.set_xticks(ticks); a1.set_xticklabels(tlabels, fontsize=8)
a1.set_ylabel("final-layer MSE vs Monte-Carlo truth"); a1.set_ylim(1e-8, 3e-3)
a1.set_title("The cumulant ladder: ARC's reference at k = 1…4, and our chain", fontsize=9)
nets = list(range(10))
y2 = [0.697, 0.770, 0.681, 0.775, 0.649, 0.726, 0.699, 0.615, 0.691, 0.701]
y2k4 = [0.835, 0.862, 0.807, 0.900, 0.740, 0.933, 0.817, 0.752, 0.816, 0.793]
a2.plot(nets, y2, "o", color=BLUE, label="young trees → κ₄ diag + (3,1): ×0.701")
a2.plot(nets, y2k4, "s", color=ORANGE, ms=4, label="κ₄ diagonal only: ×0.824")
a2.axhline(1.0, color="k", lw=0.6); a2.set_ylim(0.55, 1.22)
a2.set_xticks(nets); a2.set_xticklabels([str(n) for n in nets], fontsize=7.5)
a2.set_xlabel("network (1024 × 16)"); a2.set_ylabel("raw MSE × of our chain")
a2.legend(frameon=True, framealpha=0.95, edgecolor="none", fontsize=7, loc="upper left")
a2.set_title("The young-κ₄ reader at full width", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_ladder.png", dpi=200); plt.close(fig)

# ---- 4. the three young trees (§7)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
def node(ax, x, y, text, fc="white"):
    ax.add_patch(Circle((x, y), 0.11, fc=fc, ec="k", lw=0.8, zorder=3)); ax.text(x, y, text, ha="center", va="center", fontsize=7.5, zorder=4)
def edge(ax, p, q, label=None, ls="-"):
    ax.plot([p[0], q[0]], [p[1], q[1]], color="k", lw=0.9, ls=ls, zorder=2)
    if label:
        ax.text((p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.07, label, fontsize=7, ha="center", color=BLUE)
for ax in axes:
    ax.set_xlim(0, 2); ax.set_ylim(0, 1.4); ax.set_aspect("equal"); ax.axis("off")
# path: i -C- j -C- k -C- l, gates d1 d2 d2 d1
ax = axes[0]; P = [(0.3, 0.7), (0.77, 0.7), (1.23, 0.7), (1.7, 0.7)]
for p, q in zip(P, P[1:]): edge(ax, p, q, "C")
for p, t in zip(P, ["d₁", "d₂", "d₂", "d₁"]): node(ax, *p, t)
ax.set_title("path   12·d₁C d₂C d₂C d₁", fontsize=8)
# star: centre j (d3), three C edges to i,k,l (d1)
ax = axes[1]; c = (1.0, 0.7); L = [(0.4, 1.15), (0.4, 0.25), (1.6, 0.7)]
for p in L: edge(ax, c, p, "C")
node(ax, *c, "d₃", fc="#dbe5f1")
for p in L: node(ax, *p, "d₁")
ax.set_title("star   4·d₁C d₃ (C d₁)²", fontsize=8)
# 3+2: i (d1) -C- j (d2); j,k,l joined by the kappa3 blob; k,l gated d1
ax = axes[2]; i = (0.3, 0.7); j = (0.95, 0.7); k = (1.65, 1.1); l = (1.65, 0.3)
edge(ax, i, j, "C")
ax.add_patch(Polygon([j, k, l], closed=True, fc="#fde9d9", ec=ORANGE, lw=0.9, zorder=1))
ax.text(1.32, 0.7, "κ₃(j,k,l)", fontsize=7.5, ha="center", color=ORANGE, zorder=2)
node(ax, *i, "d₁"); node(ax, *j, "d₂", fc="#dbe5f1"); node(ax, *k, "d₁"); node(ax, *l, "d₁")
ax.set_title("3+2   12·d₁(i) C(i,j) d₂(j) κ₃(j,k,l) d₁(k) d₁(l)", fontsize=8)
fig.suptitle("The κ₄ born at one ReLU on the step's covariance C: without the 3+2 graph the young κ₄ is worse than none", fontsize=8.5, y=1.0)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_tree.png", dpi=200, bbox_inches="tight"); plt.close(fig)
print("wrote", [p.name for p in OUT.glob("fig_v4_*.png")])
