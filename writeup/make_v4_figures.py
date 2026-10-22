"""Figures of write-up v4 from logged data (run: uvx --with matplotlib python make_v4_figures.py, from writeup/).
  fig_v4_history.png  -- graded score of every k=3 version, with the pre-submission projection of the last nine
  fig_v4_costmap.png  -- the shipped estimator's FLOPs and backend time by call site (namespaced copy, one network)
  fig_v4_projection.png -- the random-projection theorem on the real matrices, and the payoff of a rank-r correction
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
OUT = HERE / "figures"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# ---- 1. graded history (AIcrowd submission pages; projections from the local 100-network runs)
hist = [("k3b", 1.915e-8, None), ("k3e", 9.562e-9, None), ("k3f", 8.286e-9, None), ("k3h", 7.401e-9, None),
        ("k3i", 6.728e-9, None), ("k3j", 5.939e-9, None), ("k3l", 5.5984e-9, None), ("k3m", 5.3906e-9, None),
        ("k3o", 5.1127e-9, 5.12e-9), ("k3p", 4.5644e-9, 4.57e-9), ("k3q", 4.1279e-9, 4.14e-9), ("k3r", 4.0579e-9, 4.06e-9),
        ("k3t", 3.9554e-9, 3.95e-9), ("k3v", 3.9102e-9, 3.91e-9), ("k3w", 3.8576e-9, 3.86e-9), ("k3x", 3.8069e-9, 3.80e-9),
        ("k3y", 3.7573e-9, 3.76e-9), ("k3z", 3.7357e-09, 3.735e-9)]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
xs = range(len(hist))
ax.semilogy(list(xs), [h[1] for h in hist], "o-", color="#1f4e79", ms=4, label="graded (50 public of 100 networks)")
px = [i for i, h in enumerate(hist) if h[2]]
ax.semilogy(px, [hist[i][2] for i in px], "x", color="#c55a11", ms=6, label="projected before submission")
ax.set_xticks(list(xs)); ax.set_xticklabels([h[0] for h in hist], rotation=0, fontsize=7.5)
ax.set_ylabel("adjusted score"); ax.legend(frameon=False, fontsize=8)
ax.set_title("Every version since k3o landed within 0.3% of its projection", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_history.png", dpi=200); plt.close(fig)

# ---- 2. cost map (experiments/2026-10-02_costmap/prof_k3y_n1.json)
d = json.load(open(EXP / "2026-10-02_costmap" / "prof_k3z_n1.json"))
ns = d["results"]["breakdowns"]["estimator"]["by_namespace"]
tot_f = sum(v["flops_used"] for v in ns.values()); tot_t = sum(v.get("flopscope_backend_time_s", 0) for v in ns.values())
groups = [("leg transport", ["transport"]), ("D pairing (non-members)", ["dpair"]),
          ("band members materialised", ["band_members"]), ("members' D pairing", ["mpair"]),
          ("band entry", ["band_entry"]), ("covariance (C W, C_pre)", ["ctransport", "cpre"]),
          ("the step's mathematics", ["nonlin", "k4levels", "k4corr", "readout", "births"]),
          ("rest", ["estimator-client", "pairprep", "thin_transport"])]
lab = [g[0] for g in groups]
fs = [sum(ns["estimator." + k]["flops_used"] for k in g[1]) / tot_f * 100 for g in groups]
ts = [sum(ns["estimator." + k].get("flopscope_backend_time_s", 0) for k in g[1]) / tot_t * 100 for g in groups]
fig, ax = plt.subplots(figsize=(7.2, 3.0))
y = range(len(groups))
ax.barh([i + 0.2 for i in y], fs, height=0.38, color="#1f4e79", label="FLOPs")
ax.barh([i - 0.2 for i in y], ts, height=0.38, color="#9dc3e6", label="backend time")
ax.set_yticks(list(y)); ax.set_yticklabels(lab); ax.invert_yaxis(); ax.set_xlabel("% of the shipped estimator's total")
for i, (f, t) in enumerate(zip(fs, ts)):
    ax.text(max(f, t) + 0.5, i, f"{f:.1f} / {t:.1f}", va="center", fontsize=7)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax.set_title("Four fifths of the budget moves κ₃; the step's mathematics is 3%", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_costmap.png", dpi=200); plt.close(fig)

# ---- 3. the random-projection theorem (cloud #11: res2/theory_m.json, res2/split_m.json) and the rank payoff
CE = EXP / "2026-10-02_cov-error" / "res2"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8))
th = json.load(open(CE / "theory_0.json"))                      # the theorem was checked on network 0's matrices
ls = sorted(int(k) for k in th)                                   # an identity: it holds for any fixed dC
a1.plot(ls, [th[str(l)]["ratio_off"] for l in ls], "o-", color="#1f4e79", ms=3,
        label="net 0: observed / predicted energy")
for m, col in ((0, "#1f4e79"), (1, "#c55a11")):
    sp = json.load(open(CE / f"split_{m}.json"))["layers"]
    sh = [(r["l"], r["all"]["S_off"] / r["all"]["S_tot"]) for r in sp if r["l"] >= 3]   # l < 3: below noise
    a1.plot([s[0] for s in sh], [s[1] for s in sh], "s--", color=col, ms=3, alpha=0.6,
            label=f"net {m}: off-diagonal share")
a1.axhline(1.0, color="grey", lw=0.6)
a1.set_xlabel("layer l (error at l+1)"); a1.set_ylim(0.5, 1.25); a1.set_xticks(range(0, 15, 2))
a1.set_ylabel("ratio / share")
a1.set_title("Variance error = ΔC₀ seen through wᵢ", fontsize=9)
a1.legend(frameon=False, fontsize=6.3, loc="lower left")
ranks = [1, 4, 16, 64, 256]
pay = {0: [1.002, 0.927, 0.897, 0.764, 0.691], 1: [1.066, 1.061, 0.985, 0.830, 0.837]}
full = {0: 0.676, 1: 0.849}
for m, col in ((0, "#1f4e79"), (1, "#c55a11")):
    a2.semilogx(ranks, pay[m], "o-", color=col, ms=3, label=f"network {m} (full: ×{full[m]:.3f})")
    a2.axhline(full[m], color=col, lw=0.6, ls=":")
a2.axhline(1.0, color="grey", lw=0.6)
a2.set_xlabel("rank r of the exact off-diagonal correction"); a2.set_ylabel("raw ×")
a2.set_title("Rank ≤ 4: 40–65% of the energy, no payoff", fontsize=9); a2.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(OUT / "fig_v4_projection.png", dpi=200); plt.close(fig)
print("figures written:", sorted(p.name for p in OUT.glob("fig_v4_*.png")))
