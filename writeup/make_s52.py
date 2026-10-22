"""Write-up §5.2: the ablation of the production estimator, one switch off at a time, from
experiments/2026-10-03_evidence (runs_mac/ preferred: Mac-vs-Mac; runs/ for arms only the cloud ran: cloud-vs-cloud).
Prints the markdown block that replaces §5.2's body in v4-s5-s7.md (run: python3 make_s52.py --write)."""
import json
import sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent
E = H.parent / "experiments" / "2026-10-03_evidence"
B = 2 ** 41
DESC = {  # arm -> (what is switched off, section)
    "cuts_off": ("the matched dead-neuron cuts (rows and birth columns)", "3.5"),
    "satrow_off": ("E2: saturated rows of D out, κ₄ level restored", "3.5"),
    "band_off": ("the riding band (old sources carried densely)", "3.6"),
    "l5_off": ("Strassen level 5 (level 4 everywhere)", "3.8"),
    "colcut_off": ("the dead-side birth-column cut alone", "3.5"),
    "marg2_off": ("marg2 (Gaussian k = 2 pair coefficient outside the dead tail)", "3.5"),
    "k4corr_off": ("the learned κ₄-diagonal corrector", "3.7"),
    "csym_binv_off": ("symmetric C_pre and the explicit triangular R⁻¹", "3.8"),
    "k31scale_1": ("K31 at the closure's own level (×1 instead of ×0.65)", "3.4"),
    "feed_off": ("the κ₄ → κ₃ feed", "3.3"),
    "paths_off": ("the path births (augmented (1,1,1) births at the first readout)", "3.4"),
    "c0d_off": ("the edge × κ₃ diagram C0·D", "3.2"),
    "marg_off": ("the marginal Edgeworth pair coefficients", "3.3"),
    "k31_off": ("the closure's κ₄ (3,1) slice K31", "3.3"),
    "star_off": ("the star-diagonal κ₄ birth", "3.4"),
    "c0k22_off": ("the augmented pair diagrams C0³ and C0∘K22", "3.4"),
}


def load(d, name):
    f = E / d / f"{name}_n32.json"
    if not f.exists():
        return None
    return {p["mlp_index"]: p for p in json.loads(f.read_text())["results"]["per_mlp"]}


def ratios(d, base, idx):
    a = np.mean([d[i]["adjusted_final_layer_score"] for i in idx]) / np.mean([base[i]["adjusted_final_layer_score"] for i in idx])
    r = np.mean([d[i]["final_layer_mse"] for i in idx]) / np.mean([base[i]["final_layer_mse"] for i in idx])
    u = np.mean([d[i]["flops_used"] for i in idx]) / np.mean([base[i]["flops_used"] for i in idx])
    w = sum(d[i]["adjusted_final_layer_score"] > base[i]["adjusted_final_layer_score"] for i in idx)
    return a, r, u, w


rows, missing = [], []
for arm, (what, sec) in DESC.items():
    src = None
    for d in ("runs_mac", "runs"):
        x = load(d, arm)
        if x is not None:
            src = (d, x, load(d, "base"))
            break
    if src is None:
        missing.append(what)
        continue
    d, x, base = src
    a, r, u, w = ratios(x, base, range(32))
    a2, _, _, w2 = ratios(x, base, range(10, 32))
    rows.append((a, what, sec, a, r, u, w, a2, w2, "M" if d == "runs_mac" else "C"))
rows.sort(key=lambda t: -t[0])
out = ["The production estimator (k3y; the designated k3z differs from it only by exact FLOP savings) with ONE switch",
       "turned off at a time, on public networks 0–31, and separately on 10–31 — none of the method's fitted choices (the",
       "corrector's training set, the cut thresholds, K31's level) was made on them [M]. Ratios against the switch on; each",
       "arm is compared with a base run on the same machine (M = laptop, C = a cloud session: per-network float32 rounding",
       "differs between the two by up to ±4%, the FLOPs by ≤3·10⁻⁶).",
       "",
       "| switched off | § | adjusted × (0–31) | raw × | util × | networks worse | adjusted × (10–31) | run |",
       "|---|---|---|---|---|---|---|---|"]
for _, what, sec, a, r, u, w, a2, w2, src in rows:
    out.append(f"| {what} | {sec} | **×{a:.3f}** | ×{r:.3f} | ×{u:.3f} | {w}/32 | ×{a2:.3f} ({w2}/22) | {src} |")
if missing:
    out.append("")
    out.append("Not yet measured: " + "; ".join(missing) + ".")
out += ["",
        "Three things stand out [M]. **The step's first-order scale-mixture set is the most valuable part of the method,",
        "and it is free:** without the marginal Edgeworth pair coefficients the error is ×1.56, without the edge × κ₃",
        "diagram C0·D ×1.30, without the closure's K31 ×1.17 — all O(n²) — then the path births ×1.10, the κ₄ → κ₃ feed",
        "×1.09 and K31's truncation level ×1.07. On the final stack each of them is load-bearing even though, as §3 showed,",
        "none of them paid alone when it was added: the consistency of §3.5 seen from the other end. **The consistency",
        "cuts are accuracy gains as well as savings** — with the matched dead cuts off, raw error is ×1.24 worse; with E2",
        "off, ×1.07. **The cost levers buy FLOPs with little or no accuracy:** the riding band costs 3% of raw error and",
        "saves 11% of the bill; level-5 Strassen and the symmetric/triangular savings change the output only by rounding.",
        "The one learned component is worth ×1.010 out of its training sample — the value it showed on the grader",
        "(×0.988⁻¹ = ×1.012). Every component worth more than 5% makes at least 30 of the 32 networks worse when removed,",
        "and the 22 networks outside every fitted choice give the same ratios to the second digit."]
block = "\n".join(out)
print(block)
if "--write" in sys.argv:
    p = H / "v4-s5-s7.md"
    s = p.read_text()
    i = s.index("### 5.2 What each component is worth on the final stack")
    j = s.index("### 5.3 Where the remaining error is")
    s = s[:i] + "### 5.2 What each component is worth on the final stack\n\n" + block + "\n\n" + s[j:]
    p.write_text(s)
    print("written")
