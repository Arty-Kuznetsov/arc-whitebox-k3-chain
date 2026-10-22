"""Task 3 frontier: projected adjusted score of every stack, from res/ (no hand transcription).
projected = 5.3906e-9 (k3m graded) x raw / 2.3462e-8 (k3m torch) x util / 24.835%.
raw: B9 / FULL on the 10 screening MLPs times the arm's ratio to its base, measured on the 10 screening MLPs ("screen")
and on the 32 fresh MLPs 10-41 ("fresh"; FULL's own ratio to B9 on the fresh set is used for the FULL rows).
util: the tailr rules only remove terms (zero FLOPs added; no product-level saving claimed). Two columns: the port-battery
model (cost_pb.util) and the model anchored to the Mac's meter (k3o 23.792%, k3q 19.73% = model x1.008 / x1.033),
with E2 metered as k3q/k3p = 0.9536 and E3 at its model ratio (0.9603). L5 = x0.9165 (the Strassen L5 meter)."""
import numpy as np

import ta  # noqa: E402  (puts the port-battery folder on the path)
import cost_pb  # noqa: E402
import pb
import ta

A = open(ta.HERE / "stack_A.txt").read().strip()
B = open(ta.HERE / "stack_B.txt").read().strip()
U_MODEL = {"B9": cost_pb.util(t_row=-2.5, t_col=-2.5, satrow=2.5), "FULL": cost_pb.util(**ta.FULL_U)}
U_METER = {"B9": 0.23792 * (0.1973 / 0.2069), "FULL": 0.1973 * (U_MODEL["FULL"] / cost_pb.util(
    **{k: v for k, v in ta.FULL_U.items() if k != "e3"}))}


def ratio(tag, base, mlps):
    per, ref = ta.per_mlp(tag), ta.per_mlp(base)
    ms = [m for m in mlps if m in per and m in ref]
    if len(ms) < len(mlps):
        return None, 0, len(ms)
    return float(np.mean([per[m] for m in ms]) / np.mean([ref[m] for m in ms])), sum(per[m] < ref[m] for m in ms), len(ms)


raw_b9 = float(np.mean([ta.per_mlp("B9")[m] for m in ta.SCREEN]))
full_b9_fresh, _, _ = ratio("FULL", "B9", ta.CONF)
full_b9_screen, _, _ = ratio("FULL", "B9", ta.SCREEN)
rows = [("B9 (matched -2.5 + E2)", "B9", "B9"), ("B9 + marg2 (m2.0)", "marg2_sat_m2.0", "B9"),
        ("B9 + marg2 (all)", "marg2_sat_all", "B9"), ("B9 + stack B (marg2 + K22 dead + star col dead)", B, "B9"),
        ("B9 + stack A (9 rules)", A, "B9"), ("FULL (#8 t4_final25a)", "FULL", "FULL"),
        ("FULL + marg2 (m2.0)", "FULL+marg2_sat_m2.0", "FULL"), ("FULL + stack B", "FULL+" + B, "FULL"),
        ("FULL + stack A", "FULL+" + A, "FULL")]
print("| stack | x base (10 screen) | x base (32 fresh) | raw (screen) | util model / meter-anchored | "
      "**projected (fresh, meter)** | projected (screen, model) | + L5 (fresh, meter) |")
print("|---|---|---|---|---|---|---|---|")
for name, tag, base in rows:
    xs, bs, ns = ratio(tag, base, ta.SCREEN) if tag != base else (1.0, 0, 10)
    xf, bf, nf = ratio(tag, base, ta.CONF) if tag != base else (1.0, 0, 32)
    b_screen = raw_b9 * (1.0 if base == "B9" else full_b9_screen)
    b_fresh = raw_b9 * (1.0 if base == "B9" else full_b9_fresh)
    um, ue = U_MODEL[base], U_METER[base]
    proj = lambda raw, u: pb.GRADED_K3M * raw / pb.K3M_RAW * u / pb.UTIL_K3M
    ps = proj(b_screen * xs, um) if xs else None
    pf = proj(b_fresh * xf, ue) if xf else None
    fmt = lambda v, f: f.format(v) if v is not None else ("not run (stack A, which contains it, was)" if nf == 0 else "pending")
    print(f"| {name} | {fmt(xs, '{:.4f}')} ({bs}/{ns}) | {fmt(xf, '{:.4f}')} ({bf}/{nf}) | {b_screen * (xs or 1):.4e} | "
          f"{100 * um:.2f}% / {100 * ue:.2f}% | **{fmt(pf, '{:.3e}')}** | {fmt(ps, '{:.3e}')} | "
          f"{fmt(pf * 0.9165 if pf else None, '{:.3e}')} |")
print(f"\nFULL / B9 raw: screen {full_b9_screen:.4f}, fresh {full_b9_fresh:.4f}")
