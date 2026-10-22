"""Markdown tables of RESULTS.md from res/ (no hand transcription): python report.py [t1|grid3|stack]
t1: every catalog arm on B9 (raw x, better, status, util from cost_cat, projected x, fresh-set x where measured)."""
import sys

import cat_grid
import cost_cat
import cost_pb
import ta

which = sys.argv[1] if len(sys.argv) > 1 else "t1"
UB9 = cost_cat.util(ta.B9)


def fresh(tag, ref):
    s = ta.summary(tag, ref, ta.CONF)
    return f"{s['x']:.4f} ({s['better']}/{s['n']})" if s else "—"


if which == "t1":
    rows = []
    for t in cat_grid.ARMS:
        if t == "marg2":
            continue
        s = ta.summary(t)
        if s is None:
            continue
        u = cost_cat.util(cat_grid.cfg_of(t))
        rows.append((s["x"] * u / UB9, t, s, u))
    print("| arm | raw x B9 (10) | better | status | util (est.) | **projected x B9** | raw x B9, 32 fresh |")
    print("|---|---|---|---|---|---|---|")
    for pr, t, s, u in sorted(rows, key=lambda r: r[2]["x"]):
        st = "survives (raw)" if s["status"] == "done" and s["x"] < 1 and s["better"] >= 7 else (
            "killed" if s["status"] in ("KILLED", "done") else s["status"])
        print(f"| {t} | {s['x']:.4f} | {s['better']}/{s['n']} | {st} | {100 * u:.2f}% | **{pr:.4f}** | {fresh(t, 'B9')} |")
elif which == "grid3":
    U_FULL = 0.1973 * (cost_pb.util(**ta.FULL_U) / cost_pb.util(**{k: v for k, v in ta.FULL_U.items() if k != "e3"}))
    u_full = cost_pb.util(**ta.FULL_U)
    print("| on FULL | raw x FULL (10) | better | status | util | **projected x FULL** |\n|---|---|---|---|---|---|")
    rows = []
    for tag in cat_grid.GRID3:
        s = ta.summary(tag, "FULL")
        kw = dict(ta.FULL_U)
        part = tag[5:]
        if part.startswith("g_"):
            t, s_, r = (float(x) for x in part[2:].split("_"))
            kw.update(t_row=t, t_col=t, satrow=s_, e3=dict(theta=r, arm="b"))
            name = f"cut {t}, E2 {s_}, recv {r}"
        else:
            kw["band"] = dict(kw["band"], a0=6, r=384)
            name = "band (6,384)"
        u = U_FULL * cost_pb.util(**kw) / u_full
        rows.append((s["x"] * u / U_FULL, name, s, u))
    print(f"| FULL (cut -2.5, E2 2.5, recv -3.0; band (5,448)) | 1.0000 | — | base | {100 * U_FULL:.2f}% | **1.0000** |")
    for pr, name, s, u in sorted(rows):
        print(f"| {name} | {s['x']:.4f} | {s['better']}/{s['n']} | {s['status']} | {100 * u:.2f}% | **{pr:.4f}** |")
elif which == "stack":
    import pb
    raw_b9 = sum(ta.per_mlp("B9")[m] for m in ta.SCREEN) / 10
    f_b9_screen = ta.summary("FULL")["x"]
    f_b9_fresh = ta.summary("FULL", "B9", ta.CONF)["x"]
    UF = 0.1973 * (cost_pb.util(**ta.FULL_U) / cost_pb.util(**{k: v for k, v in ta.FULL_U.items() if k != "e3"}))
    print("| stack | x base (10) | x base (32 fresh) | util | **projected (fresh)** | + L5 |\n|---|---|---|---|---|---|")
    for name, tag, base in (("B9", "B9", "B9"), ("B9 + marg2", "marg2_sat_m2.0", "B9"), ("B9 + marg2 + q22x2", "marg2+q22x2", "B9"),
                            ("B9 + marg2 + q22x2 + thin48 (greedy step, rejected)", "marg2+q22x2+thin48", "B9"),
                            ("FULL", "FULL", "FULL"), ("FULL + marg2", "FULL+marg2_sat_m2.0", "FULL"),
                            ("FULL + marg2 + q22x2", "FULL+marg2+q22x2", "FULL")):
        s = ta.summary(tag, base) if tag != base else {"x": 1.0, "better": 0, "n": 10}
        c = ta.summary(tag, base, ta.CONF) if tag != base else {"x": 1.0, "better": 0, "n": 32}
        u = UB9 if base == "B9" else UF
        if tag == "marg2+q22x2+thin48":
            u = cost_cat.util(cat_grid.cfg_of(tag))
        braw = raw_b9 * (1.0 if base == "B9" else f_b9_fresh)
        x = c["x"] if c and c["n"] == 32 else None
        proj = pb.GRADED_K3M * braw * x / pb.K3M_RAW * u / pb.UTIL_K3M if x else None
        cs = f"{c['x']:.4f} ({c['better']}/{c['n']})" if c else "not run"
        ps = f"{proj:.3e}" if proj else "pending"
        l5 = f"{proj * 0.9165:.3e}" if proj else "—"
        print(f"| {name} | {s['x']:.4f} ({s['better']}/{s['n']}) | {cs} | {100 * u:.2f}% | **{ps}** | {l5} |")
