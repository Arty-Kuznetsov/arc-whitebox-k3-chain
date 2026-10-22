"""Render the markdown tables of RESULTS.md straight from the JSON results (no hand transcription).

    ../../.venv-torch/bin/python make_tables.py [t1|t1b|t3|t4|all]
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

import cost_pb
import pb

HERE = Path(__file__).resolve().parent
which = sys.argv[1] if len(sys.argv) > 1 else "all"
U0 = cost_pb.util(t_row=-2.5)


def load(pattern):
    d = {}
    for f in sorted(glob.glob(str(HERE / pattern))):
        d.update(json.loads(Path(f).read_text()))
    return d


def band_stats(r):
    logs = r.get("band_log") or []
    cd = max((rec.get("cond", 0.0) for lg in logs for rec in lg), default=float("nan"))
    le = max((rec.get("leg_err", 0.0) for lg in logs for rec in lg), default=float("nan"))
    return cd, le


def worst(r, ref):
    xs = np.array(r["per_mlp"]) / np.array(ref)
    return float(xs.max())


if which in ("t1", "all"):
    d = load("t1_band_n10_*.json")
    ref = json.loads((HERE.parent / "2026-09-29_band" / "e1_n10.json").read_text())
    e1 = {(6, 448): ref["transported 6:448"], (6, 384): ref["transported 6:384"], (5, 448): ref["transported 5:448"]}
    print("| band (a0, r) | projection | QR every k | raw | x k3m | vs E1 float64 | worst MLP vs E1 | max cond(P) | max leg error |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (a0, r) in [(6, 448), (6, 384), (5, 448)]:
        for proj, ks in (("ne", ("1", "2", "3", "5", "inf")), ("qr", ("inf",))):
            for k in ks:
                tag = f"f32 ({a0},{r}) k={k} {proj}"
                if tag not in d:
                    continue
                x = d[tag]
                if "failed" in x:
                    print(f"| ({a0},{r}) | {proj} | {k} | FAILED | | | | | |")
                    continue
                cd, le = band_stats(x)
                ve = x["raw"] / e1[(a0, r)]["raw"]
                print(f"| ({a0},{r}) | {proj} | {k} | {x['raw']:.4e} | {x['x']:.4f} | {100 * (ve - 1):+.2f}% | "
                      f"x{worst(x, e1[(a0, r)]['per_mlp']):.3f} | {cd:.0f} | {le:.1e} |")
    a = d.get("anchor f64 k=1 (6,448) ne")
    if a:
        rel = max(abs(p / q - 1) for p, q in zip(a["per_mlp"], e1[(6, 448)]["per_mlp"]))
        print(f"\nanchor float64 k=1 (6,448): raw {a['raw']:.6e} vs E1 {e1[(6, 448)]['raw']:.6e}; "
              f"max per-MLP relative difference {rel:.2e}")

if which in ("t1b", "all"):
    d = load("t1b_band_n10_*.json")
    d.update(load("t1c_band_n10_*.json"))
    d1 = load("t1_band_n10_*.json")
    print("\n| band (a0, r) | entry | init | raw | x k3m | better | util (model) | util x | projected x | max cond | max leg err |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    rows = [(t, d1[t], "post") for t in d1 if " qr" in t] + [(t, d[t], "pre") for t in d]
    for tag, x, entry in rows:
        if "failed" in x:
            print(f"| {tag} | FAILED |")
            continue
        a0, r = [int(v) for v in tag.split("(")[1].split(")")[0].split(",")]
        init = "rr" if "init=rr" in tag else "cols"
        u = cost_pb.util(t_row=-2.5, band=dict(a0=a0, r=r, k=None, proj="qr", entry=entry, init=init))
        cd, le = band_stats(x)
        print(f"| ({a0},{r}) | {entry} | {init} | {x['raw']:.4e} | {x['x']:.4f} | {x['better']}/10 | {100 * u:.2f}% | "
              f"{u / U0:.4f} | **{x['x'] * u / U0:.4f}** | {cd:.0f} | {le:.1e} |")

if which in ("t3", "all"):
    for base in ("k3m", "m225"):
        b = load(f"t3_base_{base}_n10.json")
        d = {}
        for g in ("e2", "e2c", "e2ch", "e3", "e4", "mt"):
            d.update(load(f"t3_{g}_{base}_n10.json"))
        bb = b.get(f"{base} base")
        if bb is None:
            continue
        print(f"\nbase {base}: raw {bb['raw']:.4e}")
        print("| arm | raw | raw x base | better | util (model) | util x base | projected x base | projected-better |")
        print("|---|---|---|---|---|---|---|---|")
        for tag, x in d.items():
            if "failed" in x:
                print(f"| {tag} | FAILED {x['failed'][:50]} |")
                continue
            if "util_x" not in x:
                continue
            print(f"| {tag.replace(base + ' + ', '')} | {x['raw']:.4e} | {x['raw_x']:.4f} | {x['better']}/10 | "
                  f"{100 * x['util']:.2f}% | {x['util_x']:.4f} | **{x['proj_x']:.4f}** | {x['proj_better']}/10 |")

if which in ("t4", "all"):
    rows = []
    for f in sorted(glob.glob(str(HERE / "t4_*_n10.json"))):
        for tag, x in json.loads(Path(f).read_text()).items():
            if "projected" in x:
                rows.append((Path(f).stem, tag, x))
    m = json.loads((HERE / "t3_base_m225_n10.json").read_text())["m225 base"]
    u_m = cost_pb.util(t_row=-2.25, t_col=-2.25)
    adj_m = pb.GRADED_K3M * m["raw"] / pb.K3M_RAW * u_m / pb.UTIL_K3M
    print("\n| file | stack (on the matched cut) | raw | raw x k3m | better vs k3m | util (model) | util x k3m | "
          "projected adjusted | x k3m | + L5 x0.9165 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    print(f"| — | k3m (graded 332403) | {pb.K3M_RAW:.4e} | 1.0000 | — | 24.84% | 1.0000 | 5.3906e-09 | 1.000 | — |")
    print(f"| t3_base_m225 | matched −2.25 | {m['raw']:.4e} | {m['raw'] / pb.K3M_RAW:.4f} | — | {100 * u_m:.2f}% | "
          f"{u_m / pb.UTIL_K3M:.4f} | {adj_m:.4e} | {adj_m / pb.GRADED_K3M:.3f} | {adj_m * cost_pb.L5_METER:.4e} |")
    base_per = json.loads((HERE / "t3_base_k3m_n10.json").read_text())["k3m base"]["per_mlp"]
    for stem, tag, x in sorted(rows, key=lambda z: -z[2]["projected"]):
        b = sum(1 for p, q in zip(x["per_mlp"], base_per) if p < q)
        print(f"| {stem} | {tag} | {x['raw']:.4e} | {x['raw_x']:.4f} | {b}/10 | {100 * x['util']:.2f}% | "
              f"{x['util_x']:.4f} | **{x['projected']:.4e}** | {x['projected'] / pb.GRADED_K3M:.3f} | "
              f"{x['projected_l5']:.4e} |")
