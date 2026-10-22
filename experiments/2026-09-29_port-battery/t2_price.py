"""Task 2 of the port battery: price the band honestly (cost_pb.py) and apply the kill rule to every measured arm.

For every band arm measured in task 1 / 1b (JSON), the utilisation is modelled with its exact construction:
P transport, the entrant's projection (ne: G = P^T P + P^T X + LU solve; qr: QR of P + Q^T X + solve with R), the
materialisation A, B = P Z, the compressed pairing, the QR + coefficient rotation ONLY on QR layers, and (entry
"post") the entrant's own dense transport at age a0. Projected = raw ratio x util ratio (both vs k3m).
Kill: projected worse than x0.97 of the base.

    ../../.venv-torch/bin/python t2_price.py
"""
import glob
import json
import re
from pathlib import Path

import cost_pb

HERE = Path(__file__).resolve().parent
rows = {}
for f in sorted(glob.glob(str(HERE / "t1_band_n10_*.json")) + glob.glob(str(HERE / "t1b_band_n10_*.json"))
                + glob.glob(str(HERE / "t1c_band_n10_*.json"))):
    for tag, r in json.loads(Path(f).read_text()).items():
        rows[tag] = r
u0 = cost_pb.util(t_row=-2.5)
out = {}
print(f"{'arm':44s} {'raw x':>7s} {'util':>7s} {'util x':>7s} {'proj x':>7s} {'max leg err':>11s} {'max cond':>9s}  verdict")
for tag, r in rows.items():
    if "failed" in r:
        print(f"{tag:44s} FAILED {r['failed'][:60]}")
        continue
    m = re.search(r"\((\d+),(\d+)\) k=(\w+) (ne|qr)( pre)?", tag)
    if not m:
        continue
    a0, rr, k, proj, pre = int(m.group(1)), int(m.group(2)), m.group(3), m.group(4), bool(m.group(5))
    band = dict(a0=a0, r=rr, k=None if k == "inf" else int(k), proj=proj, entry="pre" if pre else "post",
                init="rr" if "init=rr" in tag else "cols")
    u = cost_pb.util(t_row=-2.5, band=band)
    logs = r.get("band_log") or []
    le = max((rec.get("leg_err", 0.0) for lg in logs for rec in lg), default=float("nan"))
    cd = max((rec.get("cond", 0.0) for lg in logs for rec in lg), default=float("nan"))
    px = r["x"] * u / u0
    verdict = "KILL" if px > 0.97 else "survives"
    if "f64" in tag:
        verdict = "anchor"
    out[tag] = dict(raw=r["raw"], raw_x=r["x"], better=r["better"], util=u, util_x=u / u0, proj_x=px,
                    max_leg_err=le, max_cond=cd, band=band, verdict=verdict)
    print(f"{tag:44s} {r['x']:7.4f} {100 * u:6.2f}% {u / u0:7.4f} {px:7.4f} {le:11.2e} {cd:9.0f}  {verdict}")
(HERE / "t2_price.json").write_text(json.dumps(out, indent=1))
