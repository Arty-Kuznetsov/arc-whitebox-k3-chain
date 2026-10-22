"""Task 3 of the port battery: the cheap battery, each arm alone on k3m AND on k3m + matched cut (-2.25). 10 MLPs.

  E2  saturated-ROW cut of the D pairing (chain.py satrow) WITH the kappa4 level restored: satrow_k4="restore" = a
      second step on the full D, only its K22 level kept (upper bound); "sep" = the O(S n^2) production form
      (the separable part of the level's linear response to the cut rows, d2-suppressed terms dropped).
      theta in {3.0 (kill test first), 3.5, 3.25, 2.75, 2.5}.
      Kill: raw worse than +1% at 3.0, or fewer than 9/10 MLPs better on the projected score (raw_i x util ratio).
  E3  receiver rows (chain.py recv=(theta_out, arm)): rows of neurons with alpha_l < theta_out at the CURRENT layer
      are not computed in the leg transport. arm "a": their t_i from the age-1 source only; arm "b": plus the
      repeated-index rule on the previous layer's (t, D); "zero": t_i and the star kappa4 row dropped (mechanism).
      theta_out in {-3.0, -3.25, -3.5}. Kill a threshold at raw worse than +0.5% (arm a) / +1.0% (arm b); arm b is
      run only where arm a is killed.
  E4  age-dependent matched cut (chain.py agecut=(a_min, t_old)): legs of age >= 4 cut at t_old in {-2.0, -1.5} on
      rows (gating) and columns (from age 4), young legs at the base threshold.

    PB_THREADS=1 ../../.venv-torch/bin/python t3_battery.py <group: e2|e3|e4> <base: k3m|m225>
"""
import json
import sys
from pathlib import Path

import numpy as np

import cost_pb
import pb

HERE = Path(__file__).resolve().parent
group = sys.argv[1]
basename = sys.argv[2] if len(sys.argv) > 2 else "k3m"
BASES = {"k3m": (pb.K3M, dict(t_row=-2.5)), "m225": (pb.matched(-2.25), dict(t_row=-2.25, t_col=-2.25))}
base, ucfg = BASES[basename]
out_path = HERE / f"t3_{group if group in ('e2c', 'e2ch', 'mt') else group[:2]}_{basename}_n10.json"
u0 = cost_pb.util(**ucfg)


BASEFILE = HERE / f"t3_base_{basename}_n10.json"   # the base, computed once (t3_bases.py) and shared
bout = pb.run_arms([(f"{basename} base", dict(base))], BASEFILE)
b = bout[f"{basename} base"]


def arm(tag, cfg, ukw):
    out = pb.run_arms([(tag, cfg)], out_path, ref_per=b["per_mlp"])
    r = out[tag]
    if "failed" in r:
        return None
    ur = cost_pb.util(**dict(ucfg, **ukw)) / u0
    xs = np.array(r["per_mlp"]) / np.array(b["per_mlp"])
    r.update(util=cost_pb.util(**dict(ucfg, **ukw)), util_x=ur, proj_x=float(r["raw"] / b["raw"] * ur),
             proj_better=int((xs * ur < 1.0).sum()), raw_x=float(r["raw"] / b["raw"]))
    out[tag] = r
    out_path.write_text(json.dumps(out, indent=1))
    print(f"      -> raw x{r['raw_x']:.4f}  util x{ur:.4f}  projected x{r['proj_x']:.4f}  "
          f"projected-better {r['proj_better']}/10", flush=True)
    return r


if group == "e2":
    r = arm(f"{basename} + E2 satrow 3.0 restore", dict(base, satrow=3.0, satrow_k4="restore"), dict(satrow=3.0))
    killed = r is None or r["raw_x"] > 1.01 or r["proj_better"] < 9
    print(f"   E2 kill test at 3.0: {'KILLED' if killed else 'survives'}", flush=True)
    if not killed:
        for th in (3.5, 3.25, 2.75, 2.5):
            arm(f"{basename} + E2 satrow {th} restore", dict(base, satrow=th, satrow_k4="restore"), dict(satrow=th))
        for th in (3.0, 2.75, 2.5):
            arm(f"{basename} + E2 satrow {th} sep", dict(base, satrow=th, satrow_k4="sep"), dict(satrow=th))
elif group == "e2b":   # re-prioritised E2 after the 3.0 kill test passed: the production form first
    for th, mode in ((3.0, "sep"), (2.5, "restore"), (2.5, "sep"), (2.75, "restore"), (2.75, "sep"),
                     (3.25, "restore"), (3.5, "restore")):
        arm(f"{basename} + E2 satrow {th} {mode}", dict(base, satrow=th, satrow_k4=mode), dict(satrow=th))
elif group == "e2c":   # the restore arms kept improving (3.0 x0.9917, 2.5 x0.9534): push the threshold further
    for th, mode in ((2.25, "restore"), (2.25, "sep"), (2.0, "restore"), (2.0, "sep")):
        arm(f"{basename} + E2 satrow {th} {mode}", dict(base, satrow=th, satrow_k4=mode), dict(satrow=th))
elif group == "e2ch":  # E2 mechanism: which channel of the cut rows carries the gain (exact level in all arms)
    for ch in ("cov", "leg"):
        arm(f"{basename} + E2 satrow 2.5 restore ch={ch}", dict(base, satrow=2.5, satrow_k4="restore", satrow_ch=ch),
            dict(satrow=2.5))
elif group == "mt":    # side probe (not in the brief, zero FLOPs): no marginal Edgeworth correction in the tails
    for th in (2.5, 3.0):
        arm(f"{basename} + margtail {th}", dict(base, margtail=th), dict())
elif group == "e2k":   # the k3m side: the kill test and the production form at the two informative thresholds
    for th, mode in ((3.0, "restore"), (3.0, "sep"), (2.5, "sep")):
        arm(f"{basename} + E2 satrow {th} {mode}", dict(base, satrow=th, satrow_k4=mode), dict(satrow=th))
elif group == "e3":
    for th in (-3.0, -3.25, -3.5):
        r = arm(f"{basename} + E3 recv {th} a", dict(base, recv=(th, "a")), dict(e3=dict(theta=th, arm="a")))
        if r is None or r["raw_x"] > 1.005:
            print(f"   E3 arm a at {th}: KILLED (> +0.5%) -> arm b", flush=True)
            rb = arm(f"{basename} + E3 recv {th} b", dict(base, recv=(th, "b")), dict(e3=dict(theta=th, arm="b")))
            if rb is None or rb["raw_x"] > 1.01:
                print(f"   E3 arm b at {th}: KILLED (> +1.0%)", flush=True)
        else:
            print(f"   E3 arm a at {th}: survives", flush=True)
    arm(f"{basename} + E3 recv -3.0 zero", dict(base, recv=(-3.0, "zero")), dict(e3=dict(theta=-3.0, arm="a")))
elif group == "e4":
    for t_old in (-2.0, -1.5):
        arm(f"{basename} + E4 agecut (4, {t_old})", dict(base, agecut=(4, t_old)), dict(e4=dict(a_min=4, t_old=t_old)))
else:
    raise SystemExit(f"unknown group {group}")
