"""Task B: today's cost-free ceiling. torch chain, float64, the 10 baked MLPs, cost ignored.
Base = B9 + marg2 (#9A arm marg2_sat_m2.0); + allages (e3age=None), thin384 (corr=split384rr), k4k3=0.5, alone and
stacked. Per-(arm, MLP) results in resB/<arm>/m<m>.json (resumable)."""
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-30_tail-accuracy"))
os.environ.setdefault("PB_THREADS", "1")
import ta  # noqa: E402
ta.RES = HERE / "resB"
BASE = dict(ta.B9, tailr=(("marg2", "sat", None, -2.0, "gauss"),))
ARMS = {"base": BASE, "allages": dict(BASE, e3age=None), "thin384": dict(BASE, corr="split384rr"),
        "k4k3_0.5": dict(BASE, k4k3=0.5), "stack": dict(BASE, e3age=None, corr="split384rr", k4k3=0.5)}
for tag, cfg in ARMS.items():
    ta.run_arm(tag, cfg, ta.SCREEN, early=False)
    print(tag, ta.summary(tag, "base", ta.SCREEN), flush=True)
print("B DONE")
