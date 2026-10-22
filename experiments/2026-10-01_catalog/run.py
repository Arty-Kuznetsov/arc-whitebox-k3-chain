"""One queue job: python run.py <tag> [screen|conf|m0,m1,...] (Prompt B; results in res/<tag>/m<m>.json).
screen = MLPs 0-9 with the lossless early stop at 4 worse vs B9 (vs FULL for FULL+ arms); conf = MLPs 10-41."""
import sys

import cat_grid
import ta

tag = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "screen"
cfg = ta.BASES.get(tag) or cat_grid.cfg_of(tag)
ref_tag = "FULL" if tag.startswith("FULL+") else "B9"
if which == "screen":
    mlps, early = ta.SCREEN, tag not in ta.BASES
elif which == "conf":
    mlps, early = ta.CONF, False
else:
    mlps, early = [int(x) for x in which.split(",")], False
ta.run_arm(tag, cfg, mlps, ref=ta.per_mlp(ref_tag) if early else None, early=early)
print(tag, which, ta.summary(tag, ref_tag, mlps))
