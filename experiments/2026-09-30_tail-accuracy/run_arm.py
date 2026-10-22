"""One job of the queue: python run_arm.py <tag> [screen|conf|full|m0,m1,...]
<tag> is a base (B9, FULL, K3M) or an arm of grid.ARMS. screen = MLPs 0-9 with early stopping against B9 (or FULL for
arms whose name starts with FULL+); conf = MLPs 10-41, no early stop; an explicit list runs those MLPs only."""
import sys

import grid
import ta

tag = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "screen"
cfg = ta.BASES.get(tag) or grid.cfg_of(tag)
if which == "screen":
    mlps, early = ta.SCREEN, tag not in ta.BASES
elif which == "conf":
    mlps, early = ta.CONF, False
else:
    mlps, early = [int(x) for x in which.split(",")], False
ref = ta.per_mlp("FULL" if tag.startswith("FULL+") else "B9") if early else None
ta.run_arm(tag, cfg, mlps, ref=ref, early=early)
s = ta.summary(tag, "FULL" if tag.startswith("FULL+") else "B9", mlps)
print(tag, which, s)
