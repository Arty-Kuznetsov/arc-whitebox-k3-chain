"""One queue job: python run.py <tag> [screen|conf|m0,m1,...]; tags = bases (B9, FULL, K3M) or kv_grid.ARMS."""
import sys

import kv
import kv_grid
import ta

tag = sys.argv[1]
which = sys.argv[2] if len(sys.argv) > 2 else "screen"
cfg = ta.BASES.get(tag) or kv_grid.cfg_of(tag)
ref_tag = "FULL" if tag.startswith("FULL+") else "B9"
if which == "screen":
    mlps, early = ta.SCREEN, tag not in ta.BASES
elif which == "conf":
    mlps, early = ta.CONF, False
else:
    mlps, early = [int(x) for x in which.split(",")], False
ta.run_arm(tag, cfg, mlps, ref=ta.per_mlp(ref_tag) if early else None, early=early)
print(tag, which, ta.summary(tag, ref_tag, mlps))
