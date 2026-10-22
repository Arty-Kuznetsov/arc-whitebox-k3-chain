"""Summary of prop3_truth.py outputs: per width, mean (sd) of measured/predicted share, the share itself, f_AD, q_v,
cos(v, m), predicted vs measured R^2 of the next layer's skew law, and Prop. 5's slope check.
usage: python p3t_summary.py"""
import json, glob
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
for n in (64, 128, 256):
    rows = [r for f in sorted(glob.glob(str(HERE / "res" / f"p3t_n{n}_*.json"))) for r in json.load(open(f))["layers"].values()]
    if not rows: continue
    g = lambda k: np.array([r[k] for r in rows])
    ratio, sm, sp = g("ratio"), g("share_meas"), g("share_pred")
    dc = g("c_meas_n") / g("c_pred_n") - 1
    print(f"n = {n}: {len(rows)} (layer, net) pairs | share meas {sm.mean():.3f} pred {sp.mean():.3f} | meas/pred {ratio.mean():.2f} (sd {ratio.std(ddof=1):.2f}) | "
          f"f_AD {g('f_ad').mean():.2f} q_v {g('q_v').mean():.2f} cos(v,m) {g('cos_v_m').min():.3f}-{g('cos_v_m').max():.3f} | "
          f"R2 pred {g('R2_pred').mean():.3f} meas {g('R2_meas').mean():.3f} (corr {np.corrcoef(g('R2_pred'), g('R2_meas'))[0,1]:.2f}) | "
          f"slope |meas/pred-1| median {np.median(np.abs(dc)):.3f}")
