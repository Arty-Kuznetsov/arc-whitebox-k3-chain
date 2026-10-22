"""Per-MLP comparison of two 100-MLP `whest run --format json` reports.

    python compare_n100.py <new.json> <old.json> [label_new] [label_old]

Prints the aggregate block (adjusted / final MSE / all-layers / util / failures / residual) and the per-MLP
distribution of the adjusted ratio, which is what decides a lever: k3j..k3l were all shipped on "N of 100
MLPs improve", not on the mean alone."""
import json, sys
import numpy as np


def load(p):
    d = json.load(open(p))
    r = d["results"]
    pm = r["per_mlp"]
    def col(*names):
        for nm in names:
            if nm in pm[0]:
                return np.array([m[nm] for m in pm], dtype=float)
        return None
    fl = col("flops_used")
    return dict(agg=r, adj=col("adjusted_final_layer_score"), fin=col("final_layer_mse"),
                util=(fl / 2 ** 41 if fl is not None else None), all_=col("all_layers_mse"),
                resid=col("residual_wall_time_s"), wall=col("wall_time_s"), keys=list(pm[0].keys()))


a, b = load(sys.argv[1]), load(sys.argv[2])
na = sys.argv[3] if len(sys.argv) > 3 else "new"
nb = sys.argv[4] if len(sys.argv) > 4 else "old"
print(f"per-MLP keys: {a['keys']}")
for k, fmt in (("adjusted_final_layer_score", "{:.4e}"), ("final_layer_mse", "{:.4e}"),
               ("all_layers_mse", "{:.4e}"), ("mean_compute_utilization", "{:.3%}"),
               ("n_failed_mlps", "{:.0f}")):
    va, vb = a["agg"].get(k), b["agg"].get(k)
    if va is None or vb is None:
        continue
    ratio = f"  x{va/vb:.4f}" if vb else ""
    print(f"{k:34s} {na}: {fmt.format(va):>12s}   {nb}: {fmt.format(vb):>12s}{ratio}")
for tag in ("resid", "wall"):
    if a[tag] is not None and b[tag] is not None:
        print(f"{tag:34s} {na}: mean {a[tag].mean():.3f} max {a[tag].max():.3f}   "
              f"{nb}: mean {b[tag].mean():.3f} max {b[tag].max():.3f}")
if a["adj"] is not None and b["adj"] is not None:
    r = a["adj"] / b["adj"]
    print(f"\nper-MLP adjusted ratio {na}/{nb}: mean {r.mean():.4f}  median {np.median(r):.4f}  "
          f"min {r.min():.4f}  max {r.max():.4f}")
    print(f"  better ({na} < {nb}): {int((r < 1).sum())}/100     worse: {int((r > 1).sum())}/100")
    rf = a["fin"] / b["fin"]
    print(f"per-MLP raw final-MSE ratio:  mean {rf.mean():.4f}  better {int((rf < 1).sum())}/100")
    if a["util"] is not None:
        print(f"per-MLP utilisation {na}: {a['util'].min():.3%} .. {a['util'].max():.3%}   "
              f"{nb}: {b['util'].min():.3%} .. {b['util'].max():.3%}")
    w = np.argsort(r)[::-1][:5]
    print("  worst 5 MLPs by adjusted ratio: " + ", ".join(f"#{i} {r[i]:.3f}" for i in w))
