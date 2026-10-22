"""Summarise k3p 10-MLP screens against k3m (re-run 09-29) and k3o (matched -2.5) on the same 10 MLPs."""
import glob, json, statistics as st
from pathlib import Path
R = Path(__file__).resolve().parent
m = json.load(open(R.parent / "2026-09-30_k3o/runs/k3m_n10_today.json"))["results"]["per_mlp"]
o = json.load(open(R.parent / "2026-09-30_k3o/runs/k3o_m25_n10.json"))["results"]["per_mlp"]
def stats(pm):
    return (st.mean(x["final_layer_mse"] for x in pm), st.mean(x["flops_used"] for x in pm) / 2 ** 41,
            st.mean(x["adjusted_final_layer_score"] for x in pm),
            st.mean(x["residual_wall_time_s"] for x in pm), max(x["residual_wall_time_s"] for x in pm),
            st.mean(x["wall_time_s"] for x in pm), max(x["wall_time_s"] for x in pm))
fm, um, am, *_ = stats(m); fo, uo, ao, *_ = stats(o)
print(f"{'arm':<14}{'raw':>11}{'x k3m':>8}{'util':>9}{'x k3o':>8}{'adjusted':>11}{'x k3m':>8}{'x k3o':>8}{'bt k3o':>7}{'resid mean/max':>16}{'wall max':>9}")
print(f"{'k3m':<14}{fm:>11.4e}{1:>8.4f}{100*um:>8.3f}%{um/uo:>8.4f}{am:>11.4e}{1:>8.4f}{am/ao:>8.4f}")
print(f"{'k3o -2.5':<14}{fo:>11.4e}{fo/fm:>8.4f}{100*uo:>8.3f}%{1:>8.4f}{ao:>11.4e}{ao/am:>8.4f}{1:>8.4f}")
for f in sorted(glob.glob(str(R / "runs/k3p_*_n10.json"))):
    try:
        r = json.load(open(f))["results"]
    except Exception:
        continue
    pm = r["per_mlp"]
    fp, up, ap, rm, rx, wm, wx = stats(pm)
    b = sum(x["adjusted_final_layer_score"] < y["adjusted_final_layer_score"] for x, y in zip(pm, o))
    tag = Path(f).name.replace("k3p_", "").replace("_n10.json", "")
    print(f"{tag:<14}{fp:>11.4e}{fp/fm:>8.4f}{100*up:>8.3f}%{up/uo:>8.4f}{ap:>11.4e}{ap/am:>8.4f}{ap/ao:>8.4f}{b:>5}/10"
          f"{rm:>8.3f}/{rx:.3f}{wx:>8.1f}  fails {r['n_failed_mlps']}")
