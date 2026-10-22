"""k3x (dpair at level 5) vs k3w on the 100 local MLPs: score, per-MLP ratios, FLOPs, wall, residual, failures."""
import json, sys
from pathlib import Path
import statistics as st

HERE = Path(__file__).resolve().parent
A = json.load(open(HERE.parent / "2026-10-02_k3w/runs/k3w_n100.json"))["results"]
B = json.load(open(HERE / "runs/k3x_dpair_n100.json"))["results"]
pa = {m["mlp_index"]: m for m in A["per_mlp"]}
pb = {m["mlp_index"]: m for m in B["per_mlp"]}
print(f"k3w adjusted {A['adjusted_final_layer_score']:.4e}  raw {A['final_layer_mse']:.4e}  util {A['mean_compute_utilization']:.4%}  fails {A['n_failed_mlps']}")
print(f"k3x adjusted {B['adjusted_final_layer_score']:.4e}  raw {B['final_layer_mse']:.4e}  util {B['mean_compute_utilization']:.4%}  fails {B['n_failed_mlps']}  {B['failure_breakdown']}")
clean = [i for i in pa if i in pb and not any(pa[i][k] or pb[i][k] for k in ("budget_exhausted", "time_exhausted", "residual_wall_time_exhausted"))]
ra = [pb[i]["final_layer_mse"] / pa[i]["final_layer_mse"] for i in clean]
fa = [pb[i]["flops_used"] / pa[i]["flops_used"] for i in clean]
ad = sum(pb[i]["adjusted_final_layer_score"] for i in clean) / sum(pa[i]["adjusted_final_layer_score"] for i in clean)
print(f"on {len(clean)} clean MLPs: raw x{st.mean(ra):.5f} (max dev {max(abs(r - 1) for r in ra):.2e})  FLOPs x{st.mean(fa):.5f}  adjusted x{ad:.5f}  better {sum(pb[i]['adjusted_final_layer_score'] < pa[i]['adjusted_final_layer_score'] for i in clean)}/{len(clean)}")
for lab, P in (("k3w", pa), ("k3x", pb)):
    w = sorted(P[i]["wall_time_s"] for i in P); r = sorted(P[i]["residual_wall_time_s"] for i in P)
    print(f"{lab}: wall median {w[len(w)//2]:.1f} p90 {w[int(.9*len(w))]:.1f} max {w[-1]:.1f} s   residual median {r[len(r)//2]:.3f} p90 {r[int(.9*len(r))]:.3f} max {r[-1]:.3f} s")
dw = sorted(pb[i]["wall_time_s"] - pa[i]["wall_time_s"] for i in clean)
print(f"per-MLP wall increment k3x - k3w: median {dw[len(dw)//2]:+.1f} s  (noisy: shared machine, concurrent jobs)")
