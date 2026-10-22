import json, sys
f = sys.argv[1] if len(sys.argv) > 1 else "experiments/2026-10-06_k4a/runs/k4a_n3_v2.json"
d = json.load(open(f)); r = d.get("results", d)
ref = json.load(open("experiments/2026-10-02_k3z/runs/k3z_n100.json")); rr = ref.get("results", ref)
refm = {x["mlp_index"]: x for x in rr["per_mlp"]}
torch_ratio = {0: 0.935, 1: 0.900, 2: 0.861, 3: 0.961, 4: 0.821, 5: 0.925, 6: 0.874, 7: 0.823, 8: 0.871, 9: 0.827}
print(f"{f}: adjusted {r['adjusted_final_layer_score']:.4e}  util {r['mean_compute_utilization']:.4%}  failed {r['n_failed_mlps']}")
e0s = e1s = 0.0
for x in r["per_mlp"]:
    i = x["mlp_index"]; b = refm[i]
    e0 = b["final_layer_mse"]; e1 = x["final_layer_mse"]; e0s += e0; e1s += e1
    print(f"  MLP {i}: raw {e1:.4e} vs k3z {e0:.4e} -> x{e1/e0:.3f} (torch x{torch_ratio.get(i)})  FLOPs x{x['flops_used']/b['flops_used']:.4f}  "
          f"wall {x['wall_time_s']:.1f} s (k3z {b['wall_time_s']:.1f})  residual {x['residual_wall_time_s']:.3f} s (k3z {b['residual_wall_time_s']:.3f})")
ids = [x["mlp_index"] for x in r["per_mlp"]]
tp = sum(torch_ratio[i] * refm[i]["final_layer_mse"] for i in ids) / sum(refm[i]["final_layer_mse"] for i in ids)
print(f"pooled raw x{e1s/e0s:.3f}  (torch pooled over these MLPs x{tp:.3f})")
