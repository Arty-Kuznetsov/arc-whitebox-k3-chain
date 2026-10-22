"""The quick-ship reference: levers that work on the CURRENT k3m layout (no wide layout, no band): k3m + E2 (+ E3).
    PB_THREADS=1 ../../.venv-torch/bin/python t4_k3m_quick.py"""
import json
from pathlib import Path

import cost_pb
import pb

HERE = Path(__file__).resolve().parent
out_p = HERE / "t4_k3m_quick_n10.json"
arms = [("k3m + E2 sep 2.5 + E3 b -3.0", dict(pb.K3M, satrow=2.5, satrow_k4="sep", recv=(-3.0, "b")),
         dict(t_row=-2.5, satrow=2.5, e3=dict(theta=-3.0, arm="b")))]
for tag, cfg, ukw in arms:
    out = pb.run_arms([(tag, cfg)], out_p)
    r = out[tag]
    u = cost_pb.util(**ukw)
    adj = pb.GRADED_K3M * r["raw"] / pb.K3M_RAW * u / pb.UTIL_K3M
    r.update(util=u, util_x=u / pb.UTIL_K3M, raw_x=r["raw"] / pb.K3M_RAW, projected=adj,
             projected_l5=adj * cost_pb.L5_METER, names=["k3m", "e2_2.5_sep", "e3_-3.0_b"], t=-2.5)
    out[tag] = r
    out_p.write_text(json.dumps(out, indent=1))
    print(f"   {tag}: raw x{r['raw_x']:.4f} util {100 * u:.2f}% projected {adj:.4e} (L5 {adj * cost_pb.L5_METER:.4e})")
