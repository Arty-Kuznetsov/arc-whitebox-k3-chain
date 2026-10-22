"""Summary of the activation test (RESULTS.md §10): per activation, Prop. 5 (A1), the radius test (A2), the Euler link (A3).
Needs vm_analyze.py's analysis of the runs: python act_summary.py  (runs vm_analyze on the files itself; torch env)."""
import json, sys, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from vm_analyze import analyze                                  # noqa: E402
out = {}
for act in ("relu", "leaky", "gelu", "silu"):
    tag = "" if act == "relu" else f"-{act}"
    A1, A1s, A2, cvm, ccm, cvc = [], [], [], [], [], []
    for k in (0, 1, 2):
        g = analyze(f"vm_s256x16-{k}{tag}_gauss_s1"); s = analyze(f"vm_s256x16-{k}{tag}_sphere_s1") if (HERE / "res" / f"vm_s256x16-{k}{tag}_sphere_s1.npz").exists() else None
        A1 += [abs(r["c_meas_n"] / r["c_pred_n"] - 1) for r in g[1:]]
        if s:
            A1s += [abs(r["c_meas_n"] / r["c_pred_n"] - 1) for r in s[1:]]
            A2 += [abs((a["c_meas_n"] - b["c_meas_n"]) - 3 * b["sig2w"]) / max(abs(a["c_meas_n"] - b["c_meas_n"]), 1e-9) for a, b in zip(g[1:], s[1:])]
        cvm.append([r["cos_v_m"] for r in g]); ccm.append([r["cos_Cm_m"] for r in g]); cvc.append([r["cos_v_Cm"] for r in g])
    cvm, ccm, cvc = (np.mean(x, 0) for x in (cvm, ccm, cvc))
    out[act] = dict(P1_median=float(np.median(A1)), P1_sphere_median=float(np.median(A1s)) if A1s else None,
                    radius_rel_dev_median=float(np.median(A2)) if A2 else None, radius_rel_dev_max=float(np.max(A2)) if A2 else None,
                    cos_v_m=cvm.tolist(), cos_Cm_m=ccm.tolist(), cos_v_Cm=cvc.tolist())
    o = out[act]
    print(f"{act:6s} | P1 median |meas/pred-1| gauss {o['P1_median']:.3f} sphere {o['P1_sphere_median'] if o['P1_sphere_median'] is None else round(o['P1_sphere_median'],3)} | "
          f"radius: |(g - s) - 3 sig2w| / |g - s| median {o['radius_rel_dev_median'] if o['radius_rel_dev_median'] is None else round(o['radius_rel_dev_median'],3)} max {o['radius_rel_dev_max'] if o['radius_rel_dev_max'] is None else round(o['radius_rel_dev_max'],3)} | "
          f"cos(v,m) l1 {cvm[1]:.3f} l8 {cvm[8]:.3f} l15 {cvm[15]:.3f} | cos(Cm,m) l1 {ccm[1]:.3f} l8 {ccm[8]:.3f} l15 {ccm[15]:.3f}")
json.dump(out, open(HERE / "res" / "act_summary.json", "w"), indent=1)
