"""Write one two-line wrapper per ablation arm of k3y (constructor kwargs; as ../2026-10-02_k3x/runs/v_k3x_dpair.py)."""
from pathlib import Path
ARMS = [("base", ""), ("k4corr_off", "k4corr=False"), ("marg2_off", "marg2=None"), ("satrow_off", "satrow=None"),
        ("band_off", "band=None"), ("cuts_off", "deadcut=None, colcut=None"), ("colcut_off", "colcut=None"),
        ("l5_off", "transport_l5=False"), ("csym_binv_off", "csym=False, binv=False"), ("k31scale_1", "k31_scale=1.0"),
        ("feed_off", "feed=False"), ("paths_off", "paths=False"), ("c0d_off", "c0d=False"), ("marg_off", "marg=False"),
        ("k31_off", "k31=False"), ("star_off", "star=False"), ("c0k22_off", "c0k22=False")]
for name, kw in ARMS:
    Path(f"arms/v_{name}.py").write_text(f'''"""k3y ablation arm {name}: {kw or 'defaults'}"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from estimator_k3y import Estimator as _E


class Estimator(_E):
    def __init__(self):
        super().__init__({kw})
''')
print(" ".join(n for n, _ in ARMS))
