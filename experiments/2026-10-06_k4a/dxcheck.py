"""Debug: at each restructured layer l, compare the recorded Dx(l, l-1) with the direct pairing of the recorded
partners LT(l) against the recorded UNGATED legs at l-1 (sources born <= l-2), synthetic 1024x16."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import flopscope as flops
import flopscope.numpy as fnp
flops.configure(symmetry_warnings=False)
import estimator_k4a as A
from gate import mlp, run
m = mlp(1024, 16, 11)
e1 = A.Estimator(young=None); e1._drec = []
run(e1, m)
for rec in e1._drec:
    if rec[0] != "dx": continue
    _, l, Dx1, idr, LT, SX1, s1, Wn, wb1, d1 = rec
    Dx1 = np.asarray(Dx1, dtype=np.float64); idr = np.asarray(idr); LT = np.asarray(LT, dtype=np.float64)
    SX1 = np.asarray(SX1, dtype=np.float64)
    rP, W = s1["rP"], s1["W"]
    Wold = Wn - wb1                       # rows of the sources born <= l-2 (they have legs at l-1)
    L1 = LT[:Wold][:, idr]; L2 = LT[Wn:Wn + Wold][:, idr]          # (Wold, n_r)
    B1 = SX1[rP:rP + Wold]; A1 = SX1[rP + W:rP + W + Wold]          # (Wold, n): ungated legs at l-1, same source order?
    Dx_true = L1.T @ B1 + L2.T @ A1                                # (n_r, n)
    rel = np.linalg.norm(Dx1 - Dx_true) / np.linalg.norm(Dx_true)
    # also: the same with the GATED legs (if the recorded Dx1 were wrongly gated)
    relg = np.linalg.norm(Dx1 - (L1.T @ (B1 * d1[None, :]) + L2.T @ (A1 * d1[None, :]))) / np.linalg.norm(Dx_true)
    print(f"layer {l:2d}: Wn={Wn} wb1={wb1} Wold={Wold} rP={rP} W={W}  ||Dx1 - Dx_true||/||Dx_true|| = {rel:.2e}   vs gated legs: {relg:.2e}")
