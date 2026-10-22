from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain
HERE = Path(__file__).resolve().parent
mc = dict(np.load(HERE / "runs" / "mc_oracle_0.npz"))
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
gt = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
_, info = run_chain(Ws, k4="memo22", corr="split24rr", record=True)
ratio = [1.0] + [float(mc["k4"][l].mean() / info[l]["K4d"].numpy().mean()) if info[l]["K4d"].abs().sum() > 0 else 1.0 for l in range(1, 16)]
print("per-layer level ratio MC/chain:", " ".join(f"{r:.2f}" for r in ratio))
base, _ = run_chain(Ws, k4="memo22", corr="split24rr")
print(f"chain                       final MSE {float(((base[-1].numpy()-gt[-1])**2).mean()):.4e}")
# iterate: rescaling changes later layers, so re-estimate ratios on the rescaled chain a few times
sc = list(ratio)
for it in range(3):
    mu, info2 = run_chain(Ws, k4="memo22", corr="split24rr", record=True, oracle={"k4scale_l": sc})
    print(f"level-rescaled (iter {it})   final MSE {float(((mu[-1].numpy()-gt[-1])**2).mean()):.4e}   residual level ratios:",
          " ".join(f"{float(mc['k4'][l].mean()/info2[l]['K4d'].numpy().mean()):.2f}" for l in range(2, 16)))
    sc = [sc[l] * (float(mc["k4"][l].mean() / info2[l]["K4d"].numpy().mean()) if l >= 1 and info2[l]["K4d"].abs().sum() > 0 else 1.0) for l in range(16)]
