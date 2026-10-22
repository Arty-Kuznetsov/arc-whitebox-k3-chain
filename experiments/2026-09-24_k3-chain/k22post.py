from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
o = np.load(HERE / "runs" / "mc_oracle22post_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
posts = []
orig = ch.nonlin_slices
def hook(*a, **kw):
    r = orig(*a, **kw); posts.append((r["K22"].clone(), r["K4"].clone(), r["K11"].clone(), r["K2"].clone())); return r
ch.nonlin_slices = hook
import sys
kw = eval(sys.argv[1]) if len(sys.argv) > 1 else {}
ch.run_chain(Ws, k4="memo22", corr="split24rr", **kw)
ch.nonlin_slices = orig
off = lambda M: M[~np.eye(M.shape[0], dtype=bool)]
for l in [int(v) for v in o["layers"]]:
    k22 = off(o[f"k22_{l}"].astype(np.float64)); C = o[f"C{l}"].astype(np.float64)
    K22c, K4c, K11c, K2c = [x.numpy() for x in posts[l]]
    x = off(K22c)
    print(f"post layer {l}: MC k22 mean {k22.mean():.3e} | chain {x.mean():.3e} (ratio {x.mean()/k22.mean():.3f}) | corr {np.corrcoef(x, k22)[0,1]:.3f} "
          f"| cov offdiag: MC mean {off(C).mean():.3e} chain {off(K11c).mean():.3e} | var mean MC {np.diag(C).mean():.4f} chain {K2c.mean():.4f}")
