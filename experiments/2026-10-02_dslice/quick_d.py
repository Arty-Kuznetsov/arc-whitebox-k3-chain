"""Early look, one oracle seed (noise NOT subtracted): the chain's D vs the oracle at layers 9, 12, MLP 0."""
import numpy as np, torch
import dslice as S
m = 0
Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
base, info = S.run(Ws, m, record="full")
A = np.load(S.HERE / "oracles" / f"d_{m}_s3101.npz")
print(f"base raw {base:.6e} (gate 2.278922e-08)")
for l in S.LD:
    Dc, DA = S.offd(info[l]["D"].numpy()), S.offd(A[f"D_{l}"])
    x, y = DA.ravel(), Dc.ravel()
    print(f"  l={l}: |D_oracle| rms {np.sqrt((x*x).mean()):.3e}  |D_chain| rms {np.sqrt((y*y).mean()):.3e}  "
          f"rel diff {np.linalg.norm(y-x)/np.linalg.norm(x):.3f}  corr {np.corrcoef(x,y)[0,1]:.4f}  slope {(x@y)/(x@x):.3f}")
    vA = A[f"var_pre_{l}"]; vc = info[l]["var"].numpy()
    print(f"        var rel diff {np.linalg.norm(vc-vA)/np.linalg.norm(vA):.2e}; t(chain) vs D-diag? D_chain diag max {np.abs(np.diag(info[l]['D'].numpy())).max():.1e}")
