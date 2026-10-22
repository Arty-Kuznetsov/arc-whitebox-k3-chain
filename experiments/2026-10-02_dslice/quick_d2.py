"""Early look, one oracle seed: the chain's D vs the oracle on the block the chain actually computes (rows and columns
of dead neurons alpha < -2.5 are zero by the matched cut; rows of saturated neurons alpha > 2.5 are zero by E2)."""
import numpy as np, torch
import dslice as S
m = 0
Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
base, info = S.run(Ws, m, record="full")
A = np.load(S.HERE / "oracles" / f"d_{m}_s3101.npz")
for l in S.LD:
    a = info[l]["mu_pre"].numpy() / np.sqrt(info[l]["var"].numpy())
    rows = (a >= -2.5) & (a <= 2.5); cols = a >= -2.5
    M = np.outer(rows, cols).astype(float); np.fill_diagonal(M, 0.0)
    Dc, DA = info[l]["D"].numpy() * M, A[f"D_{l}"] * M
    x, y = DA[M > 0], Dc[M > 0]
    core = np.outer(np.abs(a) < 1, np.abs(a) < 1)[M > 0]
    print(f"l={l}: rows kept {rows.sum()} cols kept {cols.sum()} | oracle rms {np.sqrt((x*x).mean()):.3e} chain rms "
          f"{np.sqrt((y*y).mean()):.3e} | rel diff {np.linalg.norm(y-x)/np.linalg.norm(x):.3f} corr {np.corrcoef(x,y)[0,1]:.4f} "
          f"slope {(x@y)/(x@x):.3f} | core x core: rel diff {np.linalg.norm(y[core]-x[core])/np.linalg.norm(x[core]):.3f} "
          f"corr {np.corrcoef(x[core],y[core])[0,1]:.4f}")
    # what the dropped saturated rows hold, in the oracle
    sat = np.outer(a > 2.5, cols).astype(float); np.fill_diagonal(sat, 0.0)
    print(f"      oracle energy share in saturated rows {((A[f'D_{l}']*sat)**2).sum() / ((A[f'D_{l}']*(M+sat))**2).sum():.3f}")
