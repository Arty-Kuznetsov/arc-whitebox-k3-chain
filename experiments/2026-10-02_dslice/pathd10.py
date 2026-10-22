"""pathD="full" on the SHIPPABLE base (path births at the first readout only, e3age=1): one extra product per layer
(the newest source's path leg against its tree leg), never measured here -- the 09-28 x1.025 was on the all-ages base,
where it costs one product per SOURCE. 10 baked MLPs, B9 (torch), raw per MLP."""
import numpy as np, torch
import dslice as S
rows = []
for m in range(10):
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(S.pb.BAKED / f"weights_{m}.npy")]
    p0, _ = S.run_chain(Ws, **S.B9)
    p1, _ = S.run_chain(Ws, **dict(S.B9, pathD="full"))
    r0, r1 = S.raw(p0, m), S.raw(p1, m)
    rows.append((r0, r1))
    print(f"  MLP {m}: base {r0:.4e}  pathD=full {r1:.4e}  x{r1 / r0:.4f}", flush=True)
r = np.array(rows)
print(f"mean base {r[:,0].mean():.4e}  pathD=full {r[:,1].mean():.4e}  x{r[:,1].mean() / r[:,0].mean():.4f}  "
      f"better {(r[:,1] < r[:,0]).sum()}/10  (gate: B9 10-MLP mean 2.19667e-08)")
