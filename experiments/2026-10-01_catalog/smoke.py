"""Smoke test: every catalog arm runs (n=128, depth 6 random He MLP) and changes the output (or is a documented no-op)."""
import torch

import cat_grid
from chain import run_chain

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
g = torch.Generator().manual_seed(7)
Ws = [torch.randn(128, 128, generator=g) * (2.0 / 128) ** 0.5 for _ in range(6)]
base, _ = run_chain(Ws, **cat_grid.B9)
for t in cat_grid.ARMS:
    try:
        r, _ = run_chain(Ws, **cat_grid.cfg_of(t))
        print(f"{t}: {float((r[-1] - base[-1]).abs().max() / base[-1].abs().max()):.1e}", end=" | ")
    except Exception as e:
        print(f"\n{t}: FAILED {type(e).__name__}: {str(e)[:120]}")
print()
