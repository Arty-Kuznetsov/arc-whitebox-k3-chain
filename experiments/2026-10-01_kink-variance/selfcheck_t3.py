"""Task 3 switches: off = bitwise; each consistency arm moves the output (random He MLP n=128, depth 6)."""
import torch

import kv  # noqa: F401
from chain import run_chain

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
g = torch.Generator().manual_seed(7)
Ws = [torch.randn(128, 128, generator=g) * (2.0 / 128) ** 0.5 for _ in range(6)]
base, _ = run_chain(Ws, **kv.B9)
r0, _ = run_chain(Ws, k4cons=None, k2pair=False, **kv.B9)
print("off vs base:", float((r0 - base).abs().max()))
for kw in (dict(k4cons="k4d_from_k22"), dict(k4cons="k22_from_k4d"), dict(k4cons="v_from_k22"), dict(k2pair=True)):
    r, _ = run_chain(Ws, **kw, **kv.B9)
    print(kw, f"{float((r[-1] - base[-1]).abs().max() / base[-1].abs().max()):.1e}")
