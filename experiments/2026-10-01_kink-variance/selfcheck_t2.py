"""Task 2 switches: off = bitwise; every new diagram and marg3 alone moves the output (random He MLP n=128, depth 6)."""
import torch

import kv  # noqa: F401
from chain import run_chain

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
g = torch.Generator().manual_seed(7)
Ws = [torch.randn(128, 128, generator=g) * (2.0 / 128) ** 0.5 for _ in range(6)]
base, _ = run_chain(Ws, **kv.B9)
r0, _ = run_chain(Ws, pairnew=(), marg3=False, **kv.B9)
print("off vs base:", float((r0 - base).abs().max()))
for t in ("c0k31", "d2", "ddt", "k22sq", "dk22", "c0sqd", "c0sqk22"):
    r, _ = run_chain(Ws, pairnew=(t,), **kv.B9)
    print(t, f"{float((r[-1] - base[-1]).abs().max() / base[-1].abs().max()):.1e}", end=" | ")
r, _ = run_chain(Ws, marg3=True, **kv.B9)
print("marg3", f"{float((r[-1] - base[-1]).abs().max() / base[-1].abs().max()):.1e}")
