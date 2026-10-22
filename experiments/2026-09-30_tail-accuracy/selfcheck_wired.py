"""Every tailr rule alone at theta=1.0 (small random MLP, n=128, depth 6) must change the result: no silent no-op."""
import sys
from pathlib import Path
import torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb  # noqa: E402
from chain import run_chain  # noqa: E402
n, L = 128, 6
g = torch.Generator().manual_seed(7)
Ws = [torch.randn(n, n, generator=g) * (2.0 / n) ** 0.5 for _ in range(L)]
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
base, _ = run_chain(Ws, **B9)
T = ["Dcol", "Drow", "t", "K4d", "K22", "K31row", "K31col", "qC0", "qC0sq", "qK22", "qC0K22", "qC0^3", "P1", "P2", "P34",
     "tree_col", "corr_col", "path_col", "star_col", "feed_col", "B_row", "corr_row", "path_row", "star_row"]
R = [(tm, None, "zero") for tm in T] + [(tm, ax, "zero") for tm in ("qD", "qK31", "qC0D") for ax in ("dbl", "sgl")]
R += [("marg1", None, "gauss"), ("marg2", None, "gauss")]
out = []
for tm, ax, mode in R:
    ch = []
    for side in ("dead", "sat"):
        r, _ = run_chain(Ws, tailr=[(tm, side, ax, 1.0, mode)], **B9)
        ch.append(float((r[-1] - base[-1]).abs().max() / base[-1].abs().max()))
    out.append(f"{tm}{'/' + ax if ax else ''}: dead {ch[0]:.1e} sat {ch[1]:.1e}")
print("\n".join(out))
