"""Identity self-checks of the tailr / regime-oracle switches (chain.py) on a small random He MLP (n=128, depth 6):
tailr=None vs () bitwise; every rule at theta=1e9 equal to rounding; regime oracle over all alphas = plain oracle;
empty regime = no oracle; inject_regime over all alphas = plain inject."""
import os
import sys
from pathlib import Path

import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb  # noqa: E402
from chain import run_chain  # noqa: E402

n, L = 128, 6
g = torch.Generator().manual_seed(7)
Ws = [torch.randn(n, n, generator=g) * (2.0 / n) ** 0.5 for _ in range(L)]
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
base, info = run_chain(Ws, record=True, **B9)
d = lambda a, b: float((a - b).abs().max() / b.abs().max())
r0, _ = run_chain(Ws, tailr=(), **B9)
print("tailr=() vs None: max rel", d(r0, base))
TERMS = ["Dcol", "Drow", "t", "K4d", "K22", "K31row", "K31col", "qC0", "qC0sq", "qK22", "qC0K22", "qC0^3", "P1", "P2",
         "P34", "tree_col", "corr_col", "path_col", "star_col", "feed_col", "B_row", "corr_row", "path_row", "star_row"]
rules = [(tm, "both", None, 1e9) for tm in TERMS] + [(tm, "both", ax, 1e9) for tm in ("qD", "qK31", "qC0D") for ax in ("dbl", "sgl")]
rules += [("marg1", "both", None, 1e9, "gauss"), ("marg2", "both", None, 1e9, "gauss")]
r1, _ = run_chain(Ws, tailr=rules, **B9)
print("all rules at theta=1e9: max rel", d(r1, base))
# every rule active at theta=2.0 at once must change the result (the switches are wired)
r2, _ = run_chain(Ws, tailr=[(r[0], "sat", r[2], 2.0) + tuple(r[4:]) for r in rules], **B9)
print("all rules at theta=2.0 (sat): max rel change", d(r2, base))
orc = {"var": [x["var"].numpy() * 1.01 for x in info], "k3": [x["t"].numpy() * 1.1 for x in info],
       "k4": [x["K4d"].numpy() * 1.1 for x in info], "var_layers": range(L), "k3_layers": range(L), "k4_layers": range(L)}
o1, _ = run_chain(Ws, oracle=orc, **B9)
o2, _ = run_chain(Ws, oracle=dict(orc, regime=(-1e9, 1e9)), **B9)
o3, _ = run_chain(Ws, oracle=dict(orc, regime=(1e9, 2e9)), **B9)
print("oracle regime all vs plain:", d(o2, o1), " regime empty vs no oracle:", d(o3, base), " oracle effect:", d(o1, base))
inj = [base[l] * 1.05 for l in range(L)]
i1, _ = run_chain(Ws, inject=inj, inject_upto=L - 2, **B9)
i2, _ = run_chain(Ws, inject=inj, inject_upto=L - 2, inject_regime=(-1e9, 1e9), **B9)
i3, _ = run_chain(Ws, inject=inj, inject_upto=L - 2, inject_regime=(1e9, 2e9), **B9)
print("inject regime all vs plain:", d(i2, i1), " empty vs none:", d(i3, base), " inject effect:", d(i1, base))
