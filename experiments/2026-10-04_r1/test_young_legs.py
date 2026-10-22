"""Gate for young4.py: chain.py k4young impl="legs" must reproduce impl="arc" (ARC's FactoredTensor4) -- the per-layer
K4d the step uses and the final means -- for every reader set and age, on small random MLPs."""
import sys
import torch
torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
from r1lib import BASE, weights, run_chain           # noqa: E402

ok = True
for n, seed in ((48, 3), (96, 0)):
    Ws = weights(n, 6, seed)
    for cfg in (dict(ages=2), dict(ages=1), dict(ages=3), dict(ages=2, read=("K4d",)), dict(ages=2, read=("K31",)),
                dict(ages=2, types=("path",)), dict(ages=2, types=("star",)), dict(ages=2, types=("32",)),
                dict(ages=2, couple=1)):
        base = dict(ages=2, types=("path", "star", "32"), resid=True, read=("K4d", "K31"), conv="arc")
        ya = dict(base, **cfg)
        ma, ia = run_chain([W.clone() for W in Ws], record=True, k4young=dict(ya, impl="arc"), **BASE)
        ml, il = run_chain([W.clone() for W in Ws], record=True, k4young=dict(ya, impl="legs"), **BASE)
        e_mu = float((ma - ml).abs().max() / ma.abs().max())
        e_k4 = max(float((a["K4d"] - b["K4d"]).abs().max() / a["K4d"].abs().max().clamp_min(1e-300)) for a, b in zip(ia, il))
        ok &= e_mu < 1e-10 and e_k4 < 1e-10
        print(f"n={n} {str(cfg):42s} mean rel err {e_mu:.1e}   K4d rel err {e_k4:.1e}", flush=True)
print("PASS" if ok else "FAIL")
