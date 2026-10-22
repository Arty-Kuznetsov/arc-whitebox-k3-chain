"""Self-check of the two chain.py hooks added for cloud #11 (postc, prec): identity hooks must reproduce B9 bitwise
(final-layer MSE equal as floats, all 16 layers of post means equal), and a recording hook must not perturb it.
Also checks the identity used everywhere downstream: the chain's next pre-variance is diag(W^T C W) of the post C the
hook sees. Run after the B9 gate (res/B9/m0.json, m1.json): PB_THREADS=3 python selfcheck_hooks.py"""
import json

import numpy as np
import torch

import ce
from chain import run_chain

ok = True
for m in (0, 1):
    Ws, g = ce.ta.mlp(m)
    ref = json.loads((ce.ta.RES / "B9" / f"m{m}.json").read_text())["mse"]
    mu0, info0 = run_chain(Ws, record=True, **ce.B9)
    seen = {}

    def rec(l, C):
        seen[l] = C.clone()
        return C

    mu1, info1 = run_chain(Ws, record=True, postc=rec, prec=lambda l, Cp: Cp, **ce.B9)
    mse0 = float(((mu0[-1] - g) ** 2).mean())
    mse1 = float(((mu1[-1] - g) ** 2).mean())
    same = bool(torch.equal(mu0, mu1))
    dv = max(float((torch.einsum("ji,jk,ki->i", Ws[l + 1], seen[l], Ws[l + 1]) - info1[l + 1]["var"]).abs().max()
                   / info1[l + 1]["var"].mean()) for l in range(15))
    print(f"MLP {m}: B9 res {ref:.6e} | plain {mse0:.6e} (== res: {mse0 == ref}) | hooks {mse1:.6e} "
          f"(== plain: {mse1 == mse0}, all layers equal: {same}) | max rel |diag(W^T C W) - var_next| {dv:.1e}")
    ok &= (mse0 == ref) and (mse1 == mse0) and same and dv < 1e-12
print("SELF-CHECK", "PASSED" if ok else "FAILED")
