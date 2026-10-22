"""The random-projection identity behind task 2(a). C_chain(l) and C_exact(l) depend on W_0..W_l only, so dC(l) is
independent of W_{l+1}; with w_i = column i of W_{l+1} (iid N(0, 2/n)):
    e_off,i = w_i^T dC0 w_i:  E = 0,  Var = 8 |dC0|_F^2 / n^2,  independent across i      =>  |e_off|^2 ~ (8/n) |dC0|_F^2
    e_diag,i = w_i^T diag(dC) w_i:  E = (2/n) tr dC (ONE number for every neuron),  Var = 8 sum_j dC_jj^2 / n^2
Checked here on the real matrices (dC = C_chain - C_A of one seed, any fixed matrix will do), per layer:
the ratio |e_off|^2 / ((8/n) |dC0|^2), the mean of e_off against its rms, and the coherent share of e_diag.
    python theory_check.py M  -> res2/theory_<M>.json"""
import json
import sys

import numpy as np

import ce
import t2

m = int(sys.argv[1])
Wn, _ = ce.wn(m)
Cch, _ = t2.rec(m)
CA = ce.cov(m, ce.SEEDS[0])
n = Wn[0].shape[0]
out = {}
for l in range(ce.L - 1):
    dC = np.asarray(Cch[l]) - np.asarray(CA[l])
    W = Wn[l + 1]
    ed, eo = t2.parts(W, dC)
    dC0 = dC - np.diag(np.diag(dC))
    ratio = float((eo @ eo) / (8.0 / n * (dC0 * dC0).sum()))
    lvl = 2.0 / n * float(np.trace(dC))
    out[l] = dict(ratio_off=ratio, mean_off_over_rms=float(eo.mean() / np.sqrt((eo ** 2).mean())),
                  diag_coherent_share=float(ed.mean() ** 2 / (ed ** 2).mean()), diag_mean=float(ed.mean()),
                  diag_mean_pred=lvl)
    print(f"L{l:2d}->{l + 1:2d}: |e_off|^2 / ((8/n)|dC0|^2) = {ratio:.3f} | mean(e_off)/rms {out[l]['mean_off_over_rms']:+.3f}"
          f" | e_diag: coherent share {out[l]['diag_coherent_share']:.2f}, mean {ed.mean():+.2e} (pred (2/n)tr {lvl:+.2e})",
          flush=True)
(ce.HERE / "res2" / f"theory_{m}.json").write_text(json.dumps(out, indent=1))
