# R5 — optimal CP compression of the old κ₃ tier (2026-10-04, night): a reproduction of NEGATIVE §58, stopped

`r5cp.py`: float64 research chain (research base B9 + marg2) on a He MLP n = 128, L = 12, captured at layer 10; dense
κ₃ of the sources aged ≥ 5 ("old") and of all sources (`chain._k3_dense`); non-symmetric CP-ALS (60 sweeps, SVD init,
symmetrised); relative errors of the tensor, of its diagonal t and of its (2,1) slice D, at layer 10 and after one exact
gate + transport (`res/cp_n128_L12_l10_a5_s0.json`).

| part | R | tensor | t | D | t next layer | D next layer |
|---|---|---|---|---|---|---|
| old | 32 (n/4) | 6.7% | 4.0% | 5.1% | 4.2% | 4.9% |
| old | 64 (n/2) | 3.6% | 2.3% | 2.7% | 2.1% | 2.6% |
| old | 128 (n) | 2.3% | 1.5% | 1.7% | 1.3% | 1.6% |
| old | 256 (2n) | 0.5% | 0.3% | 0.4% | 0.3% | 0.4% |
| all | 128 (n) | 5.4% | 5.3% | 4.8% | 4.7% | 4.7% |
| all | 256 (2n) | 2.8% | 2.8% | 2.5% | 2.6% | 2.4% |

**Verdict: nothing new — this is NEGATIVE §58** (cloud #4, 09-27: "a rank-n block EXISTS", n = 64: 4.29e-4 at r = n;
the tail's error budget is 3–5% relative). The old tier does compress to rank ≈ n (≈1.5–2% on t and D, inside that
budget), but §58 already closed the construction: one Gram fitting pass costs the whole merge budget, and CP has no
nestedness (adding one source to a converged block → 48.7%). Stopped after one run (~10 min) when the journal check
caught the duplicate. Rule re-learned: read NEGATIVE before starting a "new" compression (CLAUDE.md, item 2).
