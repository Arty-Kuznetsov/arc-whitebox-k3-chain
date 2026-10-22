# Two shortcuts to the frontier, both measured dead (2026-09-23)

Before committing to the expensive route (a factorized K=3 chain in flopscope), two cheap
non-standard ideas were tested against the baked GT. Both are decisive negatives, obtained in
minutes from data already on disk. Scripts: `ladder_extrap.py`, `subspace_probe.py`,
`subspace_truncate.py`. Logs in `logs/`.

## 1. Richardson extrapolation across cumulant ORDER — dead

The measured ladder (k1 3.33e-4, k2 4.50e-6, k3 4.10e-8) has ratios 74× and 110×, close to the
companion paper's `(L/n)^k` law with `n/L = 64`. If that law held **per neuron**, the step between
two orders would predict the remaining error:

    e_k = c_i * q^k  =>  e_2 = (mu^(2) - mu^(1)) * q/(q-1),   q = L/n = 1/64

and since `k=1` costs 3.5e7 FLOPs against `k=2`'s 5.2e10 (0.07%), the correction would be free.
Resummation of the depth recursion is listed in our own research notes as organizer-endorsed and
absent from the published negative maps, so it was worth one test.

| rung | corr(step, error) | fitted slope | theoretical slope | MSE after theory | after fitted |
|---|---|---|---|---|---|
| `mu^2 - mu^1` | 0.169 | **+0.0273** | −0.0159 | 0.92× (worse) | 1.08× |
| `mu^3 - mu^2` | 0.179 | **+0.0154** | −0.0159 | 0.92× (worse) | 1.03× |

**The fitted slope has the opposite sign to theory**, the correlation is 0.17, and applying the
theoretical correction makes the answer 8% worse. The `(L/n)^k` law governs the aggregate MSE but
the per-neuron error *directions* decorrelate between orders, so there is nothing to extrapolate.
Even an oracle-fitted slope buys 1.03–1.08×.

## 2. A low-dimensional subspace — the spectrum concentrates, the dynamics do not

The pre-activation covariance concentrates sharply with depth (2 MLPs):

| layer | 0 | 4 | 8 | 12 | 15 |
|---|---|---|---|---|---|
| dims for 90% of energy | 522 | 296 | 215 | 170 | **145** |
| dims for 99% | 789 | 620 | 530 | 474 | 435 |
| participation ratio | 511 | 199 | 119 | 83 | **65** |

By layer 13 the covariance is numerically rank-deficient (smallest eigenvalues hit 0 in float64).
This looked like the one lever that changes the *cost class*: the k=3 transport costs `O(n^2 R)`
with `R = 2n(l+1)`, so ~75% of the bill sits in the deep layers — exactly the layers whose subspace
is smallest. Our earlier `rank_diag` run shows the κ₃ factors concentrating the same way (top-128
singular values hold 99% of the energy at layer 15 against 62% at layer 0).

It does not survive contact. Running the chain with the covariance projected onto its top-m
**exact** eigenspace at every layer (the idealized case — a real implementation would need a cheaper
basis):

| m | 1024 | 512 | 300 | 200 | 150 | 100 | 64 |
|---|---|---|---|---|---|---|---|
| final_layer_mse | 4.78e-6 | 1.64e-2 | 5.01e-2 | 7.96e-2 | 1.00e-1 | 1.27e-1 | 1.52e-1 |
| vs untruncated | 1.00× | **3424×** | 10475× | 16662× | 20984× | 26524× | 31721× |

Self-check: `m = n` reproduces the untruncated chain.

**Mechanism.** The mean readout depends on `sigma_i = sqrt(C_ii)`, and the diagonal aggregates
*every* eigendirection. The discarded directions carry little total variance but contribute
coherently to all n diagonal entries, so truncation biases `sigma` downward for every neuron at
once. A systematic bias, not a small perturbation. **Variance concentration in the spectrum does not
imply the dynamics are low-dimensional whenever the readout is a diagonal functional.**

## Consequence
Neither shortcut reduces the cost class. The remaining route to leaderboard-frontier accuracy is the
factorized K=3 chain, where the problem is cost engineering rather than accuracy: ARC's reference
already reaches raw 4.1e-8, but at 148% of budget.

## The target, stated precisely (worth keeping in view)
The score multiplier **floors at 0.1 for C ≤ 0.1·B**, so below 10% of budget adjusted = raw × 0.1.
Every current leader sits *above* the floor and therefore pays a penalty:

| | raw MSE | utilization | adjusted | adjusted if run at ≤10% |
|---|---|---|---|---|
| #1 J2W | 1.64e-8 | 13.1% | 2.1e-9 | 1.64e-9 |
| #10 mliston | 1.91e-8 | 18.4% | 3.5e-9 | 1.91e-9 |
| 504aldo's public K=3 (#42) | 2.13e-8 | 25.3% | 5.4e-9 | 2.13e-9 |

So the unclaimed optimum is **raw ≈ 2e-8 at exactly 10% of budget**, which would win outright, and
**raw 3.5e-8 at ≤10% is enough for top-10**. ARC's reference k=3 already delivers raw 4.1e-8 — only
17% above that threshold. The entire gap to the frontier is the 15× cost reduction, not accuracy.
