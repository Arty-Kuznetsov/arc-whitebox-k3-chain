# The tail-regime hunt (cloud, Prompt A of `notes/CLOUD-BRIEF.md`) — RESULTS

Branch `cloud/tail-accuracy` (from `origin/main` 43d0012). Every number below is final-layer MSE ("raw") of the torch
chain (`experiments/2026-09-24_k3-chain/chain.py`, float64) against the baked GT (1e9 samples); FLOPs and MSE only
(CLOUD.md §2) — no wall or residual time from this box is evidence. Tables are rendered by `summarize.py` from `res/`.

## 0. Setup, gates, anchors

**New switches (off by default; `tailr=None`, `inject_regime=None`, no `oracle["regime"]` = the old code path, bitwise).**
- `tailr` = a list of rules `(term, side, axis, theta[, mode[, q]])`, parsed by `_tailr_rules`: the tail is
  `alpha < -theta` ("dead"), `alpha > theta` ("sat") or `|alpha| > theta`, with alpha = mu/sigma of the chain itself
  at the layer the term belongs to. Terms:
  - *state level* (cut before the step, so the pair cumulants, the readout AND the births all see it — the E2
    pattern): `Dcol` / `Drow` (single / repeated index of kappa3[iij]), `t` (kappa3 diagonal), `K4d`, `K22`
    (either index), `K31row` / `K31col`;
  - *pair terms of Q(p,q)* (inside `nonlin_slices` only): `qC0`, `qC0sq`, `qK22`, `qC0K22`, `qC0^3` (symmetric:
    an entry is dropped when either index is in the tail) and `qD`, `qK31`, `qC0D` with axis `dbl` (the repeated
    index of the slice in the tail) or `sgl` (the single one) — D and D^T share one rule, so Q11/Q22 stay
    symmetric; `q` restricts a rule to one power cumulant (11, 21, 22);
  - *readout*: `P1` (the mean itself), `P2`, `P34` — the Edgeworth correction of that power moment dropped;
  - *marginal coefficients*: `marg1`, `marg2` with mode `gauss` — coef(k, p) keeps only its Gaussian value;
  - *births* (the new source, birth layer): columns = hub p: `tree_col` (g), `corr_col` (dK3, alpha, ER columns),
    `path_col` (B2), `star_col` (d3 of the star kappa4 birth), `feed_col` (h); rows = the legs' other index:
    `B_row`, `corr_row` (ER), `path_row` (B2); and `star_row` (the star kappa4 birth read into K4d rows).
- `oracle["regime"] = (lo, hi)`: the oracle's per-neuron values replace the chain's only where lo <= alpha < hi;
  `inject_regime = (lo, hi)`: the same for the injected exact post-activation means.

**Identity self-checks** (`selfcheck_tailr.py`, `selfcheck_wired.py`, random He MLP n=128, depth 6, on B9):
`tailr=()` vs None 0.0 (bitwise); every rule at theta = 1e9 at once 3.6e-16 (the P regrouping); all rules active
2.4e-2 (wired); regime oracle over all alphas vs the plain oracle 0.0, empty regime vs no oracle 0.0; the same for
`inject_regime`. Every rule alone at theta = 1.0 changes the output on both sides (dead 3.8e-5 .. 3.4e-3, sat
6.1e-5 .. 4.3e-2); `qK31 dbl/sgl` reproduce `K31row/col` to all printed digits (K31 enters only the pair terms), so
the state K31 cuts are not in the grid.

**Gates** (every new switch off; `res/<base>/m*.json`, `logs/q_*`):

| gate | required | measured |
|---|---|---|
| `ablate.py k4memo 3` | 3.7533e-08 | **3.7533e-08** |
| `ablate.py k3l_t 10` | 2.3135e-08 | **2.3135e-08** |
| k3m BASE (`pb.K3M`), 10 MLPs | 2.3462e-08 | **2.3462e-08** (per MLP = #8's `t3_base_k3m` to 3e-12) |
| #8 `t4_final25a` = FULL (matched -2.5 + E2 2.5 sep + band (5,448,rr,pre) + E3 (-3.0,b)) | 2.2392e-08 | **2.23915e-08** (#8: 2.23915e-08; per MLP within 9.3e-6 = float32 band arithmetic at another thread count) |

**Anchor of this session: B9 = k3m + matched cut -2.5 + E2 (satrow 2.5, sep) = 2.1967e-08**
(x0.9363 of k3m; per MLP x1e8: 2.2789 2.2057 2.1551 2.2456 2.3033 1.9982 1.8966 2.1943 2.0301 2.6589).
All arm ratios below are raw / B9 on the same MLPs unless stated.

## 1. Regime attribution (task 1)

### 1a. Where the final error lands (`t1_attrib.py split`, `res1/split.json`, 10 MLPs, B9)
Output neurons at layer 15 by the chain's own alpha: **kink 57.9% of the final MSE (551 neurons on average), saturated
42.0% (235), dead 0.08% (237)** — per neuron the saturated outputs carry 1.83x the average error, kink 1.08x, dead
0.004x. The tails grow with depth: dead / saturated fractions 0.04 / 0.04 at layer 4, 0.13 / 0.13 at layer 8,
0.20 / 0.20 at layer 12, 0.23 / 0.23 at layer 15.

### 1b. The post-mean channel by regime (`t1_attrib.py inject R`, `res1/inject_R.json`, 10 MLPs)
Exact post-activation means (baked GT, 1e9 samples) substituted at layers 0-14 for ONE regime at a time (absolute
values, the chain's own alpha picks the neurons, everything else stays the chain's):

| regime substituted | raw | x B9 | better | per-MLP range |
|---|---|---|---|---|
| dead (alpha < -2.5) | 2.1863e-08 | 0.9953 | 7/10 | 0.987 .. 1.006 |
| saturated (alpha > 2.5) | 1.0797e-08 | 0.4915 | 10/10 | 0.432 .. 0.530 |
| **kink (abs alpha <= 2.5)** | **4.6026e-09** | **0.2095** | 10/10 | 0.180 .. 0.240 |
| all | 4.5351e-09 | 0.2065 | 10/10 | 0.178 .. 0.236 |

**~79% of B9's final error travels through the intermediate post means, and it is born in the kink regime**: exact
kink means alone recover 99% of what exact means everywhere recover (x0.2095 vs x0.2065). The saturated regime is a
*carrier* (its post mean is its pre-activation mean, W^T of the upstream means, passed on with d1 ~ 1: fixing it
fixes what it carries, x0.49, but it is not additive with kink); the dead regime neither makes nor carries error
(x0.995). This contradicts the lead of the prompt (tail-regime terms inject the error): the tails do not originate
it. By NEGATIVE 31 the one-step readout of exact moments is accurate (9.7e-11), so the kink neurons' post-mean error
is the error of their own pre-activation moments; which moment is 1c.

### 1b'. Inside the kink: core or shoulders? (same injection, finer regimes)

| regime substituted | raw | x B9 | better | per-MLP range |
|---|---|---|---|---|
| dead shoulder (-2.5 < alpha < -1) | 2.0330e-08 | 0.9255 | 9/10 | 0.836 .. 1.001 |
| **core (abs alpha < 1)** | **7.3730e-09** | **0.3356** | 10/10 | 0.283 .. 0.392 |
| saturating shoulder (1 < alpha < 2.5) | 1.1497e-08 | 0.5234 | 10/10 | 0.467 .. 0.590 |

The injections are not additive (fixing a regime also fixes what it carries), but the order is clear: **the error is
born in the core of the kink**, where the ReLU is most nonlinear (d2 largest) and the Edgeworth series is at its most
reliable — not at the shoulders where the series degrades. The saturating shoulder behaves half as carrier (d1 = Phi(alpha)
= 0.84 .. 0.99), the dead shoulder contributes x0.93.

### 1c. Which moment, in which regime (`t1_attrib.py mc`, `res1/mc_0_*.json`, MLP 0, two independent 1e7 oracles)
The MC oracle's per-neuron pre-activation diagonals (var, kappa3 = t, kappa4) replace the chain's at layers 1-15 ONLY
for the neurons of one regime (absolute values; layer 0 is Gaussian and exact in the chain). raw / B9 on MLP 0
(B9 = 2.2789e-08); columns = seed 1001 / seed 1002 / both pooled (2e7):

| channel | all | kink | saturated | dead |
|---|---|---|---|---|
| var + t + k4 (joint) | 0.2349 / 0.2349 / **0.1165** | 0.2383 / 0.2387 / **0.1204** | 1.0563 / 1.0833 / 1.0682 | 0.9919 / 0.9905 / 0.9911 |
| var | 0.7405 / 0.7536 / 0.6486 | 0.7353 / 0.7499 / 0.6444 | 1.0682 / 1.0933 / 1.0792 | 1.0014 / 1.0014 / 1.0014 |
| kappa4 diagonal | 0.7622 / 0.7812 / 0.7662 | 0.7778 / 0.7942 / 0.7805 | 0.9889 / 0.9897 / 0.9892 | 0.9895 / 0.9892 / 0.9893 |
| kappa3 diagonal | 0.9303 / 0.8897 / 0.8988 | 0.9157 / 0.8761 / 0.8845 | 1.0047 / 1.0045 / 1.0046 | 1.0021 / 1.0012 / 1.0016 |

- **Every moment channel's error is a kink-regime error**: exact kink moments alone reproduce the all-neuron arm in
  every row (joint 0.120 vs 0.117 pooled; var 0.644 vs 0.649; kappa4 0.78 vs 0.77; kappa3 0.88 vs 0.90).
- **The tails hold nothing to fix.** Dead: 0.99-1.00 in every channel. Saturated: exact kappa3/kappa4 are neutral
  (1.005 / 0.989) and exact VARIANCE is *harmful* (x1.07-1.09, and pooling does not halve the harm, so it is not only
  oracle noise): the saturated neurons pass their variance on with d1 ~ 1, and a diagonal that is exact but no longer
  consistent with the chain's own covariance rows is worse than the chain's consistent one — the consistency mechanism
  of #8's matched cut and of NEGATIVE 57, seen from the other side.
- The variance and joint arms are oracle-limited (pooling two seeds: var 0.74-0.75 -> 0.65, joint 0.235 -> 0.117), as in
  2026-09-26_accuracy: the ceilings are lower bounds set by the oracle. kappa4 is not (0.76-0.78 either way).
- Order of the channels in the kink (pooled): var x0.64 > kappa4 x0.78 > kappa3 x0.88 — the same order as the
  regime-free attribution of 2026-09-26_accuracy on the k3l base (x1.54 / x1.31 / x1.14 as gains).

MLP 1 (its two oracles are being rebuilt after the container restarts) is appended below when it lands.

MLP 1 (B9 = 2.2057e-08), same arms, seed 1001 / seed 1002 / pooled:

| channel | all | kink | saturated | dead |
|---|---|---|---|---|
| joint | 0.2146 / 0.2514 / **0.1275** | 0.2260 / 0.2584 / **0.1354** | 1.0186 / 0.9975 / 1.0072 | 1.0015 / 1.0061 / 1.0037 |
| var | 0.9354 / 0.8871 / 0.8241 | 0.9326 / 0.8844 / 0.8210 | 1.0390 / 1.0191 / 1.0281 | 0.9985 / 0.9996 / 0.9991 |
| kappa4 | 0.7882 / 0.7835 / 0.7788 | 0.7910 / 0.7854 / 0.7812 | 0.9883 / 0.9878 / 0.9880 | 1.0013 / 1.0023 / 1.0017 |
| kappa3 | 0.9179 / 0.8831 / 0.8808 | 0.9248 / 0.8862 / 0.8862 | 0.9907 / 0.9894 / 0.9900 | 1.0032 / 1.0056 / 1.0044 |

Same picture on both MLPs: every channel's error is a kink error (kink reproduces all to within the seed spread);
exact saturated variance is harmful (x1.02-1.09 on both MLPs); the dead tail is neutral. On MLP 1 the kappa4 channel
(x0.78) is ahead of the variance (x0.82 pooled), on MLP 0 behind it (x0.78 vs x0.64): the order of var and kappa4
is MLP-dependent, kappa3 is third on both.

**Task 1 verdict.** The tails neither originate the error nor hold a fixable part of it. ~79% of the final error
travels through the post means (1b), all of it born in the kink core, and the moment errors that feed it (var,
kappa4, kappa3) are kink errors on both oracle MLPs. The pair channels (covariance rows/columns, kappa3 births) have no
valid oracle at 1e7 (NEGATIVE 50); they are tested by the term restrictions of task 2, which reach the same verdict.

## 2. The exhaustive tail-restriction screen (task 2)

### 2a. Is a tail-safe readout worth an arm? One step, per regime (`t2_readout.py`, `res1/readout_0_s*.json`)
MLP 0, every layer, the MC oracle's own pre-activation (mu, var, kappa3, kappa4) per neuron -> the post mean by five
forms, scored against the MC post mean of the SAME 1e7 samples (the moments' sampling noise cancels) and against the
independent baked GT (1e9). Contribution to the layer MSE (sum over the regime / 1024), seed 1001 (seed 1002 agrees to
the printed digits within x1.3):

| layer, regime (n) | Gaussian | Edgeworth k3 only | **Edgeworth k3 + k4 (the chain's P1)** | + k3^2 (edge2) | shifted lognormal (tail-safe) | MC vs GT noise floor |
|---|---|---|---|---|---|---|
| L5 dead (64) | 4.5e-10 | 1.3e-10 | **4.9e-13** | 9.7e-13 | 7.2e-11 | 6.6e-13 |
| L5 kink (903) | 3.3e-07 | 3.8e-08 | **1.9e-10** | 1.8e-10 | 3.7e-08 | 2.0e-08 |
| L5 sat (57) | 7.3e-10 | 1.6e-10 | **7.4e-13** | 1.1e-12 | 8.9e-11 | 3.2e-09 |
| L10 dead (140) | 8.4e-10 | 2.9e-10 | **1.2e-12** | 3.2e-12 | 1.3e-10 | 4.8e-13 |
| L10 kink (723) | 3.0e-07 | 3.0e-08 | **1.1e-10** | 9.5e-11 | 2.8e-08 | 7.3e-09 |
| L10 sat (161) | 1.3e-09 | 3.5e-10 | **1.4e-12** | 3.2e-12 | 1.6e-10 | 3.8e-09 |
| L15 dead (232) | 9.4e-10 | 2.8e-10 | **2.4e-12** | 4.4e-12 | 1.0e-10 | 3.5e-13 |
| L15 kink (551) | 1.8e-07 | 2.1e-08 | **9.0e-11** | 7.1e-11 | 1.9e-08 | 4.2e-09 |
| L15 sat (241) | 6.6e-10 | 2.7e-10 | **2.0e-12** | 5.0e-12 | 1.2e-10 | 4.1e-09 |

(against the GT every form that is below the noise floor reads the floor: Edgeworth = MC-GT in every regime.)
**Closed on the one-step oracle, no chain arm needed:** fed exact moments, the chain's first-order Edgeworth readout
is accurate to ~1e-12 of layer MSE in BOTH tails and ~1e-10 in the kink (sum 9.4e-11 at L15 = NEGATIVE 31's 9.7e-11).
The large tail corrections #8 measured (d3 t/6 = 1-2x the Gaussian mean of a dead neuron) are *correct*: the Gaussian
readout of the dead tail is off by 9e-10 and the correction removes 99.7% of it. A proper-density tail form (shifted
lognormal matched to mu, var, kappa3) is 50-100x WORSE than the truncated series in the tails and 200x worse in the kink
(it cannot carry kappa4); second order (edge2) helps only in the kink (x0.8) and hurts in the tails. The same bracket
holds for the pair coefficients: `margtail` (Gaussian coef in the tails) and the base (Edgeworth coef) were equal in #8
(x0.9999), and the grid's `marg1`/`marg2` rules below measure each coefficient order and side separately.

### 2b. The screen: 149 arms on B9 (`grid.py`, `run_arm.py`, full table `grid_table.md`)
Every term x tail x theta in {2.0, 2.5, 3.0} as listed in section 0, minus the exact no-ops. Kill rule: raw worse than
B9 on 10 MLPs or better on fewer than 7/10 (early stop at 4 worse, lossless for the rule). **18 survive, 131 killed,
0 failed.** Best per term and side (x B9; K = killed, d = done; better/n):

| family | best dead-side arm | best saturated-side arm | worst arm |
|---|---|---|---|
| state D columns (E2's transpose) | 2.0: 1.0141 K | 3.0: **54.7** K | sat 2.0: 74.9 |
| state D rows | 2.0: 1.0806 K | (E2, #8) | |
| state t (kappa3 diagonal) | 3.0: 1.0282 K | 3.0: 1.1288 K | sat 2.0: 17.2 |
| state K4d | 3.0: 1.0092 K | 3.0: 1.0246 K | sat 2.0: 2.93 |
| state K22 | **2.5: 0.9973 d 7/10** | 3.0: >1 K | sat 2.0: 13.1 |
| qC0 (covariance pass-through) | 3.0: 0.9987 K 4/8 | 3.0: 7211 K | sat 2.0: 8961 |
| qC0sq | 3.0: 0.9988 d 9/10 | 3.0: 108 K | sat 2.0: 228 |
| qK22 | 3.0: 0.9991 d 9/10 | >1 K | sat 2.0: 13.1 |
| qC0K22 | 2.0: 0.9997 d 7/10 | 2.5: 0.9996 d 7/10 | 1.0003 |
| qC0^3 | 2.0: 0.9996 d 8/10 | 2.5: 0.9997 d 7/10 | 1.0002 |
| qD single index | 2.0: 1.0222 K | 3.0: 87.3 K | sat 2.0: 112 |
| qD repeated index | 2.0: 1.0691 K | 2.0: 15.0 K | |
| qC0D single / repeated | 2.0: 1.0008 K / 1.0111 K | 3.0: 1.095 K / 2.0: 1.0085 K | 1.19 |
| qK31 triple / single | 2.0: 1.0037 K / 2.0: 0.9999 d 8/10 | 2.0: 0.9976 K 4/8 / 1.05 K | |
| P1 (mean readout correction) | 3.0: 1.0121 K | 3.0: 1.05 K | sat 2.0: 7.95 |
| P2 / P3-P4 | 3.0: 0.9999 K / 1.0000 K | 3.0: 1.0031 K / 1.08 K | |
| marg1 (coef k=1 -> Gaussian) | 3.0: 1.0017 K | 3.0: 1.0047 K | sat 2.0: 1.08 |
| **marg2 (coef k=2 -> Gaussian)** | 2.5: 1.0018 K | **2.0: 0.9925 d 7/10** (3.0: 0.9976, 2.5: 0.9946) | dead 2.0: 1.0042 |
| births tree / corr / path / star / feed columns | 2.0: 1.0 / 0.9994 K / 1.0027 K / **0.9985 d 8/10** / 1.0003 K | 3.0: 0.9995 d / 1.40 K / 2.5: 0.9999 d / 1.0001 K / 2.5: 0.9997 d | corr sat 2.0: 2.47 |
| births B / corr / path / star rows | 2.0: 1.0035 K / 3.0: 0.9998 K / **2.0: 0.9997 d 9/10** / 1.0 K | 3.0: 17.4 K / 1.44 K / 1.036 K / 1.0002 K | B sat 2.0: 34.3 |

- **The saturated side is structural, not noise.** Every cut of a saturated neuron's covariance, kappa3 column,
  tree leg, correction-leg column, kappa3 or kappa4 diagonal or mean correction is harmful, from x1.02 to x9000:
  a saturated ReLU is (almost) the identity, so its terms carry the signal through. **E2's transpose is refuted**
  (saturated D columns: x54.7 even at theta = 3.0) — D_ij enters Cov(y_i, y_j) through b(1)_j = d1_j ~ 1. The only
  saturated-side survivors are the tiny birth-column cuts (tree 3.0, path 2.5, feed 2.5: <= 0.05%) and marg2.
- **The dead side is inert**: its survivors (K22, qC0sq, qK22, star columns, path rows, qC0K22, qC0^3) are worth
  0.03-0.27% each — the dead neurons neither make nor carry error (task 1), so removing their terms changes little.
- **marg2 is the one lever, and it is not a tail lever**: the curve keeps improving below 2.0 and is best with the
  correction dropped nearly everywhere (next section).

### 2c. marg2: the Edgeworth correction of the second-order pair coefficients (follow-ups, `grid.EXTRA`)
coef(2, p) = w(2,p) + w(5,p) t/6 + w(6,p) K4d/24 replaced by the Gaussian w(2,p) for the neurons with alpha > theta,
in every pair term (the k = 1 correction is kept: removing it costs x1.02-1.08):

| arm | theta (alpha > theta -> Gaussian) | x B9 | better |
|---|---|---|---|
| marg2_sat_3.0 / 2.5 / 2.0 | 3.0 / 2.5 / 2.0 | 0.9976 / 0.9946 / 0.9925 | 8 / 7 / 7 of 10 |
| marg2_sat_1.5 / 1.0 / 0.0 | 1.5 / 1.0 / 0.0 | 0.9921 / 0.9910 / 0.9864 | 7 / 7 / 7 |
| **marg2_sat_m1.0 / m2.0** | -1.0 / -2.0 (kept only in the dead tail) | **0.9813 / 0.9811** | 8 / 8 |
| marg2_sat_all | none (dropped everywhere) | 0.9836 | 9 |
| marg2_all_q11 (only in Q11) | none | 1.0908 (killed at 4) | 0/4 |
| marg2_all_q21 (only in Q21) | none | 1.1117 (killed at 4) | 0/4 |
| marg2_all_q22 (only in Q22) | none | 0.9996 | 8 |

**A consistency effect, again.** Dropping the correction in Q11 alone (x1.09) or in Q21 alone (x1.11) is bad, in all
three power cumulants together it is good (x0.984): K21 = Q21 - 2 P1 Q11 is a difference of two large terms, and the
k = 2 coefficient must be the same in both; consistently Gaussian beats consistently corrected by 1.6-1.9%. The
correction is (weakly) useful only in the dead tail (m2.0 vs all: 0.9811 vs 0.9836). Production: the Gaussian w(2,p)
replaces coef(2,p) in the pair coefficients (outside the dead tail), an O(n) change of the coefficient vectors at zero
FLOP cost — it removes terms. Both variants are being confirmed on the 32 fresh MLPs before they enter the frontier.

### 2d. Pairwise combinations of the top 10 (`pairs_top10.txt`, same kill rule vs B9, early stop)
Top 10 = the best threshold of each distinct (term, side) among the survivors: marg2 sat m2.0, K22 dead 2.5, star
columns dead 2.0, qC0sq dead 3.0, qK22 dead 3.0, tree columns sat 3.0, qC0^3 dead 2.0, qC0K22 sat 2.5, qC0K22 dead 2.0,
path rows dead 2.0 -> 45 pairs. Question: is any pair better than the product of its parts (a consistency pair, like
marg2's Q11 + Q21)? Sorted by x B9:

<!-- PAIRS -->
| arm | MLPs | raw x B9 | better |
|---|---|---|---|
| marg2_sat_m2.0+K22_dead_2.5+star_col_dead_2.0+qC0sq_dead_3.0+path_row_dead_2.0+qC0cube_dead_2.0+qC0K22_sat_2.5+qC0K22_dead_2.0+tree_col_sat_3.0 | 10 | 0.9753 | 8/10 |
| marg2_sat_m2.0+K22_dead_2.5+star_col_dead_2.0 | 10 | 0.9781 | 8/10 |
| marg2_sat_m2.0+K22_dead_2.5 | 10 | 0.9796 | 8/10 |
| marg2_sat_m2.0+star_col_dead_2.0 | 10 | 0.9796 | 9/10 |
| marg2_sat_m2.0+qC0sq_dead_3.0 | 10 | 0.9801 | 9/10 |
| marg2_sat_m2.0+qK22_dead_3.0 | 10 | 0.9804 | 9/10 |
| marg2_sat_m2.0+tree_col_sat_3.0 | 10 | 0.9805 | 8/10 |
| marg2_sat_m2.0+qC0cube_dead_2.0 | 10 | 0.9806 | 9/10 |
| marg2_sat_m2.0+qC0K22_dead_2.0 | 10 | 0.9807 | 9/10 |
| marg2_sat_m2.0+path_row_dead_2.0 | 10 | 0.9808 | 8/10 |
| marg2_sat_m2.0+qC0K22_sat_2.5 | 10 | 0.9810 | 8/10 |
| K22_dead_2.5+star_col_dead_2.0 | 10 | 0.9958 | 8/10 |
| K22_dead_2.5+qC0sq_dead_3.0 | 10 | 0.9963 | 7/10 |
| K22_dead_2.5+tree_col_sat_3.0 | 10 | 0.9968 | 7/10 |
| K22_dead_2.5+qC0cube_dead_2.0 | 10 | 0.9969 | 7/10 |
| K22_dead_2.5+path_row_dead_2.0 | 10 | 0.9970 | 7/10 |
| K22_dead_2.5+qK22_dead_3.0 | 10 | 0.9973 | 7/10 |
| star_col_dead_2.0+qC0sq_dead_3.0 | 10 | 0.9974 | 9/10 |
| K22_dead_2.5+qC0K22_dead_2.0 | 9 | 0.9977 | 5/9 |
| star_col_dead_2.0+qK22_dead_3.0 | 10 | 0.9977 | 9/10 |
| K22_dead_2.5+qC0K22_sat_2.5 | 9 | 0.9979 | 5/9 |
| qC0sq_dead_3.0+qK22_dead_3.0 | 10 | 0.9980 | 9/10 |
| star_col_dead_2.0+tree_col_sat_3.0 | 10 | 0.9980 | 8/10 |
| star_col_dead_2.0+path_row_dead_2.0 | 10 | 0.9982 | 9/10 |
| star_col_dead_2.0+qC0K22_dead_2.0 | 10 | 0.9982 | 8/10 |
| star_col_dead_2.0+qC0K22_sat_2.5 | 10 | 0.9982 | 9/10 |
| star_col_dead_2.0+qC0cube_dead_2.0 | 10 | 0.9982 | 8/10 |
| qC0sq_dead_3.0+tree_col_sat_3.0 | 10 | 0.9983 | 9/10 |
| qC0sq_dead_3.0+qC0cube_dead_2.0 | 10 | 0.9984 | 9/10 |
| qC0sq_dead_3.0+path_row_dead_2.0 | 10 | 0.9985 | 9/10 |
| qC0sq_dead_3.0+qC0K22_dead_2.0 | 10 | 0.9985 | 9/10 |
| qC0sq_dead_3.0+qC0K22_sat_2.5 | 10 | 0.9985 | 8/10 |
| qK22_dead_3.0+tree_col_sat_3.0 | 10 | 0.9986 | 8/10 |
| qK22_dead_3.0+qC0cube_dead_2.0 | 10 | 0.9987 | 8/10 |
| qK22_dead_3.0+path_row_dead_2.0 | 10 | 0.9988 | 10/10 |
| qK22_dead_3.0+qC0K22_dead_2.0 | 10 | 0.9988 | 8/10 |
| qK22_dead_3.0+qC0K22_sat_2.5 | 10 | 0.9988 | 9/10 |
| tree_col_sat_3.0+qC0K22_dead_2.0 | 10 | 0.9991 | 7/10 |
| tree_col_sat_3.0+qC0K22_sat_2.5 | 10 | 0.9991 | 7/10 |
| tree_col_sat_3.0+qC0cube_dead_2.0 | 10 | 0.9991 | 7/10 |
| tree_col_sat_3.0+path_row_dead_2.0 | 10 | 0.9992 | 7/10 |
| qC0K22_sat_2.5+path_row_dead_2.0 | 10 | 0.9993 | 7/10 |
| qC0K22_sat_2.5+qC0K22_dead_2.0 | 10 | 0.9993 | 8/10 |
| qC0cube_dead_2.0+path_row_dead_2.0 | 10 | 0.9993 | 8/10 |
| qC0cube_dead_2.0+qC0K22_sat_2.5 | 10 | 0.9993 | 7/10 |
| qC0K22_dead_2.0+path_row_dead_2.0 | 10 | 0.9994 | 7/10 |
| qC0cube_dead_2.0+qC0K22_dead_2.0 | 10 | 0.9994 | 8/10 |
<!-- /PAIRS -->

**No consistency pair among the tail cuts.** 43 of 45 pairs survive the kill rule (2 killed), and for every
finished pair the measured x is the product of its parts to within 0.9999-1.0011: the tail cuts are independent,
additive (at most slightly sub-additive) perturbations. The best pairs are marg2 + one small dead-side cut (x0.9796),
the best pair without marg2 is K22 dead 2.5 + star columns dead 2.0 (x0.9958). The stacks of section 3 are therefore
the whole story of the tail cuts, and on fresh MLPs they add 0.17% to marg2.

## 3. Confirmation on 32 fresh MLPs, stacks, frontier (task 3)

Stacks (10 screening MLPs, then the 32 fresh MLPs 10-41 that were never used for any selection):
- **stack B** = marg2 m2.0 + K22 dead 2.5 + star columns dead 2.0; **stack A** = stack B + qC0sq dead 3.0 + path rows
  dead 2.0 + qC0^3 dead 2.0 + qC0K22 sat 2.5 + qC0K22 dead 2.0 + tree columns sat 3.0 (all distinct top-10 terms;
  qK22 dead 3.0 is nested in the state K22 cut at 2.5 and left out).
- On B9: marg2 alone x0.9811 (screen) -> **x0.9838 (fresh, 29/32)**; stack A x0.9753 (screen) -> **x0.9821 (fresh,
  28/32)**. The eight small cuts regress from 0.6% on the screen to 0.17% on fresh MLPs (the winner's curse of a
  149-arm selection), marg2 keeps 86% of its screening gain.

**Frontier** (`frontier.py`; projected = 5.3906e-9 x raw / 2.3462e-8 x util / 24.835%; the tailr rules add no FLOPs;
"meter-anchored" util = the port-battery model scaled to the Mac's meter: k3o 23.792% = model x1.008, k3q 19.73% =
model x1.033 — the model is optimistic by 1-3%, more so the more levers are stacked; "fresh" uses the 32-MLP ratios,
including FULL's own raw cost over B9 on the fresh set, x1.0315 vs x1.0193 on the screen):

<!-- FRONTIER -->
| stack | x base (10 screen) | x base (32 fresh) | raw (screen) | util model / meter-anchored | **projected (fresh, meter)** | projected (screen, model) | + L5 (fresh, meter) |
|---|---|---|---|---|---|---|---|
| B9 (matched -2.5 + E2) | 1.0000 (0/10) | 1.0000 (0/32) | 2.1967e-08 | 22.02% / 22.69% | **4.611e-09** | 4.475e-09 | 4.226e-09 |
| B9 + marg2 (m2.0) | 0.9811 (8/10) | 0.9838 (29/32) | 2.1552e-08 | 22.02% / 22.69% | **4.536e-09** | 4.390e-09 | 4.157e-09 |
| B9 + marg2 (all) | 0.9836 (9/10) | 0.9867 (28/32) | 2.1606e-08 | 22.02% / 22.69% | **4.550e-09** | 4.402e-09 | 4.170e-09 |
| B9 + stack B (marg2 + K22 dead + star col dead) | 0.9781 (8/10) | not run (stack A, which contains it, was) (0/0) | 2.1485e-08 | 22.02% / 22.69% | **not run (stack A, which contains it, was)** | 4.377e-09 | not run (stack A, which contains it, was) |
| B9 + stack A (9 rules) | 0.9753 (8/10) | 0.9821 (28/32) | 2.1424e-08 | 22.02% / 22.69% | **4.528e-09** | 4.364e-09 | 4.150e-09 |
| FULL (#8 t4_final25a) | 1.0000 (0/10) | 1.0000 (0/32) | 2.2391e-08 | 18.34% / 18.95% | **3.972e-09** | 3.799e-09 | 3.640e-09 |
| FULL + marg2 (m2.0) | 0.9802 (9/10) | 0.9842 (28/32) | 2.1947e-08 | 18.34% / 18.95% | **3.909e-09** | 3.724e-09 | 3.582e-09 |
| FULL + stack B | 0.9772 (9/10) | not run (stack A, which contains it, was) (0/0) | 2.1881e-08 | 18.34% / 18.95% | **not run (stack A, which contains it, was)** | 3.713e-09 | not run (stack A, which contains it, was) |
| FULL + stack A | 0.9747 (9/10) | 0.9834 (29/32) | 2.1824e-08 | 18.34% / 18.95% | **3.906e-09** | 3.703e-09 | 3.580e-09 |

FULL / B9 raw: screen 1.0193, fresh 1.0315
<!-- /FRONTIER -->

- **FULL + marg2 lands at 3.91e-9 (3.58e-9 with Strassen L5)**, honest version: fresh-MLP ratios (x0.9842, 28/32) and
  the meter-anchored util. #8 projected the FULL stack at 3.79e-9 from the screening raw and the unanchored model; the
  same projection redone honestly is 3.97e-9 (the model's util deficit x1.03 and the band + E3's larger raw cost on
  fresh MLPs, x1.0315 vs x1.0193, account for the difference). FULL + stack A (3.906e-9) is not better than FULL + marg2
  (3.909e-9) by more than the fresh-set noise: only marg2 goes to the Mac.

## 4. Verdict (task 4)

**How much of the x1.8 accuracy gap do the tails explain? None measurable.**
1. *Where the error is born* (task 1): ~79% of the final error travels through the intermediate post means and is
   born in the kink core (exact means of abs alpha < 1 alone: x0.34; of the whole kink: x0.21; of the dead tail: x0.995).
   The moments that feed it (variance, kappa4, kappa3) are kink moments on both oracle MLPs (exact kink moments
   reproduce exact moments everywhere; exact dead-tail moments x0.99-1.00; exact saturated variance is harmful).
2. *The truncated series does not break down where it looked like it did*: fed exact moments, the first-order
   Edgeworth readout is accurate to ~1e-12 of layer MSE in both tails (1e-10 in the kink). #8's large tail corrections
   are the right corrections; a proper-density tail form is 50-100x worse.
3. *No tail restriction is worth more than a rounding error* (task 2, 149 arms + pairs): the saturated side is
   structural (cuts cost x1.02 to x9000; E2's transpose x54.7), the dead side is inert (its best cuts: <= 0.27% on the
   screen, 0.17% together on fresh MLPs), and no pair is better than the product of its parts.
4. So E2's x0.94 and the matched cut's x0.993 were never evidence of tail-injected error. What the three measured
   gains of #8 and this session have in common is **consistency between coupled channels**: the matched cut removes a
   dead neuron from the row and the column side together; E2 needs its covariance and its leg channel together
   (x0.973 and x0.986 alone, #8); marg2 is good in Q11 + Q21 + Q22 together and bad in any one alone (x1.09, x1.11);
   exact saturated variance is bad because it is inconsistent with the chain's own covariance rows. The chain is
   hurt where two channels that must share a term do not, and helped where they are made to agree.

**The one lever found, and it is not a tail lever: marg2** — the Gaussian second-order pair coefficient.
- Measured: B9 x0.9811 (8/10) -> x0.9838 on the 32 fresh MLPs (29/32); FULL x0.9802 (9/10) -> x0.9842 fresh (28/32).
  Projected FULL + marg2: **3.91e-9** (+L5 3.58e-9), from FULL 3.97e-9 — worth ~1.6%, below what one submission can
  resolve (NEGATIVE 60: < 2-3%), so it rides along with the next package rather than being judged on its own.
- **Production recipe (O(n), zero products — it removes terms):** in the nonlinear step, the marginally corrected
  pair Wick coefficients coef(k, p) = w(k,p) + w(k+3,p) t/6 + w(k+4,p) K4d/24 keep their correction for k = 1
  (always) and for k = 2 only where alpha = mu/sigma <= -2.0; for alpha > -2.0, coef(2, p) = w(2, p), p = 1, 2. The
  same coefficients in every pair term of Q11, Q21 AND Q22 (consistency: one of them alone is x1.09-1.11 worse). In the
  measured arms the E2 "sep" level fix (`coefv` in run_chain) kept its marg-corrected coef(2,2) and beta — port it
  unchanged to reproduce these numbers. Torch switch: `tailr=(("marg2", "sat", None, -2.0, "gauss"),)` on B9/FULL.
  Simpler variant without a threshold (coef(2,p) = w(2,p) everywhere): x0.9867 fresh (28/32) — also fine.
- Not recommended: the eight small dead-side cuts (stack A): 0.6% on the screen, 0.17% on fresh MLPs.

**What this means for the goal.** With every lever measured so far, the chain lands at ~3.9e-9 (3.6e-9 with L5),
rank ~20-27 — the tails cannot close the x1.8 to the top 5. The error lives in the kink-core moments (variance
pattern, kappa4, kappa3 of abs alpha < 1 neurons), and the joint exact-kink-moments ceiling is x0.12-0.14 (pooled
oracle): there is room, but it is in the non-Gaussian propagation of the kink, not in the tails. That is where the
next accuracy session should look (with the one-step-oracle-first discipline of this one: an idea that cannot beat
the chain's own readout on exact moments one step at a time will not beat it through a 16-layer chain).
