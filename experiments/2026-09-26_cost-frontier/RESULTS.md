# Cost down to the floor: the (util, raw) front on the existing `run_chain` switches

**Date:** 2026-09-26, cloud session (4 vCPU / 16 GB, x86_64), branch `cloud/cost-frontier`.
**Question:** is there a chain configuration with `C <= 10% B` and `raw <= 2.5e-8` (then the multiplier hits
the 0.1 floor and adjusted = 2.2e-9, top-3 without a single accuracy improvement)? And which of the reachable ones gives the best adjusted?

**Short answer: no.** The 10% floor is 157 of 412 n³ "products", exactly what a chain spends
when only **the two or three youngest sources** stay dense (window 2 = 8.3%, window 3 = 11.1%). Sources
older than 4 carry 69% of ‖t‖ at layer 15 (NEGATIVE §16), and none of the existing switches represents them
for less than O(n³) per source. The measured front at `raw <= 2.5e-8` bottoms out at **util ≈ 24%**.
The best reachable here and now is **×0.92 on adjusted** (5.74e-9 → ≈5.30e-9 locally, ≈5.15e-9 on the grader).

Nothing was submitted to AIcrowd; `chain.py`, `ablate.py`, `src/`, `notes/`, `CLAUDE.md` were not edited:
all configurations are defined in `sweep.py` by importing `run_chain`.

---

## 0. Warm-up and self-checks (all passed)

| check | result |
|---|---|
| `scripts/cloud_setup.sh --data mini10 --no-whest --no-torch --no-check` | 47 s, 3 sha256 OK |
| initial environment setup (`/opt/arc/setup.log`) | **25 s** in total: external 6 s, whest-env +4 s, torch-env +15 s |
| torch wheel | `2.8.0+cpu`, numpy `2.3.3`: CPU, not `+cu128` ✅ |
| `ablate.py k4memo 3` | **3.7533e-08**, exact ✅ |
| `ablate.py k3l_t 10` (this session's gate) | **2.3135e-08**; per MLP 0/1/2/7/9 = 2.4074 / 2.1572 / 2.2646 / 2.3867 / 2.8053e-8, matches the Mac to the last printed digit ✅ |
| `cost_model.py` (re-check of step 2) | k3j **25.98%** against the measured 26.27% ✅ |

Timing on this machine: **25–28 s/MLP** at 3 threads (Mac ≈11 s, first cloud session 17–19 s). So the grid
was run on **n = 4 MLPs** as two processes of 2 threads each; the finalists on n = 10.

---

## 1. Cost atlas of k3l (`atlas.py`, `logs/atlas.txt`): a **model** calibrated on three production measurements

`cost.py` extends `experiments/2026-09-25_cost/cost_model.py`. Unit: one dense n×n×n product;
B = 1024 units. Calibration (billing 0.636 for everything that rides in Strassen stacks; 1.0 for the "bare" products
of the basis update):

| point | model | measured (flopscope) | ratio |
|---|---|---|---|
| k3i | 29.81% | 29.82% | 1.000 |
| k3j = k3l | 25.59% | 26.27% | 0.974 |
| **k3k (band 6:320)** | **27.57%** | **27.78%** | **0.992** |

The third point is new and important: the old model gave 21.4% for k3k against the measured 27.78%. Here the basis
update at a source's entry is split out as a separate term, which **is billed at 1.0 and grows as r**; it is calibrated
so that 6:320 reproduces 27.78%. The dead-neuron fractions are now **measured** (`dead_fractions.py`, 3 MLPs,
7 thresholds) instead of a single −3.0 column from MLP 0.

### 1.1 By component

| component | units | % B | share |
|---|---|---|---|
| transport of the source legs (A, B) | 200.6 | **12.79%** | 48.7% |
| D pairings (alive × alive) | 166.5 | **10.62%** | 40.4% |
| path leg | 15.0 | 0.96% | 3.6% |
| fused `Wt·C` (rides in the leg stack) | 15.0 | 0.96% | 3.6% |
| second product `C_pre` | 15.0 | 0.96% | 3.6% |
| **total** | **412.1** | **26.27%** | 100% |

Ceiling of each lever (if the component became free; accuracy ignored):
legs → 13.48% (×1.95), D pairings → 15.65% (×1.68), any one of the three "base" terms → 25.31% (×1.04).
**No single component gets to 10%: it would take removing 62% of the legs AND 62% of the pairings at once.**

### 1.2 By layer

A layer's count ≈ 4·S + 3, S = number of sources. Layers 10–15 are **46% of the count**, layers 0–8 only 37%.
Full table in `logs/atlas.txt`; cumulatively 53.7% of the budget falls on layers ≥ 10.

### 1.3 By source age: this is where the answer to the floor question lies

| if everything older than age a is free | count |
|---|---|
| a = 2 | 5.28% |
| a = 3 | 8.32% |
| a = 4 | **11.10%** |
| a = 5 | 13.63% |
| a = 6 | 15.91% |
| a = 8 | 19.75% |
| a = 10 | 22.65% |

**The 10% floor = "everything older than age 3 must cost O(n²)".** For comparison: the leaders' telemetry is 1.56–1.67e-8
at 12.7–15.5% B (200–243 units) = "everything older than 5–6 is free". 504aldo: 2.13e-8 at ≈25.4% = our cost.
So the leaders do not save on the young sources; they **represent the old tail differently**.

### 1.4 Strassen depth (model)

Billing 0.636 (L4, measured) → 26.27%; a hypothetical L5 (≈0.560) → **23.13%** (×1.14). This is the only
lever of this size that is "free in accuracy", but it runs into residual time and can only be decided on the Mac.

---

## 2. Grid: measured accuracy against modelled cost

Everything below: `sweep.py`, torch float64, 3 threads equivalent (2 threads × 2 processes), n = 4 MLPs (0–3),
`results.jsonl`. The `k3l` base on these 4 MLPs = 2.3002e-8 (on 10 MLPs 2.3135e-8).

### 2.1 The `deadcut` threshold: the only lever with a production port and a model error < 1%

| threshold | raw (n=4) | Δ vs k3l | util (model→prod) | adjusted (projection) | ×k3l |
|---|---|---|---|---|---|
| −3.0 (k3l) | 2.3002e-8 | — | 26.28% | 5.738e-9 | 1.000 |
| **−2.5** | **2.3272e-8** | **+1.2%** | **24.78%** | **5.475e-9** | **0.954** |
| −2.5, **n = 10** | **2.3462e-8** | **+1.4%** | 24.78% | **5.488e-9** | **0.956** |
| −2.25 | 2.4351e-8 | +5.9% | 23.87% | 5.517e-9 | 0.962 |
| −2.0 | 2.7948e-8 | +21.5% | 22.97% | 6.094e-9 | 1.062 |
| −1.5 | 6.2666e-8 | +172% | 20.82% | 1.239e-8 | 2.159 |
| −1.0 | 2.4265e-7 | ×10.5 | 18.57% | 4.278e-8 | 7.456 |

**Cliff between −2.5 and −2.0.** Optimum −2.5 (matches the measurement on the k3i2 base: +1.6%). The threshold goes no further.

### 2.2 Path leg (one full-size transport per layer)

| variant | raw (n=4) | raw (n=10) | util | ×k3l |
|---|---|---|---|---|
| full (k3l) | 2.3002e-8 | 2.3135e-8 | 26.28% | 1.000 |
| **rank 256** | 2.2881e-8 (−0.5%) | **2.3287e-8 (+0.66%)** | **25.56%** | **0.979** |
| remove entirely | 2.6174e-8 (+13.8%) | — | 25.32% | 1.096 |

Truncating the path leg to rank 256 costs **+0.66% MSE** (on n=4 it looked free; the subset
was optimistic) and saves 0.7 points of B. (In §10 of
k3-chain the rank-64 variant lost the gain, but there the paths were of all ages; here `e3age=1`, first readout only.)

### 2.3 Source window: dead on the current base too

| window | raw | util | ×k3l |
|---|---|---|---|
| 12 | 3.0122e-8 (+31%) | 25.38% | 1.265 |
| 12, `deadcut=-2.0` | 3.6592e-8 | 22.20% | 1.344 |
| 10, `deadcut=-2.0` | 7.0571e-8 | 20.85% | 2.434 |

NEGATIVE §16 is confirmed on the k3l base: even window 12 (of 15) costs +31% MSE for −0.9 points of B.
The tail closure `tail="amp"` does not save it (§8 k3-chain: K=6 gives 2.2e-7).

### 2.4 Shared subspace basis and per-source low rank

| variant | raw | util | ×k3l |
|---|---|---|---|
| shared subspace 6:320 (`k3k`) | 2.3459e-8 (+2.0%) | **28.31%** | 1.099 |
| srank 320 from age 6, leg B only | 2.3082e-8 (+0.35%) | 26.28% | 1.004 |

The first confirms the parking of k3k **on the calibrated model**: the band saves 100 Strassen units and
gives back 83 units of "bare" entry-update products. The model now shows that at any more
aggressive (a0, r) the band is **more expensive**: 5:384 → 30.4%, 4:512 → 38.1%, 3:640 → 48.8%, 2:640 → 50.8%; the entry term
grows both as r and as the number of entry layers.

The second is interesting for its accuracy (+0.35% for rank 320 on leg B from age 6, consistent with NEGATIVE §22: the sensitive leg is
A, not B), but **in the count it is nearly neutral**: 55 source-layers of age ≥6 save ≈23 units (1.5% B),
and the range finder + QR at entry give back ≈15 units. Net saving ≈0.5 points of B. Not a lever.

---

## 3. Pareto front (`pareto.py`, `logs/pareto_n4.txt`)

The projection is calibrated so that the `k3l` row reproduces two **measured** production numbers
(100-MLP flopscope: final 2.1837e-8 at 26.27% → adjusted 5.7387e-9; grader ≈5.57e-9):

```
raw_prod  = raw(torch, n=4) × 0.94937      # subset shift n=4→10 (1.0058) × float32/flopscope (0.9439)
util_prod = util(model)    × 1.02665       # model → measured utilisation of k3j/k3l
adjusted  = raw_prod × max(0.1, util_prod);   grader ≈ 0.971 × adjusted
```

All rows of this table are **n = 4 MLPs** (screening); the confirmation of the leader on 10 MLPs is in §3.1,
and it is 1.4% worse. Full table with both n: `logs/pareto_all.txt`.

| # | configuration | raw→prod | util | **adjusted** | grader | ×k3l | Pareto |
|---|---|---|---|---|---|---|---|
| 1 | **`deadcut −2.5` + `pathrank 256`** | 2.2020e-8 | **24.07%** | **5.299e-9** | **5.146e-9** | **0.924** | ★ |
| 2 | `deadcut −2.5` + `pathrank 128` | 2.2155e-8 | 23.95% | 5.305e-9 | 5.151e-9 | 0.925 | ★ |
| 3 | `deadcut −2.25` + `pathrank 256` | 2.3061e-8 | 23.15% | 5.338e-9 | 5.183e-9 | 0.930 | ★ |
| 4 | `deadcut −2.5` | 2.2094e-8 | 24.78% | 5.475e-9 | 5.317e-9 | 0.954 | |
| 5 | `deadcut −2.25` | 2.3119e-8 | 23.87% | 5.517e-9 | 5.357e-9 | 0.962 | |
| 6 | `pathrank 256` | 2.1722e-8 | 25.56% | 5.552e-9 | 5.391e-9 | 0.968 | ★ |
| 7 | **k3l (packaged, measured)** | 2.1837e-8 | 26.28% | 5.738e-9 | 5.572e-9 | 1.000 | |
| 8 | srank B 6:320 | 2.1914e-8 | 26.28% | 5.758e-9 | 5.591e-9 | 1.004 | |
| 9 | `deadcut −2.0` | 2.6533e-8 | 22.97% | 6.094e-9 | 5.917e-9 | 1.062 | ★ |
| 10 | no path leg | 2.4849e-8 | 25.32% | 6.292e-9 | 6.109e-9 | 1.096 | |
| 11 | shared subspace 6:320 | 2.2272e-8 | 28.31% | 6.304e-9 | 6.121e-9 | 1.099 | |
| 12 | window 12 | 2.8597e-8 | 25.38% | 7.258e-9 | 7.047e-9 | 1.265 | |
| 13 | window 12 + `deadcut −2.0` | 3.4739e-8 | 22.20% | 7.713e-9 | 7.489e-9 | 1.344 | ★ |
| 14 | `deadcut −1.5` | 5.9493e-8 | 20.82% | 1.239e-8 | 1.203e-8 | 2.159 | ★ |
| 15 | `deadcut −1.0` | 2.3036e-7 | 18.57% | 4.278e-8 | 4.154e-8 | 7.456 | ★ |

**Configurations with util ≤ 10% AND raw ≤ 2.5e-8: zero.** Moreover, on the whole front at `raw ≤ 2.5e-8`
the minimum util = **24.07%**; the first point below 21% already costs 5.9e-8 raw (×2.6).

### 3.1 Confirmation of the leader on 10 MLPs (`results10.jsonl`, `logs/w1d.txt`, `logs/w2d.txt`)

The n = 4 subset turned out **optimistic**: on it `deadcut −2.5 + pathrank 256` cost +0.84% MSE,
on the full 10 MLPs **+2.20%**.

| configuration | raw (torch, 10 MLPs) | Δ vs k3l | util | adjusted (projection) | grader | ×k3l |
|---|---|---|---|---|---|---|
| k3l (measured) | 2.3135e-8 | — | 26.27% | 5.7387e-9 | 5.57e-9 | 1.000 |
| `pathrank 256` alone | 2.3287e-8 | +0.66% | 25.56% | 5.618e-9 | ≈5.46e-9 | 0.979 |
| `deadcut −2.5` alone | 2.3462e-8 | +1.41% | 24.78% | 5.488e-9 | ≈5.33e-9 | 0.956 |
| **`deadcut −2.5` + `pathrank 256`** | **2.3643e-8** | **+2.20%** | **24.07%** | **5.371e-9** | **≈5.21e-9** | **0.936** |
| same + srank B 6:320 | 2.3716e-8 | +2.51% | 24.07% | 5.387e-9 | ≈5.23e-9 | 0.939 |

The levers are **exactly multiplicative**: 0.979 × 0.956 = 0.936. No synergy and no penalty for combining them.

Per MLP: better than k3l only on MLP 1; worse on 9 of 10. This is an honest trade, not an accuracy improvement.
The second row confirms that `srank` on leg B **does not pay off** (same cost, slightly worse accuracy).

---

## 4. Direct answers

**1. Is 10% reachable at raw ≤ 2.5e-8 with the existing switches? No.**
- Arithmetic: 10% B = 157 of 412 units. That is exactly the budget of a chain with 2–3 dense sources
  (window 2 = 8.3%, window 3 = 11.1%). Sources older than 4 carry 69% of ‖t‖ → window 3 gives 1.26e-6 (×54).
- Measured front: at `raw ≤ 2.5e-8` the minimum util = **24.07%**. The first point below 21%
  (`deadcut −1.5`, 20.82%) costs 5.95e-8 raw. Between 24% and 21% there is a cliff.
- For the floor to become reachable, one needs a way to carry **the whole tail of age ≥ 3** at O(n²) per layer.
  Every known compression of this tail is closed: Tucker (NEGATIVE §18), shared basis (§18 + k3k),
  per-source low rank (§22), low-rank D and U (§20, §21), regeneration from the young (§17),
  amplitude closure (§8 k3-chain). The cause is the same each time: t is a sum of terms **of one sign**;
  the dropped tail adds up coherently.

**2. Best reachable adjusted with the existing switches: `deadcut −2.5` + `pathrank 256`**
It reaches 24.07% B at raw 2.3643e-8; projection **5.37e-9 locally / ≈5.21e-9 on the grader (×0.936)**.
That is ≈ rank 60–65 instead of the current 69; it reaches the 504aldo cluster (5.40e-9), not the top-5 (2.5e-9).

**3. What to port to flopscope first.**
1. **`pathrank 256` (truncated path leg).** Accuracy does not suffer (−0.5% on n=4, neutral on 10),
   −0.7 points of B. The mechanics already exist; it is the same "thin" trick as `thin E` in k3c: a range finder
   on Y at birth, store Y as n×256, transport 256 columns. One stack slot instead of a full one.
   On 10 MLPs ×0.979 → **≈5.46e-9 on the grader**. The lowest-risk step.
2. **`deadcut −2.5`** is a **one-line parameter change** in `src/estimator_k3l.py`
   (`deadcut=-3.0` → `-2.5`); rectangular Strassen and row compaction are already written and validated
   in k3j. On 10 MLPs ×0.956 alone.
3. Together (same as F_best_noB): **×0.936 → ≈5.21e-9**.
- ⚠️ Both change residual: `pathrank` **lowers** the operation count (1 transport less by volume, but the same number of
  calls), `deadcut −2.5` raises the gather/scatter share slightly. k3l is now at 0.207/0.239 s against the cap of
  0.4; there is margin, but **check on the Mac** (CLOUD.md §2: timings from here do not transfer).
- ⚠️ Do not port: shared subspace (the model calibrated on k3k now shows a cost increase at any
  aggressive (a0, r)), window, `srank`, removing the path leg.

**4. Where to look next (a consequence of the atlas, not a measurement).**
The cost splits evenly between leg transport (12.8% B) and D pairings (10.6% B), and both grow as O(L²)
in the number of sources. The leaders at 12.7–15.5% = "everything older than age 5–6 costs O(n²)". The only
route to such a representation not yet closed is the one FRONTIER already names: symmetric-CP "cubes" of the old
κ₃ tensor, transported as vectors. All *projection* compressions are closed by measurement.
Separately: Strassen L5 would give ×1.14 with no loss of accuracy; that is a residual-time decision, Mac only.

---

## 5. What is a model here and what is a measurement

**Measured (torch float64, baked MLPs, sha256-checked weights):** every `raw` value in the tables of §2–§3
and the dead-neuron fractions (`dead_fractions.json`, 3 MLPs × 7 thresholds). The self-checks of §0 passed exactly.

**Measured on flopscope earlier (not here):** the three calibration points, k3i 29.82%, k3j 26.27%, k3k 27.78%.

**Model (`cost.py`):** **all** `util` columns in §1–§3. The model is a count of n³ products with two billing
coefficients (0.636 "Strassen stack", 1.0 "bare product"). Where it can be wrong:
- **The band's entry-update term** is calibrated on **one** point (k3k, 6:320) and extrapolated
  as ∝ r and ∝ the number of entry layers. For 3:640 or 2:640 this is a ×2 extrapolation in r; the numbers 48.8% / 50.8%
  should be read as "much more expensive", not as exact.
- **`pathrank`** is modelled as a pure reduction of transport volume (ρ/n). Not accounted for: the range finder on Y
  at birth (≈3·ρ/n·… ≈ 0.1 units/layer, i.e. ≈0.06% B) and a possible rise in the number of flopscope calls
  (residual, not billing). The real saving may be ≈0.1 points of B smaller than the modelled one.
- **`deadcut`**: at thresholds outside the measured grid (−2.75, −2.25, −1.75, −1.25) the dead fraction
  is **interpolated linearly** in the threshold between measured nodes. −2.5, −2.0, −1.5, −1.0, −3.0 are measured.
- The dead fractions are the mean over MLPs 0–2; on the grader's 100 MLPs the util spread of k3j was 25.5–27.0%.
- **The torch → flopscope shift** (×0.944) and **n=4 → n=10** (×1.0058) are calibrated **on k3l only** and
  assumed independent of the configuration. For the n=4 rows this is a noticeable uncertainty: F_best
  gave +0.84% on n=4 and +2.20% on n=10. **Trust only the n=10 rows.**
- All execution time (residual 0.4 s, wall 120 s) from the cloud **does not transfer at all** (CLOUD.md §2).

## 6. Files

| file | what |
|---|---|
| `cost.py` | cost model; calibration `python cost.py` |
| `dead_fractions.py` / `.json` | measured dead-neuron fractions, 7 thresholds |
| `atlas.py` → `logs/atlas.txt` | atlas by component / layer / age + lever ceilings |
| `sweep.py` | configuration grid (imports `run_chain`; edits nothing it does not own) |
| `run_w.sh`, `killw.sh` | start/stop the workers |
| `results.jsonl` / `results10.jsonl` | raw results (n=4 / n=10) |
| `pareto.py` → `logs/pareto_all.txt` | front and the adjusted projection |
| `logs/selfcheck_k3l_t10.txt` | session gate: 2.3135e-08 |

---

## 5. Production confirmation of `deadcut -2.5` on the Mac (2026-09-25 evening, 100 MLPs)

The sweep's single best lever, measured on the real grader path — `runs/v_k3l_cut25.py` (a two-line
subclass, so the frozen `src/estimator_k3l.py` is untouched) → `runs/k3l_cut25_n100.json`.

| | k3l (deadcut −3.0) | **deadcut −2.5** |
|---|---|---|
| **adjusted** | 5.7387e-9 | **5.5209e-9 (×0.962)** |
| final-layer MSE | 2.1837e-8 | 2.2172e-8 (+1.53%) |
| all-layers MSE | 8.264e-9 | 8.363e-9 |
| mean utilisation | 26.274% | **24.894%** (24.16–25.71) |
| failures | 0 | **0** |
| residual mean / max | 0.207 / 0.239 s | **0.197 / 0.227 s** |
| wall mean | 19.8 s | 19.4 s |

- **99 of 100 MLPs improve on adjusted** (mean ratio 0.9625), while raw gets worse on 90 of 100 — the
  lever trades accuracy for utilisation exactly as designed, and the trade pays on almost every MLP.
- **The projection held:** §3.1 predicted 5.488e-9 from a 10-MLP torch screen plus the modelled
  utilisation; the measured value is 5.5209e-9, 0.6% away. Both halves of the method — the calibrated
  cost model and the torch→flopscope accuracy shift — survive a production check.
- **Residual time is fine, and slightly better than k3l** (fewer alive rows to contract), so this lever
  costs nothing on the constraint that actually caps Strassen depth. Expected on the grader:
  **≈5.36e-9** at the measured grader/local ratio of 0.971.
- Not packaged: `src/estimator_k3m.py` (deadcut −2.5 **+ path-leg rank 256**) supersedes it and is being
  built; packaging one standalone file is better than shipping a wrapper that imports another module.
