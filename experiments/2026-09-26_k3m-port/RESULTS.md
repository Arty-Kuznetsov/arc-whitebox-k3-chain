# k3m = k3l + deadcut −2.5 (+ a low-rank path leg, which does NOT pay off)

**Date:** 2026-09-26, cloud session (4 vCPU / 16 GB, x86_64), branch `claude/friendly-mendel-a6a3yf`.
**Task:** build `src/estimator_k3m.py` = k3l + `deadcut −2.5` + truncation of the path leg to rank 256,
the only combination of existing switches for which the previous session's Pareto front promised
adjusted 5.739e-9 → 5.371e-9 locally (≈5.21e-9 on the grader, ×0.936).

**Short answer: truncating the path leg does NOT pay off in production, and not "by a little": the sign is reversed.**
On the real flopscope meter, rank 256 costs **more** than the full transport it replaces:
util 25.169% → **25.715% (×1.0217)**. So `pathrank 256` makes both accuracy (+0.77%) and cost (+2.2%) worse.
The cause is arithmetic and general (§3.3): the previous session's model priced the discarded transport as a **bare**
n³ product (1.0 unit), whereas it rides in the Strassen stack and costs **0.602**, while the truncation machinery
(build the factorisation + rebuild Y for the (2,1) birth slice + thin transport + materialisation)
gets no Strassen discount and costs ≈3.8·r/n. **The break-even point is rank 158**, i.e. the rank has to be cut
by a factor of ≈6.5, not 4. This is the same mechanism that killed k3k.

From the other side, accuracy closes the window: the truncation penalty **saturates already by rank 128** (§4), so
every rank that saves FLOPs costs ≈1.7% raw. The best measured case is rank 64: util ×0.980
against raw ×1.017 = **×0.9966**, i.e. 0.34% instead of the promised 6.4%, and that is inside the 0.6% grader/local
calibration error. On top of that, the exact SVD hits the wall-time limit (§7).

**So k3m ships as k3l + `deadcut −2.5`**; `pathrank` is in the file, validated and
OFF by default. Measured on 100 MLPs (§6): **adjusted 5.5215e-9 (×0.9621 vs k3l)**, final
2.2174e-8, util **24.894%**, **0 failures, 99/100 MLPs better**; expected on the grader ≈5.36e-9. This
reproduces the Mac-confirmed number 5.5209e-9 to 1.1e-4, and **the utilisation matched
exactly**, so this standalone file is the one to package. Nothing was submitted to AIcrowd, `whest package` was not run; no
existing file was edited (`src/estimator_k3l.py`, the rest of `src/*`, `chain.py`, `ablate.py`,
`notes/`, `CLAUDE.md`, `CLOUD.md` are unchanged).

---

## 0. Warm-up and session gate: passed exactly

| check | result |
|---|---|
| `scripts/cloud_setup.sh --data mini10 --no-whest --no-torch --no-check` | 44 s, 3 sha256 OK |
| **`ablate.py k3l_t 10` (session gate)** | **2.3135e-08** ✅ |
| per MLP 0 / 1 / 2 / 7 / 9 | 2.4074 / 2.1572 / 2.2646 / 2.3867 / 2.8053e-8, matches the Mac and the first cloud session **to the last printed digit** |
| `whest validate --estimator src/estimator_k3m.py` | passed, 178 ms |

Timing on this machine: torch 17–21 s/MLP (Mac ≈11 s). The container restarted once in the middle of the
session: the 100-MLP run was restarted from scratch; all other numbers were obtained before the restart and
are in `logs/`.

## 1. What k3m is

`src/estimator_k3m.py` is a **standalone copy** of `estimator_k3l.py` (not a wrapper: a wrapper that imports
another module packages badly) with two changes:

* **`deadcut` default −3.0 → −2.5.** The cut machinery itself has not changed since k3j (rectangular
  Strassen, row compaction, scatter-back). The Mac already confirmed this arithmetic on 100 MLPs through
  a subclass wrapper: adjusted 5.7387e-9 → **5.5209e-9 (×0.9621)**, util 26.274% → 24.894%, 99/100 MLPs
  better, 0 failures (2026-09-26_cost-frontier/RESULTS.md §5). §6 below is the same measurement, but of the
  **file that gets packaged**.
* **`pathrank` (new, default `None`).** The path leg Y, today one full-width transport per
  layer, is carried as a rank-r factorisation, Y ≈ P V, so the transport moves only r columns:
  Wt Y = (Wt P) V. The semantics mirror `chain.py` **exactly**: the diagonal of Y is zeroed BEFORE truncation and NOT
  zeroed after it, and it is the **truncated** Y that is subtracted from the (2,1) birth slice (`dK21 −= Y∘U`).
  Two backends: `pathsvd=True` is an exact top-r SVD (`fnp.linalg.svd(k=r)`, the same operator as in the
  torch arm); `pathsvd=False` is a single-pass randomised range finder with r + chunk probes.

## 2. Port self-checks: all passed

| # | check | result |
|---|---|---|
| (a) | `deadcut=-3.0, pathrank=None` vs `estimator_k3l.py`, 1024×16 | **BITWISE equal**, FLOPs equal (26.622% B), residual 0.595 s (k3l 0.590 s) ✅ |
| | k3m default (`deadcut=-2.5`) on the same MLP | 25.169% B, final rel diff vs k3l 1.6e-05 |
| (a2) | **low-rank machinery:** `pathrank = n` (the reconstruction is exact but goes through the whole new factorised path) vs k3l | rel **6.1e-07** on all layers: the factorised transport, the materialisation, the new slot layout and the SVD branch are **exact** ✅ |
| (b) | vs the torch chain, n=128, depth 8, `thin=128` (`corr="split128"`, i.e. the correction leg is exact and cannot hide a mismatch behind the range finder) | `pathrank=None` **1.7–2.6e-07**; `pathrank=64` **1.6–3.3e-07**, within the 2–6e-7 band of all previous ports ✅ |
| | size of the truncation itself (torch p64 vs none) | **2.3e-05 … 1.1e-04**, i.e. ≈300 times the port error → the agreement is not accidental (the same stress style as k3j at threshold −1.0) |

## 3. Cost of the low-rank path leg: a MEASUREMENT, not a model

### 3.1 Per-layer billing (`path_cost_probe.py`, n=1024, Strassen L4)

Unit: one bare n×n×n product (2n³ FLOPs = 0.0977% B).

```
path leg today: rides in the "full" shared stack next to C
  shared(Wt, stack of 2 [C, Y])     1.2137 u
  shared(Wt, stack of 1 [C only])   0.6116 u
  -> the path leg's own transport: 0.6021 u per layer = 0.882% B over 15 layers
```

| r | svd(k=r) | materialisation | thin transport | rebuild Y | **TOTAL svd** | TOTAL rf | vs full |
|---|---|---|---|---|---|---|---|
| 384 | 0.7500 | 0.2333 | 0.2353 | 0.2333 | **1.4519** | 1.4853 | 2.41× / 2.47× |
| **256** | 0.5000 | 0.1576 | 0.1601 | 0.1576 | **0.9754** | 0.9714 | **1.62× / 1.61×** |
| 192 | 0.3750 | 0.1198 | 0.1224 | 0.1198 | 0.7371 | 0.7311 | 1.22× / 1.21× |
| 128 | 0.2500 | 0.0820 | 0.0848 | 0.0820 | 0.4988 | 0.5033 | 0.83× / 0.84× |
| 96 | 0.1875 | 0.0631 | 0.0660 | 0.0631 | 0.3796 | 0.3944 | 0.63× / 0.66× |
| 64 | 0.1250 | 0.0442 | 0.0472 | 0.0442 | 0.2605 | 0.2889 | 0.43× / 0.48× |
| 32 | 0.0625 | 0.0252 | 0.0284 | 0.0252 | 0.1414 | 0.1871 | 0.23× / 0.31× |

**Break-even point: rank 158** (linear in r, both backends). `fnp.linalg.svd(a, k=r)` is billed exactly
`min(4·m·n·k, economy)` = **2r/n units at weight 1.0**, and that is half of the whole cost.

### 3.2 Confirmation on the full predict (`k3m_check.py fs`, step 4; same MLP, `deadcut=-2.5`)

| pathrank | util (% B) | ×None | §3.1 prediction | wall |
|---|---|---|---|---|
| None | 25.169% | 1.0000 | — | 75.0 s |
| 384 svd / rf | 26.414% / 26.467% | 1.0495 / 1.0516 | +1.24 points ✓ | 245.6 / 137.4 s |
| **256 svd / rf** | **25.715% / 25.713%** | **1.0217 / 1.0216** | **+0.546 points ✓** | 190.0 / 115.9 s |
| 192 svd / rf | 25.366% / 25.360% | 1.0078 / 1.0076 | +0.197 points ✓ | 304.2 / 79.2 s |
| 128 svd / rf | 25.017% / 25.025% | 0.9940 / 0.9943 | −0.151 points ✓ | 75.6 / 76.4 s |
| 64 svd / rf | 24.668% / 24.711% | 0.9801 / 0.9818 | −0.500 points ✓ | 76.9 / 73.7 s |
| 48 svd / rf | 24.581% / 24.635% | 0.9766 / 0.9788 | −0.588 points ✓ | 340.6 / 119.2 s |
| 32 svd / rf | 24.493% / 24.561% | 0.9732 / 0.9758 | −0.675 points ✓ | 235.0 / 76.4 s |

(ranks 48 and 32: `cost_rank32.py`, same MLP and same harness.) The per-layer probe and the full predict
agree **to the third digit at all eight ranks**; for example rank 32: predicted 24.494%,
measured 24.493%. Break-even falls between 192 (+0.78%) and 128 (−0.60%), i.e. right on
the 158 predicted by §3.1.

### 3.3 Mechanism: where the previous session's model went wrong

The `cost.py` model treated `pathrank` as "a pure reduction of transport volume (ρ/n)" and priced the range
finder at ≈0.06% B. Both halves are wrong in the same direction:

1. **The discarded product is already discounted.** The path-leg transport rides in the Strassen L4 stack, so it
   costs 0.602 units, not 1.0. The most that can be saved is 0.602, not 1.0.
2. **The truncation machinery gets no discount, and it is three products, not one.** Besides the thin transport
   it needs: (i) a rebuild of Y for the (2,1) birth slice, because `dK21 −= Y∘U` is an elementwise operation
   and needs Y as n×n; (ii) a second materialisation of `Wt Y` = (Wt P)·V for the readout and the merge into
   L1/L2, for the same reason; (iii) building the factorisation itself, which at weight 1.0 costs 2r/n
   (SVD) or the same via sketch + QR + QᵀY.

Total ≈3.8·r/n against a base of 0.602 ⇒ cutting the rank by a factor of 4 (1024 → 256) gives 0.975 instead of 0.602.
**This is the same accounting error that buried k3k:** there the band basis update was billed at 1.0 and ate
the Strassen-discount saving. A rule worth writing down: *on this meter any projection
compression has to cut the rank at least 5–6 times before it saves anything, because the replaced
product is already multiplied by 0.60 and the compression is not.*

## 4. Truncation accuracy (torch, 10 baked MLPs, on top of `deadcut −2.5`)

| pathrank | raw | Δ vs pathrank=None |
|---|---|---|
| None | 2.3462e-08 | — |
| 256 | 2.3643e-08 | +0.77% |
| 192 | 2.3768e-08 | +1.30% |
| 128 | 2.3840e-08 | +1.61% |
| 64 | 2.3857e-08 | +1.68% |
| 32 | 2.4111e-08 | +2.77% |

**The penalty saturates on a plateau of ≈+1.7% between ranks 128 and 64** and rises again at 32, while
removing the path leg entirely costs +13.8% (cost-frontier §2.2). So the bulk of the leg's content lives in
a few dozen top singular directions, but its last ≈1.7% is spread over the whole spectrum.
Practical consequence: any rank that saves FLOPs (< 158) pays ≥1.7% raw, and below 64 it pays
more than it saves.

## 5. Combined verdict: everything measured, nothing projected

adjusted × = (raw ×, torch 10 MLPs) × (util ×, flopscope full predict). Base: k3m default.

| configuration | raw × | util × | **adjusted ×** | |
|---|---|---|---|---|
| **k3m = deadcut −2.5, pathrank None** | 1.0000 | 1.0000 | **1.0000** | ✅ ships |
| + pathrank 256 (the requested combination) | 1.0077 | **1.0217** | **1.0296** | ❌ worse on both axes |
| + pathrank 192 | 1.0130 | 1.0078 | 1.0209 | ❌ |
| + pathrank 128 | 1.0161 | 0.9940 | 1.0100 | ❌ |
| + pathrank 64 | 1.0168 | **0.9801** | **0.9966** | ⚠️ the only minimum of the curve: 0.34%, inside the calibration error |
| + pathrank 32 | 1.0277 | 0.9732 | **1.0002** | ❌ exactly break-even |

The curve has **a single shallow minimum at rank 64** (×0.9966) and is ≥ 1 everywhere else:
above 64 the machinery cost grows faster than the saving, below 64 accuracy drops faster than the saving grows.
Rank 48 (util ×0.9766, raw between 64 and 32) lies on the same flat bottom, ≈×0.997. The best this
lever can give in production is a third of a percent.

For comparison, what the previous session's Pareto front promised for `deadcut −2.5 + pathrank 256`:
util 24.78% → 24.07% and adjusted ×0.936. Measured: util 24.894% → **25.44%** (×1.0217 applied to the
100-MLP base) and adjusted **×1.0296**. The discrepancy is in sign, not in magnitude.

**All three routes to the 0.882% B that the path leg costs are closed:**
* truncation: here (break-even at rank 158, and the ranks below it cost more accuracy than they save);
* folding into the tree leg (free in the count): NEGATIVE §53, 1.1530e-7 against 2.4441e-8,
  a disaster, because B is also used by the correction leg (3B diag α) and the P2 feed, and Y/(2d2)
  is not small at depth;
* removing it entirely: +13.8% raw for −0.88 points of util (cost-frontier §2.2).

## 6. 100 MLPs: k3m on the production path

`.venv-whest/bin/whest run --estimator src/estimator_k3m.py --split mini --runner local --max-threads 4
--no-residual-wall-time-limit --wall-time-limit 600` → `runs/k3m_n100.json`
(comparisons: `logs/compare_k3m_vs_k3l.txt`, `logs/compare_k3m_vs_mac.txt`).

### 6.1 k3m vs k3l

| | k3l (`k3l_n100.json`) | **k3m** | |
|---|---|---|---|
| **adjusted** | 5.7387e-9 | **5.5215e-9** | **×0.9621** |
| final-layer MSE | 2.1837e-8 | 2.2174e-8 | ×1.0155 |
| all-layers MSE | 8.2645e-9 | 8.3635e-9 | ×1.0120 |
| mean util | 26.274% | **24.894%** | ×0.9475 |
| util range over MLPs | 25.494–27.068% | 24.161–25.713% | |
| **failures** | 0 | **0** | |

* **99 of 100 MLPs better on adjusted** (mean ratio 0.9625, median 0.9624, worst MLP #56 ×1.003),
  while raw is worse on 90 of 100: the lever trades accuracy for utilisation exactly as designed,
  and the trade wins on almost every MLP.
* Expected on the grader at the historical grader/local ratio ≈0.971: **≈5.36e-9**.

### 6.2 Clean room: cloud vs Mac on the same arithmetic

The Mac measured this arithmetic through a subclass wrapper (`v_k3l_cut25.py`); here the standalone file itself was measured.

| | Mac (wrapper) | cloud (`src/estimator_k3m.py`) | |
|---|---|---|---|
| adjusted | 5.5209e-9 | 5.5215e-9 | ×1.0001 |
| final-layer MSE | 2.2172e-8 | 2.2174e-8 | ×1.0001 |
| **mean util** | **24.894%** | **24.894%** | **×1.0000** |
| util range over MLPs | 24.161–25.713% | 24.161–25.713% | matched to the printed digits |
| failures | 0 | 0 | |

**Utilisation matched exactly, including the per-MLP bounds**: the flopscope meter is hardware-independent, as
CLOUD.md §2 promises. MSE differed by 1.1e-4 relative, and the per-MLP spread 0.9964–1.0027
is **symmetric** (48 better / 52 worse): this is re-association of sums in a different BLAS, not a systematic
shift. So the standalone file reproduces the arithmetic the Mac already confirmed, and it is the one
to package.

## 7. Residual and wall: reported, but decided on the Mac

CLOUD.md §2: timings from here do not transfer at all. What can be seen:

* **`deadcut −2.5` lowers residual**: on this box 0.488 s against 0.590 s for k3l (the same
  probe MLP, §2), consistent with the Mac measurement (0.197/0.227 against 0.207/0.239): fewer alive rows
  in the contractions. The margin to the 0.4 s cap does not get worse.
* The absolute numbers of the 100-MLP run from here (residual 0.513/0.574 s, wall 66.8/72.5 s) are **×2.6 and ×3.4
  of the Mac on the same arithmetic** (0.197/0.227 and 19.4/20.4 s). This is exactly why CLOUD.md §2
  forbids transferring timings: the residual decision and the packaging stay on the Mac.
* **`pathrank` increases residual**: +2 Strassen calls per layer (thin transport and materialisation),
  on this box 0.496 → 0.65–0.79 s. One more argument against, but it is for the Mac to decide.
* **The exact SVD hits the wall-time limit regardless of the meter.** 15 economy SVDs of 1024² per predict:
  standalone probe 5–9 s per call, full predict **190 s at rank 256 against 75 s for None**. The grader
  cap is 120 s, and the grader is ≈3.2× slower than the Mac. So even if the billing were favourable, only the
  range-finder backend could ship, and it approximates worse than the exact SVD, i.e. the +1.68% at rank 64
  would grow (final rel diff 2.84e-05 against 1.79e-05).

## 8. What is a measurement here and what is not

**Measured on flopscope (this machine; the meter is hardware-independent):** every `util` in §2, §3.1, §3.2, §5,
§6; the bitwise equality; the accuracy of the low-rank machinery. §6.2 shows that util matches the Mac
to the printed digits, and MSE to 1.1e-4.
**Measured in torch float64 (baked MLPs, sha256-checked weights):** every `raw` in §4 and §5.
**Not measured / does not transfer:** residual and wall (§7), Mac only; the grader multiplier
(grader/local ≈0.971) is taken from the submission history, not re-measured.
**Projected:** exactly one number, "util 25.44% on the 100-MLP base" in §5 (that is 24.894% × 1.0217,
i.e. a measured ratio applied to the measured Mac base).

## 9. Files

| file | what |
|---|---|
| `../../src/estimator_k3m.py` | the estimator itself (standalone; `deadcut=-2.5`, `pathrank=None`) |
| `k3m_check.py` | self-checks: `fs` (bitwise vs k3l, machinery, cost table) and `torch` (cross-check n=128) |
| `path_cost_probe.py` | per-layer billing of the path leg and of all truncation products (§3.1) |
| `cost_rank32.py` | addition to §3.2: ranks 48 and 32 |
| `torch_ranks.py` | truncation accuracy on 10 MLPs (§4); imports `run_chain`, edits nothing |
| `compare_n100.py` | per-MLP comparison of two 100-MLP `whest run` reports (validated: reproduces cost-frontier §5 exactly) |
| `logs/` | `k3m_check_fs.txt`, `k3m_check_torch.txt`, `torch_ranks.{txt,jsonl}`, `cost_pathrank*.json` |
| `runs/k3m_n100.json` | 100-MLP report of k3m (§6) |
