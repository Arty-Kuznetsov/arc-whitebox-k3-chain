# The band knobs, swept on the meter for the first time (2026-10-01)

The cost map (`notes/COST-MAP-2026-10-01.md`) showed the band governs ≈30% of the budget — `band_members` 16.71%
(linear in the rank), `band_entry` 6.81% (≈quadratic), and the members' compressed pairing ≈7% (linear), hidden in the
unnamespaced block. The shipped `(a0, r) = (5, 448)` came from the cloud's **cost model**, which on the same day was
measured wrong by 3.5× (E3), 9× (E3's gain) and in sign (Strassen L5). So both knobs were swept on the real meter.

## 1. The rank, at a0 = 5 (10 MLPs, base = k3r)

| rank | raw × | util × | **adjusted ×** | better |
|---|---|---|---|---|
| 320 | 1.3835 | 0.8873 | **1.2277** | 0/10 |
| 384 | 1.0944 | 0.9430 | **1.0320** | 1/10 |
| **448 (shipped)** | **1** | **1** | **1** | — |
| 512 | 0.9754 | 1.0594 | **1.0334** | **10/10** |

**448 is a sharp optimum.** The utilisation scaling confirms the cost map exactly (predicted ×0.943 for rank 384,
measured ×0.943), so the band really does govern ~30% of the budget — but the accuracy cost of truncating it is
steep: the slope |d log raw / d log util| is **1.54 going down** (> 1, so the product worsens) and **0.43 going up**
(< 1, so the product worsens the other way). The model picked this one correctly.

Worth keeping for the write-up: **rank 512 buys raw ×0.9754 on 10/10 MLPs**, i.e. the rank-448 truncation costs us
2.5% of accuracy. At the 10% multiplier floor that would be free; at our 20% it is not.

## 2. The entry age a0, at rank 448

<!-- A0 -->
| a0 | raw × | util × | **adjusted ×** | better |
|---|---|---|---|---|
| 4 | 1.1744 | 0.9581 | **1.1251** | 0/10 |
| **5 (shipped)** | **1** | **1** | **1** | — |
| 6 | 0.9800 | 1.0368 | **1.0160** | 8/10 |
<!-- /A0 -->

**a0 = 5 is the optimum too.** a0 = 6 keeps sources dense one layer longer and buys raw ×0.980 (8/10) for +3.7% of
utilisation — slope 0.56, below 1, so the product worsens; a0 = 4 compresses one layer earlier and is far worse
(raw ×1.174).

## 3. The pattern that holds across every lever measured on 2026-10-01

| lever | direction | raw × | util × | slope \|d log raw / d log util\| | adjusted × |
|---|---|---|---|---|---|
| band rank 448 → 512 | spend more | 0.9754 | 1.0594 | 0.43 | 1.033 |
| band a0 5 → 6 | spend more | 0.9800 | 1.0368 | 0.56 | 1.016 |
| `allages` (catalogue) | spend more | 0.9461 | 1.259 | 0.25 | 1.191 |
| `thin` 24 → 384 (catalogue) | spend more | 0.9799 | 1.221 | 0.10 | 1.196 |
| `k4k3` (catalogue) | spend more | 0.9607 | 2.395 | 0.05 | 2.300 |
| band rank 448 → 384 | spend less | 1.0944 | 0.9430 | **1.54** | 1.032 |
| band a0 5 → 4 | spend less | 1.1744 | 0.9581 | **3.76** | 1.125 |
| E3 θ = −3.0 | spend less | 1.0093 | 0.9869 | 0.70 | **0.996** |

**Below 1 in every "spend more" direction and above 1 in every "spend less" direction** — the configuration sits
exactly at the kink of its own accuracy/cost curve, and the product raw × util is minimised there. Eight independent
knobs, measured on the meter rather than on the model. The one exception, E3 at 0.70, is worth 0.4% — inside the
grader's resolution and in conflict with the band (§4 of `../2026-10-01_e3/RESULTS.md`).

