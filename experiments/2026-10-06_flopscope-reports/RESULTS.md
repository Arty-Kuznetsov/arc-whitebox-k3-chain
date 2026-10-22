# flopscope cost-model reports (2026-10-06): two reproducible observations, prepared for the public tracker

**Why.** The sponsor is fixing billing disclosures before the designation form opens (forum 18118, mohanty 2026-10-04),
and "helpful bug reports" are the first example the rules give for community-contribution awards (SPEC §6). Both
observations below come from our own engineering (write-up Appendix A.1) and are reproduced here against flopscope
0.12.1 (numpy 2.2.6) in the starter-kit environment. Neither is an exploit; both are places where the meter and the
work done disagree. **Filing them on github.com/AIcrowd/flopscope is the author's action** (the issue texts are
below, ready to paste); nothing has been posted.

## 1. `einsum` is 12–62× slower in wall time than the equivalent `matmul` + `sum` at an identical bill (`einsum_wall.py`)

| G | K | m | einsum bill | matmul+sum bill | einsum s | matmul+sum s | ratio |
|---|---|---|---|---|---|---|---|
| 16 | 16 | 32 | 16,760,832 | 16,760,832 | 0.0049 | 0.0004 | 11.8 |
| 64 | 16 | 64 | 536,608,768 | 536,608,768 | 0.1227 | 0.0025 | 48.9 |
| 128 | 8 | 64 | 536,346,624 | 536,346,624 | 0.1232 | 0.0020 | 61.8 |
| 256 | 16 | 32 | 268,173,312 | 268,173,312 | 0.0776 | 0.0019 | 40.4 |
| 2401 | 1 | 64 | 1,248,980,992 | 1,248,980,993 | 0.2944 | 0.0102 | 29.0 |

The two forms agree to float32 rounding. Why it matters: the grader caps wall time at 120 s per network and zeroes the
network's prediction when it is crossed, so an op that costs the same FLOPs but 30–60× the wall is a hidden cost that
the metric does not show. In our estimator a single `einsum` in the Strassen leaf path would have cost the 120 s cap.

## 2. `linalg.solve` bills an LU plus two triangular solves even when the matrix is triangular (`solve_triangular_billing.py`)

| n | nrhs | `solve(R, B)` billed | = LU + fwd/back | vs one triangular solve (n²·nrhs) | `inv(R)` | `inv(R) @ B` |
|---|---|---|---|---|---|---|
| 256 | 256 | 44,739,242 | 11,184,810 + 33,554,432 | ×2.67 | 33,554,432 | 33,488,896 |
| 448 | 448 | 239,774,378 | 59,943,594 + 179,830,784 | ×2.67 | 179,830,784 | 179,630,080 |
| 448 | 1024 | 470,985,386 | 59,943,594 + 411,041,792 | ×2.29 | 179,830,784 | 410,583,040 |

A triangular system needs one triangular solve (n²·nrhs multiply-adds, i.e. about n²·nrhs flops in flopscope's
convention), with no factorisation. flopscope has no `solve_triangular`, so the only way to be billed for the work
actually done is an explicit inverse applied by `matmul` — which is what our estimator does at the band entry (write-up
A.1), at a cost in numerical accuracy (R's condition number reaches 7·10⁴ in float32). Request: a
`fnp.linalg.solve_triangular(a, b, lower=…)` op billed at one triangular solve, or structure detection inside `solve`.

## Issue texts (ready to paste; the author files them)
Checked 2026-10-13 against all 108 issues of AIcrowd/flopscope (AndreasHad04 filed #267–#298 on billing and symmetry):
neither A nor B is reported by anyone. Filing them before ARC's planned flopscope release gives them the most use.


### Issue A — `einsum` wall time is 12–62× that of the equivalent `matmul` + `sum` at an identical bill

**Summary.** `fnp.einsum("gkij,gkjl->gil", P, Q)` and `fnp.sum(fnp.matmul(P, Q), axis=1)` are billed identically and
agree to float32 rounding, but the einsum takes 12–62× longer in wall time (flopscope 0.12.1, numpy 2.2.6, Apple M2 Max,
in-process). Since the evaluator enforces a 120 s wall cap per network and zeroes the prediction when it is exceeded,
equal-bill operations with very different wall times are a hidden cost that the cost model does not reflect. It would
help participants if the documentation named `einsum` as slow, or if the symmetry-aware einsum path dispatched to the
batched-matmul kernel when the equation is a plain batched contraction.

**Repro.** (the table and script above; `einsum_wall.py` is self-contained)

### Issue B — `linalg.solve` has no triangular form: a triangular system is billed as LU + two substitutions

**Summary.** `fnp.linalg.solve(R, B)` with an upper-triangular `R` bills 2n³/3 + 2·n²·nrhs (an LU factorisation plus a
forward and a back substitution), i.e. 2.3–2.7× the cost of the one triangular solve that the system needs. There is no
`solve_triangular`, so a participant who wants to be billed for the work actually done has to form an explicit inverse
and apply it with `matmul`, which bills less (n²·nrhs for the application) but is numerically worse. The rules encourage
"genuinely reducing computation"; a `fnp.linalg.solve_triangular(a, b, lower=…)` billed at one triangular solve would let
the meter follow the work.

**Repro.** (`solve_triangular_billing.py`, table above)
