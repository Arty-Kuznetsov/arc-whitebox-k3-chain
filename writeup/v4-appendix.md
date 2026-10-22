## Appendix A. Engineering: exact arithmetic, and the budgets that are not the meter

This appendix holds the parts of the work that are engineering rather than understanding. We moved them out of the
main text because the judging guidelines say performance optimisation is of less interest; we keep them because about
a fifth of the score gain since k3m came from here (k3t, k3w, k3x, k3y and k3z together ×0.931 of the ×0.693 from
5.39e-9 to 3.74e-9), and because the three caps in A.2 decided the depth of every design choice.

### A.1 Exact arithmetic: fewer billed FLOPs, the same numbers

Each item below changes the FLOP bill and moves the output only by float rounding (≤ 1.2e-6 relative); each passed a
bitwise gate against its predecessor with the switch off, and each landed on the grader as projected [M].

- **Strassen's recursion in flopscope operations** (permitted with no level limit, as the sponsor confirmed on the forum on 2026-09-11),
  batched over sources with preallocated per-level workspaces so that the number of calls does not grow ×7 per level.
  The D pairings need only Σ_k L_k R_kᵀ, and Strassen is bilinear, so the K leaf products are summed before the
  recombination, which runs once (level 4, k3h: util 32.4% → 28.9%, MSE unchanged).
- **Level 5 only where it pays for its wall.** Level 5 everywhere would bill ×0.939 but doubles the wall time (A.2).
  Measuring each contraction's FLOPs saved per second of wall added (leg transport 0.49 %/s, the C and path-leg
  transport 0.46, C_pre 0.39, the band members 0.26, the D pairing 0.27) and deepening those five recovered most of it
  inside the 120 s cap (k3t, k3w, k3x; grader wall max 95.2 s).
- **C_pre = Wᵀ(CW) is symmetric**: compute the upper block row and the lower-right block, copy the remaining block as a
  transpose — 3/4 of the product, ×0.9934 of the FLOPs at no wall cost.
- **The band entry's triangular solves.** flopscope bills `solve` as LU + back-substitution even when the matrix is
  triangular; one explicit R⁻¹ per entry, applied to the entrant by Strassen, bills ×0.9941 and is numerically clean on
  all 100 public networks even though R's condition number reaches 7·10⁴ in float32 (k3y, graded 3.7573e-9).
- **Winograd's form, and the last half products.** Winograd's variant of the recursion needs 15 matrix additions per
  level instead of 18, and the two half products of the symmetric C_pre also run at level 5: ×0.9940 of the FLOPs
  together, wall-neutral (grader wall median 91.2 s against k3y's 92.2 s; k3z, graded 3.7357e-9).

### A.2 The budgets that are not the meter

Three caps zero a network's prediction when crossed, and in practice they, not the FLOP count, set the depth of every
engineering choice: **400 ms of non-flopscope ("residual") time**, **120 s of wall time**, and **8 GB of memory**.

- Residual time is ≈20–40 µs per flopscope call regardless of the array size, so it caps the number of operations:
  the shipped estimator makes ≈24 000 calls per network and reads 0.31 s (median) / 0.33 s (max) on the grader's own
  telemetry, ≈0.73× our quiet local runner. Every batched design choice in §3.1 exists for this cap.
- Wall time caps the Strassen depth. One more level everywhere bills ×0.939 and doubles the wall (×2.2, ≈140 s projected
  against the 120 s cap) at a peak RSS of 7.21 GB of 8 GB; a 32×32 leaf has no BLAS efficiency left. The level-5 sites
  run at 8–14 billed GFLOP/s against 17–30 for level-4 and element-wise work: deeper recursion buys FLOPs with wall time.
  Five level-5 sites chosen by FLOPs saved per second of wall leave the grader at **98.8 s max / 91.2 s median**; the
  next site measured 0.094 %/s (both of its products have a 448-wide side, so its level-5 leaves are 14 wide) and was
  not taken [M]. Another team reports that a seven-level recursion of their own chain passes the grader's caps
  (forum 18218, 2026-10-04), so the limit is our implementation's wall, not the grader's rules.
- Memory: the level-5 recursion's workspace is most of the shipped estimator's **6.9 GB peak of 8** (one predict,
  measured locally). All 100 grader networks ran without a memory failure.

The leaders' public telemetry is consistent with the same constraints binding them: the top entries spend the same
77–95 s of wall for 0.58–0.79 of our FLOPs, and the first two run at the residual cap (0.37 s medians) [M].

### A.3 Designation and the private re-evaluation

One network whose wall time exceeds 120 s is scored as all zeros with a multiplier of 1, which adds ≈9·10⁻³ to the mean
over 100 networks — the whole entry. The designated submission's wall maximum is 98.8 s on the public grader. As
the second entry we designate k3w (333388, 3.8576e-9), whose maximum is 92.2 s, for that margin rather than for its score;
if no designation can be made, the rules' default pairs 333534 with our next-best public entry, k3y (333402, 3.7573e-9,
maximum 97.6 s).
