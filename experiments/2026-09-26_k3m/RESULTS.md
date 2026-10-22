# k3m — k3l with the dead-row cut at −2.5, as a standalone production file

**Date:** 2026-09-26, morning, Mac. **Package:** `submission-20260926-061347.tar.gz`
(`src/estimator_k3m.py`, sha256 `543905c638d128fb903ebade47c5cf81513170716385a5342a730687f7f7a9a7`).

## Why a file and not the wrapper

`experiments/2026-09-26_cost-frontier/runs/v_k3l_cut25.py` already measured this arithmetic on 100 MLPs
(RESULTS §5 there: adjusted **5.5209e-9**, util 24.894%, final 2.2172e-8, 0 failures, residual 0.197/0.227 s,
99/100 MLPs better than k3l). It is a two-line subclass that imports `estimator_k3l`, which packages badly —
`whest package` ships one file as `estimator.py`. So k3m is a **standalone copy** of k3l whose only change is
the default `deadcut=-3.0 → -2.5` (plus the header). Nothing else was touched, and that is proven, not asserted.

## Self-checks (`k3m_check.py`, log in `k3m_check.log`)

Fresh 1024×16 He-init MLP (seed 11), `flopscope` budget 2**50 so nothing is clipped:

| claim | result |
|---|---|
| `k3m(deadcut=-3.0)` vs `k3l()` | **bitwise equal**, FLOPs equal, 26.622% of B |
| `k3m()` vs `k3l(deadcut=-2.5)` | **bitwise equal**, FLOPs equal, 25.169% of B |

The first proves the copy is faithful; the second proves the default is the only change, so the 100-MLP
measurement above transfers with no re-run. (The residual/wall columns in the log are informational — this
synthetic MLP runs under desktop load; the authoritative residual is the 100-MLP local-runner number.)

## Package check

- `whest validate` passed (72 ms).
- The `estimator.py` inside the tarball is **byte-identical** to `src/estimator_k3m.py` (same sha256).
- `--runner subprocess`, 3 MLPs (`k3m_pkg_subproc_n3.json`): adjusted **5.7152e-9**, final 2.2924e-8,
  util 24.935%, **0 failures**, residual 0.290 / 0.357 / 0.380 s, wall 20.6 s/MLP. MLPs 0–2 are harder than the
  100-MLP mean, and the subprocess runner reads ≈2× the grader's residual (k3l: subprocess 0.30–0.33 → grader
  0.148–0.162), so this is consistent and safe.

## Expected on the grader

`5.5209e-9 × 0.9756 = ` **≈5.386e-9** — the ratio measured on 332253 (k3l: local 5.7387e-9 → graded 5.5984e-9).
Residual expected ≈0.148 s (k3l 0.156 s at a slightly larger alive set), predict wall ≈59 s of the 120 s cap.
That would be ≈ rank 60, inside the 504aldo cluster's neighbourhood and one step below it.
