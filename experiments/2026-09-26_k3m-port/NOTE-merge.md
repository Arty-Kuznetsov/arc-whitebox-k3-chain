# Why only this folder was taken from `claude/friendly-mendel-a6a3yf`

This cloud session (2026-09-26/27) was started with the old brief "build k3m = k3l + deadcut −2.5 + pathrank 256".
In parallel the Mac built and shipped its own `src/estimator_k3m.py` = k3l + `deadcut −2.5` only (submitted as
332403, graded **5.3906e-9**, the current best). The cloud's `src/estimator_k3m.py` has an extra `pathrank` kwarg
(default `None`) and is otherwise the same arithmetic — its check (a) is bitwise k3l at `deadcut=-3.0`, and its
100-MLP run reproduces the Mac's number to 1.1e-4 with **identical** utilisation (24.894%).

**Merged into `main` on 2026-09-29:** this experiment folder only, as evidence (it is the real-meter measurement behind
NEGATIVE §55). **Not merged:** the branch's `src/estimator_k3m.py` — same file name as the shipped estimator, and the
`pathrank` machinery it adds does not pay (see RESULTS.md here). If `pathrank` is ever needed, take it from the branch.
