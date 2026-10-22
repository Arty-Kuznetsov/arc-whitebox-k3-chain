"""Extended FLOP-cost model of the k=3 chain (units: one dense n x n x n product; B = 1024 units).

Differences from experiments/2026-09-25_cost/cost_model.py:
  * per-layer alive fractions come from a MEASURED alpha = mu_pre/sigma histogram (dead_fractions.py),
    for any deadcut threshold, not only alpha < -3;
  * `window`, `paths`, `pathrank` are modelled;
  * shared-subspace bands are split into the part that rides the Strassen stacks (billed 0.636, the factor
    calibrated on k3i) and the ENTRY UPDATE, which runs in plain products (billed 1.0).  k3k measured
    27.78% where the old model predicted 21.43%; the entry term is calibrated on that one production number.

Calibration points (all production, flopscope, seed-11 / 100-MLP runs):
    k3i 29.82%   k3j (deadcut -3) 26.27%   k3k (k3j + band 6:320, q=1) 27.78%
"""
import json, os

N, L = 1024, 16
B_UNITS = 1024.0           # 2^41 / (2 n^3)
BILL_S = 0.636             # billed share of a plain product for everything that rides the Strassen stacks
BILL_P = 1.000             # billed share for the plain products of a subspace entry update

# alive fractions f[l] = 1 - dead(l) at threshold -3.0, MLP 0 (sparsity_probe.py / cost_model.py).
DEAD_M3 = [0, 0, .001, .008, .020, .036, .050, .063, .081, .072, .089, .125, .142, .167, .151, .180]

_TAB = None
def alive(thr):
    """Per-layer alive fraction at deadcut threshold `thr` (None = no cut).  Uses the measured alpha table
    written by dead_fractions.py when it exists, else falls back to the -3.0 column of cost_model.py."""
    global _TAB
    if thr is None:
        return [1.0] * L
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dead_fractions.json")
    if _TAB is None and os.path.exists(p):
        with open(p) as fh:
            _TAB = json.load(fh)
    if _TAB is not None:
        key = f"{thr:.2f}"
        if key in _TAB:
            return [1.0 - d for d in _TAB[key]]
        ks = sorted(float(k) for k in _TAB)          # linear interpolation in the threshold (the alpha CDF is
        if ks[0] <= thr <= ks[-1]:                    # smooth on this grid; flagged as interpolated in RESULTS)
            hi = min(k for k in ks if k >= thr)
            lo = max(k for k in ks if k <= thr)
            if hi == lo:
                return [1.0 - d for d in _TAB[f"{lo:.2f}"]]
            w = (thr - lo) / (hi - lo)
            a, b = _TAB[f"{lo:.2f}"], _TAB[f"{hi:.2f}"]
            return [1.0 - (a[i] * (1 - w) + b[i] * w) for i in range(L)]
    if abs(thr + 3.0) < 1e-9:
        return [1.0 - d for d in DEAD_M3]
    raise KeyError(f"no measured dead fractions for threshold {thr}; run dead_fractions.py")


def units(deadcut=-3.0, window=None, bands=(), paths=True, pathrank=None, entry=6.7, rot=True, L_=L):
    """Return (strassen_units, plain_units).

    bands: tuple of (a0, r) age bands for the shared subspace, oldest-first age thresholds ascending,
           e.g. ((6, 320),) or ((4, 512), (8, 320)).  A source of age a joins the band with the largest
           a0 <= a.  `entry` = plain products charged each time a source enters a band (calibrated on k3k).
    """
    f = alive(deadcut)
    q_path = 1.0 if pathrank is None else pathrank / N
    st = pl = 0.0
    for l in range(L_):
        S = l if window is None else min(l, window)
        last = l == L_ - 1
        fin = f[l - 1] if l > 0 else 1.0
        fo = f[l]
        ages = list(range(1, S + 1))
        young, old = [], []      # old = (age, r) in a band
        for a in ages:
            r_ = None
            for a0, r0 in bands:
                if a >= a0:
                    r_ = r0
            (old.append((a, r_)) if r_ is not None else young.append(a))
        if S > 0:
            st += sum(1 if a == 1 else 2 for a in young) * fin   # dense leg transports (newborn A = Wt, free)
            st += 1.0 + (q_path if paths else 0.0)               # fused Wt.C and the path leg (full rows)
            if not last:
                st += 2 * len(young) * fo * fo                   # D pairings, alive x alive
        st += 0.0 if last else 1.0                               # C_pre second product
        for a0, r0 in {(a0, r0) for a0, r0 in bands}:
            grp = [x for x in old if x[1] == r0]
            if not grp:
                continue
            q = r0 / N
            st += len(grp) * 2 * q                               # materialise A, B = P @ coef
            st += q * fin                                        # transport the band basis P
            if not last:
                st += len(grp) * 2 * q * fo + q * fo * fo        # compressed pairings + one Z P^T
            if any(a == a0 for a, _ in grp):
                pl += entry * (r0 / 320.0)                       # entry update: Gram applies, QR, eigh, new coefs
            if rot:
                pl += 3 * len(grp) * q * q                       # coefficient rotation r^2 n per leg (plain)
    return st, pl


def util(**kw):
    st, pl = units(**kw)
    return (BILL_S * st + BILL_P * pl) / B_UNITS


if __name__ == "__main__":
    ref = [("k3i", dict(deadcut=None), 0.2982),
           ("k3j = k3l", dict(), 0.2627),
           ("k3k (band 6:320)", dict(bands=((6, 320),)), 0.2778)]
    print("calibration against production flopscope runs")
    for name, kw, meas in ref:
        u = util(**kw)
        print(f"  {name:22s} model {100*u:6.2f}%   measured {100*meas:6.2f}%   ratio {u/meas:.3f}")
