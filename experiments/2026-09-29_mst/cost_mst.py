"""Cost of the Slice-Markov tail in cost.py units (1 unit = one dense n^3 product, B = 1024 units).

Young sources (age < a0) keep their legs: transports + pairings as today.  The whole tail costs a FIXED
~3 dense products per layer regardless of how many sources it holds -- that is the entire point.
Self-check: a0 -> infinity must reproduce cost.util (no tail, everything young).
"""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26_cost-frontier"))
import cost

N, L, BU, BS, BP = cost.N, cost.L, cost.B_UNITS, cost.BILL_S, cost.BILL_P


def util(a0=3, tail_units=3.0, deadcut=None, paths=True):
    f = cost.alive(deadcut)
    st = pl = 0.0
    for l in range(L):
        S, last = l, l == L - 1
        fin = f[l - 1] if l > 0 else 1.0
        fo = f[l]
        young = [a for a in range(1, S + 1) if a0 is None or a < a0]
        has_tail = a0 is not None and S >= a0
        if S > 0:
            st += sum(1 if a == 1 else 2 for a in young) * fin
            st += 1.0 + (1.0 if paths else 0.0)
            if not last:
                st += 2 * len(young) * fo * fo
        if has_tail:
            pl += tail_units
        st += 0.0 if last else 1.0
    return (BS * st + BP * pl) / BU


if __name__ == "__main__":
    base = cost.util(deadcut=None)
    print(f"self-check a0=None: {100*util(a0=None):.3f}%  vs cost.util {100*base:.3f}%")
    print(f"\n{'a0':>4s} {'util':>8s}   raw x   -> adjusted (raw_base 2.1652e-8 graded)")
    RAW = {2: 18.7498, 3: 12.3784, 4: 8.1402, 5: 5.3279}
    for a0 in (2, 3, 4, 5, 6, 8, 10, 12):
        u = util(a0=a0)
        x = RAW.get(a0)
        adj = f"{2.1652e-8 * x * max(0.1, u):.3e}" if x else "raw pending"
        print(f"{a0:4d} {100*u:7.2f}%  {('x%.2f' % x) if x else '   ?':>8s}   {adj}")
    print(f"\nk3m today: util 24.835%, adjusted 5.3906e-9. Floor (util<=10%) needs raw x<=1 to beat it.")
