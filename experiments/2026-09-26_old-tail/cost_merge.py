"""What a merged old tail would be WORTH, in experiments/2026-09-26_cost-frontier/cost.py units.

Prices the hypothetical: sources of age >= a0 are carried as ONE block of r columns (q = r/n), so the tail
costs O(1) products per layer instead of 2 per source, plus `m` plain units per layer for the merge itself.
Imports cost.py for the billing constants and the measured alive fractions; edits nothing there.

    ../../.venv-whest/bin/python cost_merge.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-26_cost-frontier"))
import cost

N, L, BU = cost.N, cost.L, cost.B_UNITS
BS, BP = cost.BILL_S, cost.BILL_P


def units(a0=4, q=1.0, m=0.0, deadcut=-2.5, paths=True, thin=True):
    """(strassen, plain) units for the chain with sources of age >= a0 merged into one q*n-column block.
    a0=None reproduces cost.units(deadcut=deadcut) exactly (the self-check below)."""
    f = cost.alive(deadcut)
    st = pl = 0.0
    for l in range(L):
        S, last = l, l == L - 1
        fin = f[l - 1] if l > 0 else 1.0
        fo = f[l]
        ages = list(range(1, S + 1))
        young = [a for a in ages if a0 is None or a < a0]
        old = [a for a in ages if a0 is not None and a >= a0]
        if S > 0:
            st += sum(1 if a == 1 else 2 for a in young) * fin
            st += 1.0 + (1.0 if paths else 0.0)
            if not last:
                st += 2 * len(young) * fo * fo
        if old:
            st += 2 * q * fin                       # transport the merged block's two dense legs
            if not last:
                st += 2 * q * fo * fo               # its D pairings
            if thin:
                st += 0.0                           # its thin correction leg is O(n * kk), negligible
            pl += m                                 # the merge / graduation update, plain products
        st += 0.0 if last else 1.0                  # C_pre second product
    return st, pl


def util(**kw):
    st, pl = units(**kw)
    return (BS * st + BP * pl) / BU


if __name__ == "__main__":
    base = cost.util(deadcut=-2.5)
    mine = util(a0=None, deadcut=-2.5)
    print(f"self-check: a0=None reproduces cost.py  {100*mine:.3f}%  vs cost.util {100*base:.3f}%"
          f"   (diff {abs(mine-base)*100:.4f} points)")
    print(f"k3m measured on the grader: 24.835% of B;  cost.py model: {100*base:.3f}%\n")
    print("util of a merged tail, by a0 and merge cost m (q = r/n = 1.0, i.e. r = n):")
    print(f"  {'a0':>3s} " + "".join(f"{'m='+str(m):>9s}" for m in (0, 1, 2, 3, 4, 6, 8)))
    for a0 in (3, 4, 5, 6, 8):
        row = "".join(f"{100*util(a0=a0, q=1.0, m=m, deadcut=-2.5):8.2f}%" for m in (0, 1, 2, 3, 4, 6, 8))
        print(f"  {a0:3d} {row}")
    print("\nsame at r = n/2 (q = 0.5):")
    print(f"  {'a0':>3s} " + "".join(f"{'m='+str(m):>9s}" for m in (0, 1, 2, 3, 4, 6, 8)))
    for a0 in (4, 6):
        row = "".join(f"{100*util(a0=a0, q=0.5, m=m, deadcut=-2.5):8.2f}%" for m in (0, 1, 2, 3, 4, 6, 8))
        print(f"  {a0:3d} {row}")
    print("\nTargets: top-5 needs adjusted <= 2.30e-9.  At the graded raw of k3m (2.1652e-8) that is")
    print("util <= 10.6%; the multiplier floors at 10%, so util <= 10% buys nothing more.")
    for tag, u in (("k3m today", base), ("merge a0=4, m=2", util(a0=4, m=2, deadcut=-2.5)),
                   ("merge a0=6, m=2", util(a0=6, m=2, deadcut=-2.5))):
        for raw, rn in ((2.1652e-8, "k3m raw"), (2.4e-8, "raw +10.8%")):
            print(f"  {tag:18s} util {100*u:6.2f}%  x {rn:11s} -> adjusted {raw*max(0.1,u):.3e}")
