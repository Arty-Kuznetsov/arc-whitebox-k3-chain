"""Cost model of the k3 chain per MLP (units: one dense n x n x n product = 2n^3 FLOPs; B = 1024 units).
Calibrated on k3i (29.82% measured). Dead fractions: MLP 0, alpha < -3 (sparsity_probe.py).
Shared subspace for sources of age >= a0 with basis rank r: their legs are P @ coef (no per-source transport);
per layer they cost the materialisation of A and B (2 r/n) and compressed pairings (2 f r/n), plus band upkeep."""
import numpy as np
n, L = 1024, 16
dead = np.array([0, 0, .001, .008, .020, .036, .050, .063, .081, .072, .089, .125, .142, .167, .151, .180])
f = 1 - dead
def model(cut=False, a0=None, r=None, rot=True):
    tot = 0.0
    for l in range(L):
        S = l
        last = l == L - 1
        fin = f[l - 1] if (cut and l > 0) else 1.0
        fo = f[l] if cut else 1.0
        young = [a for a in range(1, S + 1) if a0 is None or a < a0]
        old = [a for a in range(1, S + 1) if a0 is not None and a >= a0]
        if S > 0:
            legs = sum(1 if a == 1 else 2 for a in young)       # newborn's A is a free copy of Wt
            tot += legs * fin + 2                                 # + fused C and the path leg (full)
            if not last:
                tot += 2 * len(young) * fo * fo                   # D pairings, alive x alive
        tot += 0 if last else 1                                   # C_pre second product
        if old:
            q = r / n
            tot += len(old) * 2 * q                               # materialise A, B = P coef
            tot += q * fin                                        # transport the basis P
            if not last:
                tot += len(old) * 2 * q * fo + q * fo * fo        # compressed pairings + one final P^T product
            if a0 in old:
                tot += 4 * q                                      # entering source: project its legs, basis update
            if rot:
                tot += 3 * len(old) * q * q                       # rotate band coefficients to the updated basis
    return tot
b = 0.2982 * 1024 / model()
print(f"billing factor calibrated on k3i: {b:.3f}")
for name, kw in (("k3i", {}), ("k3j (cut)", dict(cut=True))):
    print(f"{name:28s} {100 * b * model(**kw) / 1024:.2f}% of B")
for a0, r in ((5, 384), (5, 512), (4, 512), (4, 384), (6, 320), (6, 384)):
    print(f"k3j + shared {a0}:{r:<4d}           {100 * b * model(cut=True, a0=a0, r=r) / 1024:.2f}% of B")
