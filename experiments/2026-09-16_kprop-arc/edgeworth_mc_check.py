"""Re-verify the first-order Edgeworth term for the ReLU mean, with a logged script.

The claim (METHOD.md §1): for a pre-activation z with mean mu, std sigma and standardized skew s,
    E[ReLU(z)] ~= mu*Phi(a) + sigma*phi(a) - s*sigma*a*phi(a)/6,        a = mu/sigma.

The notes quoted an earlier 40M-sample Monte Carlo check whose script was not kept. This replaces it.

Test family (exact, no sampling noise): z = mu + sigma*(G - k)/sqrt(k), G ~ Gamma(k, 1).
Standardized skew s = 2/sqrt(k), standardized 4th cumulant 6/k = 1.5 s^2. The ReLU mean has a
closed form through the regularized upper incomplete gamma Q:
    z > 0  <=>  G > t = k - a*sqrt(k)
    E[ReLU(z)] = (mu - sigma*sqrt(k)) * Q(k, t) + sigma*sqrt(k) * Q(k+1, t)
(using E[G 1{G>t}] = k Q(k+1, t)). A Monte-Carlo control with N = 4e7 samples checks that closed form
at a few points, so the reference itself is verified independently.

Run (from experiments/2026-09-16_kprop-arc/):
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src" uv run --project "$ARC" python edgeworth_mc_check.py | tee logs/edgeworth_mc_check.txt
"""
import math

import torch

torch.set_num_threads(3)
torch.set_default_dtype(torch.float64)


def Phi(a):
    return 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))


def phi(a):
    return math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)


def Q(k, t):
    if t <= 0.0:
        return 1.0
    return float(torch.special.gammaincc(torch.tensor(k), torch.tensor(t)))


def exact_relu_mean(mu, sigma, k):
    a = mu / sigma
    t = k - a * math.sqrt(k)
    return (mu - sigma * math.sqrt(k)) * Q(k, t) + sigma * math.sqrt(k) * Q(k + 1.0, t)


def mc_relu_mean(mu, sigma, k, n=40_000_000, chunk=5_000_000, seed=0):
    dist_k = torch.tensor(k)
    tot, tot2 = 0.0, 0.0
    for start in range(0, n, chunk):
        m = min(chunk, n - start)
        torch.manual_seed(seed * 1000 + start // chunk)  # _standard_gamma takes no generator
        G = torch._standard_gamma(dist_k.expand(m))
        z = mu + sigma * (G - k) / math.sqrt(k)
        r = torch.clamp(z, min=0.0)
        tot += float(r.sum()); tot2 += float((r * r).sum())
        del G, z, r
    mean = tot / n
    se = math.sqrt(max(tot2 / n - mean * mean, 0.0) / n)
    return mean, se


print("STEP 1  closed-form reference vs Monte Carlo (N = 4e7), sigma = 1")
print("     a     skew |      closed form           MC        MC s.e.   |diff|/s.e.")
for a, s in [(1.0, 0.3), (-1.0, 0.3), (0.5, 0.1), (2.0, 0.2)]:
    k = (2.0 / s) ** 2
    ex = exact_relu_mean(a, 1.0, k)
    mc, se = mc_relu_mean(a, 1.0, k)
    print(f" {a:5.2f}  {s:6.3f} | {ex:14.9f} {mc:14.9f} {se:10.2e}   {abs(ex - mc) / se:6.2f}")

print()
print("STEP 2  Gaussian closure vs first-order Edgeworth, against the exact ReLU mean (sigma = 1)")
print("  err_G = Gaussian closure - exact ;  err_E = Edgeworth - exact ;  removed = 1 - |err_E|/|err_G|")
print("     a    skew |        err_G        err_E   removed |  err_E / skew^2")
rows = []
for s in (0.05, 0.1, 0.2, 0.3):
    k = (2.0 / s) ** 2
    for a in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        mu, sigma = a, 1.0
        ex = exact_relu_mean(mu, sigma, k)
        gauss = mu * Phi(a) + sigma * phi(a)
        edge = gauss - s * sigma * a * phi(a) / 6.0
        eg, ee = gauss - ex, edge - ex
        removed = 1.0 - abs(ee) / abs(eg) if abs(eg) > 0 else float("nan")
        rows.append((a, s, eg, ee, removed))
        print(f" {a:5.2f}  {s:5.2f} | {eg:12.4e} {ee:12.4e}   {removed:6.3f} | {ee / (s * s):12.4e}")
    print()

nz = [r for r in rows if r[0] != 0.0]
print("SUMMARY")
print(f"  at a != 0, fraction of the Gaussian-closure error removed: "
      f"min {min(r[4] for r in nz):.3f}  median {sorted(r[4] for r in nz)[len(nz) // 2]:.3f}  "
      f"max {max(r[4] for r in nz):.3f}")
for s in (0.05, 0.1, 0.2, 0.3):
    sub = [r for r in nz if r[1] == s]
    print(f"  skew {s:4.2f}: removed min {min(r[4] for r in sub):.3f} max {max(r[4] for r in sub):.3f}")
z0 = [r for r in rows if r[0] == 0.0]
print("  at a = 0 the correction is identically zero; residual err_G there = "
      + ", ".join(f"{r[2]:.2e} (skew {r[1]})" for r in z0))
print("  err_E / skew^2 roughly constant in skew at fixed a  =>  the residual is second order, as expected")
