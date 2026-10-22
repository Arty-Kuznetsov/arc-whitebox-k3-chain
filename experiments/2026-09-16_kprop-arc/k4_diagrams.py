"""Validate the Wick diagrams for the 4th cumulant against Monte Carlo at small width.

The cumulant rule that produced the MC-verified Edgeworth mean term extends to 4th order:
    Delta mu = (1/24) * E[d^4 ReLU] * kappa4[i,i,i,i],   E[d^4 ReLU] = (alpha^2 - 1) phi / sigma^3
so the missing ingredient is kappa4 of the pre-activation. Its leading connected diagrams on four
vertices are trees with three edges. Two topologies, contracted with the next layer's weight column
u_i (all four legs land on the same output neuron i), with v = C diag(d1) u and p = u * d2 * v:

  STAR  (one hub of degree 3, three leaves)   ->  S_i = sum_a d3_a * u_i[a] * v_i[a]^3
  PATH  (degrees 1,2,2,1)                     ->  P_i = p_i^T C p_i

Rather than trust hand-derived combinatorial prefactors, this fits them against Monte Carlo at
width 64, depth 2, where 2e7 samples resolve kappa4 well. Clean near-integer coefficients confirm the
derivation; messy ones mean a topology is missing. The same run re-checks the kappa3 formula, whose
prefactor is known to be 3 -- that is the control.

Run:  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
      env PYTHONPATH="$ARC/src" uv run --project "$ARC" python k4_diagrams.py
"""
import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: torch.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: torch.special.ndtr(a)

n = 64
N_TOTAL = 20_000_000
CHUNK = 250_000
rng = torch.Generator().manual_seed(0)

print(f"width {n}, depth 2, {N_TOTAL:,} Monte-Carlo samples\n")
rows = []
for trial in range(3):
    W0 = torch.randn(n, n, generator=rng) * np.sqrt(2.0 / n)
    W1 = torch.randn(n, n, generator=rng) * np.sqrt(2.0 / n)

    # ---- Monte Carlo cumulants of the layer-1 pre-activation ----
    s1 = torch.zeros(n); s2 = torch.zeros(n); s3 = torch.zeros(n); s4 = torch.zeros(n)
    done = 0
    while done < N_TOTAL:
        m = min(CHUNK, N_TOTAL - done)
        x = torch.randn(m, n, generator=rng)
        z1 = torch.relu(x @ W0) @ W1
        s1 += z1.sum(0); zz = z1 * z1
        s2 += zz.sum(0); s3 += (zz * z1).sum(0); s4 += (zz * zz).sum(0)
        done += m
    m1 = s1 / N_TOTAL; m2 = s2 / N_TOTAL; m3 = s3 / N_TOTAL; m4 = s4 / N_TOTAL
    c2 = m2 - m1 * m1
    c3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    c4 = m4 - 4 * m1 * m3 - 3 * c2 * c2 + 12 * m1 * m1 * m2 - 6 * m1 ** 4 - 3 * m1 ** 4 + 3 * m1 ** 4
    # kappa4 = mu4c - 3*mu2c^2 with central moments
    mu4c = m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4
    c4 = mu4c - 3 * c2 * c2

    # ---- analytic layer-0 state and the two diagram templates ----
    C0 = W0.T @ W0
    var0 = C0.diagonal().clamp_min(1e-12)
    sig0 = var0.sqrt()
    al0 = torch.zeros(n)                       # layer-0 pre-activations are exactly zero-mean
    d1 = nPhi(al0); d2 = nphi(al0) / sig0; d3 = -al0 * nphi(al0) / var0
    A = W1                                     # A[a, i] = u_i[a]
    V = C0 @ (W1 * d1[:, None])                # V[b, i] = sum_a C0[b,a] d1_a W1[a,i]
    k3_tpl = ((W1 * d2[:, None]) * V * V).sum(dim=0)          # prefactor 3 expected
    star = ((W1 * d3[:, None]) * V * V * V).sum(dim=0)
    P = W1 * d2[:, None] * V
    path = (P * (C0 @ P)).sum(dim=0)

    rows.append((c3, c4, k3_tpl, star, path, sig0))
    print(f"  trial {trial}: mean|kappa3|={c3.abs().mean():.3e}  mean|kappa4|={c4.abs().mean():.3e}  "
          f"|star|={star.abs().mean():.3e}  |path|={path.abs().mean():.3e}", flush=True)

c3 = torch.cat([r[0] for r in rows]); c4 = torch.cat([r[1] for r in rows])
k3t = torch.cat([r[2] for r in rows]); st = torch.cat([r[3] for r in rows])
pt = torch.cat([r[4] for r in rows])

print("\n--- control: kappa3 prefactor (derivation says 3) ---")
b3 = float((k3t @ c3) / (k3t @ k3t))
r3 = float(torch.corrcoef(torch.stack([k3t, c3]))[0, 1])
print(f"  fitted prefactor = {b3:.4f}   corr = {r3:.5f}   "
      f"residual/|kappa3| = {float(((c3 - 3*k3t)**2).mean().sqrt() / c3.abs().mean()):.4f}")

print("\n--- kappa4: least squares on [star, path] ---")
X = torch.stack([st, pt], dim=1)
coef = torch.linalg.lstsq(X, c4.unsqueeze(1)).solution.squeeze(1)
pred = X @ coef
print(f"  star coefficient = {float(coef[0]):.4f}   path coefficient = {float(coef[1]):.4f}")
print(f"  corr(prediction, kappa4) = {float(torch.corrcoef(torch.stack([pred, c4]))[0,1]):.5f}")
print(f"  residual RMS / |kappa4| RMS = "
      f"{float(((c4-pred)**2).mean().sqrt() / (c4**2).mean().sqrt()):.4f}")
for name, col in (("star only", st), ("path only", pt)):
    b = float((col @ c4) / (col @ col))
    r = float(torch.corrcoef(torch.stack([col, c4]))[0, 1])
    print(f"  {name:10s}: best coefficient {b:9.4f}   corr {r:+.5f}")
print("\nNote: at layer 0 alpha = 0 exactly, so d3 = -alpha*phi/sigma^2 vanishes and the STAR term is")
print("identically zero there. If star is ~0, only the path coefficient is identified by this test.")
