"""FRONTIER A1, quantitative test of the Gaussian scale-mixture explanation of the post-ReLU kappa4 (2,2) deficit.
Model for a pair (i, j) of pre-activations with the C0 coupling switched off: z_i = mu_i + sqrt(s) sigma_i xi_i, xi iid
N(0,1), one common scalar s with E s = 1, Var s = v, kappa3(s) = k3s (a shifted gamma matching the three). Then the
post (2,2) joint cumulant is EXACTLY
    kappa(y_i, y_i, y_j, y_j) = Cov_s(A_i, A_j) - 2 Cov_s(m_i, m_j)^2,
    m_i(s) = E[y_i | s],  A_i(s) = E[(y_i - E y_i)^2 | s],
while the SIMPLE step keeps the first order in v: A_i'(1) A_j'(1) v = (Phi_i - P1_i d2_i)(Phi_j - P1_j d2_j) K22_ij.
If the pair-mean ratio exact/first-order is ~1/0.977, 1/0.965, 1/0.955 at layers 6/10/14 (the measured one-step
deficits with MC inputs, logs/k31_step_0.txt), the missing term is the higher-order scale statistics (radial kappa6...).
v and k3s are taken from the MC oracle two ways: (a) the common-scale statistic q (minus its Gaussian part), (b) the
pair-mean of K22_ij / (var_i var_j) with k3s from (a) scaled."""
import math
from pathlib import Path
import numpy as np
from scipy import stats, special
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz")
o31 = np.load(HERE / "runs" / "mc_oracle31_0.npz")

def relu_moments(mu, var):
    """E[ReLU(Z)], E[ReLU(Z)^2] for Z ~ N(mu, var) (arrays broadcast)."""
    s = np.sqrt(var); a = mu / s
    Ph, ph = special.ndtr(a), np.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)
    return mu * Ph + s * ph, (mu * mu + var) * Ph + mu * s * ph

def scale_nodes(v, k3s, m=4000):
    """Quadrature of a shifted gamma s = s0 + Gamma(k, theta) with mean 1, variance v, kappa3 k3s (equal-mass nodes)."""
    theta = k3s / (2 * v); k = v / theta ** 2; s0 = 1 - k * theta
    u = (np.arange(m) + 0.5) / m
    s = s0 + stats.gamma.ppf(u, k, scale=theta)
    return s, np.full(m, 1.0 / m), s0

rng = np.random.default_rng(0)
print("layer | v(q) k3s(q) skew | v(K22) | s0 | pair-mean ratio exact/first-order (model a, model b) | measured 1/deficit")
meas = {6: 0.9761, 10: 0.9627, 14: 0.9557}
for l in (6, 10, 14):
    mu, var = mc["mean"][l].astype(np.float64), mc["var"][l].astype(np.float64)
    C = o31[f"C{l}"].astype(np.float64); n = C.shape[0]
    rho = C / np.sqrt(np.outer(np.diag(C), np.diag(C)))
    qv, qk3 = o31[f"q{l}"][1], o31[f"q{l}"][2]
    gv = 2.0 * (rho ** 2).sum() / n ** 2                     # Gaussian part of Var(q)
    gk3 = 8.0 * np.trace(rho @ rho @ rho) / n ** 3            # Gaussian part of kappa3(q)
    v_a, k3_a = qv - gv, qk3 - gk3
    K22 = o31[f"k22_{l}"].astype(np.float64)
    off = ~np.eye(n, dtype=bool)
    v_b = float((K22 / np.outer(var, var))[off].mean())
    res = []
    for v, k3s in ((v_a, k3_a), (v_b, k3_a * (v_b / v_a) ** 2)):
        s, wq, s0 = scale_nodes(v, k3s)
        m1, m2 = relu_moments(mu[None, :], s[:, None] * var[None, :])        # (nodes, n)
        Ey, Ey2 = wq @ m1, wq @ m2
        A = m2 - 2 * Ey[None, :] * m1 + Ey[None, :] ** 2                      # E[(y - E y)^2 | s]
        dA, dm = A - wq @ A, m1 - Ey[None, :]
        # first-order-in-v slope of A at s = 1 (numerical derivative), the SIMPLE pass-through coefficient
        h = 1e-4
        a1p, a2p = relu_moments(mu, (1 + h) * var); a1m, a2m = relu_moments(mu, (1 - h) * var)
        A1 = ((a2p - 2 * Ey * a1p) - (a2m - 2 * Ey * a1m)) / (2 * h)
        idx = rng.choice(n, size=(20000, 2))
        idx = idx[idx[:, 0] != idx[:, 1]]
        i, j = idx[:, 0], idx[:, 1]
        exact = (wq @ (dA[:, i] * dA[:, j])) - 2 * (wq @ (dm[:, i] * dm[:, j])) ** 2
        first = A1[i] * A1[j] * v
        res.append((exact.mean() / first.mean(), s0))
    print(f"{l:5d} | {v_a:.4f} {k3_a:.4f} {k3_a / v_a ** 1.5:.2f} | {v_b:.4f} | {res[0][1]:.3f} | "
          f"{res[0][0]:.4f}, {res[1][0]:.4f} | {1 / meas[l]:.4f}")
