"""Our working estimator for the ARC WhiteBox Estimation Challenge 2026 (Phase 2).

This is the file we iterate on and submit. It STARTS as a faithful copy of the
starter kit's full covariance-propagation baseline (examples/03), which is the
only bundled example that beats sampling:

    final_layer_mse = 4.05e-6   (adjusted ≈ 4.05e-7)   at 2.351% of the FLOP budget

That 4.05e-6 is the number to beat — see ../notes/PLAN.md for the attack vectors.
Keep the float32 seeding and the symmetric re-tag (../notes/SPEC.md) intact: they
are accuracy-neutral FLOP savings, and at ≤10% of budget accuracy is the only lever.

Local fast loop (Monte-Carlo convergence table, NOT the baked ground truth):
    uv run --project ../external/whest-starterkit python src/estimator.py
Authoritative scoring is `whest run --dataset ... --split mini`; see ../README.md.
"""

from __future__ import annotations

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

# If any diagonal entry of the covariance exceeds this, rescale to keep the
# arithmetic inside float32 range. At 1024x16 He weights this guard never fires
# (max diagonal stays in [0.26, 1.37]); carried for deeper/differently-scaled nets.
_COV_RESCALE_THRESHOLD = 1e30


class Estimator(BaseEstimator):
    """Full covariance propagation through ReLU layers.

    Linear layer (exact):   mu_pre = W.T mu ;  cov_pre = W.T cov W
    ReLU (approximate gain): cov_post[i,j] ≈ Phi(a_i) Phi(a_j) cov_pre[i,j],
                             diagonal replaced by the exact rectified-Gaussian variance.

    TODO(improve): the gain approximation on the off-diagonal is the first thing to
    replace with the exact bivariate rectified-Gaussian second moment (PLAN.md, vector A).
    """

    def __init__(
        self,
        n_corr_terms: int = 1,
        rho_clip: float = 0.999,
        edgeworth: bool = True,
        skew_clip: float = 1.0,
    ) -> None:
        self._setup_rng = None  # set from ctx.seed inside setup()
        # Tetrachoric/Hermite correction to the ReLU covariance (see notes/PLAN.md).
        # n_corr_terms=0 reproduces the gain-method covariance baseline exactly.
        # Default 1: the 2026-09-13 width sweep shows the n=1 term captures the whole
        # correction (K=2,3 add <1e-8); it helps once width>=512 (the CLT regime) and is
        # ~free. Confirm the actual 1024x16 gain on baked GT before finalizing a submission.
        self.n_corr_terms = int(n_corr_terms)
        self.rho_clip = float(rho_clip)
        # Edgeworth mean correction (see the block at Step 5b and notes/RESULTS-edgeworth.md).
        # Measured on the baked 1e9 GT: 1.67x lower final_layer_mse for +1.46% of budget, which
        # stays under the 10% multiplier floor, so it is free on the metric. No fitted constants.
        self.edgeworth = bool(edgeworth)
        # Validity guard only. The first-order Edgeworth term is a small-skew expansion; this clips
        # the standardized skew used by the correction. Measured |skew| at 1024x16 is <= 0.1, so the
        # clip never binds at the competition shape; it exists so degenerate tiny-width probes (the
        # validator uses width=4, depth=2) cannot produce a blow-up. It is NOT a tuned parameter.
        self.skew_clip = float(skew_clip)

    def setup(self, ctx: SetupContext) -> None:
        # Submission-level RNG (unused by this deterministic method; carried for the contract).
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _rng = fnp.random.default_rng(mlp.seed)  # per-MLP RNG (unused here; kept for contract)
        _ = _rng
        _ = budget
        width = mlp.width

        # --- Step 1: input distribution = standard multivariate normal (mu=0, cov=I). ---
        # float32 on purpose: float64 bills 2x and would carry through the O(width^3) einsum.
        mu = fnp.zeros(width, dtype=fnp.float32)
        cov = flops.as_symmetric(fnp.eye(width, dtype=fnp.float32), symmetry=(0, 1))
        log_scale = 0.0

        rows = []
        # Previous layer's (pre-activation covariance, d1, d2) for the Edgeworth term. None on the
        # first layer, where the pre-activations are exactly Gaussian so the correction is zero.
        prev = None
        for w in mlp.weights:  # w: (width, width), indexed [input, output]
            # --- Step 2: overflow guard (rescale if covariance grows too large). ---
            cov_diag = fnp.diag(cov)
            max_var_np = float(fnp.max(cov_diag))
            if max_var_np > _COV_RESCALE_THRESHOLD:
                s = float(fnp.sqrt(max_var_np))
                mu = mu / s
                cov = cov / (s * s)
                log_scale += float(fnp.log(s))
                prev = None  # stored state is in the old scale; drop it rather than mix units

            # --- Step 3: linear layer. Single einsum keeps the symmetric (discounted) rate. ---
            mu_pre = w.T @ mu
            cov_pre = fnp.einsum("ij,ia,jb->ab", cov, w, w)

            var_pre = fnp.maximum(fnp.diag(cov_pre), 1e-12)
            sigma_pre = fnp.sqrt(var_pre)

            # --- Step 4: alpha = mu/sigma; PDF/CDF (cast back to float32 after the scipy-matching promote). ---
            alpha = mu_pre / sigma_pre
            phi_alpha = flops.stats.norm.pdf(alpha).astype(fnp.float32)
            Phi_alpha = flops.stats.norm.cdf(alpha).astype(fnp.float32)

            # --- Step 5: post-ReLU mean under the Gaussian closure (exact per neuron). ---
            mu_gauss = mu_pre * Phi_alpha + sigma_pre * phi_alpha

            # --- Step 5b: Edgeworth correction to the mean (the one real change vs k=2) ---
            # The Gaussian-closure mean is exact only for Gaussian pre-activations. Its leading
            # error in the pre-activation's own third cumulant is, with a = mu/sigma and
            # s = kappa3_iii / sigma**3 the standardized skew,
            #     mu_post = mu_gauss - s * sigma * a * phi(a) / 6.
            # kappa3_iii is a VECTOR, and it does not have to be transported: to leading order the
            # previous ReLU generates it from that layer's own mean and covariance (Wick/cluster
            # expansion, derivative coefficients d1 = Phi(a), d2 = phi(a)/sigma, checked against the
            # ARC reference wick.py). Contracting the leading connected diagram with this layer's
            # weights collapses to one matmul:
            #     Vt = cov_prev @ (w * d1_prev),      kappa3_iii = 3 * sum_c (w*d2_prev) * Vt**2
            # Cost: n*n*(2n-1) per layer. Derivation + measurements:
            # experiments/2026-09-16_kprop-arc/RESULTS-edgeworth.md.
            if self.edgeworth and prev is not None:
                cov_prev, d1_prev, d2_prev = prev
                Vt = cov_prev @ (w * d1_prev[:, None])          # (input c, output i)
                kappa3 = 3.0 * fnp.sum((w * d2_prev[:, None]) * Vt * Vt, axis=0)
                skew = kappa3 / (var_pre * sigma_pre)
                skew = fnp.maximum(fnp.minimum(skew, self.skew_clip), -self.skew_clip)
                mu = mu_gauss - skew * sigma_pre * alpha * phi_alpha / 6.0
            else:
                mu = mu_gauss

            # --- Step 6: post-ReLU diagonal variance (exact per neuron). ---
            # Deliberately built from mu_gauss, not the corrected mean: the correction targets the
            # mean only. Measured (RESULTS-layer-blame.md): the covariance carries none of the k=2
            # final-layer error, and perturbing it alone makes the iterate worse.
            ez2 = (mu_pre * mu_pre + var_pre) * Phi_alpha + mu_pre * sigma_pre * phi_alpha
            var_post = fnp.maximum(ez2 - mu_gauss * mu_gauss, 0.0)

            # --- Step 7: post-ReLU off-diagonal covariance (tetrachoric / Hermite series) ---
            # Price's theorem:  Cov(ReLU_i,ReLU_j) = sigma_i sigma_j * int_0^rho Phi2(a_i,a_j;r) dr.
            # Expanding Phi2 in its tetrachoric (Hermite) series and integrating term by term:
            #   cov_off[i,j] = cov_pre[i,j] * [ Phi(a_i)Phi(a_j)
            #                                   + sum_{n>=1} c_n * q_n(i) q_n(j) * rho_ij**n ]
            # with c_n = 1/((n+1) n!), q_n(i) = phi(a_i) He_{n-1}(a_i), He = probabilists' Hermite.
            # The n=0 term is exactly the gain method; higher n correct it. All terms are
            # O(width^2), symmetric, and bounded (|rho|<=rho_clip). Derivation: notes/PLAN.md.
            _zero32 = fnp.zeros((), dtype=fnp.float32)
            valid = sigma_pre > 1e-12
            gain = fnp.where(valid, Phi_alpha, _zero32)
            factor = fnp.outer(gain, gain)  # n=0 (gain) term

            if self.n_corr_terms > 0:
                inv_sigma = fnp.where(valid, 1.0 / sigma_pre, _zero32)
                R = cov_pre * fnp.outer(inv_sigma, inv_sigma)  # pre-activation correlation matrix
                R = fnp.maximum(fnp.minimum(R, self.rho_clip), -self.rho_clip)
                phi_masked = fnp.where(valid, phi_alpha, _zero32)
                r_pow = R  # R**n, starts at n=1
                he_prev = fnp.zeros(width, dtype=fnp.float32)  # He_{-1} (unused at n=1)
                he_curr = fnp.ones(width, dtype=fnp.float32)  # He_0 = 1
                coeff = 0.5  # c_1 = 1/((1+1)*1!)
                for n in range(1, self.n_corr_terms + 1):
                    q = phi_masked * he_curr  # q_n = phi(alpha) * He_{n-1}(alpha)
                    factor = factor + coeff * fnp.multiply(fnp.outer(q, q), r_pow)
                    # advance Hermite (He_n = alpha*He_{n-1} - (n-1)*He_{n-2}), power, coefficient
                    he_prev, he_curr = he_curr, alpha * he_curr - float(n - 1) * he_prev
                    r_pow = r_pow * R  # R**(n+1)
                    coeff = coeff / float(n + 2)  # c_{n+1} = c_n / (n+2)

            cov = fnp.multiply(factor, cov_pre)
            fnp.fill_diagonal(cov, var_post)

            # --- Step 7b: re-validate + re-tag symmetry (the write above voided the tag). ---
            cov = flops.as_symmetric(cov, symmetry=(0, 1))

            # --- Step 7c: carry this layer's state for the next layer's Edgeworth term. ---
            if self.edgeworth:
                prev = (cov_pre, Phi_alpha, phi_alpha / sigma_pre)

            # --- Step 8: record mean in original (unscaled) coordinates. ---
            scale_factor = float(fnp.exp(log_scale))
            rows.append(mu * scale_factor)

        return fnp.stack(rows, axis=0)


if __name__ == "__main__":
    # Local Stage-1 loop: compares against a FRESH Monte-Carlo estimate (not the baked
    # 1e9 ground truth). Run via the starter kit's env so flopscope/whestbench resolve:
    #   uv run --project ../external/whest-starterkit python src/estimator.py
    import sys
    from pathlib import Path

    starterkit = Path(__file__).resolve().parent.parent / "external" / "whest-starterkit"
    sys.path.insert(0, str(starterkit))
    from local_engine import build_mlp, compare_against_monte_carlo  # type: ignore

    mlp = build_mlp(width=1024, depth=16, seed=0)  # competition shape
    compare_against_monte_carlo(Estimator(), mlp)
