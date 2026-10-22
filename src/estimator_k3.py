"""Third-cumulant chain estimator for the ARC White-Box Estimation Challenge 2026 (Phase 2).

The algorithm is the SIMPLE k=3 cumulant propagation of Wu, Lecomte, Winer, Robinson, Hilton & Christiano
(arXiv:2605.05179; reference code github.com/alignment-research-center/mlp_cumulant_propagation, MIT), which we
re-derived as explicit formulas and re-implemented from scratch in a cheaper "source" representation
(experiments/2026-09-24_k3-chain/: step_check.py verifies every dense slice against the reference to 1e-15,
chain.py reproduces its per-layer means to 1e-16). One change of ours: the fourth cumulant is carried as its
(2,2)-slice matrix and diagonal, transported by the direct pairings (W∘W), instead of one radial scalar
(measured 1.13x better on the baked ground truth).

State after each ReLU layer: mean mu, covariance C, and one "source" per earlier ReLU layer m. A source is
born with legs A = I, B = diag(d1) C0 (C0: pre-activation covariance with zeroed diagonal), E = V (correction
that makes the repeated-index slices of the post-activation kappa3 exact) and hub vector g = d2; its kappa3 is
Sym(B (x) 3A (x) B diag(g)) + Sym(E (x) A (x) A). Every linear layer multiplies all legs by W^T; every ReLU
gates them by d1 (the leading pass-through term). Per source and layer: 3 transports + 3 products for the
(2,1) slice. All products go through a batched Strassen-Winograd whose Python-call count is O(levels).
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_F32 = fnp.float32


# ------------------------------------------------------------------------------------------------ batched Strassen
def _bs(A, B, lv):
    """A: (P, Sa, m, k), B: (P, S, k, p) with Sa in {1, S}; returns (P, S, m, p). Each recursion level runs once
    for the whole stack: 7 half-size operand pairs are stacked on axis 0, leaves are one batched matmul."""
    m, k = A.shape[-2], A.shape[-1]
    p = B.shape[-1]
    if lv <= 0 or m % 2 or k % 2 or p % 2 or min(m, k, p) < 64:
        return fnp.matmul(A, B)
    hm, hk, hp = m // 2, k // 2, p // 2
    a11, a12, a21, a22 = A[..., :hm, :hk], A[..., :hm, hk:], A[..., hm:, :hk], A[..., hm:, hk:]
    b11, b12, b21, b22 = B[..., :hk, :hp], B[..., :hk, hp:], B[..., hk:, :hp], B[..., hk:, hp:]
    # pure quadrants are copied first: concatenating non-contiguous views costs untracked (residual) time
    L = fnp.concatenate([a11 + a22, a21 + a22, fnp.copy(a11), fnp.copy(a22), a11 + a12, a21 - a11, a12 - a22], axis=0)
    R = fnp.concatenate([b11 + b22, fnp.copy(b11), b12 - b22, b21 - b11, fnp.copy(b22), b11 + b12, b21 + b22], axis=0)
    M = _bs(L, R, lv - 1)
    P = A.shape[0]
    m1, m2, m3, m4, m5, m6, m7 = (M[i * P:(i + 1) * P] for i in range(7))
    top = fnp.concatenate([m1 + m4 - m5 + m7, m3 + m5], axis=-1)
    bot = fnp.concatenate([m2 + m4, m1 - m2 + m3 + m6], axis=-1)
    return fnp.concatenate([top, bot], axis=-2)


def _bmm(A, B, lv):
    """Batched product: A (m,k) or (S,m,k), B (S,k,p) -> (S,m,p)."""
    if A.ndim == 2:
        A = A[None]
    return _bs(A[None], B[None], lv)[0]


# ------------------------------------------------------------------------------------------------ nonlinear step
def _wick(mu, var):
    """w[(k,p)] = E[d^k ReLU(Z)^p] for Z ~ N(mu, var), k = 0..4, p = 1..4; returns (w, d1, d2)."""
    s = fnp.sqrt(fnp.maximum(var, 1e-12))
    a = mu / s
    Ph = flops.stats.norm.cdf(a).astype(_F32)
    ph = flops.stats.norm.pdf(a).astype(_F32)
    sph = s * ph
    mm = mu * mu
    m1 = mu * Ph + sph
    m2 = (mm + var) * Ph + mu * sph
    m3 = (mm * mu + 3.0 * mu * var) * Ph + (mm + 2.0 * var) * sph
    m4 = (mm * mm + 6.0 * mm * var + 3.0 * var * var) * Ph + (mm * mu + 5.0 * mu * var) * sph
    inv_s = 1.0 / s
    d2 = ph * inv_s
    d3 = -a * d2 * inv_s
    d4 = (a * a - 1.0) * d2 * inv_s * inv_s
    d = {1: Ph, 2: d2, 3: d3, 4: d4}
    w = {(0, 1): m1, (0, 2): m2, (0, 3): m3, (0, 4): m4}
    for k in range(1, 5):
        w[(k, 1)] = d[k]
    mom = {1: m1, 2: m2, 3: m3}
    for p in (2, 3, 4):
        for k in range(1, 5):
            if k < p:
                w[(k, p)] = (math.factorial(p) / math.factorial(p - k)) * mom[p - k]
            else:
                w[(k, p)] = float(math.factorial(p)) * d[k - p + 1]
    return w, Ph, d2


def _nonlin(mu, var, C0, t, D, K4d, K22, need_pairs):
    """Post-activation slices from the pre-activation mean, variance, off-diagonal covariance C0, kappa3 slices
    t = k3[iii], D = k3[iij] (zero diagonal) and kappa4 slices K4d = k4[iiii], K22 = k4[iijj] (zero diagonal)."""
    w, d1, d2 = _wick(mu, var)
    P = {}
    for p in range(1, 5):
        v = w[(0, p)] + w[(3, p)] * (t / 6.0)
        if K4d is not None:
            v = v + w[(4, p)] * (K4d / 24.0)
        P[p] = v
    out = {"K1": P[1], "d1": d1, "d2": d2}
    if not need_pairs:
        return out
    C0sq = C0 * C0
    DT = D.T

    def Q(p, q):
        a1, a2 = w[(1, p)][:, None], w[(2, p)][:, None]
        b1, b2 = w[(1, q)][None, :], w[(2, q)][None, :]
        M = (a1 * b1) * C0 + (0.5 * a1 * b2) * DT + (0.5 * a2 * b1) * D + (0.5 * a2 * b2) * C0sq
        if K22 is not None:
            M = M + (0.25 * a2 * b2) * K22
        return M

    Q11, Q21, Q22 = Q(1, 1), Q(2, 1), Q(2, 2)
    P1, P2, P3, P4 = P[1], P[2], P[3], P[4]
    P1sq = P1 * P1
    out["K2"] = P2 - P1sq
    out["K3"] = P3 - 3.0 * P2 * P1 + 2.0 * P1sq * P1
    out["K4"] = P4 - 4.0 * P3 * P1 - 3.0 * P2 * P2 + 12.0 * P2 * P1sq - 6.0 * P1sq * P1sq
    K21 = Q21 - 2.0 * P1[:, None] * Q11
    K22o = (Q22 - 2.0 * P1[None, :] * K21 - 2.0 * P1[:, None] * K21.T - 2.0 * Q11 * Q11
            - 4.0 * (P1[:, None] * P1[None, :]) * Q11)
    zero = fnp.zeros(mu.shape[0], dtype=_F32)
    fnp.fill_diagonal(Q11, zero)
    fnp.fill_diagonal(K21, zero)
    fnp.fill_diagonal(K22o, zero)
    out.update(K11=Q11, K21=K21, K22=K22o)
    return out


class Estimator(BaseEstimator):
    def __init__(self, strassen_levels: int = 4, chunk: int = 12, k4_memo: bool = True) -> None:
        self._setup_rng = None
        self.lv = int(strassen_levels)
        self.chunk = int(chunk)
        self.k4_memo = bool(k4_memo)

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _transport(self, Wt, mats):
        """Wt @ X for every X in mats, batched in chunks."""
        out = []
        for c in range(0, len(mats), self.chunk):
            stack = fnp.stack(mats[c:c + self.chunk], axis=0)
            res = _bmm(Wt, stack, self.lv)
            out.extend(res[i] for i in range(res.shape[0]))
        return out

    def _sum_products(self, lefts, rights):
        """sum_i lefts[i] @ rights[i]^T, batched in chunks."""
        total = None
        for c in range(0, len(lefts), self.chunk):
            Ls = fnp.stack(lefts[c:c + self.chunk], axis=0)
            Rs = fnp.swapaxes(fnp.stack(rights[c:c + self.chunk], axis=0), 1, 2)
            part = fnp.sum(_bmm(Ls, Rs, self.lv), axis=0)
            total = part if total is None else total + part
        return total

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _rng = fnp.random.default_rng(mlp.seed)  # deterministic method; kept for the seeding contract
        _ = (_rng, budget)
        n = mlp.width
        L = len(mlp.weights)
        mu = fnp.zeros(n, dtype=_F32)
        C = flops.as_symmetric(fnp.eye(n, dtype=_F32), symmetry=(0, 1))
        sources = []  # dicts: A (None = identity), B, E, g
        K4post = K22post = None
        rows = []
        for l, w in enumerate(mlp.weights):
            last = l == L - 1
            Wt = w.T
            # ---- linear step
            mu_pre = Wt @ mu
            C_pre = fnp.einsum("ij,ia,jb->ab", C, w, w)
            var = fnp.maximum(fnp.diag(C_pre), 1e-12)
            C0 = C_pre - fnp.diag(fnp.diag(C_pre))
            if sources:
                mats, slots = [], []
                for si, s in enumerate(sources):
                    if s["A"] is not None:
                        mats.append(s["A"]); slots.append((si, "A"))
                    mats.append(s["B"]); slots.append((si, "B"))
                    mats.append(s["E"]); slots.append((si, "E"))
                moved = self._transport(Wt, mats)
                for (si, key), X in zip(slots, moved):
                    sources[si][key] = X
                for s in sources:
                    if s["A"] is None:
                        s["A"] = Wt
            # ---- kappa3 slices of the pre-activation
            t = fnp.zeros(n, dtype=_F32)
            D = None
            if sources:
                lefts, rights = [], []
                for s in sources:
                    A, B, E, g = s["A"], s["B"], s["E"], s["g"]
                    Ag = A * g[None, :]
                    AgB = Ag * B
                    t = t + 3.0 * fnp.sum(AgB * B, axis=1) + fnp.sum(E * A * A, axis=1)
                    if not last:
                        lefts += [2.0 * AgB, B * B * g[None, :] + (2.0 / 3.0) * (E * A), (1.0 / 3.0) * (A * A)]
                        rights += [B, A, E]
                if not last:
                    D = self._sum_products(lefts, rights)
                    fnp.fill_diagonal(D, fnp.zeros(n, dtype=_F32))
            if D is None and not last:
                D = fnp.zeros((n, n), dtype=_F32)
            # ---- kappa4 slices of the pre-activation (direct pairings of the tracked post slices)
            K4d = K22 = None
            if K4post is not None:
                Wsq = w * w
                X = K22post + fnp.diag(K4post)
                X = flops.as_symmetric(0.5 * (X + X.T), symmetry=(0, 1))
                M = fnp.einsum("ij,ia,jb->ab", X, Wsq, Wsq)
                K4d = 3.0 * fnp.diag(M) - 2.0 * ((Wsq * Wsq).T @ K4post)
                if not last:
                    K22 = M - fnp.diag(fnp.diag(M))
            # ---- nonlinear step
            o = _nonlin(mu_pre, var, C0, t, D, K4d, K22, need_pairs=not last)
            mu = o["K1"]
            rows.append(mu)
            if last:
                break
            C = o["K11"] + fnp.diag(o["K2"])
            C = flops.as_symmetric(0.5 * (C + C.T), symmetry=(0, 1))
            if self.k4_memo:
                K4post, K22post = o["K4"], o["K22"]
            d1, d2 = o["d1"], o["d2"]
            gate = d1[:, None]
            for s in sources:
                s["A"] = s["A"] * gate
                s["B"] = s["B"] * gate
                s["E"] = s["E"] * gate
            # ---- birth
            U = gate * C0
            d11 = d1 * d1
            dK3 = o["K3"] - d11 * d1 * t
            dK21 = o["K21"] - (d11[:, None] * d1[None, :]) * D - (d11[:, None] * d2[None, :]) * (C0 * C0)
            fnp.fill_diagonal(dK21, fnp.zeros(n, dtype=_F32))
            V = 3.0 * dK21.T + fnp.diag(dK3)
            sources.append({"A": None, "B": U, "E": V, "g": d2})
        return fnp.stack(rows, axis=0)
