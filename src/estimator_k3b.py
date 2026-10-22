"""Third-cumulant chain estimator, buffered layout (ARC White-Box Estimation Challenge 2026, Phase 2).

Same algorithm as estimator_k3.py (the SIMPLE k=3 cumulant chain of Wu et al., arXiv:2605.05179, re-derived and
re-implemented by us in the "source" representation; memoryless (2,2)-slice kappa4 transport), laid out so the
number of flopscope calls per layer does not grow with the number of sources: all source legs live in one
preallocated buffer, ordered [B_s, E_s, A_s] per source s, and every per-source operation is one batched call
writing into preallocated workspaces (out=). This keeps untracked (residual) time low: the grader zeroes an MLP
whose non-flopscope time exceeds 400 ms.
"""

from __future__ import annotations

import math

import flopscope as flops
import flopscope.numpy as fnp
from whestbench import BaseEstimator, SetupContext
from whestbench.domain import MLP

_F32 = fnp.float32


class _Strassen:
    """Batched Strassen-Winograd with preallocated per-level workspaces: out = A @ B for stacks
    A (G, Ka, m, m), B (G, K, m, m), out (G, K, m, m), Ka in {1, K}. Every level issues a fixed number of
    flopscope calls (7 + 7 combinations written with out=, one recursion, 10 recombination updates written
    into quadrant views of out), so neither allocations nor Python-call counts grow with the stack."""

    def __init__(self, n, levels, chunk):
        self.levels, self.chunk = levels, chunk
        self.bufs = []
        m, G = n, 1
        for _ in range(levels):
            if m % 2 or m < 128:
                break
            m //= 2
            G *= 7
            self.bufs.append((fnp.empty((G, chunk, m, m), dtype=_F32),
                              fnp.empty((G, chunk, m, m), dtype=_F32),
                              fnp.empty((G, chunk, m, m), dtype=_F32)))

    def _rec(self, A, B, out, d):
        G, Ka, m = A.shape[0], A.shape[1], A.shape[2]
        K = B.shape[1]
        if d >= len(self.bufs):
            fnp.matmul(A, B, out=out)
            return
        h = m // 2
        LA_, LB_, M_ = self.bufs[d]
        LA, LB, M = LA_[:7 * G, :Ka], LB_[:7 * G, :K], M_[:7 * G, :K]
        a11, a12, a21, a22 = A[:, :, :h, :h], A[:, :, :h, h:], A[:, :, h:, :h], A[:, :, h:, h:]
        b11, b12, b21, b22 = B[:, :, :h, :h], B[:, :, :h, h:], B[:, :, h:, :h], B[:, :, h:, h:]
        sl = [slice(i * G, (i + 1) * G) for i in range(7)]
        fnp.add(a11, a22, out=LA[sl[0]]); fnp.add(a21, a22, out=LA[sl[1]]); fnp.multiply(a11, 1.0, out=LA[sl[2]])
        fnp.multiply(a22, 1.0, out=LA[sl[3]]); fnp.add(a11, a12, out=LA[sl[4]]); fnp.subtract(a21, a11, out=LA[sl[5]])
        fnp.subtract(a12, a22, out=LA[sl[6]])
        fnp.add(b11, b22, out=LB[sl[0]]); fnp.multiply(b11, 1.0, out=LB[sl[1]]); fnp.subtract(b12, b22, out=LB[sl[2]])
        fnp.subtract(b21, b11, out=LB[sl[3]]); fnp.multiply(b22, 1.0, out=LB[sl[4]]); fnp.add(b11, b12, out=LB[sl[5]])
        fnp.add(b21, b22, out=LB[sl[6]])
        self._rec(LA, LB, M, d + 1)
        m1, m2, m3, m4, m5, m6, m7 = (M[x] for x in sl)
        o11, o12, o21, o22 = out[:, :, :h, :h], out[:, :, :h, h:], out[:, :, h:, :h], out[:, :, h:, h:]
        fnp.add(m1, m4, out=o11); fnp.subtract(o11, m5, out=o11); fnp.add(o11, m7, out=o11)
        fnp.add(m3, m5, out=o12)
        fnp.add(m2, m4, out=o21)
        fnp.subtract(m1, m2, out=o22); fnp.add(o22, m3, out=o22); fnp.add(o22, m6, out=o22)

    def shared(self, W, X, out):
        """out[i] = W @ X[i] for a stack X (k, m, m)."""
        k = X.shape[0]
        for c in range(0, k, self.chunk):
            e = min(c + self.chunk, k)
            self._rec(W[None, None], X[None, c:e], out[None, c:e], 0)

    def paired(self, Xa, Xb, out):
        """out[i] = Xa[i] @ Xb[i]."""
        k = Xa.shape[0]
        for c in range(0, k, self.chunk):
            e = min(c + self.chunk, k)
            self._rec(Xa[None, c:e], Xb[None, c:e], out[None, c:e], 0)


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
    def __init__(self, k4_memo: bool = True, strassen_levels: int = 3, chunk: int = 16) -> None:
        self._setup_rng = None
        self.k4_memo = bool(k4_memo)
        self.lv = int(strassen_levels)
        self.chunk = int(chunk)

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        _rng = fnp.random.default_rng(mlp.seed)  # deterministic method; kept for the seeding contract
        _ = (_rng, budget)
        n = mlp.width
        L = len(mlp.weights)
        Smax = max(L - 1, 1)
        # leg buffers (ping-pong) and workspaces; slot 3s+0 = B_s, 3s+1 = E_s, 3s+2 = A_s
        # fnp.empty, not zeros: flopscope tags an all-zero buffer as symmetric, and out= writes then fail.
        # Only slots that were written are ever read.
        leg = [fnp.empty((3 * Smax, n, n), dtype=_F32), fnp.empty((3 * Smax, n, n), dtype=_F32)]
        left = fnp.empty((3 * Smax, n, n), dtype=_F32)
        prod = fnp.empty((3 * Smax, n, n), dtype=_F32)
        tmp = fnp.empty((Smax, n, n), dtype=_F32)
        G = fnp.empty((Smax, n), dtype=_F32)  # hub vectors g_s
        sw = _Strassen(n, self.lv, self.chunk) if self.lv > 0 else None
        cur = 0
        S = 0  # live sources
        mu = fnp.zeros(n, dtype=_F32)
        C = fnp.eye(n, dtype=_F32)
        K4post = K22post = None
        rows = []
        for l, w in enumerate(mlp.weights):
            last = l == L - 1
            Wt = w.T
            mu_pre = Wt @ mu
            C_pre = Wt @ (C @ w)
            var = fnp.maximum(fnp.diag(C_pre), 1e-12)
            C0 = C_pre - fnp.diag(fnp.diag(C_pre))
            t = fnp.zeros(n, dtype=_F32)
            D = None
            if S > 0:
                nxt = 1 - cur
                X, Y = leg[cur], leg[nxt]
                k = 3 * S - 1  # every live slot except the newborn's A (identity)
                if sw is None:
                    fnp.matmul(Wt, X[:k], out=Y[:k])
                else:
                    sw.shared(Wt, X[:k], Y[:k])
                fnp.multiply(Wt, 1.0, out=Y[k])  # newborn A = W^T I (flopscope arrays are immutable: write via out=)
                cur = nxt
                X = leg[cur]
                Bs, Es, As = X[0:3 * S:3], X[1:3 * S:3], X[2:3 * S:3]
                g3 = G[:S][:, None, :]
                # kappa3 diagonal: 3 sum_p g A B^2 + sum_p E A^2
                L1 = left[0:3 * S:3]           # pairs with B:  2 A g B
                L3 = left[1:3 * S:3]           # pairs with E:  (1/3) A^2
                L2 = left[2:3 * S:3]           # pairs with A:  g B^2 + (2/3) E A
                T = tmp[:S]
                fnp.multiply(As, g3, out=L1)
                fnp.multiply(L1, Bs, out=L1)            # A g B
                fnp.multiply(L1, Bs, out=T)             # A g B^2
                t = 3.0 * fnp.sum(fnp.sum(T, axis=2), axis=0)
                fnp.multiply(As, As, out=L3)            # A^2
                fnp.multiply(L3, Es, out=T)             # E A^2
                t = t + fnp.sum(fnp.sum(T, axis=2), axis=0)
                if not last:
                    fnp.multiply(L1, 2.0, out=L1)
                    fnp.multiply(L3, 1.0 / 3.0, out=L3)
                    fnp.multiply(Bs, Bs, out=L2)
                    fnp.multiply(L2, g3, out=L2)
                    fnp.multiply(Es, As, out=T)
                    fnp.multiply(T, 2.0 / 3.0, out=T)
                    fnp.add(L2, T, out=L2)
                    if sw is None:
                        fnp.matmul(left[:3 * S], fnp.swapaxes(X[:3 * S], 1, 2), out=prod[:3 * S])
                    else:
                        sw.paired(left[:3 * S], fnp.swapaxes(X[:3 * S], 1, 2), prod[:3 * S])
                    D = fnp.sum(prod[:3 * S], axis=0)
                    fnp.fill_diagonal(D, fnp.zeros(n, dtype=_F32))
            if D is None and not last:
                D = fnp.zeros((n, n), dtype=_F32)
            K4d = K22 = None
            if K4post is not None:
                Wsq = w * w
                Xk = K22post + fnp.diag(K4post)
                M = Wsq.T @ (Xk @ Wsq)
                K4d = 3.0 * fnp.diag(M) - 2.0 * ((Wsq * Wsq).T @ K4post)
                if not last:
                    K22 = M - fnp.diag(fnp.diag(M))
            o = _nonlin(mu_pre, var, C0, t, D, K4d, K22, need_pairs=not last)
            mu = o["K1"]
            rows.append(mu)
            if last:
                break
            C = o["K11"] + fnp.diag(o["K2"])
            if self.k4_memo:
                K4post, K22post = o["K4"], o["K22"]
            d1, d2 = o["d1"], o["d2"]
            X = leg[cur]
            if S > 0:
                fnp.multiply(X[:3 * S], d1[None, :, None], out=X[:3 * S])  # ReLU pass-through gate
            # birth of source S: B = diag(d1) C0, E = V, A = identity (materialised at the next transport)
            d11 = d1 * d1
            fnp.multiply(d1[:, None], C0, out=X[3 * S])
            dK3 = o["K3"] - d11 * d1 * t
            dK21 = o["K21"] - (d11[:, None] * d1[None, :]) * D - (d11[:, None] * d2[None, :]) * (C0 * C0)
            fnp.fill_diagonal(dK21, fnp.zeros(n, dtype=_F32))
            fnp.add(3.0 * dK21.T, fnp.diag(dK3), out=X[3 * S + 1])
            fnp.multiply(d2, 1.0, out=G[S])
            S += 1
        return fnp.stack(rows, axis=0)
