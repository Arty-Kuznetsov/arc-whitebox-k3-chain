"""Third-cumulant chain estimator with thin correction legs (ARC White-Box Estimation Challenge 2026, Phase 2).

v6 of our k=3 chain = estimator_k3e.py + the remaining first-order terms of the Gaussian scale-mixture closure
(experiments/2026-09-24_k3-chain, ablate.py "feedP2_mk": 10 baked MLPs 3.1496e-8 -> 2.7989e-8, all 10 better):
  * marg: marginal Edgeworth correction of the pair Wick coefficients a_k, b_k (k <= 2):
    w(k,p) -> w(k,p) + w(k+3,p) t/6 + w(k+4,p) K4d/24;
  * k31: the kappa4 (3,1) slice of the closure, K31_ij = kappa4[iiij] = 3 v var_i C0_ij, in every Q(p,q) as
    (1/6)(a3 b1 K31 + a1 b3 K31^T); (a) and (b) must come together (each alone is neutral or harmful);
  * feed (P2): the all-distinct kappa3 births from the kappa4 (2,1,1) slice, 1/2 d2_x d1_y d1_z v var_x C_yz: a
    source Sym(h (x) D1 C0 D1) with hub vector h = v/2 d2 var. Transported, J (D1 C0 D1) J^T = B diag(d1_birth) A^T,
    so with m = A h and q = rowsum(B o d1_birth o A):  t += 3 m q,  D += 2 diag(m) B diag(d1_birth) A^T + q m^T
    (the first D term rides the existing L2 A^T product; everything else is O(n^2)). At birth its (2,1) slice
    2 (h d1)_i C0_ij d1_j moves into alpha (alpha -> alpha - 2 h d1), so R is unchanged.
    (The kappa4 -> kappa3 channel was reported as the main lever of 504aldo's public factorized chain, forum #18218;
    this form and implementation are ours, derived from the scale-mixture closure.)
With marg=False, k31=False, feed=False the arithmetic is that of estimator_k3e.py, operation for operation.

Header of estimator_k3e.py (v5):
v5 of our k=3 chain = estimator_k3d.py with two accuracy changes found with a Monte-Carlo oracle
(experiments/2026-09-24_k3-chain: k31_step.py, k4_pattern.py, ablate.py; 10 baked MLPs: final MSE x1.13 better):
  * c0d: the second-order diagram "edge x kappa3 (2,1) slice", 1/2 (a2 b3 D^T + a3 b2 D) o C0, in every pair power
    cumulant Q(p,q). With exact (MC) inputs it closes the per-step deficit of the post-ReLU kappa4 (2,2) slice
    (0.956 -> 1.004 of MC at layer 14, corr 0.974 -> 0.999). O(n^2), no products.
  * closure: a Gaussian scale-mixture closure for the kappa4 slices, K4d_i = 3 v var_i^2 and
    K22_ij = v (var_i var_j + 2 C0_ij^2) (MC: the kappa4 diagonal correlates 0.86 with var_i^2 at depth, the chain's
    memo22 pattern 0.23). Only the levels come from the memoryless transport M = Wsq^T X Wsq, and those are computed
    in O(n^2): sum(M) = u^T X u exactly (u = row sums of W o W), tr(M) with the mean-field G_ab = u_a u_b / n off the
    diagonal (reproduces the exact levels to 4 digits) -> the two dense kappa4 products per layer are gone.
With closure=False and c0d=False the arithmetic is that of estimator_k3d.py, operation for operation.

Header of estimator_k3d.py (v4):
v4 of our k=3 chain = estimator_k3c.py with two changes that only lower untracked (residual) time and FLOPs; the
numbers are those of k3c up to float32 rounding (experiments/2026-09-24_k3-chain/k3d_check.py):
  * the four base products per layer (C_pre = Wt C w and the kappa4 slice Wsq^T Xk Wsq) run as two paired Strassen
    products of the stacks [Wt, Wsq^T] @ [C, Xk] and [Wt, Wsq^T] @ (that)^T, using that C and Xk are symmetric;
    layer 0 (C = I) needs one product, and the last layer only the diagonals (O(n^2), no product);
  * the workspaces are cached on the instance and reused by the next predict() of the same shape, so no predict()
    frees them. Every region read is written earlier in the same call (k3d(batch_base=False) is bitwise equal to k3c,
    also with a reused workspace), so no information carries over between MLPs.

Header of estimator_k3c.py (v3):
Our k=3 chain (see estimator_k3b.py for the buffered layout and the attribution). New there: the correction
leg E of every source is no longer a dense n x n matrix. The (2,1) correction splits exactly as
    dK21 = diag(alpha) C0 diag(d1) + R,   alpha = w(1,2) - 2 P1 d1,
and the first part is carried for free as 3 B diag(alpha) (a column rescaling of the tree leg B), while the remainder
R is numerically low-rank: a rank-24 randomized range-finder approximation changes the final MSE by <1%
(experiments/2026-09-24_k3-chain, "split16rr"). Per source and layer this is 4 dense products instead of 6.
Original header of estimator_k3b.py follows.

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


def _wick(mu, var, want_hi=False):
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
    if want_hi:  # d5, d6 for the marginal correction of the pair coefficients
        a2_ = a * a
        d[5] = -(a2_ * a - 3.0 * a) * d2 * inv_s * inv_s * inv_s
        d[6] = (a2_ * a2_ - 6.0 * a2_ + 3.0) * d2 * inv_s * inv_s * inv_s * inv_s
    w = {(0, 1): m1, (0, 2): m2, (0, 3): m3, (0, 4): m4}
    for k in range(1, 5):
        w[(k, 1)] = d[k]
    mom = {1: m1, 2: m2, 3: m3}
    for p in (2, 3, 4):
        for k in range(1, 5):
            if p > 2 and k < 3:  # w(1..2, 3..4) are never used
                continue
            if k < p:
                w[(k, p)] = (math.factorial(p) / math.factorial(p - k)) * mom[p - k]
            else:
                w[(k, p)] = float(math.factorial(p)) * d[k - p + 1]
    if want_hi:
        w[(5, 1)], w[(6, 1)] = d[5], d[6]
        w[(5, 2)], w[(6, 2)] = 2.0 * d[4], 2.0 * d[5]
    return w, Ph, d2


def _nonlin(mu, var, C0, t, D, K4d, K22, need_pairs, want_w12=False, c0d=False, marg=False, v31=None):
    w, d1, d2 = _wick(mu, var, want_hi=marg and need_pairs)
    t6 = t / 6.0
    k24 = K4d / 24.0 if K4d is not None else None
    P = {}
    for p in range(1, 5):
        v = w[(0, p)] + w[(3, p)] * t6
        if K4d is not None:
            v = v + w[(4, p)] * k24
        P[p] = v
    out = {"K1": P[1], "d1": d1, "d2": d2, "w12": w[(1, 2)]}
    if not need_pairs:
        return out
    C0sq = C0 * C0
    DT = D.T
    if c0d:
        DC0 = D * C0
        DTC0 = DT * C0
    cf = {}
    for p in (1, 2):
        for k in (1, 2):
            c = w[(k, p)]
            if marg:  # marginal Edgeworth correction of the pair coefficient
                c = c + w[(k + 3, p)] * t6
                if K4d is not None:
                    c = c + w[(k + 4, p)] * k24
            cf[(k, p)] = c
    K31 = K31T = None
    if v31 is not None:  # kappa4 (3,1) slice of the scale-mixture closure: kappa4[iiij] = 3 v var_i C0_ij
        K31 = (3.0 * v31 * var)[:, None] * C0
        K31T = K31.T

    def Q(p, q):
        a1, a2 = cf[(1, p)][:, None], cf[(2, p)][:, None]
        b1, b2 = cf[(1, q)][None, :], cf[(2, q)][None, :]
        M = (a1 * b1) * C0 + (0.5 * a1 * b2) * DT + (0.5 * a2 * b1) * D + (0.5 * a2 * b2) * C0sq
        if K22 is not None:
            M = M + (0.25 * a2 * b2) * K22
        if K31 is not None:
            a3, b3 = w[(3, p)][:, None], w[(3, q)][None, :]
            M = M + ((1.0 / 6.0) * a3 * b1) * K31 + ((1.0 / 6.0) * a1 * b3) * K31T
        if c0d:  # edge x kappa3 (2,1)/(1,2) slice
            a3, b3 = w[(3, p)][:, None], w[(3, q)][None, :]
            M = M + (0.5 * a2 * b3) * DTC0 + (0.5 * a3 * b2) * DC0
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
    def __init__(self, k4_memo: bool = True, strassen_levels: int = 3, chunk: int = 16, thin: int = 24,
                 base_strassen: bool = True, batch_base: bool = True, keep_workspace: bool = True,
                 closure: bool = True, c0d: bool = True, marg: bool = True, k31: bool = True,
                 feed: bool = True, fuse_base: bool = True) -> None:
        self._setup_rng = None
        self.fuse_base = bool(fuse_base)
        self.marg = bool(marg)
        self.k31 = bool(k31)
        self.feed = bool(feed)
        self.closure = bool(closure)
        self.c0d = bool(c0d)
        self.base_strassen = bool(base_strassen)
        self.batch_base = bool(batch_base)
        self.keep_workspace = bool(keep_workspace)
        self.k4_memo = bool(k4_memo)
        self.lv = int(strassen_levels)
        self.chunk = int(chunk)
        self.kk = int(thin)
        self._ws = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _workspace(self, n, L, kk):
        """All big buffers. With keep_workspace they are cached on the instance and reused by the next predict() of the
        same shape, so no predict() frees them (freeing the ~2.5 GB of workspaces cost 30-40 ms of residual time per
        MLP). Every region that predict() reads is written earlier in the same call, so nothing carries over between
        MLPs; each predict() is also correct on a fresh instance (the grader spreads MLPs over several workers)."""
        key = (n, L, kk, self.lv, self.chunk)
        if self.keep_workspace and self._ws is not None and self._ws[0] == key:
            return self._ws[1]
        self._ws = None  # drop a workspace of another shape before allocating
        Smax = max(L - 1, 1)
        ws = {
            # dense legs (ping-pong), slot 2s = B_s, 2s+1 = A_s; thin legs H_s (n x kk), fixed right factors Ut_s
            "leg": [fnp.empty((2 * Smax, n, n), dtype=_F32), fnp.empty((2 * Smax, n, n), dtype=_F32)],
            "Hb": [fnp.empty((Smax, n, kk), dtype=_F32), fnp.empty((Smax, n, kk), dtype=_F32)],
            "Ut": fnp.empty((Smax, kk, n), dtype=_F32),
            "left": fnp.empty((2 * Smax, n, n), dtype=_F32),
            "prod": fnp.empty((2 * Smax, n, n), dtype=_F32),
            "Eb": fnp.empty((Smax, n, n), dtype=_F32),
            "tmp": fnp.empty((Smax, n, n), dtype=_F32),
            "thinw": fnp.empty((Smax, n, kk), dtype=_F32),
            "G": fnp.empty((Smax, n), dtype=_F32),      # hub vectors g_s = d2 at birth
            "K3v": fnp.empty((Smax, n), dtype=_F32),    # dK3 at birth
            "AL": fnp.empty((Smax, n), dtype=_F32),     # alpha at birth
            "HH": fnp.empty((Smax, n), dtype=_F32),     # feed hub vectors h = v/2 d2 var at birth
            "D1B": fnp.empty((Smax, n), dtype=_F32),    # d1 at birth (for J D1 C0 D1 J^T = B diag(d1b) A^T)
            "MQ": fnp.empty((2, Smax, n), dtype=_F32),  # feed m = A h and q = rowsum(B d1b A)
            "sw": _Strassen(n, self.lv, self.chunk) if self.lv > 0 else None,
            "mm1": fnp.empty((1, n, n), dtype=_F32),
            "mm2": fnp.empty((1, n, n), dtype=_F32),
            # batched base products: BA = [Wt, Wt*Wt], BB = [C, Xk], BP = BA @ BB, BO = BA @ BP^T
            "BA": fnp.empty((2, n, n), dtype=_F32),
            "BB": fnp.empty((2, n, n), dtype=_F32),
            "BP": fnp.empty((2, n, n), dtype=_F32),
            "BO": fnp.empty((2, n, n), dtype=_F32),
        }
        if self.keep_workspace:
            self._ws = (key, ws)
        return ws

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)  # probes of the randomized range finder
        _ = budget
        n = mlp.width
        L = len(mlp.weights)
        kk = max(1, min(self.kk, n))
        ws = self._workspace(n, L, kk)
        leg, Hb, Ut, left, prod = ws["leg"], ws["Hb"], ws["Ut"], ws["left"], ws["prod"]
        Eb, tmp, thinw, G, K3v, AL = ws["Eb"], ws["tmp"], ws["thinw"], ws["G"], ws["K3v"], ws["AL"]
        HH, D1B, MQ = ws["HH"], ws["D1B"], ws["MQ"]
        v_cur = None
        sw, mm1, mm2 = ws["sw"], ws["mm1"], ws["mm2"]
        BA, BB, BP, BO = ws["BA"], ws["BB"], ws["BP"], ws["BO"]
        use_sw = sw is not None and self.base_strassen

        def mm(a, b):  # single n x n product through the workspace Strassen (plain matmul if disabled)
            if not use_sw:
                return a @ b
            fnp.multiply(b, 1.0, out=mm1[0])
            sw.shared(a, mm1, mm2)
            return mm2[0] * 1.0

        def bprod(a, b, out):  # out[i] = a[i] @ b[i] for stacks of 1-2 matrices
            if use_sw:
                sw.paired(a, b, out)
            else:
                fnp.matmul(a, b, out=out)

        cur = 0
        S = 0
        mu = fnp.zeros(n, dtype=_F32)
        C = None if (self.closure or self.batch_base) else fnp.eye(n, dtype=_F32)  # batched paths: C in BB[0]
        K4post = K22post = None
        rows = []
        for l, w in enumerate(mlp.weights):
            last = l == L - 1
            Wt = w.T
            mu_pre = Wt @ mu
            t = fnp.zeros(n, dtype=_F32)
            D = None
            fused = self.fuse_base and self.closure and S > 0
            if S > 0:
                nxt = 1 - cur
                X, Y = leg[cur], leg[nxt]
                k = 2 * S - 1  # every dense slot except the newborn's A (identity)
                kt = k + 1 if fused else k  # fused: slot k holds C (written at the end of the previous layer)
                if sw is None:
                    fnp.matmul(Wt, X[:kt], out=Y[:kt])
                else:
                    sw.shared(Wt, X[:kt], Y[:kt])
                if fused:  # Y[k] = Wt C rode the leg transport; finish C_pre = Wt (Wt C)^T before Y[k] becomes A = Wt
                    fnp.multiply(Wt, 1.0, out=BA[0])
                    Pst = Y[k:k + 1]
                    if last:
                        var = fnp.maximum(fnp.sum(Pst[0] * BA[0], axis=1), 1e-12)
                        C0 = None
                    else:
                        bprod(BA[:1], fnp.swapaxes(Pst, 1, 2), BO[:1])
                        C_pre = BO[0]
                        var = fnp.maximum(fnp.diag(C_pre), 1e-12)
                        C0 = C_pre - fnp.diag(fnp.diag(C_pre))
                fnp.multiply(Wt, 1.0, out=Y[k])
                fnp.matmul(Wt, Hb[cur][:S], out=Hb[nxt][:S])
                cur = nxt
                X = leg[cur]
                H = Hb[cur][:S]
                Bs, As = X[0:2 * S:2], X[1:2 * S:2]
                g3 = G[:S][:, None, :]
                E = Eb[:S]
                T = tmp[:S]
                # E = A diag(dK3) + 3 B diag(alpha) + 3 H Ut
                fnp.matmul(H, Ut[:S], out=E)
                fnp.multiply(E, 3.0, out=E)
                fnp.multiply(As, K3v[:S][:, None, :], out=T)
                fnp.add(E, T, out=E)
                fnp.multiply(Bs, AL[:S][:, None, :], out=T)
                fnp.multiply(T, 3.0, out=T)
                fnp.add(E, T, out=E)
                L1 = left[0:2 * S:2]   # pairs with B: 2 A g B + A^2 alpha
                L2 = left[1:2 * S:2]   # pairs with A: g B^2 + (2/3) E A + (1/3) A^2 dK3
                fnp.multiply(As, g3, out=L1)
                fnp.multiply(L1, Bs, out=L1)               # A g B
                fnp.multiply(L1, Bs, out=T)                # A g B^2
                t = 3.0 * fnp.sum(fnp.sum(T, axis=2), axis=0)
                fnp.multiply(As, As, out=L2)               # A^2 (temporarily in L2)
                fnp.multiply(L2, E, out=T)                 # E A^2
                t = t + fnp.sum(fnp.sum(T, axis=2), axis=0)
                if self.feed:  # P2 sources: m = A h, q = rowsum(B d1b A), t += 3 m q
                    mv, qv = MQ[0, :S], MQ[1, :S]
                    fnp.matmul(As, HH[:S][:, :, None], out=mv[:, :, None])
                    fnp.multiply(Bs, D1B[:S][:, None, :], out=T)
                    fnp.multiply(T, As, out=T)
                    fnp.sum(T, axis=2, out=qv)
                    t = t + 3.0 * fnp.sum(mv * qv, axis=0)
                if not last:
                    fnp.multiply(L2, AL[:S][:, None, :], out=T)   # A^2 alpha
                    fnp.multiply(L1, 2.0, out=L1)
                    fnp.add(L1, T, out=L1)
                    # thin term (1/3) A^2 ER^T = (A^2 Ut^T) H^T, with A^2 still in L2
                    fnp.matmul(L2, fnp.swapaxes(Ut[:S], 1, 2), out=thinw[:S])
                    fnp.multiply(L2, K3v[:S][:, None, :], out=T)  # A^2 dK3
                    fnp.multiply(T, 1.0 / 3.0, out=T)
                    fnp.multiply(Bs, Bs, out=L2)
                    fnp.multiply(L2, g3, out=L2)
                    fnp.add(L2, T, out=L2)
                    fnp.multiply(E, As, out=T)
                    fnp.multiply(T, 2.0 / 3.0, out=T)
                    fnp.add(L2, T, out=L2)
                    if self.feed:  # 2 diag(m) B diag(d1b) rides the L2 A^T product
                        fnp.multiply(Bs, D1B[:S][:, None, :], out=T)
                        fnp.multiply(T, MQ[0, :S][:, :, None], out=T)
                        fnp.multiply(T, 2.0, out=T)
                        fnp.add(L2, T, out=L2)
                    if sw is None:
                        fnp.matmul(left[:2 * S], fnp.swapaxes(X[:2 * S], 1, 2), out=prod[:2 * S])
                    else:
                        sw.paired(left[:2 * S], fnp.swapaxes(X[:2 * S], 1, 2), prod[:2 * S])
                    D = fnp.sum(prod[:2 * S], axis=0)
                    fnp.matmul(thinw[:S], fnp.swapaxes(H, 1, 2), out=prod[:S])
                    D = D + fnp.sum(prod[:S], axis=0)
                    if self.feed:  # + q m^T summed over sources
                        D = D + fnp.swapaxes(MQ[1, :S], 0, 1) @ MQ[0, :S]
                    fnp.fill_diagonal(D, fnp.zeros(n, dtype=_F32))
            if D is None and not last:
                D = fnp.zeros((n, n), dtype=_F32)
            # ---- base products: C_pre = Wt C w and, from layer 1, the kappa4 slice M = Wsq^T Xk Wsq
            K4d = K22 = None
            v_cur = None
            have4 = K4post is not None
            if self.closure:
                # only the C products: Wt C then Wt (Wt C)^T (C symmetric); layer 0: C = I; last layer: diagonal only.
                # fused (fuse_base, layers >= 1): Wt C rode the leg transport and C_pre / var / C0 already exist
                if not fused:
                    fnp.multiply(Wt, 1.0, out=BA[0])
                    if l == 0:
                        Pst = BA[:1]
                    else:
                        bprod(BA[:1], BB[:1], BP[:1])                      # BB[0] = C, written below
                        Pst = BP[:1]
                    if last:
                        var = fnp.maximum(fnp.sum(Pst[0] * BA[0], axis=1), 1e-12)
                        C0 = None
                    else:
                        bprod(BA[:1], fnp.swapaxes(Pst, 1, 2), BO[:1])
                        C_pre = BO[0]
                        var = fnp.maximum(fnp.diag(C_pre), 1e-12)
                        C0 = C_pre - fnp.diag(fnp.diag(C_pre))
                if have4:
                    # levels of the memoryless transport M = Wsq^T X Wsq, X = K22post + diag(K4post), in O(n^2)
                    Wsq = w * w
                    u = fnp.sum(Wsq, axis=1)
                    Gd = fnp.sum(Wsq * Wsq, axis=1)
                    Ku2 = fnp.sum(K4post * u * u)
                    Ssum = fnp.sum((K22post @ u) * u) + Ku2                  # sum(M) = u^T X u
                    trM = (Ssum - Ku2) / n + fnp.sum(K4post * Gd)            # tr(M), mean-field off-diagonal G
                    k4m = 3.0 * trM / n - 2.0 * fnp.sum(Gd * K4post) / n      # mean of the transported K4 diagonal
                    s4 = var * var
                    K4d = s4 * (k4m / fnp.mean(s4))                           # scale-mixture pattern 3 v var_i^2
                    v_cur = k4m / (3.0 * fnp.mean(s4))                       # the scale-mixture level v
                    if not last:
                        pat = var[:, None] * var[None, :] + 2.0 * (C0 * C0)
                        fnp.fill_diagonal(pat, fnp.zeros(n, dtype=_F32))
                        K22 = pat * ((Ssum - trM) / fnp.sum(pat))            # v (var_i var_j + 2 C0_ij^2)
            elif self.batch_base:
                # two paired products of the stacks [Wt, Wsq^T] @ [C, Xk] and [Wt, Wsq^T] @ (that)^T, using that
                # C and Xk are symmetric (C w = (Wt C)^T); at layer 0 C = I, so the first stage is Wt itself; at the
                # last layer only the diagonals are needed: diag(P w) = sum(P * Wt, axis=1) in O(n^2)
                nb = 2 if have4 else 1
                fnp.multiply(Wt, 1.0, out=BA[0])
                if have4:
                    fnp.multiply(Wt, Wt, out=BA[1])                        # Wsq^T
                    fnp.add(K22post, fnp.diag(K4post), out=BB[1])           # Xk
                if l == 0:
                    Pst = BA[:1]
                else:
                    bprod(BA[:nb], BB[:nb], BP[:nb])                      # BB[0] = C, written below
                    Pst = BP[:nb]
                if last:
                    var = fnp.maximum(fnp.sum(Pst[0] * BA[0], axis=1), 1e-12)
                    C0 = None
                    if have4:
                        dM = fnp.sum(Pst[1] * BA[1], axis=1)
                else:
                    bprod(BA[:nb], fnp.swapaxes(Pst, 1, 2), BO[:nb])
                    C_pre = BO[0]
                    var = fnp.maximum(fnp.diag(C_pre), 1e-12)
                    C0 = C_pre - fnp.diag(fnp.diag(C_pre))
                    if have4:
                        M = BO[1]
                        dM = fnp.diag(M)
                        K22 = M - fnp.diag(fnp.diag(M))
                if have4:
                    K4d = 3.0 * dM - 2.0 * ((BA[1] * BA[1]) @ K4post)
            else:  # the k3c path, operation for operation (self-check)
                C_pre = mm(Wt, mm(C, w))
                var = fnp.maximum(fnp.diag(C_pre), 1e-12)
                C0 = C_pre - fnp.diag(fnp.diag(C_pre))
                if have4:
                    Wsq = w * w
                    Xk = K22post + fnp.diag(K4post)
                    M = mm(Wsq.T, mm(Xk, Wsq))
                    K4d = 3.0 * fnp.diag(M) - 2.0 * ((Wsq * Wsq).T @ K4post)
                    if not last:
                        K22 = M - fnp.diag(fnp.diag(M))
            o = _nonlin(mu_pre, var, C0, t, D, K4d, K22, need_pairs=not last, want_w12=True, c0d=self.c0d,
                        marg=self.marg, v31=(v_cur if (self.k31 and have4 and self.closure) else None))
            mu = o["K1"]
            rows.append(mu)
            if last:
                break
            if self.closure and self.fuse_base:
                fnp.add(o["K11"], fnp.diag(o["K2"]), out=leg[cur][2 * S + 1])  # C of the next layer, in the newborn-A slot
            elif self.closure or self.batch_base:
                fnp.add(o["K11"], fnp.diag(o["K2"]), out=BB[0])            # C of the next layer
            else:
                C = o["K11"] + fnp.diag(o["K2"])
            if self.k4_memo:
                K4post, K22post = o["K4"], o["K22"]
            d1, d2 = o["d1"], o["d2"]
            X = leg[cur]
            if S > 0:
                fnp.multiply(X[:2 * S], d1[None, :, None], out=X[:2 * S])
                fnp.multiply(Hb[cur][:S], d1[None, :, None], out=Hb[cur][:S])
            # ---- birth of source S
            d11 = d1 * d1
            fnp.multiply(d1[:, None], C0, out=X[2 * S])                 # B = diag(d1) C0
            dK3 = o["K3"] - d11 * d1 * t
            alpha = o["w12"] - 2.0 * o["K1"] * d1
            dK21 = o["K21"] - (d11[:, None] * d1[None, :]) * D - (d11[:, None] * d2[None, :]) * (C0 * C0)
            R = dK21 - (alpha[:, None] * d1[None, :]) * C0
            fnp.fill_diagonal(R, fnp.zeros(n, dtype=_F32))
            Om = rng.standard_normal((n, kk), dtype=_F32)
            Yr = R @ Om
            Yr = R @ (R.T @ Yr)
            Qr, _r = fnp.linalg.qr(Yr)
            fnp.multiply(R.T @ Qr, 1.0, out=Hb[cur][S])                 # H = R^T Q  (E_R = 3 H Ut)
            fnp.multiply(Qr.T, 1.0, out=Ut[S])                           # Ut = Q^T
            fnp.multiply(d2, 1.0, out=G[S])
            fnp.multiply(dK3, 1.0, out=K3v[S])
            if self.feed and have4 and self.closure:
                h = (0.5 * v_cur) * (d2 * var)
                fnp.multiply(h, 1.0, out=HH[S])
                alpha = alpha - 2.0 * (h * d1)                             # R unchanged, P2's (2,1) slice into alpha
            elif self.feed:
                fnp.multiply(d1, 0.0, out=HH[S])
            if self.feed:
                fnp.multiply(d1, 1.0, out=D1B[S])
            fnp.multiply(alpha, 1.0, out=AL[S])
            S += 1
        return fnp.stack(rows, axis=0)
