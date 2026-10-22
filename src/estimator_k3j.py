"""Third-cumulant chain estimator with thin correction legs (ARC White-Box Estimation Challenge 2026, Phase 2).

v9 = estimator_k3i.py + a dead-row cut (experiments/2026-09-25_cost): neurons in the dead regime, alpha = mu/sigma < -3
(d1 = Phi(alpha) < 1.35e-3), are gated to exactly 0 after the nonlinear step, so their rows of every carried leg are
zero and are dropped from the next leg transport (rectangular Strassen over the alive rows, padded to a multiple of
16 with the least-dead neurons); the D pairings are computed only for alive x alive pairs and the dead rows/columns of
D are 0 (torch chain.py deadcut=-3: +0.3% final MSE on 10 MLPs). C and the path leg are transported in full. The
workspace Strassen is generalised to rectangular (m x k)(k x p) blocks through views of the same buffers.
With deadcut=None the arithmetic is that of estimator_k3i.py, operation for operation.

Header of estimator_k3i.py (v8):

v8 = estimator_k3h.py + three accuracy terms (experiments/2026-09-24_k3-chain, ablate.py "k3i": 10 baked MLPs
2.7989e-8 -> 2.4919e-8, all 10 better):
  * paths: the O(n)-rank path births of the augmented term set of ARC's reference (mlp_kprop factor_k3.py,
    augment=True): every path x-p-z whose other edge is C0 shares the hub leg A and the tree leg B, so together they
    are Sym(T Y (x) 3A (x) B) with Y = d1 D^T d3 + d2 D d2 + 1/2 d2 K22 d3 + d2 C0^2 d3 (hub weights inside; checked
    against ARC's code to 5e-16). Only its diagonal readout at the first transport is kept (t += 3 rowsum(A o B o Wt Y)
    for the newest source; one extra matrix in the leg transport per layer), and its (2,1)-slice contributions
    (X o A) B^T + (X o B) A^T ride the existing L1, L2 pairings (free; the third, (A o B) X^T, is dropped); its (2,1)
    birth slice Y o U is removed from the correction leg, as for any birth.
  * star: the star-diagonal kappa4 birth of the newest source, K4d += 4 rowsum(d3_birth o A o B^3) (O(n^2)).
  * c0k22: the augmented pair diagrams we did not have, 1/6 a3 b3 C0^3 (triple edge) and 1/4 a3 b3 C0 o K22 (edge x
    kappa4 (2,2) slice), in every Q(p,q) (O(n^2)).
With paths=False, star=False, c0k22=False the arithmetic is that of estimator_k3h.py, operation for operation.

Header of estimator_k3h.py (v7):

v7 = estimator_k3f.py with a Strassen layout for deeper recursion: the D pairings only need the SUM of the
per-source products, so the leaf products are summed before the recombination (einsum), and the per-level
buffers are shared between the transport and the pairing modes. Strassen L4 at chunk 16 then fits in ~2.7 GB
of workspace with the same number of calls as L3. Same algorithm as k3f (float32 summation order differs).

Header of estimator_k3f.py (v6):
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
    """Batched Strassen-Winograd with preallocated per-level workspaces (k3h layout).
    shared(W, X, out):    out[i] = W @ X[i]                       (the leg transports)
    paired(Xa, Xb, out):  out[i] = Xa[i] @ Xb[i], for stacks of ONE matrix only (the base product)
    paired_sum(Xa, Xb, out): out = sum_i Xa[i] @ Xb[i]            (the D pairings: only the sum is needed)
    Strassen is bilinear, so paired_sum adds the K leaf products before the recombination (one einsum per leaf):
    its product buffer M has no K axis and the recombination runs once. Per level three buffers X1, X2 (chunk
    matrices per group) and X3 (one per group) serve both modes: shared uses LA = X3, LB = X1, M = X2; paired_sum
    uses LA = X1, LB = X2, M = X3. Every level issues a fixed number of flopscope calls."""

    def __init__(self, n, levels, chunk):
        self.levels, self.chunk = levels, chunk
        self.bufs = []
        self.regions = []
        m, G = n, 1
        for _ in range(levels):
            if m % 2 or m < 128:
                break
            m //= 2
            G *= 7
            half = G * chunk * m * m
            R = fnp.empty((2 * half,), dtype=_F32)   # X1 and X2 of this level live in one flat region
            self.regions.append(R)
            self.bufs.append((R[:half].reshape((G, chunk, m, m)), R[half:].reshape((G, chunk, m, m)),
                              fnp.empty((G, 1, m, m), dtype=_F32)))
        # scratch for the paired_sum leaf (matmul, then sum over k): the level L-2 region is free while the leaf
        # runs (its LA/LB were consumed by the level L-1 combos) and is 8/7 of the leaf size
        self.leaf_tmp = None
        L = len(self.bufs)
        if L >= 2:
            Gl, ml = 7 ** L, n // 2 ** L
            self.leaf_tmp = self.regions[L - 2][:Gl * chunk * ml * ml].reshape((Gl, chunk, ml, ml))
        elif L == 1:
            self.leaf_tmp = fnp.empty((7, chunk, n // 2, n // 2), dtype=_F32)

    @staticmethod
    def _combos(A, B, LA, LB, G):
        hm, hk, hp = A.shape[2] // 2, A.shape[3] // 2, B.shape[3] // 2
        a11, a12, a21, a22 = A[:, :, :hm, :hk], A[:, :, :hm, hk:], A[:, :, hm:, :hk], A[:, :, hm:, hk:]
        b11, b12, b21, b22 = B[:, :, :hk, :hp], B[:, :, :hk, hp:], B[:, :, hk:, :hp], B[:, :, hk:, hp:]
        sl = [slice(i * G, (i + 1) * G) for i in range(7)]
        fnp.add(a11, a22, out=LA[sl[0]]); fnp.add(a21, a22, out=LA[sl[1]]); fnp.multiply(a11, 1.0, out=LA[sl[2]])
        fnp.multiply(a22, 1.0, out=LA[sl[3]]); fnp.add(a11, a12, out=LA[sl[4]]); fnp.subtract(a21, a11, out=LA[sl[5]])
        fnp.subtract(a12, a22, out=LA[sl[6]])
        fnp.add(b11, b22, out=LB[sl[0]]); fnp.multiply(b11, 1.0, out=LB[sl[1]]); fnp.subtract(b12, b22, out=LB[sl[2]])
        fnp.subtract(b21, b11, out=LB[sl[3]]); fnp.multiply(b22, 1.0, out=LB[sl[4]]); fnp.add(b11, b12, out=LB[sl[5]])
        fnp.add(b21, b22, out=LB[sl[6]])
        return sl

    @staticmethod
    def _recombine(M, out, sl):
        hm, hp = out.shape[-2] // 2, out.shape[-1] // 2
        m1, m2, m3, m4, m5, m6, m7 = (M[x] for x in sl)
        o11, o12, o21, o22 = out[..., :hm, :hp], out[..., :hm, hp:], out[..., hm:, :hp], out[..., hm:, hp:]
        fnp.add(m1, m4, out=o11); fnp.subtract(o11, m5, out=o11); fnp.add(o11, m7, out=o11)
        fnp.add(m3, m5, out=o12)
        fnp.add(m2, m4, out=o21)
        fnp.subtract(m1, m2, out=o22); fnp.add(o22, m3, out=o22); fnp.add(o22, m6, out=o22)

    def _rec(self, A, B, out, d):
        """out = A @ B, A (G, 1, m, k), B (G, K, k, p), out (G, K, m, p); m, k, p divisible by 2**levels."""
        G, K = A.shape[0], B.shape[1]
        m, k, p = A.shape[2], A.shape[3], B.shape[3]
        if d >= len(self.bufs):
            fnp.matmul(A, B, out=out)
            return
        X1, X2, X3 = self.bufs[d]
        LA = X3[:7 * G, :, :m // 2, :k // 2]
        LB = X1[:7 * G, :K, :k // 2, :p // 2]
        M = X2[:7 * G, :K, :m // 2, :p // 2]
        sl = self._combos(A, B, LA, LB, G)
        self._rec(LA, LB, M, d + 1)
        self._recombine(M, out, sl)

    def _rec_sum(self, A, B, out, d):
        """out[g] = sum_k A[g, k] @ B[g, k], A (G, K, m, k), B (G, K, k, p), out (G, m, p)."""
        G, K = A.shape[0], A.shape[1]
        m, k, p = A.shape[2], A.shape[3], B.shape[3]
        if d >= len(self.bufs):
            if self.leaf_tmp is None:  # no recursion at all
                fnp.einsum('gkij,gkjl->gil', A, B, out=out)
            else:
                T = self.leaf_tmp[:G, :K, :m, :p]
                fnp.matmul(A, B, out=T)
                fnp.sum(T, axis=1, out=out)
            return
        X1, X2, X3 = self.bufs[d]
        LA = X1[:7 * G, :K, :m // 2, :k // 2]
        LB = X2[:7 * G, :K, :k // 2, :p // 2]
        M = X3[:7 * G, 0, :m // 2, :p // 2]
        sl = self._combos(A, B, LA, LB, G)
        self._rec_sum(LA, LB, M, d + 1)
        self._recombine(M, out, sl)

    def shared(self, W, X, out):
        """out[i] = W @ X[i] for a stack X (k, m, m)."""
        k = X.shape[0]
        for c in range(0, k, self.chunk):
            e = min(c + self.chunk, k)
            self._rec(W[None, None], X[None, c:e], out[None, c:e], 0)

    def paired(self, Xa, Xb, out):
        """out[0] = Xa[0] @ Xb[0] (stacks of one matrix)."""
        assert Xa.shape[0] == 1
        self._rec(Xa[None], Xb[None], out[None], 0)

    def paired_sum(self, Xa, Xb, out, tmp):
        """out = sum_i Xa[i] @ Xb[i]; tmp: an (m, m) scratch for chunks after the first."""
        k = Xa.shape[0]
        for c in range(0, k, self.chunk):
            e = min(c + self.chunk, k)
            tgt = out if c == 0 else tmp
            self._rec_sum(Xa[None, c:e], Xb[None, c:e], tgt[None], 0)
            if c > 0:
                fnp.add(out, tmp, out=out)


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


def _nonlin(mu, var, C0, t, D, K4d, K22, need_pairs, want_w12=False, c0d=False, marg=False, v31=None, c0k22=False):
    w, d1, d2 = _wick(mu, var, want_hi=marg and need_pairs)
    t6 = t / 6.0
    k24 = K4d / 24.0 if K4d is not None else None
    P = {}
    for p in range(1, 5):
        v = w[(0, p)] + w[(3, p)] * t6
        if K4d is not None:
            v = v + w[(4, p)] * k24
        P[p] = v
    out = {"K1": P[1], "d1": d1, "d2": d2, "w12": w[(1, 2)], "d3": w[(3, 1)]}
    if not need_pairs:
        return out
    C0sq = C0 * C0
    DT = D.T
    if c0d:
        DC0 = D * C0
        DTC0 = DT * C0
    C0K22 = C0 * K22 if (c0k22 and K22 is not None) else None
    C0cube = C0sq * C0 if c0k22 else None
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
        if C0cube is not None:  # augmented pair diagrams: triple edge, and edge x kappa4 (2,2) slice
            a3, b3 = w[(3, p)][:, None], w[(3, q)][None, :]
            M = M + ((1.0 / 6.0) * a3 * b3) * C0cube
            if C0K22 is not None:
                M = M + (0.25 * a3 * b3) * C0K22
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
    def __init__(self, k4_memo: bool = True, strassen_levels: int = 4, chunk: int = 16, thin: int = 24,
                 base_strassen: bool = True, batch_base: bool = True, keep_workspace: bool = True,
                 closure: bool = True, c0d: bool = True, marg: bool = True, k31: bool = True,
                 feed: bool = True, fuse_base: bool = True, paths: bool = True, star: bool = True,
                 c0k22: bool = True, deadcut: float | None = -3.0) -> None:
        self._setup_rng = None
        self.paths = bool(paths)
        self.star = bool(star)
        self.c0k22 = bool(c0k22)
        self.deadcut = None if deadcut is None else float(deadcut)
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
            "leg": [fnp.empty((2 * Smax + 1, n, n), dtype=_F32), fnp.empty((2 * Smax + 1, n, n), dtype=_F32)],
            "D3B": fnp.empty((n,), dtype=_F32),         # d3 at birth of the newest source (star-diagonal kappa4 birth)
            # dead-row cut: flat regions reshaped per layer to the alive size (contiguous views)
            "GX": fnp.empty(((2 * Smax + 1) * n * n,), dtype=_F32),   # gathered alive rows of the legs (transport)
            "GL": fnp.empty((2 * Smax * n * n,), dtype=_F32),         # gathered alive rows of the left pairing stack
            "GR": fnp.empty((2 * Smax * n * n,), dtype=_F32),         # gathered alive rows of the right pairing stack
            "Wg": fnp.empty((n * n,), dtype=_F32),                    # gathered alive columns of Wt
            "Dext": fnp.empty(((n + 1) * (n + 1),), dtype=_F32),      # alive x alive D block + a zero row/column
            "Dtc": fnp.empty((n * n,), dtype=_F32),                   # paired_sum chunk scratch at the alive size
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
            "Dsum": fnp.empty((n, n), dtype=_F32),      # the D pairings, summed inside the Strassen recursion
            "Dtmp": fnp.empty((n, n), dtype=_F32),
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
        quantum = 2 ** len(sw.bufs) if sw is not None else 1
        cut_prev = None  # (idx, n_alive_padded) of the previous layer when its dead rows are dropped

        def alive_cut(mu_pre, var):
            """mask (1 alive / 0 dead, alpha < deadcut); if some are dead: indices of the alive neurons padded with the
            least-dead ones to a multiple of the Strassen quantum, the row map for scattering back (dead -> zero row)."""
            a = mu_pre / fnp.sqrt(var)
            mask = fnp.where(a >= self.deadcut, 1.0, 0.0).astype(_F32)
            n_alive = int(round(float(fnp.sum(mask))))
            npd = min(n, ((n_alive + quantum - 1) // quantum) * quantum)
            if npd >= n:
                return mask, None
            order = fnp.argsort(-a)
            idx = order[:npd]
            rmap = fnp.minimum(fnp.argsort(order), npd)
            return mask, (idx, npd, rmap)
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
            mask_cur = cut_cur = None
            fused = self.fuse_base and self.closure and S > 0
            if S > 0:
                nxt = 1 - cur
                X, Y = leg[cur], leg[nxt]
                k = 2 * S - 1  # every dense slot except the newborn's A (identity)
                kt = k + 1 if fused else k  # fused: slot k holds C (written at the end of the previous layer)
                kx = 2 * S + 1 if self.paths else kt  # paths: slot 2S holds the newest source's path leg Y
                if cut_prev is not None and k > 0:
                    # legs 0..k-1 have exactly zero dead rows: contract over the alive rows only; C and the path leg
                    # (slots k..kx-1, dead rows not zero) are transported in full
                    idp, npp = cut_prev[0], cut_prev[1]
                    GXv = ws["GX"][:k * npp * n].reshape((k, npp, n))
                    fnp.take(X[:k], idp, axis=1, out=GXv)
                    Wgv = ws["Wg"][:n * npp].reshape((n, npp))
                    fnp.take(Wt, idp, axis=1, out=Wgv)
                    if sw is None:
                        fnp.matmul(Wgv, GXv, out=Y[:k])
                        if kx > k:
                            fnp.matmul(Wt, X[k:kx], out=Y[k:kx])
                    else:
                        sw.shared(Wgv, GXv, Y[:k])
                        if kx > k:
                            sw.shared(Wt, X[k:kx], Y[k:kx])
                elif sw is None:
                    fnp.matmul(Wt, X[:kx], out=Y[:kx])
                else:
                    sw.shared(Wt, X[:kx], Y[:kx])
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
                        if self.deadcut is not None:
                            mask_cur, cut_cur = alive_cut(mu_pre, var)
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
                if self.paths:  # path births of the newest source, first readout: t += 3 rowsum(A o B o (Wt Y))
                    t = t + 3.0 * fnp.sum(X[2 * S - 1] * X[2 * S - 2] * X[2 * S], axis=1)
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
                    if self.paths:  # path births of the newest source in D: (X o A) B^T + (X o B) A^T ride L1, L2
                        Xp = X[2 * S]
                        fnp.add(left[2 * S - 2], Xp * X[2 * S - 1], out=left[2 * S - 2])
                        fnp.add(left[2 * S - 1], Xp * X[2 * S - 2], out=left[2 * S - 1])
                    Dsum, Dtmp = ws["Dsum"], ws["Dtmp"]
                    if self.deadcut is not None and cut_cur is not None and sw is not None:
                        # only alive x alive pairs (dead rows/columns of D are zeroed below), scattered back through
                        # rmap, which sends every neuron outside the alive block to an appended zero row/column
                        idc, npc, rmap = cut_cur
                        GLv = ws["GL"][:2 * S * npc * n].reshape((2 * S, npc, n))
                        GRv = ws["GR"][:2 * S * npc * n].reshape((2 * S, npc, n))
                        fnp.take(left[:2 * S], idc, axis=1, out=GLv)
                        fnp.take(X[:2 * S], idc, axis=1, out=GRv)
                        Dv = ws["Dext"][:(npc + 1) * (npc + 1)].reshape((npc + 1, npc + 1))
                        sw.paired_sum(GLv, fnp.swapaxes(GRv, 1, 2), Dv[:npc, :npc],
                                      ws["Dtc"][:npc * npc].reshape((npc, npc)))
                        zrow = fnp.zeros(npc + 1, dtype=_F32)
                        fnp.multiply(zrow, 1.0, out=Dv[npc])
                        fnp.multiply(zrow, 1.0, out=Dv[:, npc])
                        D = fnp.take(fnp.take(Dv, rmap, axis=0), rmap, axis=1)
                    elif sw is None:
                        fnp.einsum('kij,kjl->il', left[:2 * S], fnp.swapaxes(X[:2 * S], 1, 2), out=Dsum)
                        D = Dsum * 1.0
                    else:
                        sw.paired_sum(left[:2 * S], fnp.swapaxes(X[:2 * S], 1, 2), Dsum, Dtmp)
                        D = Dsum * 1.0
                    fnp.matmul(thinw[:S], fnp.swapaxes(H, 1, 2), out=prod[:S])
                    D = D + fnp.sum(prod[:S], axis=0)
                    if self.feed:  # + q m^T summed over sources
                        D = D + fnp.swapaxes(MQ[1, :S], 0, 1) @ MQ[0, :S]
                    if self.deadcut is not None and mask_cur is not None:  # dead rows and columns of D are 0
                        D = D * (mask_cur[:, None] * mask_cur[None, :])
                    fnp.fill_diagonal(D, fnp.zeros(n, dtype=_F32))
            if D is None and not last:
                D = fnp.zeros((n, n), dtype=_F32)
            # ---- base products: C_pre = Wt C w and, from layer 1, the kappa4 slice M = Wsq^T Xk Wsq
            K4d = K22 = None
            v_cur = v_feed = None
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
                    v_feed = v_cur
                    if self.star and S > 0:  # star-diagonal kappa4 birth of the newest source: 4 d3 A B^3
                        Xs = leg[cur]
                        Bn = Xs[2 * S - 2]
                        K4d = K4d + 4.0 * fnp.sum(ws["D3B"][None, :] * Xs[2 * S - 1] * (Bn * Bn * Bn), axis=1)
                        v_feed = fnp.mean(K4d) / (3.0 * fnp.mean(s4))
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
            if self.deadcut is not None and mask_cur is None and not last:
                mask_cur, cut_cur = alive_cut(mu_pre, var)
            o = _nonlin(mu_pre, var, C0, t, D, K4d, K22, need_pairs=not last, want_w12=True, c0d=self.c0d,
                        marg=self.marg, v31=(v_cur if (self.k31 and have4 and self.closure) else None),
                        c0k22=self.c0k22)
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
            if self.deadcut is not None:  # dead neurons gated to exactly 0 (their leg rows are then dropped)
                d1 = d1 * mask_cur
                cut_prev = cut_cur
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
            if self.paths:  # path births (augmented term set): leg Y for the next transport, (2,1) slice Y o U removed
                d3 = o["d3"]
                Yp = ((d1[:, None] * D.T) * d3[None, :] + (d2[:, None] * D) * d2[None, :]
                      + (d2[:, None] * (C0 * C0)) * d3[None, :])
                if K22 is not None:
                    Yp = Yp + ((0.5 * d2)[:, None] * K22) * d3[None, :]
                fnp.fill_diagonal(Yp, fnp.zeros(n, dtype=_F32))
                dK21 = dK21 - Yp * X[2 * S]
                fnp.multiply(Yp, 1.0, out=X[2 * S + 2])
            if self.star:
                fnp.multiply(o["d3"], 1.0, out=ws["D3B"])
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
                h = (0.5 * v_feed) * (d2 * var)
                fnp.multiply(h, 1.0, out=HH[S])
                alpha = alpha - 2.0 * (h * d1)                             # R unchanged, P2's (2,1) slice into alpha
            elif self.feed:
                fnp.multiply(d1, 0.0, out=HH[S])
            if self.feed:
                fnp.multiply(d1, 1.0, out=D1B[S])
            fnp.multiply(alpha, 1.0, out=AL[S])
            S += 1
        return fnp.stack(rows, axis=0)
