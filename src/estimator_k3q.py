"""Third-cumulant chain estimator with thin correction legs (ARC White-Box Estimation Challenge 2026, Phase 2).

v15 = estimator_k3p.py + the SATURATED-ROW cut of the D pairing with the kappa4 level restored (the cloud's E2, port
battery 2026-09-29, `chain.py satrow + satrow_k4="sep"`): rows i of the (2,1) slice D_ij = kappa3[i,i,j] of neurons
saturated NOW (alpha_i > satrow) are not computed (the pairing's left factor loses those columns) and are zero in D; the
kappa4 level the next layer reads only through Ssum = u^T K22post u (u = row sums of W_{l+1} o W_{l+1}) is restored by
dS = sum_{i sat} u_i coef(2,2)_i (D_full v)_i, v = u o (coef(1,2) - 2 P1 coef(1,1)) (marginally corrected pair
coefficients), with D_full v assembled from the legs by matrix-vector products (O(S n^2)); dS is added to the next
layer's Ssum, which is exactly the torch shift K22post += k22_fix off the diagonal. satrow=None is k3p, operation for
operation. Default satrow = 2.5 (10 MLPs on the production path, on top of k3p: raw x0.9431, util x0.9534, adjusted
x0.8990 on 10/10; 3.0 x0.9537; 2.0 x1.097 -- the cloud's curve: 2.5 is the optimum; experiments/2026-09-30_k3q).

Header of estimator_k3p.py (v14):

v14 = estimator_k3o.py + a TRANSPORTED BAND BASIS for old sources (experiments/2026-09-29_band E1 and the cloud's
experiments/2026-09-29_port-battery, task 1-2): sources of age >= a0 live as legs = P Z in ONE shared basis P (n x r),
transported with the legs (W^T diag(d1) (P Z) = (W^T diag(d1) P) Z, so the coefficients Z never change), and are
materialised every layer by one product [Z_B^T; Z_A^T] P^T for the readout and the element-wise pairing partners;
their D pairing is compressed, D_m = (L [Z_B Z_A]^T) P^T. A source enters at the END of the layer before it reaches age
a0 ("pre" entry, never transported densely at age a0), projected once on the gated P through a QR of P,
Z = R^-1 Q^T X for its A, B and thin H; P is never re-orthonormalised and the coefficients are never rotated
(the cloud: float32 is exact to <=2e-5 on the legs with this projection, cond(P) ~3e4 at depth). The first basis is the
dominant r-subspace of the first entrant's [A B] ("rr", one power step) or its first r hub columns ("cols").
In the transposed layout of k3o the basis rides the leg transport as r extra rows P^T. band=None is k3o, operation
for operation (experiments/2026-09-30_k3p/k3p_check.py). Default band = (5, 448, "rr"): the best of four on 10 MLPs
on the production path (adjusted x0.8905 vs k3o on 10/10; raw +2.4%, util x0.870; experiments/2026-09-30_k3p).

Header of estimator_k3o.py (v13):
v13 = estimator_k3m.py + the MATCHED dead-side cut: besides the dead-row cut (neurons with alpha = mu/sigma < deadcut
gated to exactly 0 and dropped from the contractions), a source born at layer s keeps only the hub columns (parents p)
with alpha_p >= colcut. A dead parent has d2_p ~ 0, and with the row cut it is already treated as dead on the row axis;
cutting its column at the same threshold removes the inconsistency (experiments/2026-09-29_colcut: torch, 10 MLPs, the
matched cut beats the row cut alone at every threshold, x0.9928 on 10/10 at -2.5; METHOD section 22).

The per-source column widths do not fit the batched (2S, n, n) leg stack of k3m, so the legs are stored TRANSPOSED as
one wide stack of hub rows: XT = [B_0^T; ...; B_{S-1}^T; A_0^T; ...; A_{S-1}^T], source s owning a block of w_s rows
(its kept hubs) in each region, and every per-hub vector (g, dK3, alpha, h, d1 at birth, Ut^T) indexed by the same rows.
Then the leg transport is ONE row-stack product XT' = XT W over the alive rows of the previous layer (a right-shared
Strassen over row chunks), and the D pairing D = L X^T = LT^T XT is ONE contraction over all hub rows, so the cost of
both scales with the total number of kept hubs. The newborn's B^T rows wait in their own buffer and are gathered in
between the two regions at the next transport; its hub leg A = I[:, P] becomes A^T = W[P, :] without a product.
Defaults: deadcut = colcut = -2.5 (the matched cut; 10 MLPs on the production path: raw x0.9937, util 23.79% = x0.956,
adjusted x0.950 on 10/10 vs k3m). With colcut=None every hub is kept and the arithmetic is k3m's (float32 rounding
differs: the products are the transposed ones), experiments/2026-09-30_k3o/k3o_check.py.

Only the production path of k3m is kept (closure, fused base, memo kappa4 levels); the switches that selected the
older paths are gone.

Header of estimator_k3m.py (v12):

v12 = estimator_k3l.py with the dead-row threshold at deadcut = -2.5 instead of -3.0 (experiments/2026-09-26_cost-frontier,
confirmed on 100 MLPs: adjusted 5.5209e-9, x0.962 of k3l; graded 332403 = 5.3906e-9).

Lineage (headers of the earlier files): k3l = k3j with the kappa4 (3,1) slice of the scale-mixture closure x0.65;
k3j = k3i + the dead-row cut with rectangular Strassen; k3i = k3h + path births at the first readout (the augmented term
set of ARC's reference, mlp_kprop factor_k3.py augment=True), the star-diagonal kappa4 birth and the C0^3 / C0 o K22
pair diagrams; k3h = k3f with Strassen L4 (sum-of-products pairings); k3f = k3e + marginal Edgeworth pair coefficients,
the closure's kappa4 (3,1) slice and the kappa4 -> kappa3 feed (P2) (the channel was reported as the main lever of
504aldo's public factorized chain, forum #18218; this form and implementation are ours); k3e = k3d + the C0 . D
diagram and the Gaussian scale-mixture kappa4 closure with O(n^2) levels; k3d = k3c with batched base products and
cached workspaces; k3c = the buffered chain with thin correction legs (rank-24 randomized range finder).

Same algorithm family as ARC's reference (the SIMPLE k=3 cumulant chain of Wu et al., arXiv:2605.05179, MIT), re-derived
and re-implemented by us in the "source" representation; the additions listed above are ours.
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
    shared(W, X, out):    out[i] = W @ X[i]
    rshared(X, W, out):   out[i] = X[i] @ W                       (k3o: the leg transports in the transposed layout)
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

    def _rec_r(self, A, B, out, d):
        """out = A @ B with the SHARED operand on the right: A (G, K, m, k), B (G, 1, k, p), out (G, K, m, p).
        The mirror of _rec: the varying stack's combos go to X1 (K axis), the shared one's to X3, products to X2."""
        G, K = A.shape[0], A.shape[1]
        m, k, p = A.shape[2], A.shape[3], B.shape[3]
        if d >= len(self.bufs):
            fnp.matmul(A, B, out=out)
            return
        X1, X2, X3 = self.bufs[d]
        LA = X1[:7 * G, :K, :m // 2, :k // 2]
        LB = X3[:7 * G, :, :k // 2, :p // 2]
        M = X2[:7 * G, :K, :m // 2, :p // 2]
        sl = self._combos(A, B, LA, LB, G)
        self._rec_r(LA, LB, M, d + 1)
        self._recombine(M, out, sl)

    def rshared(self, X, W, out):
        """out[i] = X[i] @ W for a stack X (K, c, k) and one shared W (k, p); c, k, p divisible by 2**levels."""
        K = X.shape[0]
        for c in range(0, K, self.chunk):
            e = min(c + self.chunk, K)
            self._rec_r(X[None, c:e], W[None, None], out[None, c:e], 0)

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
    out.update(K11=Q11, K21=K21, K22=K22o, cf11=cf[(1, 1)], cf12=cf[(1, 2)], cf22=cf[(2, 2)])
    return out




class Estimator(BaseEstimator):
    def __init__(self, strassen_levels: int = 4, chunk: int = 16, thin: int = 24, keep_workspace: bool = True,
                 c0d: bool = True, marg: bool = True, k31: bool = True, feed: bool = True, paths: bool = True,
                 star: bool = True, c0k22: bool = True, deadcut: float | None = -2.5, colcut: float | None = -2.5,
                 k31_scale: float = 0.65, band: tuple | None = (5, 448, "rr"), satrow: float | None = 2.5) -> None:
        self._setup_rng = None
        self.paths = bool(paths)
        self.star = bool(star)
        self.c0k22 = bool(c0k22)
        self.deadcut = None if deadcut is None else float(deadcut)
        self.colcut = None if colcut is None else float(colcut)
        self.k31_scale = float(k31_scale)
        self.marg = bool(marg)
        self.k31 = bool(k31)
        self.feed = bool(feed)
        self.c0d = bool(c0d)
        self.keep_workspace = bool(keep_workspace)
        self.lv = int(strassen_levels)
        self.chunk = int(chunk)
        self.kk = int(thin)
        # band = (a0, r, init): sources of age >= a0 live as legs P Z in one shared transported basis P (n x r);
        # init "rr" (range finder on the first entrant's [A B]) or "cols" (its first r hub columns of A)
        self.band = None if band is None else (int(band[0]), int(band[1]), str(band[2]) if len(band) > 2 else "rr")
        self.satrow = None if satrow is None else float(satrow)
        self._ws = None

    def setup(self, ctx: SetupContext) -> None:
        self._setup_rng = fnp.random.default_rng(ctx.seed)

    def _workspace(self, n, L, kk):
        """All big buffers, cached on the instance and reused by the next predict() of the same shape (no frees inside
        predict). Every region predict() reads is written earlier in the same call, so nothing carries over between
        MLPs; the zero block Z is written once and only read."""
        r = self.band[1] if self.band is not None else 0
        key = (n, L, kk, self.lv, self.chunk, r)
        if self.keep_workspace and self._ws is not None and self._ws[0] == key:
            return self._ws[1]
        self._ws = None  # drop a workspace of another shape before allocating
        Smax = max(L - 1, 1)
        Wcap = Smax * n          # hub rows per region, at most (every source keeps every hub)
        CAPR = 2 * Wcap + n + r  # rows of a leg stack, with the band basis and the Strassen padding
        Z = fnp.empty((n, n), dtype=_F32)
        fnp.multiply(fnp.zeros((n, n), dtype=_F32), 1.0, out=Z)
        ws = {
            # the legs, transposed: [P^T (band basis, r rows); B^T of the non-members; A^T of the non-members]
            "XT": [fnp.empty((CAPR, n), dtype=_F32), fnp.empty((CAPR, n), dtype=_F32)],
            "LT": fnp.empty((CAPR, n), dtype=_F32),     # the non-members' pairing partners, [L1^T; L2^T]
            "GX": fnp.empty((CAPR * n,), dtype=_F32),   # gathered alive columns of the legs (transport input)
            "GL": fnp.empty((CAPR * n,), dtype=_F32),   # gathered alive columns of LT (pairing)
            "GR": fnp.empty((CAPR * n,), dtype=_F32),   # gathered alive columns of XT (pairing)
            "Wg": fnp.empty((n * n,), dtype=_F32),      # alive rows of W
            "ET": fnp.empty((Wcap, n), dtype=_F32),     # the correction leg E^T (global hub rows)
            "T": fnp.empty((Wcap, n), dtype=_F32),
            # per-hub vectors in global hub order (source s owns rows [o_s, o_s + w_s)): g = d2, dK3, alpha, feed h,
            # d1 at birth, thin Ut^T
            "G": fnp.empty((Wcap,), dtype=_F32),
            "K3": fnp.empty((Wcap,), dtype=_F32),
            "AL": fnp.empty((Wcap,), dtype=_F32),
            "HH": fnp.empty((Wcap,), dtype=_F32),
            "D1B": fnp.empty((Wcap,), dtype=_F32),
            "UtT": fnp.empty((Wcap, kk), dtype=_F32),
            "D3B": fnp.empty((n,), dtype=_F32),         # d3 at birth of the newest source (star-diagonal kappa4 birth)
            "Bn": fnp.empty((n, n), dtype=_F32),        # B^T rows of the newborn, until its first transport
            # C (n rows) and the newest source's path leg Y^T (w rows): transported with the full W
            "CY": fnp.empty((2 * n + 64, n), dtype=_F32),
            "CYo": fnp.empty((2 * n + 64, n), dtype=_F32),
            "Hb": [fnp.empty((Smax, n, kk), dtype=_F32), fnp.empty((Smax, n, kk), dtype=_F32)],
            "thinw": fnp.empty((Smax, n, kk), dtype=_F32),
            "prod": fnp.empty((Smax, n, n), dtype=_F32),
            "MQ": fnp.empty((2, Smax, n), dtype=_F32),  # feed m = A h and q = rowsum(B d1b A)
            "Dsum": fnp.empty((n, n), dtype=_F32),
            "Dtmp": fnp.empty((n, n), dtype=_F32),
            "Dext": fnp.empty(((n + 1) * (n + 1),), dtype=_F32),   # alive x alive D block + a zero row/column
            "Dtc": fnp.empty((n * n,), dtype=_F32),                  # paired_sum chunk scratch at the alive size
            "sw": _Strassen(n, self.lv, self.chunk) if self.lv > 0 else None,
            "BA": fnp.empty((1, n, n), dtype=_F32),
            "BO": fnp.empty((1, n, n), dtype=_F32),
            "Z": Z,
        }
        if r:
            ws.update({
                "PTb": fnp.empty((r, n), dtype=_F32),               # the basis at its creation (first entry)
                "ZBT": fnp.empty((Wcap, r), dtype=_F32),            # members' coefficients Z^T (global hub rows)
                "ZAT": fnp.empty((Wcap, r), dtype=_F32),
                "ZS": fnp.empty((2 * Wcap + n, r), dtype=_F32),     # [Z_B^T; Z_A^T] of the members, contiguous
                "ZH": fnp.empty((Smax, r, kk), dtype=_F32),         # members' thin-leg coefficients
                "XM": fnp.empty((2 * Wcap + n, n), dtype=_F32),     # members' materialised legs [B^T; A^T]
                "LM": fnp.empty((2 * Wcap + n, n), dtype=_F32),     # members' pairing partners [L1^T; L2^T]
                "GLm": fnp.empty(((2 * Wcap + n) * n,), dtype=_F32),
                "E2": fnp.empty((2 * n + 64, n), dtype=_F32),       # the entrant's [B^T; A^T] for its projection
                "E2o": fnp.empty((2 * n + 64, r), dtype=_F32),
                "Ymr": fnp.empty((n * r,), dtype=_F32),             # compressed pairing L Z^T (alive x r)
                "PTg": fnp.empty((r * n,), dtype=_F32),             # alive columns of P^T
            })
        if self.keep_workspace:
            self._ws = (key, ws)
        return ws

    def predict(self, mlp: MLP, budget: int) -> fnp.ndarray:
        rng = fnp.random.default_rng(mlp.seed)  # probes of the randomized range finder (and of the band's first basis)
        _ = budget
        n = mlp.width
        L = len(mlp.weights)
        kk = max(1, min(self.kk, n))
        ws = self._workspace(n, L, kk)
        sw = ws["sw"]
        XTb, Hb, LT = ws["XT"], ws["Hb"], ws["LT"]
        ETb, Tb = ws["ET"], ws["T"]
        Gw, K3w, ALw, HHw, D1Bw, UtT = ws["G"], ws["K3"], ws["AL"], ws["HH"], ws["D1B"], ws["UtT"]
        thinw, prod, MQ = ws["thinw"], ws["prod"], ws["MQ"]
        BA, BO, Z = ws["BA"], ws["BO"], ws["Z"]
        CY, CYo, Bn = ws["CY"], ws["CYo"], ws["Bn"]
        quantum = 2 ** len(sw.bufs) if sw is not None else 1
        band = self.band
        if band is not None and (band[1] > n // 2 or band[1] % quantum):
            band = None  # the basis must stay well inside the alive rows (and Strassen-divisible): no band on small MLPs
        if band is not None:
            a0b, rb, initb = band
            ZBT, ZAT, ZS, ZH, XM, LM = ws["ZBT"], ws["ZAT"], ws["ZS"], ws["ZH"], ws["XM"], ws["LM"]

        def bprod(a, b, out):  # out[0] = a[0] @ b[0]
            if sw is not None:
                sw.paired(a, b, out)
            else:
                fnp.matmul(a, b, out=out)

        def chunks(R):
            """Split R stacked rows into K equal chunks of c rows (c <= n, a multiple of the Strassen quantum)."""
            K = max(1, -(-R // n))
            c = -(-R // K)
            c = -(-c // quantum) * quantum
            return K, c, K * c

        def rsh(X2, Wr, out2, K, c):  # out2 = X2 @ Wr, the row stack X2 (K c rows) against one shared Wr
            if sw is None:
                fnp.matmul(X2, Wr, out=out2)
            else:
                sw.rshared(X2.reshape((K, c, X2.shape[1])), Wr, out2.reshape((K, c, out2.shape[1])))

        def zero_rows(A, r0, r1):  # A[r0:r1] = 0 (the Strassen padding must be exactly zero)
            if r1 > r0:
                fnp.multiply(Z[:r1 - r0, :A.shape[1]], 1.0, out=A[r0:r1])

        def rowsum_pair(La, Xb, R2, cols, out, tmp):
            """out = La[:R2]^T Xb[:R2] (cols x p) over R2 stacked hub rows, through paired_sum on row chunks."""
            K2, c2, R2p = chunks(R2)
            zero_rows(La, R2, R2p)
            zero_rows(Xb, R2, R2p)
            sw.paired_sum(fnp.swapaxes(La[:R2p].reshape((K2, c2, cols)), 1, 2),
                          Xb[:R2p].reshape((K2, c2, Xb.shape[1])), out, tmp)

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

        def region_readout(BT, AT, v0, v1, ET, T, L1, L2, srcs, H):
            """E, the readout t and the first half of L1, L2 for one aligned (B^T, A^T) region whose hub rows are the
            global hub rows [v0, v1); srcs = [(s, local row, width)]. Returns this region's part of t."""
            gv, k3v, alv = Gw[v0:v1][:, None], K3w[v0:v1][:, None], ALw[v0:v1][:, None]
            # E = A diag(dK3) + 3 B diag(alpha) + 3 H Ut  (transposed; the thin part per source)
            for s, lr, ww in srcs:
                fnp.matmul(UtT[v0 + lr:v0 + lr + ww], fnp.swapaxes(H[s], 0, 1), out=ET[lr:lr + ww])
            fnp.multiply(ET, 3.0, out=ET)
            fnp.multiply(AT, k3v, out=T)
            fnp.add(ET, T, out=ET)
            fnp.multiply(BT, alv, out=T)
            fnp.multiply(T, 3.0, out=T)
            fnp.add(ET, T, out=ET)
            fnp.multiply(AT, gv, out=L1)
            fnp.multiply(L1, BT, out=L1)               # A g B
            fnp.multiply(L1, BT, out=T)                # A g B^2
            tp = 3.0 * fnp.sum(T, axis=0)
            fnp.multiply(AT, AT, out=L2)               # A^2 (temporarily in L2)
            fnp.multiply(L2, ET, out=T)                # E A^2
            tp = tp + fnp.sum(T, axis=0)
            if self.feed:  # P2 sources: m = A h, q = rowsum(B d1b A)
                fnp.multiply(BT, D1Bw[v0:v1][:, None], out=T)
                fnp.multiply(T, AT, out=T)
                for s, lr, ww in srcs:
                    fnp.matmul(HHw[v0 + lr:v0 + lr + ww], AT[lr:lr + ww], out=MQ[0, s])
                    fnp.sum(T[lr:lr + ww], axis=0, out=MQ[1, s])
            return tp

        def region_pairs(BT, AT, v0, v1, ET, T, L1, L2, srcs):
            """The pairing partners L1 = 2 A g B + A^2 alpha, L2 = g B^2 + (1/3) A^2 dK3 + (2/3) E A (+ the feed's
            2 diag(m) B diag(d1b)) for one region, and the thin factors thinw[s] = (A^2 Ut^T)_s."""
            gv, k3v, alv = Gw[v0:v1][:, None], K3w[v0:v1][:, None], ALw[v0:v1][:, None]
            fnp.multiply(L2, alv, out=T)                  # A^2 alpha
            fnp.multiply(L1, 2.0, out=L1)
            fnp.add(L1, T, out=L1)
            for s, lr, ww in srcs:  # thin term (1/3) A^2 ER^T = (A^2 Ut^T) H^T, with A^2 still in L2
                fnp.matmul(fnp.swapaxes(L2[lr:lr + ww], 0, 1), UtT[v0 + lr:v0 + lr + ww], out=thinw[s])
            fnp.multiply(L2, k3v, out=T)                  # A^2 dK3
            fnp.multiply(T, 1.0 / 3.0, out=T)
            fnp.multiply(BT, BT, out=L2)
            fnp.multiply(L2, gv, out=L2)
            fnp.add(L2, T, out=L2)
            fnp.multiply(ET, AT, out=T)
            fnp.multiply(T, 2.0 / 3.0, out=T)
            fnp.add(L2, T, out=L2)
            if self.feed:  # 2 diag(m) B diag(d1b) rides the L2 A^T product
                fnp.multiply(BT, D1Bw[v0:v1][:, None], out=T)
                for s, lr, ww in srcs:
                    fnp.multiply(T[lr:lr + ww], MQ[0, s][None, :], out=T[lr:lr + ww])
                fnp.multiply(T, 2.0, out=T)
                fnp.add(L2, T, out=L2)

        cur = 0
        S = 0
        WB = 0            # hub rows per region = sum of the sources' widths
        blocks = []       # (global hub offset, width) of every source, in source order
        Pn, wn = None, 0  # kept hubs and width of the newest source
        m = 0             # band members: the m oldest sources
        PTsrc = None      # the band basis P^T (r x n) as it stands at the end of the layer (None before the first entry)
        st_rP, st_o0, st_W = 0, 0, 0  # layout of the non-member storage XT[cur]: P^T rows, first non-member offset, rows
        cut_prev = None   # (idx, n_alive_padded, rmap) of the previous layer when its dead rows are dropped
        mu = fnp.zeros(n, dtype=_F32)
        K4post = K22post = None
        dS_carry = None   # E2: the kappa4-level correction of the saturated-row cut, added to the next layer's Ssum
        zn = fnp.zeros(n, dtype=_F32)
        rows = []
        for l, w in enumerate(mlp.weights):
            last = l == L - 1
            Wt = w.T
            mu_pre = Wt @ mu
            t = fnp.zeros(n, dtype=_F32)
            D = None
            mask_cur = cut_cur = None
            fnp.multiply(Wt, 1.0, out=BA[0])
            XT = XTb[cur]
            if S > 0:
                nxt = 1 - cur
                XN = XTb[nxt]
                # (1) C and the newest source's path leg, with the full W (their dead rows are not zero):
                #     [C; Y^T] W = [(Wt C)^T; (Wt Y)^T]
                Rc = n + (wn if self.paths else 0)
                Kc, cc, Rcp = chunks(Rc)
                zero_rows(CY, Rc, Rcp)
                rsh(CY[:Rcp], w, CYo[:Rcp], Kc, cc)
                P2 = CYo[:n]
                if last:
                    var = fnp.maximum(fnp.sum(P2 * w, axis=0), 1e-12)
                    C0 = None
                else:
                    bprod(BA, P2[None], BO)                                  # C_pre = Wt C W
                    C_pre = BO[0]
                    var = fnp.maximum(fnp.diag(C_pre), 1e-12)
                    C0 = C_pre - fnp.diag(fnp.diag(C_pre))
                    if self.deadcut is not None:
                        mask_cur, cut_cur = alive_cut(mu_pre, var)
                # (2) the transport, contracted over the alive rows of the previous layer (dead rows of the legs and of
                #     P are exactly zero): [P^T; B^T of the old non-members; B^T of the newborn; A^T of the old
                #     non-members] W. The storage XT[cur] still holds the rows of sources that entered the band at the
                #     end of the previous layer: they are skipped.
                om = blocks[m][0]                    # global offset of the first non-member
                sk = om - st_o0                      # rows of the storage to skip (the entrant)
                Wo = st_W - sk                       # old non-members' rows per region
                rPn = rb if PTsrc is not None else 0
                R = rPn + 2 * Wo + wn
                K_, c_, Rp = chunks(R)
                if cut_prev is not None:
                    idp, npp = cut_prev[0], cut_prev[1]
                    Wg = ws["Wg"][:npp * n].reshape((npp, n))
                    fnp.take(w, idp, axis=0, out=Wg)
                    GX = ws["GX"][:Rp * npp].reshape((Rp, npp))
                    if rPn:
                        fnp.take(PTsrc, idp, axis=1, out=GX[:rPn])
                    if Wo:
                        fnp.take(XT[st_rP + sk:st_rP + st_W], idp, axis=1, out=GX[rPn:rPn + Wo])
                    fnp.take(Bn[:wn], idp, axis=1, out=GX[rPn + Wo:rPn + Wo + wn])
                    if Wo:
                        fnp.take(XT[st_rP + st_W + sk:st_rP + 2 * st_W], idp, axis=1, out=GX[rPn + Wo + wn:R])
                else:
                    Wg = w
                    GX = ws["GX"][:Rp * n].reshape((Rp, n))
                    if rPn:
                        fnp.multiply(PTsrc, 1.0, out=GX[:rPn])
                    if Wo:
                        fnp.multiply(XT[st_rP + sk:st_rP + st_W], 1.0, out=GX[rPn:rPn + Wo])
                    fnp.multiply(Bn[:wn], 1.0, out=GX[rPn + Wo:rPn + Wo + wn])
                    if Wo:
                        fnp.multiply(XT[st_rP + st_W + sk:st_rP + 2 * st_W], 1.0, out=GX[rPn + Wo + wn:R])
                zero_rows(GX, R, Rp)
                rsh(GX, Wg, XN[:Rp], K_, c_)
                fnp.take(w, Pn, axis=0, out=XN[R:R + wn])     # the newborn's hub leg: Wt I[:, P] -> A^T = W[P, :]
                Wn = Wo + wn                                    # non-members' rows per region now (= WB - om)
                st_rP, st_o0, st_W = rPn, om, Wn
                fnp.matmul(Wt, Hb[cur][m:S], out=Hb[nxt][m:S])
                PT = XN[:rPn] if rPn else None
                if m:  # (3) the band members, materialised: [B^T; A^T] = [Z_B^T; Z_A^T] P^T, and their thin legs P Z_H
                    fnp.multiply(ZBT[:om], 1.0, out=ZS[:om])
                    fnp.multiply(ZAT[:om], 1.0, out=ZS[om:2 * om])
                    Km, cm_, Rmp = chunks(2 * om)
                    zero_rows(ZS, 2 * om, Rmp)
                    rsh(ZS[:Rmp], PT, XM[:Rmp], Km, cm_)
                    fnp.matmul(fnp.swapaxes(PT, 0, 1), ZH[:m], out=Hb[nxt][:m])
                cur = nxt
                XT = XTb[cur]
                H = Hb[cur][:S]
                # (4) readout: the non-member region, then the members'
                BTn, ATn = XT[rPn:rPn + Wn], XT[rPn + Wn:rPn + 2 * Wn]
                L1n, L2n = LT[:Wn], LT[Wn:2 * Wn]
                srcs_n = [(s, blocks[s][0] - om, blocks[s][1]) for s in range(m, S)]
                t = region_readout(BTn, ATn, om, WB, ETb[om:WB], Tb[om:WB], L1n, L2n, srcs_n, H)
                if m:
                    BTm, ATm = XM[:om], XM[om:2 * om]
                    L1m, L2m = LM[:om], LM[om:2 * om]
                    srcs_m = [(s, blocks[s][0], blocks[s][1]) for s in range(m)]
                    t = t + region_readout(BTm, ATm, 0, om, ETb[:om], Tb[:om], L1m, L2m, srcs_m, H)
                if self.feed:
                    mv, qv = MQ[0, :S], MQ[1, :S]
                    t = t + 3.0 * fnp.sum(mv * qv, axis=0)
                if self.paths:  # path births of the newest source (a non-member), first readout
                    ob, ww = blocks[-1][0] - om, blocks[-1][1]
                    XpT = CYo[n:n + ww]
                    t = t + 3.0 * fnp.sum(ATn[ob:ob + ww] * BTn[ob:ob + ww] * XpT, axis=0)
                if not last:
                    region_pairs(BTn, ATn, om, WB, ETb[om:WB], Tb[om:WB], L1n, L2n, srcs_n)
                    if m:
                        region_pairs(BTm, ATm, 0, om, ETb[:om], Tb[:om], L1m, L2m, srcs_m)
                    if self.paths:  # path births of the newest source in D: (X o A) B^T + (X o B) A^T ride L1, L2
                        fnp.add(L1n[ob:ob + ww], XpT * ATn[ob:ob + ww], out=L1n[ob:ob + ww])
                        fnp.add(L2n[ob:ob + ww], XpT * BTn[ob:ob + ww], out=L2n[ob:ob + ww])
                    # (5) the D pairing: non-members dense, D = LT^T XT over their 2 Wn hub rows; members compressed,
                    #     D_m = (LM^T [Z_B^T; Z_A^T]) P^T
                    Dsum, Dtmp = ws["Dsum"], ws["Dtmp"]
                    XNn = XT[rPn:rPn + 2 * Wn]
                    satm = None
                    if self.satrow is not None:
                        # E2: rows of saturated neurons (alpha > satrow, the head of the alpha-descending order) are not
                        # computed; the row block starts at the saturated count rounded DOWN to the Strassen quantum (the
                        # few saturated rows it still holds are zeroed with the mask below)
                        a_now = mu_pre / fnp.sqrt(var)
                        satm = fnp.where(a_now > self.satrow, 1.0, 0.0).astype(_F32)
                        n_sat = int(round(float(fnp.sum(satm))))
                        order_ = fnp.argsort(-a_now)
                        if cut_cur is not None:
                            idc, npc, rmap = cut_cur
                        else:
                            idc, npc = order_, n
                            rmap = fnp.argsort(order_)
                        s0 = min((n_sat // quantum) * quantum, npc - quantum)
                        n_r = npc - s0
                        idr = order_[s0:npc]
                        pos_ = fnp.argsort(order_) - s0
                        rmap_r = fnp.where((pos_ >= 0) & (pos_ < n_r), pos_, n_r)
                    if self.satrow is not None and sw is not None:
                        R2 = 2 * Wn
                        K2, c2, R2p = chunks(R2)
                        GL = ws["GL"][:R2p * n_r].reshape((R2p, n_r))
                        GR = ws["GR"][:R2p * npc].reshape((R2p, npc))
                        fnp.take(LT[:R2], idr, axis=1, out=GL[:R2])
                        fnp.take(XNn, idc, axis=1, out=GR[:R2])
                        zero_rows(GL, R2, R2p)
                        zero_rows(GR, R2, R2p)
                        Dv = ws["Dext"][:(n_r + 1) * (npc + 1)].reshape((n_r + 1, npc + 1))
                        sw.paired_sum(fnp.swapaxes(GL.reshape((K2, c2, n_r)), 1, 2), GR.reshape((K2, c2, npc)),
                                      Dv[:n_r, :npc], ws["Dtc"][:n_r * npc].reshape((n_r, npc)))
                        if m:
                            Rm2 = 2 * om
                            Km2, cm2, Rm2p = chunks(Rm2)
                            GLm = ws["GLm"][:Rm2p * n_r].reshape((Rm2p, n_r))
                            fnp.take(LM[:Rm2], idr, axis=1, out=GLm[:Rm2])
                            Ym = ws["Ymr"][:n_r * rb].reshape((n_r, rb))
                            rowsum_pair(GLm, ZS, Rm2, n_r, Ym, ws["Dtc"][:n_r * rb].reshape((n_r, rb)))
                            PTg = ws["PTg"][:rb * npc].reshape((rb, npc))
                            fnp.take(PT, idc, axis=1, out=PTg)
                            bprod(Ym[None], PTg[None], ws["Dtc"][:n_r * npc].reshape((1, n_r, npc)))
                            fnp.add(Dv[:n_r, :npc], ws["Dtc"][:n_r * npc].reshape((n_r, npc)), out=Dv[:n_r, :npc])
                        fnp.multiply(fnp.zeros(npc + 1, dtype=_F32), 1.0, out=Dv[n_r])
                        fnp.multiply(fnp.zeros(n_r + 1, dtype=_F32), 1.0, out=Dv[:, npc])
                        D = fnp.take(fnp.take(Dv, rmap_r, axis=0), rmap, axis=1)
                    elif self.deadcut is not None and cut_cur is not None and sw is not None:
                        idc, npc, rmap = cut_cur
                        R2 = 2 * Wn
                        K2, c2, R2p = chunks(R2)
                        GL = ws["GL"][:R2p * npc].reshape((R2p, npc))
                        GR = ws["GR"][:R2p * npc].reshape((R2p, npc))
                        fnp.take(LT[:R2], idc, axis=1, out=GL[:R2])
                        fnp.take(XNn, idc, axis=1, out=GR[:R2])
                        zero_rows(GL, R2, R2p)
                        zero_rows(GR, R2, R2p)
                        Dv = ws["Dext"][:(npc + 1) * (npc + 1)].reshape((npc + 1, npc + 1))
                        sw.paired_sum(fnp.swapaxes(GL.reshape((K2, c2, npc)), 1, 2), GR.reshape((K2, c2, npc)),
                                      Dv[:npc, :npc], ws["Dtc"][:npc * npc].reshape((npc, npc)))
                        if m:
                            Rm2 = 2 * om
                            Km2, cm2, Rm2p = chunks(Rm2)
                            GLm = ws["GLm"][:Rm2p * npc].reshape((Rm2p, npc))
                            fnp.take(LM[:Rm2], idc, axis=1, out=GLm[:Rm2])
                            Ym = ws["Ymr"][:npc * rb].reshape((npc, rb))
                            rowsum_pair(GLm, ZS, Rm2, npc, Ym, ws["Dtc"][:npc * rb].reshape((npc, rb)))
                            PTg = ws["PTg"][:rb * npc].reshape((rb, npc))
                            fnp.take(PT, idc, axis=1, out=PTg)
                            bprod(Ym[None], PTg[None], ws["Dtc"][:npc * npc].reshape((1, npc, npc)))
                            fnp.add(Dv[:npc, :npc], ws["Dtc"][:npc * npc].reshape((npc, npc)), out=Dv[:npc, :npc])
                        zrow = fnp.zeros(npc + 1, dtype=_F32)
                        fnp.multiply(zrow, 1.0, out=Dv[npc])
                        fnp.multiply(zrow, 1.0, out=Dv[:, npc])
                        D = fnp.take(fnp.take(Dv, rmap, axis=0), rmap, axis=1)
                    elif sw is None:
                        D = fnp.swapaxes(LT[:2 * Wn], 0, 1) @ XNn
                        if m:
                            D = D + (fnp.swapaxes(LM[:2 * om], 0, 1) @ ZS[:2 * om]) @ PT
                    else:
                        R2 = 2 * Wn
                        K2, c2, R2p = chunks(R2)
                        zero_rows(LT, R2, R2p)
                        zero_rows(XT, rPn + R2, rPn + R2p)
                        sw.paired_sum(fnp.swapaxes(LT[:R2p].reshape((K2, c2, n)), 1, 2),
                                      XT[rPn:rPn + R2p].reshape((K2, c2, n)), Dsum, Dtmp)
                        D = Dsum * 1.0
                        if m:
                            Ym = ws["Ymr"][:n * rb].reshape((n, rb))
                            rowsum_pair(LM, ZS, 2 * om, n, Ym, ws["Dtc"][:n * rb].reshape((n, rb)))
                            bprod(Ym[None], PT[None], ws["Dtmp"][None])
                            D = D + ws["Dtmp"]
                    fnp.matmul(thinw[:S], fnp.swapaxes(H, 1, 2), out=prod[:S])
                    D = D + fnp.sum(prod[:S], axis=0)
                    if self.feed:  # + q m^T summed over sources
                        D = D + fnp.swapaxes(MQ[1, :S], 0, 1) @ MQ[0, :S]
                    if satm is not None:  # dead rows and columns, and the saturated rows, of D are 0
                        mrow = (1.0 - satm) if mask_cur is None else mask_cur * (1.0 - satm)
                        mcol = mask_cur if mask_cur is not None else fnp.ones(n, dtype=_F32)
                        Dfull_parts = (XNn, mcol)
                        D = D * (mrow[:, None] * mcol[None, :])
                    elif self.deadcut is not None and mask_cur is not None:  # dead rows and columns of D are 0
                        D = D * (mask_cur[:, None] * mask_cur[None, :])
                    fnp.fill_diagonal(D, zn)
            else:
                # layer 0: C = I, so C_pre = Wt Wt^T
                if last:
                    var = fnp.maximum(fnp.sum(BA[0] * BA[0], axis=1), 1e-12)
                    C0 = None
                else:
                    bprod(BA, fnp.swapaxes(BA, 1, 2), BO)
                    C_pre = BO[0]
                    var = fnp.maximum(fnp.diag(C_pre), 1e-12)
                    C0 = C_pre - fnp.diag(fnp.diag(C_pre))
            if D is None and not last:
                D = fnp.zeros((n, n), dtype=_F32)
            # ---- kappa4: the levels of the memoryless transport M = Wsq^T X Wsq, X = K22post + diag(K4post), in O(n^2),
            #      with the scale-mixture pattern (K4d = 3 v var^2, K22 = v (var_i var_j + 2 C0^2))
            K4d = K22 = None
            v_cur = v_feed = None
            have4 = K4post is not None
            if have4:
                Wsq = w * w
                u = fnp.sum(Wsq, axis=1)
                Gd = fnp.sum(Wsq * Wsq, axis=1)
                Ku2 = fnp.sum(K4post * u * u)
                Ssum = fnp.sum((K22post @ u) * u) + Ku2                  # sum(M) = u^T X u
                if dS_carry is not None:  # E2: the kappa4 level of the saturated rows, restored (= K22post += dS / ...)
                    Ssum = Ssum + dS_carry
                    dS_carry = None
                trM = (Ssum - Ku2) / n + fnp.sum(K4post * Gd)            # tr(M), mean-field off-diagonal G
                k4m = 3.0 * trM / n - 2.0 * fnp.sum(Gd * K4post) / n      # mean of the transported K4 diagonal
                s4 = var * var
                K4d = s4 * (k4m / fnp.mean(s4))                           # scale-mixture pattern 3 v var_i^2
                v_cur = k4m / (3.0 * fnp.mean(s4))                       # the scale-mixture level v
                v_feed = v_cur
                if self.star and S > 0:  # star-diagonal kappa4 birth of the newest source: 4 d3 A B^3
                    ob, ww = blocks[-1][0] - om, blocks[-1][1]
                    Bn_ = BTn[ob:ob + ww]
                    K4d = K4d + 4.0 * fnp.sum(ws["D3B"][:ww][:, None] * ATn[ob:ob + ww] * (Bn_ * Bn_ * Bn_),
                                              axis=0)
                    v_feed = fnp.mean(K4d) / (3.0 * fnp.mean(s4))
                if not last:
                    pat = var[:, None] * var[None, :] + 2.0 * (C0 * C0)
                    fnp.fill_diagonal(pat, zn)
                    K22 = pat * ((Ssum - trM) / fnp.sum(pat))            # v (var_i var_j + 2 C0_ij^2)
            if self.deadcut is not None and mask_cur is None and not last:
                mask_cur, cut_cur = alive_cut(mu_pre, var)
            o = _nonlin(mu_pre, var, C0, t, D, K4d, K22, need_pairs=not last, want_w12=True, c0d=self.c0d,
                        marg=self.marg, v31=((v_cur * self.k31_scale if self.k31_scale != 1.0 else v_cur)
                                             if (self.k31 and have4) else None),
                        c0k22=self.c0k22)
            mu = o["K1"]
            rows.append(mu)
            if last:
                break
            if self.satrow is not None and S > 0:
                # E2 level restoration: dS = sum_{i sat} u_i cf22_i (D_full v)_i, v = u o (cf12 - 2 P1 cf11), with u the
                # row sums of W_{l+1} o W_{l+1} and D_full the dead-masked, zero-diagonal D BEFORE the row cut, whose rows
                # are assembled from the (still ungated) legs by matrix-vector products
                XNn_, mcol_ = Dfull_parts
                wn1 = mlp.weights[l + 1]
                u1 = fnp.sum(wn1 * wn1, axis=1)
                vv = u1 * (o["cf12"] - 2.0 * o["K1"] * o["cf11"]) * mcol_
                # (row vector) @ (contiguous matrix) everywhere: a transposed view of a 100 MB stack would be copied
                # outside flopscope's timed region, i.e. in residual time
                z = (XNn_ @ vv) @ LT[:2 * Wn]                       # = LT^T (XT v)
                tq = ws["GR"][:2 * Wn * n].reshape((2 * Wn, n))   # a free buffer: no 100 MB temporary either
                fnp.multiply(LT[:2 * Wn], XNn_, out=tq)
                dg = fnp.sum(tq, axis=0)
                if m:
                    z = z + (ZS[:2 * om] @ (PT @ vv)) @ LM[:2 * om]
                    tq = ws["GLm"][:2 * om * n].reshape((2 * om, n))
                    fnp.multiply(LM[:2 * om], XM[:2 * om], out=tq)
                    dg = dg + fnp.sum(tq, axis=0)
                hv = fnp.matmul(vv[None, None, :], H)                # (S, 1, kk) = (H_s^T v)^T
                z = z + fnp.sum(fnp.matmul(thinw[:S], fnp.swapaxes(hv, 1, 2)), axis=0)[:, 0]
                dg = dg + fnp.sum(fnp.sum(thinw[:S] * H, axis=2), axis=0)
                if self.feed:
                    z = z + (MQ[0, :S] @ vv) @ MQ[1, :S]
                    dg = dg + fnp.sum(MQ[1, :S] * MQ[0, :S], axis=0)
                dS_carry = fnp.sum(satm * u1 * o["cf22"] * (z - dg * vv))
            fnp.add(o["K11"], fnp.diag(o["K2"]), out=CY[:n])            # C of the next layer
            K4post, K22post = o["K4"], o["K22"]
            d1, d2 = o["d1"], o["d2"]
            if self.deadcut is not None:  # dead neurons gated to exactly 0 (their leg rows are then dropped)
                d1 = d1 * mask_cur
                cut_prev = cut_cur
            if S > 0:  # gate the stored legs (and the band basis) of this layer
                fnp.multiply(XT[:st_rP + 2 * st_W], d1[None, :], out=XT[:st_rP + 2 * st_W])
                fnp.multiply(Hb[cur][m:S], d1[None, :, None], out=Hb[cur][m:S])
                if st_rP:
                    PTsrc = XT[:st_rP]
            # ---- band entry ("pre"): the source that reaches age a0 at the next layer is projected onto the gated
            #      basis through a QR of P, Z = R^-1 Q^T X, for its A, B and thin H; it is never transported densely again
            if band is not None and S > 0 and 0 <= l + 1 - a0b == m and m < S:
                oe, we = blocks[m]
                re = oe - st_o0
                bt = XT[st_rP + re:st_rP + re + we]
                at = XT[st_rP + st_W + re:st_rP + st_W + re + we]
                if PTsrc is None:  # the first entrant defines the basis
                    PTb = ws["PTb"]
                    if initb == "rr":  # dominant r-subspace of its [A B] (range finder, one power step)
                        XBT = ws["E2"][:2 * we]
                        fnp.multiply(at, 1.0, out=XBT[:we])
                        fnp.multiply(bt, 1.0, out=XBT[we:2 * we])
                        Omb = rng.standard_normal((2 * we, rb), dtype=_F32)
                        Y0 = fnp.swapaxes(XBT, 0, 1) @ Omb
                        Y0 = fnp.swapaxes(XBT, 0, 1) @ (XBT @ Y0)
                    else:  # its first r hub columns of A
                        Y0 = fnp.swapaxes(at[:rb], 0, 1)
                    Q0, _r0 = fnp.linalg.qr(Y0)
                    fnp.multiply(fnp.swapaxes(Q0, 0, 1), 1.0, out=PTb)
                    PTsrc = PTb
                Qb, Rb = fnp.linalg.qr(fnp.swapaxes(PTsrc, 0, 1))
                E2 = ws["E2"]
                fnp.multiply(bt, 1.0, out=E2[:we])
                fnp.multiply(at, 1.0, out=E2[we:2 * we])
                Ke, ce, Rep = chunks(2 * we)
                zero_rows(E2, 2 * we, Rep)
                E2o = ws["E2o"]
                rsh(E2[:Rep], Qb, E2o[:Rep], Ke, ce)                       # [B^T; A^T] Q
                Zb = fnp.linalg.solve(Rb, fnp.swapaxes(E2o[:2 * we], 0, 1))   # Z = R^-1 Q^T [B A]
                fnp.multiply(fnp.swapaxes(Zb[:, :we], 0, 1), 1.0, out=ZBT[oe:oe + we])
                fnp.multiply(fnp.swapaxes(Zb[:, we:], 0, 1), 1.0, out=ZAT[oe:oe + we])
                fnp.multiply(fnp.linalg.solve(Rb, fnp.swapaxes(Qb, 0, 1) @ Hb[cur][m]), 1.0, out=ZH[m])
                m += 1
            # ---- birth of source S; its hubs are the parents the column cut keeps (alpha_p >= colcut)
            if self.colcut is None:
                P = fnp.arange(n)
                wb = n
            else:
                ab = mu_pre / fnp.sqrt(var)
                wb = max(1, int(round(float(fnp.sum(fnp.where(ab >= self.colcut, 1.0, 0.0))))))
                P = fnp.argsort(-ab)[:wb]
            U = d1[:, None] * C0                                          # B = diag(d1) C0
            fnp.take(fnp.swapaxes(U, 0, 1), P, axis=0, out=Bn[:wb])       # its B^T rows, kept hubs only
            d11 = d1 * d1
            dK3 = o["K3"] - d11 * d1 * t
            alpha = o["w12"] - 2.0 * o["K1"] * d1
            dK21 = o["K21"] - (d11[:, None] * d1[None, :]) * D - (d11[:, None] * d2[None, :]) * (C0 * C0)
            if self.paths:  # path births (augmented term set): leg Y for the next transport, (2,1) slice Y o U removed
                d3 = o["d3"]
                Yp = ((d1[:, None] * D.T) * d3[None, :] + (d2[:, None] * D) * d2[None, :]
                      + (d2[:, None] * (C0 * C0)) * d3[None, :])
                if K22 is not None:
                    Yp = Yp + ((0.5 * d2)[:, None] * K22) * d3[None, :]
                fnp.fill_diagonal(Yp, zn)
                dK21 = dK21 - Yp * U
                fnp.take(fnp.swapaxes(Yp, 0, 1), P, axis=0, out=CY[n:n + wb])
            if self.star:
                fnp.take(o["d3"], P, out=ws["D3B"][:wb])
            R_ = dK21 - (alpha[:, None] * d1[None, :]) * C0
            fnp.fill_diagonal(R_, zn)
            Om = rng.standard_normal((n, kk), dtype=_F32)
            Yr = R_ @ Om
            Yr = R_ @ (R_.T @ Yr)
            Qr, _r = fnp.linalg.qr(Yr)
            fnp.multiply(R_.T @ Qr, 1.0, out=Hb[cur][S])                 # H = R^T Q  (E_R = 3 H Ut)
            fnp.take(Qr, P, axis=0, out=UtT[WB:WB + wb])                  # Ut^T = Q, kept hubs only
            fnp.take(d2, P, out=Gw[WB:WB + wb])
            fnp.take(dK3, P, out=K3w[WB:WB + wb])
            if self.feed and have4:
                h = (0.5 * v_feed) * (d2 * var)
                fnp.take(h, P, out=HHw[WB:WB + wb])
                alpha = alpha - 2.0 * (h * d1)                             # R unchanged, P2's (2,1) slice into alpha
            elif self.feed:
                fnp.multiply(Gw[WB:WB + wb], 0.0, out=HHw[WB:WB + wb])
            if self.feed:
                fnp.take(d1, P, out=D1Bw[WB:WB + wb])
            fnp.take(alpha, P, out=ALw[WB:WB + wb])
            blocks.append((WB, wb))
            WB += wb
            Pn, wn = P, wb
            S += 1
        return fnp.stack(rows, axis=0)
