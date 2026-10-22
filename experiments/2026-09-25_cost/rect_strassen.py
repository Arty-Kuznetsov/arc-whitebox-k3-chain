"""Rectangular generalisation of the k3h/k3i workspace Strassen (_Strassen in src/estimator_k3i.py).
Blocks may be (m x k) @ (k x p) with m, k, p <= n and all divisible by 2**levels; the per-level buffers of the square
version (sized for n/2**d) are used through views [..., :a, :b], so memory and the number of calls are unchanged.
  shared(W, X, out):         out[i] = W @ X[i]          W (m x k), X[i] (k x p)   -- leg transport over alive rows
  paired_sum(Xa, Xb, out):   out = sum_i Xa[i] @ Xb[i]  Xa[i] (m x k), Xb[i] (k x p) -- D pairings, alive x alive"""
import flopscope.numpy as fnp
_F32 = fnp.float32


class RectStrassen:
    def __init__(self, n, levels, chunk):
        self.levels, self.chunk = levels, chunk
        self.bufs, self.regions = [], []
        m, G = n, 1
        for _ in range(levels):
            if m % 2 or m < 128:
                break
            m //= 2
            G *= 7
            half = G * chunk * m * m
            R = fnp.empty((2 * half,), dtype=_F32)
            self.regions.append(R)
            self.bufs.append((R[:half].reshape((G, chunk, m, m)), R[half:].reshape((G, chunk, m, m)),
                              fnp.empty((G, 1, m, m), dtype=_F32)))
        L = len(self.bufs)
        self.leaf_tmp = None
        if L >= 2:
            Gl, ml = 7 ** L, n // 2 ** L
            self.leaf_tmp = self.regions[L - 2][:Gl * chunk * ml * ml].reshape((Gl, chunk, ml, ml))
        elif L == 1:
            self.leaf_tmp = fnp.empty((7, chunk, n // 2, n // 2), dtype=_F32)

    @staticmethod
    def _combos(A, B, LA, LB, G):
        hm, hk = A.shape[2] // 2, A.shape[3] // 2
        hp = B.shape[3] // 2
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
        """out = A @ B, A (G, 1, m, k), B (G, K, k, p), out (G, K, m, p)."""
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
        k = X.shape[0]
        for c in range(0, k, self.chunk):
            e = min(c + self.chunk, k)
            self._rec(W[None, None], X[None, c:e], out[None, c:e], 0)

    def paired_sum(self, Xa, Xb, out, tmp):
        k = Xa.shape[0]
        for c in range(0, k, self.chunk):
            e = min(c + self.chunk, k)
            tgt = out if c == 0 else tmp
            self._rec_sum(Xa[None, c:e], Xb[None, c:e], tgt[None], 0)
            if c > 0:
                fnp.add(out, tmp, out=out)
