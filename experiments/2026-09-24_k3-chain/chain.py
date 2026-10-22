"""Our own implementation of the SIMPLE k=3 cumulant chain (Wu et al. 2605.05179 / ARC's mlp_kprop), in the
"source" representation, with switches for the cost/accuracy experiments.

Every ReLU layer m gives birth to one source: dense legs A = T (transport since birth; identity at birth),
B = T U with U = diag(d1) C0 (C0 = pre-activation covariance with zeroed diagonal), E = T V with V the
correction that makes the repeated-index slices of the post-activation kappa3 exact, and the hub vector g = d2.
The source's kappa3 is Sym(B (x) 3A (x) B diag(g)) + Sym(E (x) A (x) A). Carried sources pass through a ReLU
by their leading term (every leg gated by d1). kappa4 is tracked, as in the reference SIMPLE kind, as one
radial scalar a4 per layer: kappa4 = a4 * Sym(g (x) g) with metric g = 2 I after a linear step.

The nonlinear step for the dense slices is the explicit formula set verified against mlp_kprop to 1e-15
(step_check.py). Options (all default to the faithful reference algorithm):
    window      keep only the newest `window` sources (None = all)
    k4          track the radial kappa4 scalar
    corr        'full' | 'diag' | 'none': correction leg E (exact repeated slices / only the (3,) part / none)
Challenge convention: y = ReLU(W^T x), W[l] is (n_in, n_out).
"""
import math

import torch


def _Phi(a):
    return 0.5 * (1 + torch.erf(a / math.sqrt(2)))


def _phi(a):
    return torch.exp(-0.5 * a * a) / math.sqrt(2 * math.pi)


def relu_wick(mu, var):
    """w[(k, p)] = E[d^k ReLU(Z)^p], Z ~ N(mu, var), k = 0..4, p = 1..4 (closed forms)."""
    s = var.clamp_min(1e-30).sqrt()
    a = mu / s
    Ph, ph = _Phi(a), _phi(a)
    m1 = mu * Ph + s * ph                                   # E[Z; Z>0]
    m2 = (mu * mu + var) * Ph + mu * s * ph                 # E[Z^2; Z>0]
    m3 = (mu ** 3 + 3 * mu * var) * Ph + (mu * mu + 2 * var) * s * ph
    m4 = (mu ** 4 + 6 * mu * mu * var + 3 * var * var) * Ph + (mu ** 3 + 5 * mu * var) * s * ph
    d = {0: None, 1: Ph, 2: ph / s, 3: -a * ph / (s * s), 4: (a * a - 1) * ph / s ** 3,
         5: -(a ** 3 - 3 * a) * ph / s ** 4, 6: (a ** 4 - 6 * a * a + 3) * ph / s ** 5,
         7: -(a ** 5 - 10 * a ** 3 + 15 * a) * ph / s ** 6}   # k = 7 (cloud #10 marg3): new keys only
    w = {(0, 1): m1, (0, 2): m2, (0, 3): m3, (0, 4): m4}
    for k in range(1, 8):
        w[(k, 1)] = d[k]
    # d^k (ReLU^p) = p!/(p-k)! Z^(p-k) 1{Z>0} for k < p, and p! d^(k-p+1) ReLU for k >= p
    mom = {1: m1, 2: m2, 3: m3}
    for p in (2, 3, 4):
        for k in range(1, 8):
            if k < p:
                w[(k, p)] = math.factorial(p) / math.factorial(p - k) * mom[p - k]
            else:
                w[(k, p)] = math.factorial(p) * d[k - p + 1]
    return w, Ph, ph / s


def nonlin_slices(mu, C, t, D, K4d, K22, edge2=False, aug=False, tet=2, q22x=0, c0d=False, K31=None, marg=False, k31q=None, pairx=None, margtail=None, tailr=None, pairnew=None, marg3=False, k2pair=False):
    """Dense part of the nonlinear step (verified vs mlp_kprop). Returns post-activation
    K1 (mean), K2 (variance), K11 (off-diag cov), K3 = k3[iii], K21 = k3[iij], K4 = k4[iiii], K22 = k4[iijj]."""
    var = torch.diagonal(C)
    w, d1, d2 = relu_wick(mu, var)
    C0 = C - torch.diag(var)
    P = {p: w[(0, p)] + w[(3, p)] * t / 6 + w[(4, p)] * K4d / 24 for p in range(1, 5)}
    TR = _tailr_rules(tailr, mu / var.clamp_min(1e-30).sqrt()) if tailr else None

    def tkeep(term):  # tailr: product of the keep vectors of the rules for `term` (None when there is none)
        k_ = None
        for _, keep_, _, _ in (TR or {}).get(term, ()):
            k_ = keep_ if k_ is None else k_ * keep_
        return k_

    if TR is not None:  # tailr readout rules: the Edgeworth correction of the power moments dropped in the tail
        for p_ in range(1, 5):
            kp_ = tkeep("P1" if p_ == 1 else "P2" if p_ == 2 else "P34")
            if kp_ is not None:
                P[p_] = w[(0, p_)] + (w[(3, p_)] * t / 6 + w[(4, p_)] * K4d / 24) * kp_
    if edge2:  # second-order Edgeworth: the disconnected kappa3 x kappa3 term on a single index
        P = {p: P[p] + w[(6, p)] * t * t / 72 for p in range(1, 5)}

    tailkeep = None
    if margtail is not None:  # port battery side probe: no marginal Edgeworth correction in the tails |alpha| >= margtail
        tailkeep = ((mu / var.clamp_min(1e-30).sqrt()).abs() < margtail).to(mu.dtype)

    def coef(k, p, qq):
        c = w[(k, p)]
        if marg and k <= 2 and (marg is True or qq in str(marg)):
            # marginal Edgeworth correction of the pair Wick coefficient (first order in kappa3/kappa4 diagonals)
            dc = w[(k + 3, p)] * t / 6 + (w[(k + 4, p)] * K4d / 24 if K4d is not None else 0.0)
            if TR is not None and f"marg{k}" in TR:  # tailr "gauss": the tail coefficient loses its correction
                for _, keep_, _, qs_ in TR[f"marg{k}"]:  # (q = "all" or the power cumulant, e.g. "21")
                    if qs_ == "all" or qs_ == qq:
                        dc = dc * keep_
            c = c + (dc * tailkeep if tailkeep is not None else dc)
        if marg3 and k == 3:  # cloud #10: the k = 3 Wick coefficient's own marginal Edgeworth correction
            c = c + w[(6, p)] * t / 6 + (w[(7, p)] * K4d / 24 if K4d is not None else 0.0)
        return c

    def Q(p, q):
        a = lambda k: coef(k, p, f"{p}{q}")[:, None]
        b = lambda k: coef(k, q, f"{p}{q}")[None, :]

        def X(v, term, dbl):
            # tailr pair-term rule: drop the term's entries whose index is in the tail. dbl = where the REPEATED index
            # of the slice sits in this term ("row"/"col"; axis "dbl" cuts it, "sgl" the other one); None = a
            # symmetric term, cut when either index is in the tail. D and D^T (etc.) share one rule, so Q11/Q22 stay
            # symmetric. q = "all" or the one power cumulant the rule acts in.
            for axis, keep_, mode, qs in (TR or {}).get(term, ()):
                if mode != "zero" or (qs != "all" and qs != f"{p}{q}"):
                    continue
                if dbl is None:
                    v = v * (keep_[:, None] * keep_[None, :])
                elif (axis == "dbl") == (dbl == "row"):
                    v = v * keep_[:, None]
                else:
                    v = v * keep_[None, :]
            return v

        M = (X(a(1) * b(1) * C0, "qC0", None) + X(0.5 * a(1) * b(2) * D.T, "qD", "col")
             + X(0.5 * a(2) * b(1) * D, "qD", "row") + X(0.5 * a(2) * b(2) * C0 * C0, "qC0sq", None))
        if K22 is not None:
            M = M + X(0.25 * a(2) * b(2) * K22, "qK22", None)
        if K31 is not None and (k31q is None or f"{p}{q}" in k31q):  # kappa4 (3,1) slice K31[i, j] = kappa4[iiij]
            M = M + (1.0 / 6.0) * (X(a(3) * b(1) * K31, "qK31", "row") + X(a(1) * b(3) * K31.T, "qK31", "col"))
        if c0d and (c0d is True or f"{p}{q}" in str(c0d)):  # edge x kappa3 (2,1)/(1,2) slice; c0d=True or e.g. "22"
            M = M + 0.5 * (X(a(2) * b(3) * D.T, "qC0D", "col") + X(a(3) * b(2) * D, "qC0D", "row")) * C0
        if pairx and "k22" in pairx and K22 is not None:  # AUGMENT dense pair diagram: edge x kappa4 (2,2)
            M = M + X(0.25 * a(3) * b(3) * C0 * K22, "qC0K22", None)
        if pairnew:  # cloud #10: next-order pair diagrams absent from the expansion. From the bivariate cumulant
            # exponential E[f g] = E_marg[exp(sum_ab k_ab d_x^a d_y^b / (a! b!)) f g]: k_11 = C0, k_21 = D, k_12 = D^T,
            # k_31 = K31, k_22 = K22; a product of cumulants carries a(sum of x-orders) b(sum of y-orders)
            if "c0k31" in pairnew and K31 is not None:   # k11 k31 / 6
                M = M + (1.0 / 6.0) * (a(4) * b(2) * C0 * K31 + a(2) * b(4) * C0 * K31.T)
            if "d2" in pairnew:                          # k21^2 / 8
                M = M + 0.125 * (a(4) * b(2) * D * D + a(2) * b(4) * D.T * D.T)
            if "ddt" in pairnew:                         # k21 k12 / 4
                M = M + 0.25 * a(3) * b(3) * D * D.T
            if "k22sq" in pairnew and K22 is not None:   # k22^2 / 32
                M = M + (1.0 / 32.0) * a(4) * b(4) * K22 * K22
            if "dk22" in pairnew and K22 is not None:    # k21 k22 / 8
                M = M + 0.125 * (a(4) * b(3) * D * K22 + a(3) * b(4) * D.T * K22)
            if "c0sqd" in pairnew:                       # k11^2 k21 / 4
                M = M + 0.25 * (a(4) * b(3) * C0 * C0 * D + a(3) * b(4) * C0 * C0 * D.T)
            if "c0sqk22" in pairnew and K22 is not None:  # k11^2 k22 / 8
                M = M + 0.125 * a(4) * b(4) * C0 * C0 * K22
        for r in range(3, tet + 1):  # higher Gaussian (tetrachoric) terms of the pair covariance: w_r w_r C0^r / r!
            M = M + X((1.0 / math.factorial(r)) * a(r) * b(r) * C0 ** r, f"qC0^{r}", None)
        if aug:  # the O(n^2) dense diagrams of the AUGMENT term set (no kappa4 (3,1) slice: not tracked)
            M = M + (1.0 / 6.0) * a(3) * b(3) * C0 ** 3                                  # triple edge
            M = M + (1.0 / 6.0) * (a(1) * b(4) * t[None, :] + a(4) * b(1) * t[:, None]) * C0  # kappa3 diag x edge
            M = M + 0.5 * (a(2) * b(3) * D.T + a(3) * b(2) * D) * C0                       # edge x kappa3 (1,2)
            M = M + (1.0 / 24.0) * (a(1) * b(5) * K4d[None, :] + a(5) * b(1) * K4d[:, None]) * C0  # kappa4 diag x edge
            if K22 is not None:
                M = M + 0.25 * a(3) * b(3) * C0 * K22                                       # edge x kappa4 (2,2)
        return M

    Q11, Q21, Q22 = Q(1, 1), Q(2, 1), Q(2, 2)
    if q22x:
        # second-order (two kappa3 blocks) diagrams of the (2,2) power cumulant, kappa4 channel only
        a_ = lambda k: w[(k, 2)][:, None]
        b_ = lambda k: w[(k, 2)][None, :]
        X = 0.25 * a_(3) * b_(3) * D * D.T + 0.125 * (a_(4) * b_(2) * D * D + a_(2) * b_(4) * D.T * D.T)
        if q22x >= 2:  # + diagonal-kappa3 x (2,1)/(1,2) blocks
            X = X + (1.0 / 12.0) * (a_(4) * b_(2) * t[:, None] * D.T + a_(2) * b_(4) * t[None, :] * D)
        Q22 = Q22 + X
    P1, P2, P3, P4 = P[1], P[2], P[3], P[4]
    K3 = P3 - 3 * P2 * P1 + 2 * P1 ** 3
    K4 = P4 - 4 * P3 * P1 - 3 * P2 * P2 + 12 * P2 * P1 * P1 - 6 * P1 ** 4
    K21 = Q21 - 2 * P1[:, None] * Q11
    K22o = Q22 - 2 * P1[None, :] * K21 - 2 * P1[:, None] * K21.T - 2 * Q11 * Q11 - 4 * P1[:, None] * P1[None, :] * Q11
    for M in (Q11, K21, K22o):
        M.fill_diagonal_(0.0)
    # marginally corrected gating factors (E[f'], E[f''] of ReLU and E[d ReLU^2] under the Edgeworth marginal)
    k4c = K4d if K4d is not None else torch.zeros_like(t)
    d1c = w[(1, 1)] + w[(4, 1)] * t / 6 + w[(5, 1)] * k4c / 24
    d2c = w[(2, 1)] + w[(5, 1)] * t / 6 + w[(6, 1)] * k4c / 24
    w12c = w[(1, 2)] + w[(4, 2)] * t / 6 + w[(5, 2)] * k4c / 24
    K2pair = None
    if k2pair:  # cloud #10 consistency audit: Var f(x) as the pair expansion's own diagonal, sum_k coef(k,1)^2 var^k / k!
        K2pair = sum(coef(k, 1, "11") ** 2 * var ** k / math.factorial(k) for k in range(1, tet + 1))
    return dict(K1=P1, K2=P2 - P1 * P1, K2pair=K2pair, K11=Q11, K3=K3, K21=K21, K4=K4, K22=K22o, d1=d1, d2=d2, C0=C0, d3=w[(3, 1)], w12=w[(1, 2)], K22in=K22,
                d1c=d1c, d2c=d2c, w12c=w12c)


def _tailr_rules(tailr, alpha):
    """tailr (experiments/2026-09-30_tail-accuracy): rules (term, side, axis, theta[, mode[, q]]) -> {term: [(axis, keep,
    mode, q)]}, keep = 1 outside the tail. side "dead": alpha < -theta, "sat": alpha > theta, "both": |alpha| > theta,
    with alpha the chain's own mu / sigma of the layer the term belongs to. mode "zero" (the tail part of the term is
    dropped) or "gauss" (marg rules: the tail coefficient keeps only its Gaussian value). q: "all" or "11"/"21"/"22"."""
    out = {}
    for r in tailr:
        term, side, axis, th = r[0], r[1], r[2], float(r[3])
        mode = r[4] if len(r) > 4 else "zero"
        q = r[5] if len(r) > 5 else "all"
        tail = (alpha < -th) if side == "dead" else (alpha > th) if side == "sat" else (alpha.abs() > th)
        out.setdefault(term, []).append((axis, (~tail).to(alpha.dtype), mode, q))
    return out


def _tail_transform(t, D, t_old, D_old, tailx, n):
    """experiments/2026-09-26_old-tail: transform the OLD TAIL's separately accumulated (t_old, D_old), then add.
    Measures how accurate a merged tail would have to be, before any merge is built.
      exact  - add unchanged (self-check: reproduces the base to float64 rounding; only the sum order moves)
      dropD  - the tail feeds t only          (cross-checked against d21_ages = a0-1)
      dropT  - the tail feeds D only
      rankD  - keep a rank-`rank` truncation of the tail's D
      diagD  - keep only diag(D_old)  (NOTE: read_slices zeroes D's diagonal, so this degenerates to dropD)
      scaleT - t_old *= 1 + eps                     (COHERENT error: every neuron the same way)
      noiseT - t_old *= 1 + eps N(0,1) per neuron, redrawn per layer, seeded by `seed`  (INCOHERENT)
      scaleTD / noiseTD - the same, applied to BOTH t_old and D_old: what a real merge would do, since a
               merge errs in every contraction the tail feeds, not only in the diagonal readout
    NEGATIVE 57 measured that the readout discounts an incoherent error by ~sqrt(n) and amplifies a coherent
    one, so scaleT and noiseT at the same eps bracket what a merge's error is allowed to look like."""
    mode = tailx["mode"]
    layer = int(tailx.get("_layer", 0))
    tailx["_layer"] = layer + 1
    if mode == "exact":
        pass
    elif mode == "dropD":
        D_old = torch.zeros_like(D_old)
    elif mode == "dropT":
        t_old = torch.zeros_like(t_old)
    elif mode == "diagD":
        D_old = torch.diag(torch.diagonal(D_old))
    elif mode == "rankD":
        r = int(tailx["rank"])
        U_, S_, V_ = torch.linalg.svd(D_old)
        D_old = (U_[:, :r] * S_[:r]) @ V_[:r]
    elif mode == "scaleT":
        t_old = t_old * (1.0 + tailx["eps"])
    elif mode == "scaleTD":
        t_old = t_old * (1.0 + tailx["eps"])
        D_old = D_old * (1.0 + tailx["eps"])
    elif mode == "noiseT":
        g_ = torch.Generator().manual_seed(int(tailx["seed"]) * 1000003 + layer)
        t_old = t_old * (1.0 + tailx["eps"] * torch.randn(n, generator=g_, dtype=t_old.dtype))
    elif mode == "noiseTD":
        g_ = torch.Generator().manual_seed(int(tailx["seed"]) * 1000003 + layer)
        t_old = t_old * (1.0 + tailx["eps"] * torch.randn(n, generator=g_, dtype=t_old.dtype))
        D_old = D_old * (1.0 + tailx["eps"] * torch.randn(n, n, generator=g_, dtype=D_old.dtype))
    else:
        raise ValueError(f"unknown tailx mode {mode!r}")
    return t + t_old, D + D_old


_BAND_KEYS = ("A", "B", "E", "ER")


def _band_enter(cfg, st, se, P, P64, l, rec):
    """Project one source's legs onto the band basis P (initialising P from its A if there is none yet)."""
    r = int(cfg["r"])
    dt = cfg.get("dtype", torch.float32)
    proj = cfg.get("proj", "ne")
    shadow = bool(cfg.get("shadow", False))
    if P is None:  # first entry: the basis is orth(first r columns of the entrant's A), as in E1
        if cfg.get("init", "cols") == "rr":  # or the entrant's dominant r-subspace of [A B] (range finder, 1 power step)
            g_ = torch.Generator().manual_seed(9100 + l)
            XB = torch.cat([se["A"], se["B"]], dim=1)
            Y0 = XB @ torch.randn(XB.shape[1], r, generator=g_)
            Y0 = XB @ (XB.T @ Y0)
            P, _ = torch.linalg.qr(Y0.to(dt))
            if shadow:
                P64, _ = torch.linalg.qr(Y0)
        else:
            P, _ = torch.linalg.qr(se["A"][:, :r].to(dt))
            if shadow:
                P64, _ = torch.linalg.qr(se["A"][:, :r])
        st["l0"] = l
        rec["init"] = True
    if proj == "ne":
        G = P.T @ P
    else:
        Qp, Rp = torch.linalg.qr(P)
    if shadow:
        G64 = P64.T @ P64
    for key in _BAND_KEYS:
        X = se.get(key)
        if X is None:
            continue
        if shadow:
            se["Y" + key] = torch.linalg.solve(G64, P64.T @ X)
        X = X.to(dt)
        if proj == "ne":
            Z = torch.linalg.solve(G, P.T @ X)
        else:
            Z = torch.linalg.solve(Rp, Qp.T @ X)
        se["Z" + key] = Z
    se["bz"] = True
    return P, P64


def _band_enter_pre(cfg, st, sources, l):
    """entry="pre": at the END of layer l (after the gating), the source that reaches age a0 at layer l+1 is projected
    onto the gated basis, so its dense transport at age a0 is saved (it moves through P from then on)."""
    a0 = int(cfg["a0"])
    ent = [s_ for s_ in sources if l + 1 - s_["born"] == a0 and not s_.get("bz")]
    if not ent:
        return
    rec = {"l": l + 0.5, "pre_entry": True}
    P, P64 = _band_enter(cfg, st, ent[0], st.get("P"), st.get("P64"), l, rec)
    st["P"] = P
    if cfg.get("shadow", False):
        st["P64"] = P64
    cfg.setdefault("log", []).append(rec)


def _band_step(cfg, st, sources, l, Wt):
    """experiments/2026-09-29_port-battery, task 1: the TRANSPORTED BAND in coefficient form (production mirror).
    Sources of age >= a0 live as legs = P Z: ONE shared basis P (n x r), transported like a leg (gated by d1 at
    the end of every layer, Wt @ P at the next), and FIXED per-leg coefficients Z (r x n), because
    Wt diag(d1) (P Z) = (Wt diag(d1) P) Z. No per-layer QR: P is re-orthonormalised only every k layers counted
    from the first entry (k=None: never after the initial QR), and only then are the coefficients rotated,
    Z <- R Z (the r^2 n per band leg that decides the cost). The entrant (the source reaching age a0 at this
    layer; every layer >= a0 has exactly one) is projected ONCE by least squares, on the current, possibly
    un-orthonormalised P:
        proj="ne": Z = (P^T P)^-1 P^T X      (normal equations, one r x r LU solve -- fnp.linalg.solve)
        proj="qr": P = Q R (not kept), Z = R^-1 Q^T X   (a QR of P for the projection only; members not rotated)
    The first basis is orth(first r columns of the first entrant's A), exactly as ssp=(a0, r, 0, 0) (E1).
    All band arithmetic (transport of P, QR, rotation, projection, materialisation P Z) runs in cfg["dtype"]
    (float32 = production); the materialised legs are cast back to float64 for the rest of the chain.
    With dtype=float64 and k=1 this is E1's ssp=(a0, r, 0, 0) in exact arithmetic (the anchor)."""
    a0, r = int(cfg["a0"]), int(cfg["r"])
    k = cfg.get("k", 1)
    dt = cfg.get("dtype", torch.float32)
    proj = cfg.get("proj", "ne")
    shadow = bool(cfg.get("shadow", False))   # diagnostic: the same band in float64, to measure the leg error
    log = cfg.setdefault("log", [])
    P = st.get("P")
    P64 = st.get("P64")
    rec = {"l": l, "members": sum(1 for s_ in sources if s_.get("bz"))}
    if P is not None:
        P = Wt.to(dt) @ P
        if shadow:
            P64 = Wt @ P64
        if cfg.get("cond", True):
            sv = torch.linalg.svdvals(P.double())
            rec["cond"] = float(sv[0] / sv[-1])
        if k is not None and (l - st["l0"]) % int(k) == 0:   # lazy re-orthonormalisation + coefficient rotation
            Q, R = torch.linalg.qr(P)
            P = Q
            if shadow:
                Q64, R64 = torch.linalg.qr(P64)
                P64 = Q64
            for s_ in sources:
                if s_.get("bz"):
                    for key in _BAND_KEYS:
                        if s_.get("Z" + key) is not None:
                            s_["Z" + key] = R @ s_["Z" + key]
                            if shadow:
                                s_["Y" + key] = R64 @ s_["Y" + key]
            rec["qr"] = True
    if cfg.get("entry", "post") == "post":
        ent = [s_ for s_ in sources if l - s_["born"] == a0 and not s_.get("bz")]
        if ent:
            P, P64 = _band_enter(cfg, st, ent[0], P, P64, l, rec)
    st["P"] = P
    if shadow:
        st["P64"] = P64
    err = 0.0
    for s_ in sources:  # materialise every member's legs for the readout and the pairings
        if s_.get("bz"):
            for key in _BAND_KEYS:
                if s_.get("Z" + key) is not None:
                    s_[key] = (P @ s_["Z" + key]).to(torch.float64)
                    if shadow:
                        ref = P64 @ s_["Y" + key]
                        err = max(err, float((s_[key] - ref).norm() / ref.norm().clamp_min(1e-300)))
    if shadow:
        rec["leg_err"] = err   # max over members and legs of ||legs - legs_float64|| / ||legs_float64||
    rr_ = cfg.get("readrank")
    if rr_ is not None:
        # 2026-10-04 (H3, the old tier's READ cost): the members' readout and pairings at rank r' < r -- every member's
        # materialised legs projected on the top-r' left singular subspace of all members' legs at THIS layer (a
        # per-layer re-fit; Z keeps the full r, so nothing is lost for the next layer). Production would read through
        # Q' (n x r') at r'/r of the members' materialisation and compressed-pairing cost.
        mem = [s_ for s_ in sources if s_.get("bz")]
        if mem:
            X = torch.cat([s_[key] for s_ in mem for key in _BAND_KEYS if s_.get(key) is not None], dim=1)
            U_, S_, _ = torch.linalg.svd(X, full_matrices=False)
            k_ = min(int(rr_), U_.shape[1])
            Q_ = U_[:, :k_]
            rec["readrank_energy"] = float((S_[:k_] ** 2).sum() / (S_ ** 2).sum())
            for s_ in mem:
                for key in _BAND_KEYS:
                    if s_.get(key) is not None:
                        s_[key] = Q_ @ (Q_.T @ s_[key])
    log.append(rec)


def read_slices(sources, n, d21_ages=None, tailx=None):
    """kappa3[iii] and kappa3[iij] (zero diagonal) of the pre-activation, summed over sources.
    d21_ages: if set, only the newest `d21_ages` sources contribute to the (2,1) slice (the diagonal uses all).
    tailx: if set, the contributions of sources of AGE >= tailx["a0"] are accumulated separately and transformed
    by `_tail_transform` before being added in. tailx=None is the base path, operation for operation."""
    t = torch.zeros(n)
    D = torch.zeros(n, n)
    S = len(sources)
    a0 = tailx["a0"] if tailx is not None else None
    t_old = torch.zeros(n) if a0 is not None else None
    D_old = torch.zeros(n, n) if a0 is not None else None
    for idx, s in enumerate(sources):
        if a0 is not None and S - idx >= a0:   # this source is in the old tail: accumulate it apart
            tA, DA = t_old, D_old
        else:
            tA, DA = t, D
        A, B, E, g = s["A"], s["B"], s["E"], s["g"]
        if s.get("ER") is not None:
            E = A * s["dK3"][None, :] + 3.0 * B * s["alpha"][None, :] + s["ER"]
        tA += 3 * ((A * B * B) @ g)
        if E is not None:
            tA += (E * A * A).sum(1)
        B2 = s.get("B2")
        if B2 is not None:  # edge x kappa3(2,1) births: Sym over 6 of (1/2 d3)_p A_p B_p B2_p (hub p)
            g3 = s["g3"]
            AgB = A * g3[None, :] * B
            tA += 3.0 * (AgB * B2).sum(1)
        h = s.get("h")
        if h is not None:  # kappa4->kappa3 feed, part P2: Sym(h (x) M), M = D1 C0 D1 at birth; J M J^T = B diag(d1b) A^T
            m = A @ h
            Bd = B * s["d1b"][None, :]
            q = (Bd * A).sum(1)
            tA += 3.0 * m * q
        if d21_ages is not None and S - idx > d21_ages:
            continue
        if h is not None:
            DA += (2.0 * m[:, None] * Bd) @ A.T + q[:, None] * m[None, :]
        if B2 is not None:
            pD = s.get("pathD", "full")
            if pD in ("full", "merge"):
                DA += (A * g3[None, :] * B2) @ B.T + (B * B2 * g3[None, :]) @ A.T
            if pD == "full":
                DA += AgB @ B2.T
        L1 = 2 * A * B * g
        L2 = B * B * g
        DA += L1 @ B.T + L2 @ A.T
        if E is not None:
            DA += (2.0 / 3.0) * (E * A) @ A.T + (1.0 / 3.0) * (A * A) @ E.T
    if a0 is not None:
        t, D = _tail_transform(t, D, t_old, D_old, tailx, n)
    D.fill_diagonal_(0.0)
    return t, D



def _eleg(s_):
    """The source's correction leg E at its current layer (linear in the legs): E, or A dK3 + 3 B alpha + ER."""
    if s_.get("ER") is None:
        return s_["E"]
    return s_["A"] * s_["dK3"][None, :] + 3.0 * s_["B"] * s_["alpha"][None, :] + s_["ER"]


def _sym3_place(X):
    """X[i,j,k] symmetric in (j,k) with the distinguished leg first -> the sum over the 3 placements of that leg."""
    return X + X.permute(1, 0, 2) + X.permute(1, 2, 0)


def _k3_dense(sources, n, T3d=None, layer=None, couple=None):
    """2026-10-04 (R2, small widths only): the chain's pre-activation kappa3 as a dense n^3 tensor, summed over the
    sources' CP families exactly as read_slices reads them (t and D of the result equal read_slices'):
    tree Sym(A (x) B (x) B) g, correction (1/3) Sym(E (x) A (x) A), path (1/2) Sym6(A g3 (x) B (x) B2), feed
    Sym(m (x) M) with m = A h, M = (B d1b) A^T; plus the dense3 tensor. couple=a keeps only sources of age <= a."""
    T = torch.zeros(n, n, n) if T3d is None else T3d.clone()
    for s in sources:
        if couple is not None and layer is not None and layer - s["born"] > couple:
            continue
        A, B, g = s["A"], s["B"], s["g"]
        if A is None:
            continue
        T += _sym3_place(torch.einsum("ip,jp,kp->ijk", A * g[None, :], B, B))
        E = _eleg(s)
        if E is not None:
            T += _sym3_place(torch.einsum("ip,jp,kp->ijk", E, A, A)) / 3.0
        if s.get("B2") is not None:
            X = torch.einsum("ip,jp,kp->ijk", A * s["g3"][None, :], B, s["B2"])
            T += 0.5 * (X + X.permute(0, 2, 1) + X.permute(1, 0, 2) + X.permute(1, 2, 0) + X.permute(2, 0, 1)
                        + X.permute(2, 1, 0))
        if s.get("h") is not None:
            m = A @ s["h"]
            M = (B * s["d1b"][None, :]) @ A.T
            M = 0.5 * (M + M.T)
            T += m[:, None, None] * M[None, :, :] + m[None, :, None] * M[:, None, :] + m[None, None, :] * M[:, :, None]
    return T


def _k4_trees(C0, d1, d2, d3, K3, types=("path", "star", "32"), zero_diag=True):
    """2026-10-04 (R2): the kappa4 TREES born at a ReLU, as ARC's factored k=4 builds them (factor_k4.py, the
    two-leg C0 block; SIMPLE budget 3), from the chain's own state: path 12 d1 C0 d2 C0 d2 C0 d1, star
    4 d1 C0 d3 (C0 d1)^2, 3+2 12 d1 C0 d2 k3[j,k,l] d1 d1 (K3: dense kappa3; None = no 3+2).
    zero_diag=True: the edges are C0 (zero diagonal), the trees' all-distinct content only (pass K3 with its repeated
    entries zeroed); False: ARC's own convention -- the full C as the edge and the full kappa3 (ARC then replaces
    the trees' repeated slices by its correction block; test_trees.py checks this mode against the reference).
    Returns ARC's FactoredTensor4 Sym(sum_r A_r (x) B_r) (r = the hub j), one n-column block per type."""
    from mlp_kprop.factor_k4 import FactoredTensor4
    n = C0.shape[0]
    ii = torch.arange(n)
    C0z = C0.clone()
    if zero_diag:
        C0z.fill_diagonal_(0.0)
    A = torch.zeros(n, n, n)
    A[:, ii, ii] = C0z * d1[:, None]                       # A[i, j, r=j] = d1_i C0_ij
    As, Bs = [], []
    for t in types:
        if t == "path":
            Bjkl = 12.0 * (C0z[:, :, None] * C0z[None, :, :]) * (d2[:, None, None] * d2[None, :, None] * d1[None, None, :])
        elif t == "star":
            Bjkl = 4.0 * (C0z[:, :, None] * C0z[:, None, :]) * (d3[:, None, None] * d1[None, :, None] * d1[None, None, :])
        elif t == "32":
            if K3 is None:
                continue
            Bjkl = 12.0 * K3 * (d2[:, None, None] * d1[None, :, None] * d1[None, None, :])
        else:
            raise ValueError(f"unknown tree type {t!r}")
        As.append(A)
        Bs.append(Bjkl.permute(1, 2, 0).contiguous())     # B[k, l, r=j]
    if not As:
        return None
    return FactoredTensor4(n, factors=(torch.cat(As, dim=2), torch.cat(Bs, dim=2)))

def run_chain(Ws, window=None, k4=True, corr="full", metric=2.0, record=False, inject=None, inject_upto=-1, edge2=False, aug=False, d21_ages=None, tet=2, k4b=0, k4star=True, k4scale=1.0, dlr=0, srank=None, oracle=None, k4k3=0.0, q22x=0, c0d=False, k4pat=False, k22pat=False, k31pat=False, k4lvl=False, tail=None, tail_log=None, marg=False, k4feed=None, margleg=False, e3=False, e3age=None, k4bparts=None, shared_sub=None, paths=None, k4mult=None, pathrank=None, pairx=None, pathD="full", pathsub=True, pathfold=False, deadcut=None, deadC=False, ss_iter=None, ss_over=32, ssp=None, k31mult=1.0, feedmult=1.0, k22c=1.0, sdmult=1.0, varfix=None, varfix_late=None, tailx=None, colcut=None, satrow=None, stride=None, bandz=None, satrow_k4=None, recv=None, agecut=None, satrow_ch=None, margtail=None, tailr=None, inject_regime=None, pairnew=None, marg3=False, k4cons=None, k2pair=False, mcorr=None, postc=None, prec=None, t3k4=None, t3k4mult=1.0, zlegs=False, zmult=1.0, dense3=None, t3k4_last=False, k4young=None):
    """Ws: list of (n_in, n_out) challenge weights. Returns (depth, n) post-activation means.
    k4young (2026-10-04, R2; small widths -- it uses ARC's factored kappa4): the YOUNG kappa4 trees (path, star, 3+2)
    born at every ReLU from the chain's own state, carried exactly for `ages` layers (transport W, gating d1) and read
    into the pre-activation K4d and K31 on top of the closure: dict(ages=2, types=("path", "star", "32"),
    resid=True (add only the part beyond the closure's pattern: 3 var^2 / 3 var_i C0_ij), read=("K4d", "K31"),
    mult=1.0, couple=None (the 3+2's kappa3: all sources; an int a = sources of age <= a only), conv="pure" (edges C0,
    the all-distinct kappa3) or "arc" (ARC's tree blocks exactly: full C, full kappa3)).
    k4pat: replace the per-neuron pattern of the transported kappa4 diagonal by var_i^2 (scale-mixture closure),
    keeping its mean level; k22pat: same for the (2,2) slice with the pattern var_i var_j + 2 C0_ij^2.
    tail='amp' (with window=K): sources older than K are absorbed into the amplitude-mode closure (see below)."""
    tail_alpha = 0.0
    ss_state = {}  # production-like shared subspace: band basis P per band, transported and gated like a leg
    n = Ws[0].shape[0]
    mu, C = torch.zeros(n), torch.eye(n)
    a4 = 0.0
    K4post = K22post = None
    pending = None
    pending_t3 = None
    T3d = None   # dense3: a dense kappa3 tensor riding the chain (small widths only)
    young4 = []  # k4young: the live young kappa4 tree objects, [(birth layer, FactoredTensor4)]
    y4legs = None  # k4young impl="legs": young4.Young4 (the same trees on dense legs; full widths)
    y4chain = None  # k4young impl="chain": young4.Young4Chain (K4d only, from what the production chain holds)
    if k4young and k4young.get("impl", "arc") == "legs":
        from young4 import Young4
        y4legs = Young4(k4young.get("ages", 2), k4young.get("types", ("path", "star", "32")),
                        k4young.get("couple"), k4young.get("k31_32", "full"))
    elif k4young and k4young.get("impl", "arc") == "chain":
        from young4 import Young4Chain
        y4chain = Young4Chain(k4young.get("ages", 2), k4young.get("types", ("pathdiag", "star", "32")),
                              k4young.get("ages32"), k4young.get("last32", True), k4young.get("only_last32", False),
                              k4young.get("dx", "full"))
        if k4young.get("rec") is not None:
            y4chain.rec = k4young["rec"]
    sources = []
    stride_state = {"tau": None, "Delta": None, "d1": None, "checkpoints": []}
    bz_state = {}  # bandz: the shared band basis P (coefficient form, experiments/2026-09-29_port-battery)
    recv_state = {}  # recv (E3, arm b): the previous layer's (t, D) and gate d1, for the repeated-index rule
    rows, info = [], []
    L = len(Ws)
    for l, W in enumerate(Ws):
        Wt = W.T
        mu_pre = Wt @ mu
        C_pre = Wt @ C @ W
        if prec is not None:  # cloud #11 (2026-10-02_cov-error): a callable (l, C_pre) -> C_pre at the top of the layer,
            C_pre = prec(l, C_pre)  # before every other use (legs, D, K4d, the step). prec=None: the old path, bitwise
        if oracle is not None and l in oracle.get("C_layers", ()):  # oracle: exact pre-activation covariance (full)
            C_pre = torch.tensor(oracle["C"][l], dtype=torch.float64)
            C_pre = 0.5 * (C_pre + C_pre.T)
        if oracle is not None and l in oracle.get("C0_layers", ()):  # oracle: exact off-diagonal only
            Cm = torch.tensor(oracle["C"][l], dtype=torch.float64)
            Cm = 0.5 * (Cm + Cm.T)
            C_pre = Cm - torch.diag(torch.diagonal(Cm)) + torch.diag(torch.diagonal(C_pre))
        if oracle is not None and l in oracle.get("var0_layers", ()):  # oracle: exact DIAGONAL, here (early)
            # "var_layers" substitutes the diagonal after the source loop, so only nonlin_slices sees
            # it; this one substitutes it before, so the legs, D and K4d are built from it too. The
            # pair tells an error in the variance apart from an inconsistency between the two stages.
            C_pre = C_pre - torch.diag(torch.diagonal(C_pre)) + torch.diag(torch.tensor(oracle["var"][l]))
        if varfix is not None:  # multiplicative correction of the pre-activation variance PATTERN, applied EARLY
            # (2026-09-26_variance). varfix is (L, n) of factors g, or a callable (l, v, mu_pre) -> g; the
            # corrected diagonal is v * g and everything downstream (legs, D, K4d, nonlin_slices) sees it.
            v0 = torch.diagonal(C_pre)
            g = (varfix(l, v0, mu_pre) if callable(varfix)
                 else torch.as_tensor(varfix[l], dtype=torch.float64))
            C_pre = C_pre - torch.diag(v0) + torch.diag(v0 * g)
        if young4:  # k4young: the young kappa4 trees ride the linear step (ARC convention: W is (n_out, n_in))
            young4 = [(m_, y_.contract_W(Wt)) for m_, y_ in young4]
        if y4legs is not None:
            y4legs.transport(Wt)
        Wsq_t = (Wt * Wt) if zlegs else None
        if T3d is not None:   # dense3: T(l)_ijk = sum_abc W_ai W_bj W_ck T(l-1)_abc
            T3d = torch.einsum("abc,ai,bj,ck->ijk", T3d, W, W, W)
        for s in sources:
            if s.get("bz"):  # bandz member: its legs are P Z, transported through P in _band_step
                continue
            if zlegs and s.get("Z") is not None:
                # the b = c part of each single pair (it belongs to the (2,2) class, which the closure carries), from
                # the gated legs of the previous layer: S = (W o W)^T (Y o Y'), one per (family, ordering)
                Ag, Bg, Eg = s["A"], s["B"], _eleg(s)
                s["S"] = [Wsq_t @ (Bg * Bg), Wsq_t @ (Ag * Bg)] + ([Wsq_t @ (Ag * Ag), Wsq_t @ (Eg * Ag)]
                                                                 if Eg is not None else [])
                s["Z"] = [Wsq_t @ z for z in s["Z"]]   # the doubled index of the (2,1,1) kappa4 content rides W o W
            s["A"] = Wt.clone() if s["A"] is None else Wt @ s["A"]
            s["B"] = Wt @ s["B"]
            if s.get("B2") is not None:
                if e3age is not None and l - s["born"] > e3age:
                    if pathfold:  # fold the path leg into the tree leg: Sym(B~ x 3A x B~ g) = tree + path + O(Y^2)
                        s["B"] = s["B"] + Wt @ (s["B2"] / (2.0 * s["g"]).clamp_min(1e-30)[None, :])
                    s["B2"] = None
                else:
                    s["B2"] = Wt @ s["B2"]
            if s["E"] is not None:
                s["E"] = Wt @ s["E"]
            if s.get("ER") is not None:
                s["ER"] = Wt @ s["ER"]
        if agecut is not None:  # E4 (port battery): the harsher column cut, applied when a source reaches age a_min
            for s_ in sources:
                if l - s_["born"] == int(agecut[0]) and not s_.get("agecut_done"):
                    cm_ = (s_["a_birth"] >= float(agecut[1])).to(torch.float64)
                    for key in ("B", "ER", "B2"):
                        if s_.get(key) is not None:
                            s_[key] = s_[key] * cm_[None, :]
                    for key in ("dK3", "alpha", "g", "d3", "d1b", "h", "g3"):
                        if s_.get(key) is not None:
                            s_[key] = s_[key] * cm_
                    s_["agecut_done"] = True
        if bandz is not None and sources:
            _band_step(bandz, bz_state, sources, l, Wt)
        if shared_sub and sources:
            # age-gated SHARED subspace for old sources: the legs (A, B, ER) of every source with age >= a are projected
            # onto the top-r eigenvectors of their summed Gram matrix (one basis per age band)
            bands = [(shared_sub[k], shared_sub[k + 1]) for k in range(0, len(shared_sub), 2)]
            for bi, (a_min, r_) in enumerate(bands):
                a_max = bands[bi + 1][0] if bi + 1 < len(bands) else 10 ** 9
                grp = [s_ for s_ in sources if a_min <= l - s_.get("born", l) < a_max]
                if not grp:
                    continue
                Gm = torch.zeros(n, n)
                for s_ in grp:
                    for key in ("A", "B", "ER"):
                        if s_.get(key) is not None:
                            Gm += s_[key] @ s_[key].T
                if ss_iter is None:
                    ev, U_ = torch.linalg.eigh(Gm)
                    P = U_[:, -r_:]
                else:
                    # production-like: start from the transported previous basis (+ ss_over random vectors), ss_iter
                    # subspace iterations with the band Gram, Rayleigh-Ritz, keep the top r
                    g_ = torch.Generator().manual_seed(7000 + 31 * l + bi)
                    X0 = torch.randn(n, r_ + ss_over, generator=g_)
                    if bi in ss_state:
                        Pt = Wt @ ss_state[bi]
                        X0[:, :r_] = Pt
                    Xq, _ = torch.linalg.qr(X0)
                    for _ in range(ss_iter if bi in ss_state else ss_iter + 1):
                        Xq, _ = torch.linalg.qr(Gm @ Xq)
                    Tm = Xq.T @ Gm @ Xq
                    ev_, V_ = torch.linalg.eigh(0.5 * (Tm + Tm.T))
                    P = Xq @ V_[:, -r_:]
                    ss_state[bi] = P
                for s_ in grp:
                    for key in ("A", "B", "ER"):
                        if s_.get(key) is not None:
                            s_[key] = P @ (P.T @ s_[key])
        if ssp is not None and sources:
            # production mirror of the shared subspace (k3k): band = sources of age >= a0; at the layer where a source
            # enters, the band basis is the top-r Rayleigh-Ritz subspace of the band Gram (A A^T + B B^T + ER ER^T)
            # after q subspace iterations started from [transported basis | first `over` columns of the entrant's A]
            # (the entrant's first r+over columns at the first entry); all band legs are projected onto it.
            a0_, r_, q_, ov_ = ssp
            band_ = [s_ for s_ in sources if l - s_["born"] >= a0_]
            ent_ = [s_ for s_ in band_ if l - s_["born"] == a0_]
            if ent_:
                se_ = ent_[0]
                def gapply(Xg):
                    Yg = torch.zeros_like(Xg)
                    for s_ in band_:
                        for key in ("A", "B", "ER"):
                            if s_.get(key) is not None:
                                Yg += s_[key] @ (s_[key].T @ Xg)
                    return Yg
                if "P" in ss_state:
                    X0 = torch.cat([Wt @ ss_state["P"], se_["A"][:, :ov_]], dim=1)
                else:
                    X0 = se_["A"][:, :r_ + ov_].clone()
                Xq, _ = torch.linalg.qr(X0)
                for _ in range(q_):
                    Xq, _ = torch.linalg.qr(gapply(Xq))
                Tm = Xq.T @ gapply(Xq)
                ev_, V_ = torch.linalg.eigh(0.5 * (Tm + Tm.T))
                Pn = Xq @ V_[:, -r_:]
                ss_state["P"] = Pn
                for s_ in band_:
                    for key in ("A", "B", "ER"):
                        if s_.get(key) is not None:
                            s_[key] = Pn @ (Pn.T @ s_[key])
        if srank:
            # per-source low rank: when a source reaches a listed age, project all its legs onto the top-r left
            # singular vectors of its hub leg A (equivalent to transporting that basis from then on)
            S = len(sources)
            for idx, s_ in enumerate(sources):
                age = S - idx
                if isinstance(age, int) and age in srank:
                    r = srank[age]
                    Us, Ss, _ = torch.linalg.svd(torch.cat([s_["A"], s_["B"]], dim=1), full_matrices=False)
                    P = Us[:, :r]
                    for key in srank.get("legs", ("A", "B", "E")):
                        if s_.get(key) is not None:
                            s_[key] = P @ (P.T @ s_[key])
        tail_fit = None
        if tail is not None and tail.startswith("oracle") and window is not None and len(sources) > window:
            # oracle bound for the amplitude form: keep every source, but replace the old part (age > window) of t and/or
            # D by its per-layer least-squares fit on the amplitude form ("oracle", "oracle_t", "oracle_D")
            vp_ = torch.diagonal(C_pre)
            amp_ = mu_pre * vp_
            Damp_ = 2.0 * mu_pre[:, None] * (C_pre - torch.diag(vp_)) + vp_[:, None] * mu_pre[None, :]
            Damp_.fill_diagonal_(0.0)
            t_o, D_o = read_slices(sources[:-window], n)
            tail_fit = (t_o, D_o, amp_ * float(t_o @ amp_ / (amp_ @ amp_)),
                        Damp_ * float((D_o * Damp_).sum() / (Damp_ * Damp_).sum()))
        if tail is not None and not tail.startswith("oracle") and window is not None and len(sources) > window:
            # amplitude-mode closure of the old tail (positive homogeneity): sources older than `window` are absorbed
            # into one coefficient alpha: t += alpha mu_i var_i, D += (alpha/3)(2 mu_i C0_ij + mu_j var_i).
            # alpha is the least-squares projection of the absorbed sources' t onto mu*var at absorption time.
            vp_ = torch.diagonal(C_pre)
            amp_ = mu_pre * vp_
            t_o, D_o = read_slices(sources[:-window], n)
            c_ = float(t_o @ amp_ / (amp_ @ amp_))
            tail_alpha = tail_alpha + (c_ if tail == "amp" else 0.0)
            if tail_log is not None:
                tail_log.append((l, c_, tail_alpha))
            del sources[:-window]
        if stride is not None and sources:
            # STRIDED old tier (2026-09-29, the completeness critic's gap 1). Sources of age >= a0 form the old tier.
            # Its (t, D) contribution is taken EXACTLY only at checkpoint layers (every k-th layer after a0, and the
            # last layer); in between it is pushed forward from the previous layer's old-tier slices by the exact
            # repeated-index rule (experiments/2026-09-28_state-closure/closure.py, section 1), i.e. everything but
            # the all-distinct term. Sources that JUST reached age a0 are added exactly (their legs are exact then).
            # Production meaning: mode "mst" = the old legs are jumped k layers at once through one shared operator,
            # so they are not materialised between checkpoints; mode "Donly" = legs still move every layer (t_old
            # stays exact and O(n^2)), only the old tier's D PAIRINGS are skipped between checkpoints.
            a0s, ks = int(stride[0]), int(stride[1])
            smode = stride[2] if len(stride) > 2 else "mst"
            def _rs(sub):
                return read_slices(sub, n, d21_ages, None) if sub else (torch.zeros(n), torch.zeros(n, n))
            t_y, D_y = _rs([s_ for s_ in sources if l - s_["born"] < a0s])
            t_n, D_n = _rs([s_ for s_ in sources if l - s_["born"] == a0s])
            t_o, D_o = _rs([s_ for s_ in sources if l - s_["born"] > a0s])
            ck = (l == L - 1) or stride_state["tau"] is None or ((l - a0s) % ks == 0)
            if ck:
                tau, Delta = t_n + t_o, D_n + D_o
            else:
                g1 = stride_state["d1"]
                tt = g1 * g1 * g1 * stride_state["tau"]
                Dt = (g1 * g1)[:, None] * stride_state["Delta"] * g1[None, :]
                MM = Wt * Wt
                D_rep = (MM * tt[None, :] + MM @ Dt + 2.0 * Wt * (Wt @ Dt.T)) @ Wt.T
                D_rep.fill_diagonal_(0.0)
                if smode == "Donly":
                    tau = t_n + t_o
                else:
                    tau = (MM * Wt) @ tt + 3.0 * ((MM @ Dt) * Wt).sum(1) + t_n
                Delta = D_rep + D_n
            stride_state["tau"], stride_state["Delta"] = tau, Delta
            stride_state["checkpoints"].append((l, bool(ck)))
            t, D = t_y + tau, D_y + Delta
        elif agecut is not None and sources:
            # E4 (port battery): age-dependent matched cut -- the D contribution of OLD sources (age >= a_min) only over
            # the neurons alive under the harsher threshold t_old (their leg rows were cut at the previous gating)
            a_min_, t_old_ = int(agecut[0]), float(agecut[1])
            yng_ = [s_ for s_ in sources if l - s_["born"] < a_min_]
            old_ = [s_ for s_ in sources if l - s_["born"] >= a_min_]
            t_y, D_y = read_slices(yng_, n, d21_ages, None) if yng_ else (torch.zeros(n), torch.zeros(n, n))
            t_o, D_o = read_slices(old_, n, d21_ages, None) if old_ else (torch.zeros(n), torch.zeros(n, n))
            keep_o = ((mu_pre / torch.diagonal(C_pre).clamp_min(1e-30).sqrt()) >= t_old_).to(torch.float64)
            t, D = t_y + t_o, D_y + D_o * (keep_o[:, None] * keep_o[None, :])
        else:
            t, D = read_slices(sources, n, d21_ages, tailx) if sources else (torch.zeros(n), torch.zeros(n, n))
        if T3d is not None:   # dense3 readout: t_i += T_iii, D_ij += T_iij
            ii = torch.arange(n)
            t = t + T3d[ii, ii, ii]
            Dd = T3d[ii, ii, :].clone(); Dd.fill_diagonal_(0.0)
            D = D + Dd
        dead = None
        if deadcut is not None:  # neurons in the dead regime (alpha < deadcut) at this layer: skip them in D
            dead = (mu_pre / torch.diagonal(C_pre).clamp_min(1e-30).sqrt()) < deadcut
            D[dead, :] = 0.0
            D[:, dead] = 0.0
        recv_mask = None
        if recv is not None and sources:
            # E3 (port battery): RECEIVER rows -- neurons with alpha_l < theta_out at the CURRENT layer have their leg rows
            # not computed (2-D compaction of the leg transport; theta_out < deadcut, so these rows are gated to 0 at the
            # end of this layer anyway and only this layer's readout of them changes). Their t_i: "zero" = dropped;
            # "a" = from the newest (age-1) source only, whose B, A = Wt and path leg are computed in full; "b" = "a" plus
            # the exact repeated-index rule on the previous layer's (t, D) for all older sources (no all-distinct term).
            th_out_, arm_ = float(recv[0]), recv[1]
            recv_mask = (mu_pre / torch.diagonal(C_pre).clamp_min(1e-30).sqrt()) < th_out_
            if arm_ == "zero":
                t_r = torch.zeros(n)
            else:
                t_r, _ = read_slices(sources[-1:], n, None, None)
                if arm_ == "b" and recv_state.get("t") is not None:
                    g1_ = recv_state["d1"]
                    tt_ = g1_ * g1_ * g1_ * recv_state["t"]
                    Dt_ = (g1_ * g1_)[:, None] * recv_state["D"] * g1_[None, :]
                    MM_ = Wt * Wt
                    t_r = t_r + (MM_ * Wt) @ tt_ + 3.0 * ((MM_ @ Dt_) * Wt).sum(1)
            t = torch.where(recv_mask, t_r, t)
        if tail_fit is not None:
            t_o, D_o, t_f, D_f = tail_fit
            if tail in ("oracle", "oracle_t"):
                t = t - t_o + t_f
            if tail in ("oracle", "oracle_D"):
                D = D - D_o + D_f
        if tail is not None and tail_alpha != 0.0:
            vp_ = torch.diagonal(C_pre)
            t = t + tail_alpha * mu_pre * vp_
            Dt = (tail_alpha / 3.0) * (2.0 * mu_pre[:, None] * (C_pre - torch.diag(vp_)) + vp_[:, None] * mu_pre[None, :])
            Dt.fill_diagonal_(0.0)
            D = D + Dt
        if dlr and sources:  # sensitivity test: replace the (2,1) slice by its best rank-dlr approximation
            Us, Ss, Vs = torch.linalg.svd(D)
            D = (Us[:, :dlr] * Ss[:dlr]) @ Vs[:dlr]
            D.fill_diagonal_(0.0)
        if k4 == "memo22" and K4post is not None and k4lvl:
            # only the LEVELS of the memo22 transport, in O(n^2) (used with k4pat and k22pat, which only need them):
            # sum(M) = u^T X u exactly (u = row sums of W*W); tr(M) = sum_ab X_ab G_ab, G = Wsq Wsq^T, with the
            # mean-field G_ab ~ u_a u_b / n off the diagonal and the exact G_aa = sum_i W_ai^4
            Wsq = W * W
            u = Wsq.sum(1)
            X = K22post + torch.diag(K4post)
            Ssum = u @ (X @ u) if False else (X * torch.outer(u, u)).sum()
            Gd = (Wsq * Wsq).sum(1)
            trM = (Ssum - (K4post * u * u).sum()) / n + (K4post * Gd).sum()
            w4k = (Wsq * Wsq).T @ K4post
            K4d = torch.full((n,), float(3.0 * trM / n)) - 2.0 * w4k
            K22 = torch.full((n, n), float((Ssum - trM) / (n * (n - 1))))
            K22.fill_diagonal_(0.0)
        elif k4 == "memo22" and K4post is not None:
            # memoryless transport of the tracked kappa4 slices: direct pairings only (exchange terms are O(1/n))
            Wsq = W * W
            M = Wsq.T @ (K22post + torch.diag(K4post)) @ Wsq
            w4k = (Wsq * Wsq).T @ K4post
            K4d = 3.0 * torch.diagonal(M) - 2.0 * w4k
            K22 = M.clone()
            K22.fill_diagonal_(0.0)
        elif k4 and k4 != "memo22" and a4 != 0.0:
            K4d = torch.full((n,), a4 * metric * metric)
            K22 = torch.full((n, n), a4 * metric * metric / 3.0)
            K22.fill_diagonal_(0.0)
        else:
            K4d, K22 = torch.zeros(n), None
        if k4pat and K4d is not None and float(K4d.abs().sum()) > 0:
            s4 = torch.diagonal(C_pre) ** 2
            K4d = s4 * (K4d.mean() / s4.mean())
        if k22pat and K22 is not None:
            vp = torch.diagonal(C_pre)
            C0p = C_pre - torch.diag(vp)
            pat = torch.outer(vp, vp) + (2.0 * k22c) * C0p * C0p
            pat.fill_diagonal_(0.0)
            K22 = pat * (K22.sum() / pat.sum())
        k4cons_v = None
        if k4cons and K4d is not None and K22 is not None and float(K4d.abs().sum()) > 0:
            # cloud #10 consistency audit: the kappa4 LEVEL enters twice -- the diagonal K4d (level from trM) and the
            # (2,2) slice K22 (level from Ssum - trM) -- and the scale-mixture v of K31 and the feed is read off K4d.
            # Force one onto the other (no fitted constant: both levels are the chain's own).
            vpc = torch.diagonal(C_pre)
            C0c = C_pre - torch.diag(vpc)
            patc = torch.outer(vpc, vpc) + (2.0 * k22c) * C0c * C0c
            patc.fill_diagonal_(0.0)
            v22 = float(K22.sum() / patc.sum())
            vd = float(K4d.mean() / (3.0 * (vpc * vpc).mean()))
            if k4cons == "k4d_from_k22":
                K4d = 3.0 * v22 * vpc * vpc
            elif k4cons == "k22_from_k4d":
                K22 = vd * patc
            elif k4cons == "v_from_k22":
                k4cons_v = v22
            else:
                raise ValueError(f"unknown k4cons {k4cons!r}")
        if k4mult is not None and K4d is not None:  # level test: scale the (closed) kappa4 slices at every layer
            fac = float(k4mult(l)) if callable(k4mult) else float(k4mult)
            K4d = K4d * fac
            if K22 is not None:
                K22 = K22 * fac
        K31 = None
        if k31pat and K4d is not None and float(K4d.abs().sum()) > 0:
            vp = torch.diagonal(C_pre)
            v_ = K4d.mean() / (3.0 * (vp * vp).mean())       # scale-mixture level v
            if k4cons_v is not None:
                v_ = k4cons_v
            K31 = (3.0 * k31mult) * v_ * vp[:, None] * (C_pre - torch.diag(vp))
        if k4k3 and pending is not None and sources:
            # kappa3 x edge: dk4[i] = 12 sum_a W_ai d2_a K_ai (C0 D1 W)_ai, K_ai = k3_pre^m[a, v_i, v_i], v_i = D1 W[:, i]
            d1m, d2m, C0m, legs_m = pending
            Kmat = torch.zeros(n, n)  # K[a, i]
            for (Am, Bm, Em, gm), s_ in zip(legs_m, sources[:len(legs_m)]):
                A2, B2 = s_["A"], s_["B"]          # next-layer legs: A2[i,p] = (A^T v_i)_p etc.
                E2 = s_["E"] if s_.get("ER") is None else A2 * s_["dK3"][None, :] + 3.0 * B2 * s_["alpha"][None, :] + s_["ER"]
                Kmat += (1.0 / 3.0) * (Am * gm[None, :]) @ (B2 * B2).T.T.T if False else 0.0
                Kmat += (1.0 / 3.0) * (Am @ ((B2 * B2) * gm[None, :]).T + 2.0 * Bm @ ((A2 * B2) * gm[None, :]).T)
                Kmat += (1.0 / 3.0) * (Em @ (A2 * A2).T + 2.0 * Am @ (E2 * A2).T)
            V = d1m[:, None] * W
            CDW = C0m @ V
            dk4 = 12.0 * (W * d2m[:, None] * Kmat * CDW).sum(0)
            K4d = K4d + k4k3 * dk4 if K4d is not None else k4k3 * dk4
        if t3k4 and pending is not None and sources and K4d is not None and (not t3k4_last or l == len(Ws) - 1):
            # 2026-10-03, the kappa3 -> kappa4 TRANSFER through the self-variance (korder/k3to211.py): Leonov-Shiryaev on
            # k(y_a, y_a, y_b, y_c) gives k4post(a,a,b,c) += 2 var_a d1_a d2_a d1_b d1_c k3pre(a,b,c) -- the full kappa3
            # tensor (all-distinct part included) feeding the (2,1,1) class of kappa4, the class that carries most of the
            # next kappa4 diagonal's pattern. Its projection on K4d(l+1) (the (2,1,1) class, b != c):
            #   dk4_i = 12 sum_a W_ai^2 h_a [K_ai - sum_b D[b,a] v_bi^2],  h = var d1 d2 (layer m), K_ai = k3pre^m[a, v_i, v_i]
            # "exact": W_ai^2 as is (one extra pairing-size product); "mf": W_ai^2 -> its mean 2/n (O(n w), free).
            d1m, d2m, C0m, legs_m = pending
            varm, Dm = pending_t3
            hm = varm * d1m * d2m
            V = d1m[:, None] * W
            if t3k4 == "exact":
                Kt = torch.zeros(n, n)
                for (Am, Bm, Em, gm), s_ in zip(legs_m, sources[:len(legs_m)]):
                    A2, B2 = s_["A"], s_["B"]
                    E2 = s_["E"] if s_.get("ER") is None else A2 * s_["dK3"][None, :] + 3.0 * B2 * s_["alpha"][None, :] + s_["ER"]
                    Kt += (1.0 / 3.0) * (Am @ ((B2 * B2) * gm[None, :]).T + 2.0 * Bm @ ((A2 * B2) * gm[None, :]).T)
                    Kt += (1.0 / 3.0) * (Em @ (A2 * A2).T + 2.0 * Am @ (E2 * A2).T)
                if Dm is not None:
                    Kt = Kt - Dm.T @ (V * V)
                dk4 = 12.0 * (W * W * hm[:, None] * Kt).sum(0)
            elif t3k4 == "mf":
                acc = torch.zeros(n)
                for (Am, Bm, Em, gm), s_ in zip(legs_m, sources[:len(legs_m)]):
                    A2, B2 = s_["A"], s_["B"]
                    E2 = s_["E"] if s_.get("ER") is None else A2 * s_["dK3"][None, :] + 3.0 * B2 * s_["alpha"][None, :] + s_["ER"]
                    acc += (1.0 / 3.0) * (((hm @ Am) * gm) @ (B2 * B2).T + 2.0 * ((hm @ Bm) * gm) @ (A2 * B2).T)
                    acc += (1.0 / 3.0) * ((hm @ Em) @ (A2 * A2).T + 2.0 * (hm @ Am) @ (E2 * A2).T)
                if Dm is not None:
                    acc = acc - (Dm @ hm) @ (V * V)
                dk4 = 12.0 * (2.0 / n) * acc
            else:
                raise ValueError(f"unknown t3k4 {t3k4!r}")
            K4d = K4d + t3k4mult * dk4
        if zlegs and sources and K4d is not None:
            # 2026-10-03: the (2,1,1) class of the post-activation kappa4, carried. Its content is born each layer by the
            # kappa3 -> kappa4 transfer through the self-variance, k4post(a,a,b,c) += 2 h_a d1_b d1_c k3pre(a,b,c),
            # h = var d1 d2 (korder/k3to211.py: corr 0.62-0.81 with ARC k=4's slice; with the gated inheritance
            # 0.81-0.90). For a source Sym(A x B x B) g the doubled index sits on A or on B: doubled legs
            # ZA = (2/3) g h o A (singles B, B) and ZB = (2/3) g h o B (singles A, B, twice). The doubled index is gated by
            # d1^2 and rides W o W (the p = q path of the (2,1,1) -> (2,1,1) transport); the singles are the source's own
            # legs. Its (2,1,1)-class part of the pre-activation kappa4 diagonal: 6 sum_p (ZA B^2 + 2 ZB A B).
            dz = torch.zeros(n)
            for s_ in sources:
                if s_.get("Z") is None or s_["A"] is None:
                    continue
                A_, B_, E_ = s_["A"], s_["B"], _eleg(s_)
                Z = s_["Z"]; S = s_.get("S")
                nd = zlegs == "nodiag" and S is not None
                # family 1: Sym(A x B x B) g -- doubled on A (singles B, B), doubled on B (singles A, B, twice)
                dz += (Z[0] * (B_ * B_ - (S[0] if nd else 0.0))).sum(1) + 2.0 * (Z[1] * (A_ * B_ - (S[1] if nd else 0.0))).sum(1)
                if len(Z) > 2 and E_ is not None:
                    # family 2: Sym(E x A x A) -- doubled on E (singles A, A), doubled on A (singles E, A, twice)
                    dz += (Z[2] * (A_ * A_ - (S[2] if nd else 0.0))).sum(1) + 2.0 * (Z[3] * (E_ * A_ - (S[3] if nd else 0.0))).sum(1)
            K4d = K4d + (6.0 * zmult) * dz
        if k4b and sources:
            # all-distinct kappa4 born at earlier ReLUs (tree diagrams: 12 paths + 4 stars), read into the
            # pre-activation (2,2) slice (direct pairings) and diagonal through the kappa3 legs A, B
            addd = torch.zeros(n)
            add22 = torch.zeros(n, n)
            parts = k4bparts if k4bparts is not None else ("pd", "p22", "sd", "s22")
            for s in sources[-k4b:]:
                A, B, g = s["A"], s["B"], s["g"]
                if "pd" in parts or "p22" in parts:
                    X = A * B * g
                    XC = X @ s["C0"]
                    if "p22" in parts:
                        add22 += 4.0 * (XC @ X.T)
                    if "pd" in parts:
                        addd += 12.0 * (XC * X).sum(1)
                if k4star:
                    Y = A * B * s["d3"]
                    BB = B * B
                    if "s22" in parts:
                        Z = Y @ BB.T
                        add22 += 2.0 * (Z + Z.T)
                    if "sd" in parts:
                        addd += (4.0 * sdmult) * (Y * BB).sum(1)
            add22.fill_diagonal_(0.0)
            if recv_mask is not None and recv[1] == "zero":
                addd = addd * (~recv_mask).to(torch.float64)
            if tailr:  # tailr "star_row": the star kappa4 birth read into the tail neurons' diagonal is dropped
                for _, keep, _, _ in _tailr_rules(tailr, mu_pre / torch.sqrt(torch.clamp(
                        torch.diagonal(C_pre), min=1e-12))).get("star_row", ()):
                    addd = addd * keep
            K4d = K4d + k4scale * addd
            K22 = (K22 if K22 is not None else torch.zeros(n, n)) + k4scale * add22
        if y4chain is not None and y4chain.gens and K4d is not None:
            # 2026-10-04 (R3): the production form of the young-kappa4 K4d reader (young4.Young4Chain)
            rm_ = None
            if k4young.get("rowcut") is not None:   # k4a port check: emulate production's dead/saturated row cuts
                al_ = mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))
                rm_ = ((al_ >= k4young["rowcut"][0]) & (al_ <= k4young["rowcut"][1])).to(torch.float64)
            rm_read = None if k4young.get("rowcut_after") else rm_
            Y4c_ = y4chain.read(sources, _eleg, l, last=(l == L - 1), rowmask=rm_read)
            P4c_ = 3.0 * torch.diagonal(C_pre) ** 2
            if rm_ is not None and k4young.get("cproj") == "mask":   # the projection coefficient from the uncut rows only
                c4_ = float((Y4c_ * P4c_ * rm_).sum() / (P4c_ * P4c_ * rm_).sum())
            else:
                c4_ = float((Y4c_ * P4c_).sum() / (P4c_ * P4c_).sum())
            Y4r_ = Y4c_ - c4_ * P4c_
            if rm_ is not None and k4young.get("rowcut_after"):     # the rows' values dropped AFTER the full projection
                Y4r_ = Y4r_ * rm_ + (0.0 if k4young.get("rowcut_after") == "all" else 0.0)
            K4d = K4d + float(k4young.get("mult", 1.0)) * Y4r_
        if k4young and (young4 or (y4legs is not None and y4legs.gens)) and K4d is not None:
            # 2026-10-04 (R2): read the young kappa4 trees into the pre-activation K4d and K31 (on top of the closure;
            # resid: only the part beyond the closure's pattern, so the closure keeps its own level and pattern)
            vy_ = torch.diagonal(C_pre)
            C0y_ = C_pre - torch.diag(vy_)
            rd_ = k4young.get("read", ("K4d", "K31"))
            mt_ = float(k4young.get("mult", 1.0))
            if y4legs is not None:
                Y4l_, Y31l_ = y4legs.read(rd_)
            if "K4d" in rd_:
                Y4 = Y4l_ if y4legs is not None else sum(y_.get_dslice((4,)) for _, y_ in young4)
                if k4young.get("resid", True):
                    P4 = 3.0 * vy_ * vy_
                    Y4 = Y4 - float((Y4 * P4).sum() / (P4 * P4).sum()) * P4
                K4d = K4d + mt_ * Y4
            if "K31" in rd_ and K31 is not None:
                Y31 = Y31l_ if y4legs is not None else sum(y_.get_dslice((3, 1)) for _, y_ in young4)
                Y31 = Y31 - torch.diag(torch.diagonal(Y31))
                if k4young.get("resid", True):
                    P31 = 3.0 * vy_[:, None] * C0y_
                    Y31 = Y31 - float((Y31 * P31).sum() / (P31 * P31).sum()) * P31
                K31 = K31 + mt_ * Y31
        if oracle is not None and "k4scale_l" in oracle and K4d is not None:  # oracle LEVEL rescaling of kappa4
            r = oracle["k4scale_l"][l]
            K4d = K4d * r
            if K22 is not None:
                K22 = K22 * r
        if oracle is not None:  # oracle substitution of per-neuron pre-activation statistics at chosen layers
            rg_ = oracle.get("regime")  # (lo, hi): substitute only where lo <= alpha < hi (the chain's own alpha)
            if rg_ is not None:
                a_or = mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))
                m_or = (a_or >= rg_[0]) & (a_or < rg_[1])
            sub_ = (lambda cur, new: new) if rg_ is None else (lambda cur, new: torch.where(m_or, new, cur))
            if l in oracle.get("k4_layers", ()):
                K4d = sub_(K4d, torch.tensor(oracle["k4"][l]))
            if l in oracle.get("k3_layers", ()):
                t = sub_(t, torch.tensor(oracle["k3"][l]))
            if "t_scale" in oracle:
                t = t * float(oracle["t_scale"][l])
            if "D_scale" in oracle:
                D = D * float(oracle["D_scale"][l])
            if l in oracle.get("D_layers", ()):
                D = torch.tensor(oracle["D"][l], dtype=torch.float64)
                D.fill_diagonal_(0.0)
            if l in oracle.get("K22_layers", ()) and K22 is not None:  # 2026-10-02: the (2,2) slice kappa4[i,i,j,j]
                K22 = torch.tensor(oracle["K22"][l], dtype=torch.float64)   # of the pre-activations (d_oracle.py)
                K22.fill_diagonal_(0.0)
            if l in oracle.get("K31_layers", ()) and K31 is not None:  # 2026-10-04 (R1): the (3,1) slice kappa4[i,i,i,j]
                K31 = torch.tensor(oracle["K31"][l], dtype=torch.float64)   # of the pre-activations, replacing the
                K31.fill_diagonal_(0.0)                                     # closure's 3 k31mult v var_i C0_ij
            if l in oracle.get("k4add_layers", ()):   # 2026-10-04 (R1): ADDITIVE terms on top of the chain's own closure
                K4d = K4d + torch.tensor(oracle["k4add"][l], dtype=torch.float64)   # (what a young-kappa4 carrier adds)
            if l in oracle.get("K31add_layers", ()) and K31 is not None:
                K31 = K31 + torch.tensor(oracle["K31add"][l], dtype=torch.float64)
                K31.fill_diagonal_(0.0)
            if l in oracle.get("var_layers", ()):
                C_pre = C_pre - torch.diag(torch.diagonal(C_pre)) + torch.diag(sub_(torch.diagonal(C_pre),
                                                                                    torch.tensor(oracle["var"][l])))
        if mcorr is not None:  # ONLINE moment corrector (2026-10-01): a callable of the chain's OWN current state
            # returning the PREDICTED ERROR of each channel, which is subtracted. Unlike `oracle` this is a function
            # of the live trajectory, so the correction stays self-consistent after the trajectory moves -- which is
            # what a shipped corrector would be. Returns a dict with any of "var", "k3", "k4" (torch vectors of n);
            # anything absent is left alone. mcorr=None is the old path, bitwise.
            _v = torch.diagonal(C_pre)
            dd = mcorr(l, dict(mu_pre=mu_pre, var=_v, t=t, K4d=K4d, D=D, W=Ws[l], C=C_pre, K22=K22,
                               W_next=(Ws[l + 1] if l + 1 < len(Ws) else Ws[l]), n=n, L=L))
            if dd:
                if "k3" in dd:
                    t = t - dd["k3"]
                if "D" in dd:   # 2026-10-14 (skew closure): the (2,1) slice too; absent = the old path, bitwise
                    D = D - dd["D"]
                if "k4" in dd and K4d is not None:
                    K4d = K4d - dd["k4"]
                if "var" in dd:
                    C_pre = C_pre - torch.diag(_v) + torch.diag(_v - dd["var"])
        if varfix_late is not None:  # same correction as `varfix`, but applied HERE: only nonlin_slices sees the
            # corrected diagonal, the legs / D / K4d keep the chain's own variance. This is the stage at which the
            # exact-variance oracle arm is x1.54, and it is O(n) in production (2026-09-26_variance).
            v0 = torch.diagonal(C_pre)
            g = (varfix_late(l, v0, mu_pre) if callable(varfix_late)
                 else torch.as_tensor(varfix_late[l], dtype=torch.float64))
            C_pre = C_pre - torch.diag(v0) + torch.diag(v0 * g)
        if tailr:  # tail-regime restrictions (2026-09-30_tail-accuracy), state level: the tail part of a slice is not
            # used by this step at all (pair cumulants, readout AND births), consistently -- the E2 pattern
            for term, lst in _tailr_rules(tailr, mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))).items():
                for _, keep, _, _ in lst:
                    if term == "Dcol":
                        D = D * keep[None, :]
                    elif term == "Drow":
                        D = D * keep[:, None]
                    elif term == "t":
                        t = t * keep
                    elif term == "K4d" and K4d is not None:
                        K4d = K4d * keep
                    elif term == "K22" and K22 is not None:
                        K22 = K22 * (keep[:, None] * keep[None, :])
                    elif term == "K31row" and K31 is not None:
                        K31 = K31 * keep[:, None]
                    elif term == "K31col" and K31 is not None:
                        K31 = K31 * keep[None, :]
        if satrow is not None and D is not None:  # saturated-ROW cut of the (2,1) slice (2026-09-29): rows i of D =
            # kappa3[i,i,j] for neurons saturated NOW (alpha_i > satrow). In production these rows of the D pairing
            # sum_s L1_s B_s^T + L2_s A_s^T are simply not computed (left-factor rows skipped, legs untouched).
            a_now = mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))
            D_full = D if satrow_k4 in ("restore", "sep") else None
            sat_mask = a_now > satrow
            D = D * (~sat_mask).to(torch.float64)[:, None]
        else:
            D_full = None
        o = nonlin_slices(mu_pre, C_pre, t, D, K4d, K22, edge2=edge2, aug=aug, tet=tet, q22x=q22x, c0d=c0d, K31=K31,
                          marg=marg, k31q=(k31pat if isinstance(k31pat, str) else None), pairx=pairx, margtail=margtail,
                          tailr=tailr, pairnew=pairnew, marg3=marg3, k2pair=k2pair)
        k22_fix = None
        if D_full is not None and l < L - 1 and satrow_k4 == "sep":
            # E2 production form of the restoration: with k4lvl + k22pat the next layer reads K22post only through
            # Ssum = sum_ij u_i u_j K22post_ij (u = row sums of W_{l+1} o W_{l+1}). Its dependence on the cut rows i
            # (saturated: coef(2,1)_i = d2_i, w(3,p)_i ~ 0) is, dropping the d2-suppressed terms,
            #     dSsum = sum_{i in sat} u_i coef(2,2)_i (D (u o beta))_i,   beta = coef(1,2) - 2 P1 coef(1,1)
            # (beta is the correction leg's alpha, marginally corrected) -- O(S n^2) in production through
            # D_sat v = sum_s L1_s[sat] (B_s^T v) + L2_s[sat] (A_s^T v) + thin + feed terms.
            wv, _, _ = relu_wick(mu_pre, torch.diagonal(C_pre))
            k4c_ = K4d if K4d is not None else torch.zeros(n)
            tk_ = (((mu_pre / torch.diagonal(C_pre).clamp_min(1e-30).sqrt()).abs() < margtail).to(torch.float64)
                   if margtail is not None else 1.0)
            def coefv(k, p_):
                c_ = wv[(k, p_)]
                if marg and k <= 2:
                    c_ = c_ + tk_ * (wv[(k + 3, p_)] * t / 6 + wv[(k + 4, p_)] * k4c_ / 24)
                return c_
            P1v = wv[(0, 1)] + wv[(3, 1)] * t / 6 + wv[(4, 1)] * k4c_ / 24
            un = (Ws[l + 1] * Ws[l + 1]).sum(1)
            beta = coefv(1, 2) - 2.0 * P1v * coefv(1, 1)
            Dsat = D_full[sat_mask]
            dS = float((un[sat_mask] * coefv(2, 2)[sat_mask] * (Dsat @ (un * beta))).sum())
            k22_fix = dS / float(un.sum() ** 2 - (un * un).sum())
        if D_full is not None and l < L - 1 and satrow_k4 == "restore":
            # E2 (port battery): restore the kappa4 LEVEL -- a second step on the full D, keep only its K22 (with k4lvl and
            # k22pat the next layer reads only the u-weighted sum of K22post, i.e. the level; everything else, including
            # the births, uses the cut D). Production: the level is linear in the cut rows up to d2-suppressed terms, so
            # the restoration is O(S n^2) (alpha^T D_sat beta through B_s^T beta, A_s^T beta).
            o_full = nonlin_slices(mu_pre, C_pre, t, D_full, K4d, K22, edge2=edge2, aug=aug, tet=tet, q22x=q22x,
                                   c0d=c0d, K31=K31, marg=marg, k31q=(k31pat if isinstance(k31pat, str) else None),
                                   pairx=pairx, margtail=margtail, tailr=tailr, pairnew=pairnew, marg3=marg3, k2pair=k2pair)
            if satrow_ch is not None:
                # E2 channel decomposition (diagnostic): the cut D reaches only ONE channel, everything else (and the
                # level) comes from the full-D step. "cov": the next covariance K11 from the cut step, births on the
                # full D. "leg": K21 from the cut step AND the births (correction leg dK21, path leg) on the cut D --
                # the two carry the pass-through that cancels, so they are tested together.
                o_cut = o
                o = dict(o_full)
                if satrow_ch == "cov":
                    o["K11"] = o_cut["K11"]
                    D = D_full
                elif satrow_ch == "leg":
                    o["K21"] = o_cut["K21"]
                else:
                    raise ValueError(f"unknown satrow_ch {satrow_ch!r}")
            else:
                o["K22"] = o_full["K22"]
        if recv is not None:
            recv_state["t"], recv_state["D"] = t, D
        mu = o["K1"]
        rows.append(mu.clone())
        if record:
            info.append(dict(t=t.clone(), D=D.clone(), mu_pre=mu_pre.clone(), var=torch.diagonal(C_pre).clone(),
                             K4d=K4d.clone() if K4d is not None else torch.zeros(n)))
            if record == "full":  # pair slices too (float32): C_pre, K22, K31 used by the step
                info[-1].update(C=C_pre.float().clone(), K22=(K22.float().clone() if K22 is not None else None),
                                K31=(K31.float().clone() if K31 is not None else None))
        if l == L - 1:
            break
        if inject is not None and l <= inject_upto:
            if inject_regime is None:
                mu = inject[l].clone()  # oracle: exact post-activation mean; everything else stays the chain's own
            else:  # only the neurons whose (chain) alpha is in [lo, hi)
                a_in = mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))
                mu = torch.where((a_in >= inject_regime[0]) & (a_in < inject_regime[1]), inject[l], mu)
        C = o["K11"] + torch.diag(o["K2"] if o.get("K2pair") is None else o["K2pair"])
        if dead is not None and deadC:  # also drop the dead neurons from the next covariance product
            C[dead, :] = 0.0
            C[:, dead] = 0.0
        if postc is not None:  # cloud #11: a callable (l, C) -> C on the POST-activation covariance, the only thing that
            C = postc(l, C)    # feeds the next layer's C_pre = W^T C W (oracle substitution of dC and of its parts).
        a4 = 3.0 * float(o["K4"].sum() + o["K22"].sum()) / (n * (n + 2)) if k4 else 0.0
        K4post, K22post = o["K4"], o["K22"]
        if k22_fix is not None:  # E2 "sep": the level correction as a constant off-diagonal shift (only the level is read)
            K22post = K22post + k22_fix * (1.0 - torch.eye(n))
        if k4k3 or t3k4:
            legs_m = []
            for s_ in sources:
                Em = s_["E"] if s_.get("ER") is None else s_["A"] * s_["dK3"][None, :] + 3.0 * s_["B"] * s_["alpha"][None, :] + s_["ER"]
                legs_m.append((s_["A"].clone(), s_["B"].clone(), Em.clone(), s_["g"]))
            pending = (o["d1"], o["d2"], o["C0"], legs_m)
            pending_t3 = (torch.diagonal(C_pre).clone(), None if D is None else D.clone())
        d1, d2, C0 = o["d1"], o["d2"], o["C0"]
        w12_ = o["w12"]
        if margleg:  # gate the legs and build the births with the marginally corrected factors
            d1, d2, w12_ = o["d1c"], o["d2c"], o["w12c"]
        # kappa4 -> kappa3 feed (all-distinct births from the kappa4 (2,1,1) slice of the scale-mixture closure,
        # kappa4[aabc] = v (var_a C_bc + 2 C_ab C_ac)): P1 = v x tree (g -> (1+v) d2), P2 = Sym(h (x) D1 C0 D1),
        # h = v/2 d2 var
        feed_g, feed_h = 1.0, None
        if k4feed and K4d is not None and float(K4d.abs().sum()) > 0:
            vp_ = torch.diagonal(C_pre)
            v_ = float(K4d.mean() / (3.0 * (vp_ * vp_).mean()))
            if k4cons_v is not None:
                v_ = k4cons_v
            if "P1" in k4feed:
                feed_g = 1.0 + v_
            if "P2" in k4feed:
                feed_h = (0.5 * feedmult) * v_ * d2 * vp_
        # pass-through of carried sources: every leg gated by d1
        if dead is not None:  # dead rows are dropped from the next contractions (exactly zero instead of ~d1)
            d1 = torch.where(dead, torch.zeros_like(d1), d1)
        if y4chain is not None:   # the production form: births need the sources at pre(l), not yet gated
            y4chain.gate_and_retire(l)
            y4chain.birth(l, C_pre, d1, o["d2"], o["d3"], sources, _eleg)
        elif y4legs is not None:   # the same, on legs (young4.py)
            y4legs.gate_and_retire(d1, l)
            if k4young.get("conv", "pure") != "arc":
                raise ValueError("k4young impl='legs' implements ARC's convention only (conv='arc')")
            y4legs.birth(l, C_pre, d1, o["d2"], o["d3"], sources, _eleg)
        elif k4young:
            # 2026-10-04 (R2): the live young trees pass the ReLU gated by d1 on every index (as the legs);
            # those already read at their last age retire into the closure; this layer's trees are born
            ages_ = int(k4young.get("ages", 2))
            young4 = [(m_, y_) for m_, y_ in young4 if l - m_ < ages_]
            for _, y_ in young4:
                y_.contract_wick_(d1)
            ty_ = tuple(k4young.get("types", ("path", "star", "32")))
            K3y = None
            arc_ = k4young.get("conv", "pure") == "arc"   # "arc": ARC's convention (full C edge, full kappa3)
            if "32" in ty_ and (sources or T3d is not None):
                K3y = _k3_dense(sources, n, T3d, layer=l, couple=k4young.get("couple"))
                if not arc_:
                    iy_ = torch.arange(n)
                    K3y[iy_, iy_, :] = 0.0
                    K3y[iy_, :, iy_] = 0.0
                    K3y[:, iy_, iy_] = 0.0
            yb_ = _k4_trees(C_pre if arc_ else o["C0"], d1, o["d2"], o["d3"], K3y, ty_, zero_diag=not arc_)
            if yb_ is not None:
                young4.append((l, yb_))
        stride_state["d1"] = d1.clone()
        for bi_ in ss_state:
            ss_state[bi_] = d1[:, None] * ss_state[bi_]
        d1_old = None
        if agecut is not None:  # E4: legs that are old at the next layer are gated with the harsher threshold
            d1_old = d1 * ((mu_pre / torch.diagonal(C_pre).clamp_min(1e-30).sqrt()) >= float(agecut[1])).to(torch.float64)
        if recv is not None:
            recv_state["d1"] = d1.clone()
        if bz_state.get("P") is not None:  # bandz: the basis is gated like a leg; members' Z never change
            g_bz = d1_old if (d1_old is not None and int(bandz["a0"]) >= int(agecut[0])) else d1
            bz_state["P"] = g_bz.to(bz_state["P"].dtype)[:, None] * bz_state["P"]
            if bz_state.get("P64") is not None:  # the float64 shadow (diagnostic only)
                bz_state["P64"] = g_bz[:, None] * bz_state["P64"]
        if T3d is not None:   # dense3: the pass-through of the carried dense kappa3, every index gated by d1
            T3d = T3d * d1[:, None, None] * d1[None, :, None] * d1[None, None, :]
        if dense3 is not None and l in dense3.get("feedS", {}) and K4d is not None:
            # dense3 feed oracle: kappa3 births from a GIVEN pre-activation (2,1,1) slice S[a,b,c] = k4pre(a,a,b,c):
            # B(a,b,c) = 1/2 sum over the doubled index x in {a,b,c} of d2_x d1_y d1_z S(x,y,z) (a,b,c distinct), minus
            # (mode "replace") the same births from the closure's form v var_x C0_yz that the P2 feed already carries.
            S_ = torch.as_tensor(dense3["feedS"][l], dtype=torch.float64)
            vp_d = torch.diagonal(C_pre)
            if dense3.get("mode", "replace") == "replace":
                v_d = float(K4d.mean() / (3.0 * (vp_d * vp_d).mean()))
                C0d = C_pre - torch.diag(vp_d)
                S_ = S_ - (feedmult * v_d) * vp_d[:, None, None] * C0d[None, :, :]
            d1d, d2d = o["d1"], o["d2"]
            X = 0.5 * d2d[:, None, None] * d1d[None, :, None] * d1d[None, None, :] * S_
            Bd = X + X.permute(1, 0, 2) + X.permute(1, 2, 0)   # doubled index in position 1, 2, 3
            ii = torch.arange(n)
            Bd[ii, ii, :] = 0.0; Bd[ii, :, ii] = 0.0; Bd[:, ii, ii] = 0.0
            T3d = Bd if T3d is None else T3d + Bd
        h_z = (torch.diagonal(C_pre) * o["d1"] * o["d2"]) if zlegs else None
        for s in sources:
            if s.get("bz"):
                continue
            g_ = d1 if (d1_old is None or l + 1 - s["born"] < int(agecut[0])) else d1_old
            if zlegs and s["A"] is not None:   # doubled legs: inherited part gated by d1^2, then this layer's transfer
                Ec = _eleg(s)
                births = [(2.0 / 3.0) * (h_z[:, None] * s["A"]) * s["g"][None, :],
                          (2.0 / 3.0) * (h_z[:, None] * s["B"]) * s["g"][None, :]]
                if Ec is not None:
                    births += [(2.0 / 3.0) * (h_z[:, None] * Ec), (2.0 / 3.0) * (h_z[:, None] * s["A"])]
                if s.get("Z") is None:
                    s["Z"] = births
                else:
                    s["Z"] = [(g_ * g_)[:, None] * z + b_ for z, b_ in zip(s["Z"], births)]
            s["A"] = g_[:, None] * s["A"]
            s["B"] = g_[:, None] * s["B"]
            if s["E"] is not None:
                s["E"] = g_[:, None] * s["E"]
            if s.get("ER") is not None:
                s["ER"] = g_[:, None] * s["ER"]
            if s.get("B2") is not None:
                s["B2"] = g_[:, None] * s["B2"]
        if bandz is not None and bandz.get("entry", "post") == "pre":
            _band_enter_pre(bandz, bz_state, sources, l)
        # new source: tree legs from the zero-diagonal covariance, correction leg for exact repeated slices
        U = d1[:, None] * C0
        E = None
        if corr.startswith("split"):
            # production form: E = A diag(dK3) + 3 B diag(alpha) + ER, with ER = T (3 R_k^T) the thin remainder
            k = int(corr[5:].replace("rr", ""))
            alpha = w12_ - 2.0 * o["K1"] * d1
            dK3 = o["K3"] - d1 ** 3 * t
            dK21 = o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D - (d1 * d1)[:, None] * C0 * C0 * (feed_g * d2)[None, :]
            if e3:  # the E3 source's (2,1) slice at birth: d1_i^2 C0_ij D[j,i] d3_j
                dK21 = dK21 - (d1 * d1)[:, None] * C0 * D.T * o["d3"][None, :]
            Ypath = None
            if paths:
                # all O(n)-rank path births of the augmented term set (ARC mlp_kprop factor_k3, augment=True), in the
                # normalisation of the reference: every path x-p-z whose other edge is C0 shares the hub leg A and the
                # tree leg B, so together they are Sym(T Y (x) 3A (x) B) with ONE extra leg T Y, hub weights inside Y:
                #   e3 : d1_x D[p,x] d3_p   (edge x kappa3(2,1), hub on the doubled index)
                #   e3t: d2_x D[x,p] d2_p   (edge x kappa3(2,1), doubled index at the end)
                #   k22: 1/2 d2_x K22[x,p] d3_p      cc: d2_x C0[x,p]^2 d3_p   (kappa4 (2,2) / doubled-edge paths)
                d3_ = o["d3"]
                Ypath = torch.zeros(n, n)
                if "e3" in paths:
                    Ypath += d1[:, None] * D.T * d3_[None, :]
                if "e3t" in paths:
                    Ypath += d2[:, None] * D * d2[None, :]
                if "k22" in paths and K22 is not None:
                    Ypath += 0.5 * d2[:, None] * K22 * d3_[None, :]
                if "cc" in paths:
                    Ypath += d2[:, None] * C0 * C0 * d3_[None, :]
                Ypath.fill_diagonal_(0.0)
                if pathrank:  # low-rank path leg (thin in production): best rank-k approximation of Y at birth
                    Us_, Ss_, Vs_ = torch.linalg.svd(Ypath)
                    Ypath = (Us_[:, :pathrank] * Ss_[:pathrank]) @ Vs_[:pathrank]
                if pathsub:
                    dK21 = dK21 - Ypath * U  # its (2,1) slice at birth: kappa3[i,i,j] = Y_ij U_ij (hub j)
            dK21.fill_diagonal_(0.0)
            R = dK21 - alpha[:, None] * C0 * d1[None, :]
            if feed_h is not None:  # P2 makes 2 (h d1)_i C0_ij d1_j of the (2,1) slice: it moves into alpha, R unchanged
                alpha = alpha - 2.0 * feed_h * d1
            if corr.endswith("rr"):  # randomized range finder (production): k + 8 probes, one power iteration
                g_ = torch.Generator().manual_seed(1000 + l)
                Om = torch.randn(n, k + 8, generator=g_)
                Yr = R @ Om
                Yr = R @ (R.T @ Yr)
                Qr, _ = torch.linalg.qr(Yr)
                Qr = Qr[:, :k] if False else Qr
                R = Qr @ (Qr.T @ R)
            else:
                Us, Ss, Vs = torch.linalg.svd(R)
                R = (Us[:, :k] * Ss[:k]) @ Vs[:k]
            src = dict(A=None, B=U, E=None, ER=3.0 * R.T, dK3=dK3, alpha=alpha, g=feed_g * d2, C0=C0, d3=o["d3"],
                       h=feed_h, d1b=d1, B2=(Ypath if Ypath is not None else (d1[:, None] * D.T if e3 else None)),
                       g3=(torch.ones(n) if Ypath is not None else (0.5 * o["d3"] if e3 else None)), pathD=pathD,
                       born=l)
            if agecut is not None:
                src["a_birth"] = mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))
            if colcut is not None:  # birth-COLUMN cut (2026-09-29): parents p with |alpha_p| > colcut have d2_p ~ 0.
                # Every use of a column's A leg is multiplied by that column's B, g, E (= A dK3 + 3 B alpha + ER), feed
                # weight h or path leg, so zeroing those removes the column EXACTLY; production then drops it from the
                # transports and pairings. The transpose of the dead-ROW cut (deadcut).
                a_birth = mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))
                if isinstance(colcut, tuple):   # (lo, hi): keep lo <= alpha_p <= hi -- one-sided cuts
                    cm = ((a_birth >= colcut[0]) & (a_birth <= colcut[1])).to(torch.float64)
                else:
                    cm = (a_birth.abs() <= colcut).to(torch.float64)
                src["colmask"] = cm
                src["B"] = src["B"] * cm[None, :]
                src["ER"] = src["ER"] * cm[None, :]
                if src["B2"] is not None:
                    src["B2"] = src["B2"] * cm[None, :]
                for key in ("dK3", "alpha", "g", "d3", "d1b"):
                    src[key] = src[key] * cm
                for key in ("h", "g3"):
                    if src[key] is not None:
                        src[key] = src[key] * cm
            if tailr:  # tail-regime restrictions of the births: columns = hub p, rows = the leg's other index (layer l)
                for term, lst in _tailr_rules(tailr, mu_pre / torch.sqrt(torch.clamp(torch.diagonal(C_pre), min=1e-12))).items():
                    for _, keep, _, _ in lst:
                        if term == "tree_col":
                            src["g"] = src["g"] * keep
                        elif term == "corr_col":
                            src["dK3"], src["alpha"], src["ER"] = src["dK3"] * keep, src["alpha"] * keep, src["ER"] * keep[None, :]
                        elif term == "path_col" and src["B2"] is not None:
                            src["B2"] = src["B2"] * keep[None, :]
                        elif term == "star_col":
                            src["d3"] = src["d3"] * keep
                        elif term == "feed_col" and src["h"] is not None:
                            src["h"] = src["h"] * keep
                        elif term == "B_row":
                            src["B"] = src["B"] * keep[:, None]
                        elif term == "corr_row":
                            src["ER"] = src["ER"] * keep[:, None]
                        elif term == "path_row" and src["B2"] is not None:
                            src["B2"] = src["B2"] * keep[:, None]
            sources.append(src)
            if window is not None and tail is None:
                del sources[:-window]
            continue
        if corr.startswith("ulr") and ("from" not in corr or l >= int(corr.split("from")[1])):
            # tree legs from a rank-k approximation of U = diag(d1) C0; correction E exact relative to that tree
            k = int(corr[3:].split("from")[0])
            Us, Ss, Vs = torch.linalg.svd(d1[:, None] * C0)
            U = (Us[:, :k] * Ss[:k]) @ Vs[:k]
            Ud = torch.diagonal(U)
            dK3 = o["K3"] - d1 ** 3 * t - 3.0 * Ud * Ud * d2
            dK21 = (o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D
                    - 2.0 * (Ud * d2)[:, None] * U.T - U * U * d2[None, :])
            dK21.fill_diagonal_(0.0)
            E = torch.diag(dK3) + 3.0 * dK21.T
        elif corr.startswith("ulr"):
            dK3 = o["K3"] - d1 ** 3 * t
            dK21 = o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D - (d1 * d1)[:, None] * C0 * C0 * d2[None, :]
            dK21.fill_diagonal_(0.0)
            E = torch.diag(dK3) + 3.0 * dK21.T
        elif corr.startswith("free"):
            # exact split of the (2,1) correction: dK21 = diag(alpha) C0 diag(d1) + R. The first part is carried for
            # free as 3 B diag(alpha) (a column rescaling of the tree leg); only R needs a leg ("freelr{k}": rank k).
            alpha = w12_ - 2.0 * o["K1"] * d1
            dK3 = o["K3"] - d1 ** 3 * t
            dK21 = o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D - (d1 * d1)[:, None] * C0 * C0 * (feed_g * d2)[None, :]
            dK21.fill_diagonal_(0.0)
            Fpart = alpha[:, None] * C0 * d1[None, :]
            R = dK21 - Fpart
            if feed_h is not None:  # P2 makes 2 (h d1)_i C0_ij d1_j of the (2,1) slice: remove it from the free part (R unchanged)
                Fpart = Fpart - 2.0 * (feed_h * d1)[:, None] * C0 * d1[None, :]
            if corr.startswith("freelr"):
                k = int(corr[6:])
                if k == 0:
                    R = torch.zeros_like(R)
                else:
                    Us, Ss, Vs = torch.linalg.svd(R)
                    R = (Us[:, :k] * Ss[:k]) @ Vs[:k]
                R.fill_diagonal_(0.0)
            E = torch.diag(dK3) + 3.0 * (Fpart + R).T
            if k4feed:
                sources.append(dict(A=None, B=U, E=E, g=feed_g * d2, C0=C0, d3=o["d3"], h=feed_h, d1b=d1))
                if window is not None and tail is None:
                    del sources[:-window]
                continue
        elif corr.startswith("fc"):
            # tree legs from the FULL covariance; correction = exact repeated slices minus those of pass-through
            # and full-C tree; its (2,1) part optionally truncated to rank k ("fclr{k}")
            Cd = torch.diagonal(C_pre)
            Cf = C0 + torch.diag(Cd)
            U = d1[:, None] * Cf
            dK3 = o["K3"] - d1 ** 3 * t - 3.0 * (d1 * Cd) ** 2 * d2
            dK21 = (o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D
                    - 2.0 * (d1 * Cd * d2)[:, None] * (C0 * d1[None, :])
                    - (d1 * d1)[:, None] * C0 * C0 * d2[None, :])
            dK21.fill_diagonal_(0.0)
            if corr.startswith("fclr"):
                k = int(corr[4:])
                Us, Ss, Vs = torch.linalg.svd(dK21)
                dK21 = (Us[:, :k] * Ss[:k]) @ Vs[:k]
                dK21.fill_diagonal_(0.0)
            E = torch.diag(dK3) + 3.0 * dK21.T
        elif corr.startswith("fullC"):
            # cheaper variant: tree legs from the FULL covariance (its diagonal approximates the repeated slices),
            # optional diagonal-only correction (E = A diag(v) is free in production: it merges into existing products)
            Cf = C0 + torch.diag(o["K2"] * 0 + torch.diagonal(C_pre))
            U = d1[:, None] * Cf
            if corr == "fullC_diag":
                rep3_tree = 3.0 * (d1 * torch.diagonal(C_pre)) ** 2 * d2
                E = torch.diag(o["K3"] - d1 ** 3 * t - rep3_tree)
        elif corr != "none":
            d13 = d1 ** 3
            dK3 = o["K3"] - d13 * t
            V = torch.diag(dK3)
            if corr == "full" or corr.startswith("lr"):
                dK21 = o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D - (d1 * d1)[:, None] * C0 * C0 * d2[None, :]
                dK21.fill_diagonal_(0.0)
                if corr.startswith("lr"):  # rank-k truncation of the (2,1) correction (test: exact SVD)
                    k = int(corr[2:])
                    Us, Ss, Vs = torch.linalg.svd(dK21)
                    dK21 = (Us[:, :k] * Ss[:k]) @ Vs[:k]
                V = V + 3.0 * dK21.T
            E = V
        sources.append(dict(A=None, B=U, E=E, g=d2, C0=C0, d3=o["d3"]))
        if window is not None and tail is None:
            del sources[:-window]
    return torch.stack(rows), info


if __name__ == "__main__":
    # Self-check against the reference on a random small MLP: per-layer means must agree to ~1e-12.
    import sys
    torch.set_grad_enabled(False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(3)
    from mlp_kprop.mlp import MLP
    from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    torch.manual_seed(0)
    Ws = [torch.randn(n, n) * (2.0 / n) ** 0.5 for _ in range(depth)]
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE, factor=True,
                   output_all=True, up_to_layer=f"act{depth-1}")
    ref = torch.stack([Kb[f"act{l}"][1].core for l in range(depth)])
    from mlp_kprop.wick import relu_wick_coef
    mu_t, var_t = torch.randn(n), torch.rand(n) + 0.3
    w_mine, _, _ = relu_wick(mu_t, var_t)
    ew = max(float((w_mine[(k, p)] - relu_wick_coef(mu_t, var_t, k, p)).abs().max()) for k in range(5) for p in range(1, 5))
    print(f"wick coefficients vs reference: max abs diff {ew:.2e}")
    mine, info = run_chain(Ws, record=True)
    for l in range(depth):
        e = float((mine[l] - ref[l]).abs().max() / ref[l].abs().max())
        et = 0.0
        if 3 in Kb[f"pre{l}"]:
            rt = Kb[f"pre{l}"][3].get_dslice((3,))
            et = float((info[l]["t"] - rt).abs().max() / rt.abs().max())
        print(f"layer {l}: mean rel err {e:.2e}   kappa3(3,) rel err {et:.2e}")


# ---------------------------------------------------------------------------------------------------------------
# Tucker-compressed old sources: young sources (age <= young) stay dense; older ones are folded into one symmetric
# core G (r x r x r) in a shared basis P (n x r) that is transported like a leg (P <- W^T diag(d1) P).
# ---------------------------------------------------------------------------------------------------------------
def _sym3(T):
    return (T + T.permute(0, 2, 1) + T.permute(1, 0, 2) + T.permute(1, 2, 0) + T.permute(2, 0, 1) + T.permute(2, 1, 0)) / 6.0


def _rot(G, R):
    """G_abc <- sum R_aa' R_bb' R_cc' G_a'b'c'."""
    G = torch.einsum("ax,xbc->abc", R, G)
    G = torch.einsum("bx,axc->abc", R, G)
    return torch.einsum("cx,abx->abc", R, G)


def _source_core(P, s):
    """Project a dense source onto basis P (orthonormal columns) and return its symmetric core."""
    al, be = P.T @ s["A"], P.T @ s["B"]
    raw = torch.einsum("ap,bp,cp->abc", be * (3.0 * s["g"]), al, be)
    if s["E"] is not None:
        raw = raw + torch.einsum("ap,bp,cp->abc", P.T @ s["E"], al, al)
    return _sym3(raw)


def _range(M, k, gen):
    """Orthonormal basis of the dominant k-dim column space of M (randomized range finder, 1 power step)."""
    Om = torch.randn(M.shape[1], k + 8, generator=gen)
    Y = M @ Om
    Y = M @ (M.T @ Y)
    Q, _ = torch.linalg.qr(Y)
    return Q[:, :k]


def read_old(P, G):
    M = torch.einsum("ia,abc->ibc", P, G)
    t = torch.einsum("ibc,ib,ic->i", M, P, P)
    N = torch.einsum("ibc,ib->ic", M, P)
    D = N @ P.T
    D.fill_diagonal_(0.0)
    return t, D


def run_chain_tucker(Ws, young=2, rank=128, k_new=None, k4=True, corr="full", metric=2.0, seed=0):
    n = Ws[0].shape[0]
    gen = torch.Generator().manual_seed(seed)
    k_new = rank if k_new is None else k_new
    mu, C = torch.zeros(n), torch.eye(n)
    a4 = 0.0
    sources, P, G = [], None, None
    rows = []
    L = len(Ws)
    for l, W in enumerate(Ws):
        Wt = W.T
        mu_pre = Wt @ mu
        C_pre = Wt @ C @ W
        for s in sources:
            s["A"] = Wt.clone() if s["A"] is None else Wt @ s["A"]
            s["B"] = Wt @ s["B"]
            if s["E"] is not None:
                s["E"] = Wt @ s["E"]
        if P is not None:
            P, R = torch.linalg.qr(Wt @ P)
            G = _rot(G, R)
        # fold sources older than `young` into the compressed core
        while len(sources) > young:
            s = sources.pop(0)
            Qn = _range(torch.cat([s["A"], s["B"]], dim=1), k_new, gen)
            if P is None:
                P, G = Qn, torch.zeros(k_new, k_new, k_new)
            else:
                Pa, Ra = torch.linalg.qr(torch.cat([P, Qn], dim=1))
                # embed old core into the augmented basis: old coordinates x -> Pa^T P x
                E0 = Pa.T @ P
                G = torch.einsum("ax,by,cz,xyz->abc", E0, E0, E0, G)
                P = Pa
            G = G + _source_core(P, s)
            if P.shape[1] > rank:  # HOSVD truncation back to `rank`
                G1 = G.reshape(G.shape[0], -1)
                U, S, _ = torch.linalg.svd(G1 @ G1.T)
                Ur = U[:, :rank]
                G = _rot(G, Ur.T)
                P = P @ Ur
        t, D = read_slices(sources, n) if sources else (torch.zeros(n), torch.zeros(n, n))
        if P is not None:
            to, Do = read_old(P, G)
            t, D = t + to, D + Do
        if k4 and a4 != 0.0:
            K4d = torch.full((n,), a4 * metric * metric)
            K22 = torch.full((n, n), a4 * metric * metric / 3.0)
            K22.fill_diagonal_(0.0)
        else:
            K4d, K22 = torch.zeros(n), None
        o = nonlin_slices(mu_pre, C_pre, t, D, K4d, K22)
        mu = o["K1"]
        rows.append(mu.clone())
        if l == L - 1:
            break
        C = o["K11"] + torch.diag(o["K2"])
        a4 = 3.0 * float(o["K4"].sum() + o["K22"].sum()) / (n * (n + 2)) if k4 else 0.0
        d1, d2, C0 = o["d1"], o["d2"], o["C0"]
        for s in sources:
            s["A"] = d1[:, None] * s["A"]
            s["B"] = d1[:, None] * s["B"]
            if s["E"] is not None:
                s["E"] = d1[:, None] * s["E"]
        if P is not None:
            P = d1[:, None] * P  # re-orthonormalised at the next linear step
        U = d1[:, None] * C0
        E = None
        if corr != "none":
            dK3 = o["K3"] - d1 ** 3 * t
            V = torch.diag(dK3)
            if corr == "full":
                dK21 = o["K21"] - (d1 * d1)[:, None] * d1[None, :] * D - (d1 * d1)[:, None] * C0 * C0 * d2[None, :]
                dK21.fill_diagonal_(0.0)
                V = V + 3.0 * dK21.T
            E = V
        sources.append(dict(A=None, B=U, E=E, g=d2))
    return torch.stack(rows)
