"""The 39 kappa4-corrector features in flopscope numpy -- the SAME definitions the torch leave-one-out validated
(k4split.own + k4split.pair + the layer index), computed from objects the production estimator already holds at the
point just before _nonlin. Shared by the recording run (training) and the shipped estimator (inference), so there is
no train/serve skew."""
import flopscope as flops
import flopscope.numpy as fnp

_F32 = fnp.float32
NF = 39


def k4feats(mu_pre, var, t, K4d, C0, D, K22, w_l, w_next, l, n, out):
    """out: an (n, 39) float32 buffer. C0 has a zero diagonal; D, K22 are n x n."""
    v = fnp.maximum(var, 1e-12)
    sd = fnp.sqrt(v)
    a = mu_pre / sd
    ac = fnp.clip(a, -8.0, 8.0)
    ac2 = ac * ac
    phi = flops.stats.norm.pdf(a).astype(_F32)
    Ph = flops.stats.norm.cdf(a).astype(_F32)
    order = fnp.argsort(-a)
    rank = fnp.argsort(order).astype(_F32) / float(n)
    cols = [a, ac2, ac2 * ac, ac2 * ac2, phi, Ph, a * phi, (ac2 - 1.0) * phi, sd, v,
            fnp.clip(t / (sd * sd * sd + 1e-30), -50.0, 50.0), fnp.clip(K4d / (v * v + 1e-30), -50.0, 50.0),
            fnp.sum(w_next * w_next, axis=1), fnp.sum(w_l * w_l, axis=0), rank, mu_pre, t, K4d, Ph * v,
            t * ac, K4d * ac]
    aC = fnp.abs(C0)
    C2 = C0 * C0
    s2 = fnp.sum(C2, axis=1)
    cols += [fnp.sum(aC, axis=1), s2, fnp.sum(C2 * C0, axis=1), fnp.sum(C0, axis=1), s2 / v, fnp.max(aC, axis=1),
             fnp.sum(C2 * C2, axis=1), fnp.sum(aC * fnp.sqrt(aC), axis=1)]
    aD = fnp.abs(D)
    D2 = D * D
    cols += [fnp.sum(aD, axis=1), fnp.sum(D2, axis=1), fnp.sum(D, axis=1),
             fnp.sum(aD, axis=0), fnp.sum(D2, axis=0), fnp.sum(D, axis=0)]
    cols += [fnp.sum(fnp.abs(K22), axis=1), fnp.sum(K22 * K22, axis=1), fnp.sum(K22, axis=1)]
    for j, c in enumerate(cols):
        fnp.multiply(c, 1.0, out=out[:, j])
    fnp.multiply(fnp.ones(n, dtype=_F32), float(l) / 16.0, out=out[:, 38])
    return out
