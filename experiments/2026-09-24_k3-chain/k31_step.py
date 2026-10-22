"""One-step test of the post-ReLU kappa4 (2,2) deficit (FRONTIER A1), extending k22_step.py with the MC data of
mc_oracle31.py: feed the nonlinear step exact (MC) pre-activation slices and add candidate terms one at a time.
Target: the MC post (2,2) slice (pair-mean ratio, corr), plus the post covariance and (2,1) slice as side checks.
Candidates:
  marg   marginal Edgeworth correction of the pair Wick coefficients, a_k -> w(k,p) + w(k+3,p) t/6 + w(k+4,p) K4d/24
  k31    the kappa4 (3,1) slice: + (1/6) [a3 b1 K31 + a1 b3 K31^T] in every Q(p,q)
  k22sq  second order in K22: + (1/32) a4 b4 K22^2;  c0k22: + (1/4) a3 b3 C0 K22 + (1/8) a4 b4 C0^2 K22
Usage: python k31_step.py [layer ...]   (needs runs/mc_oracle31_0.npz, runs/mc_oracle_0.npz)"""
import sys, math
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
import chain as ch
HERE = Path(__file__).resolve().parent
mc = np.load(HERE / "runs" / "mc_oracle_0.npz")
o31 = np.load(HERE / "runs" / "mc_oracle31_0.npz")
W = np.load(HERE.parent / "2026-09-16_kprop-arc" / "baked" / "weights_0.npy")
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
layers = [int(v) for v in sys.argv[1:]] or [int(v) for v in o31["layers"]]

calls = []
orig = ch.nonlin_slices
def hook(mu, C, t, D, K4d, K22, **kw):
    calls.append((mu.clone(), C.clone(), t.clone(), D.clone(), None if K4d is None else K4d.clone(),
                  None if K22 is None else K22.clone(), kw))
    return orig(mu, C, t, D, K4d, K22, **kw)
ch.nonlin_slices = hook
ch.run_chain(Ws, k4="memo22", corr="split24rr")
ch.nonlin_slices = orig


def step(mu, C, t, D, K4d, K22, K31=None, marg=False, k22sq=False, c0k22=False, tet=2, c0d=False, tc0=False, k3sq=False):
    """ch.nonlin_slices (default path, tet=2) + candidate terms; returns K11, K21, K22 (zero diagonals)."""
    var = torch.diagonal(C)
    w, d1, d2 = ch.relu_wick(mu, var)
    C0 = C - torch.diag(var)
    P = {p: w[(0, p)] + w[(3, p)] * t / 6 + w[(4, p)] * K4d / 24 for p in range(1, 5)}

    def coef(k, p):
        c = w[(k, p)]
        if marg and k <= 2:
            c = c + w[(k + 3, p)] * t / 6 + w[(k + 4, p)] * K4d / 24
        return c

    def Q(p, q):
        a = lambda k: coef(k, p)[:, None]
        b = lambda k: coef(k, q)[None, :]
        M = a(1) * b(1) * C0 + 0.5 * a(1) * b(2) * D.T + 0.5 * a(2) * b(1) * D + 0.5 * a(2) * b(2) * C0 * C0
        M = M + 0.25 * a(2) * b(2) * K22
        if K31 is not None:
            M = M + (1.0 / 6.0) * (a(3) * b(1) * K31 + a(1) * b(3) * K31.T)
        if k22sq:
            M = M + (1.0 / 32.0) * a(4) * b(4) * K22 * K22
        if c0k22:
            M = M + 0.25 * a(3) * b(3) * C0 * K22 + 0.125 * a(4) * b(4) * C0 * C0 * K22
        for r in range(3, tet + 1):   # higher Gaussian (Mehler) orders w_r w_r C0^r / r!
            M = M + (1.0 / math.factorial(r)) * a(r) * b(r) * C0 ** r
        if c0d:                        # edge x kappa3 (2,1)/(1,2) slices
            M = M + 0.5 * (a(2) * b(3) * D.T + a(3) * b(2) * D) * C0
        if tc0:                        # kappa3 diagonal x edge
            M = M + (1.0 / 6.0) * (a(1) * b(4) * t[None, :] + a(4) * b(1) * t[:, None]) * C0
        if k3sq:                       # two kappa3 (2,1)-slice blocks: D D^T, D^2, (D^T)^2
            M = M + 0.25 * a(3) * b(3) * D * D.T + 0.125 * (a(4) * b(2) * D * D + a(2) * b(4) * D.T * D.T)
        return M

    Q11, Q21, Q22 = Q(1, 1), Q(2, 1), Q(2, 2)
    P1 = P[1]
    K21 = Q21 - 2 * P1[:, None] * Q11
    K22o = Q22 - 2 * P1[None, :] * K21 - 2 * P1[:, None] * K21.T - 2 * Q11 * Q11 - 4 * P1[:, None] * P1[None, :] * Q11
    for M in (Q11, K21, K22o):
        M.fill_diagonal_(0.0)
    return dict(K11=Q11, K21=K21, K22=K22o)


offm = lambda n: ~np.eye(n, dtype=bool)
for l in layers:
    mu, C, t, D, K4d, K22, kw = calls[l]
    n = mu.shape[0]
    off = offm(n)
    # self-check: the default path of step() reproduces ch.nonlin_slices
    r0 = orig(mu, C, t, D, K4d, K22, **kw); r1 = step(mu, C, t, D, K4d, K22)
    err = max(float((r0[k] - r1[k]).abs().max()) for k in ("K11", "K21", "K22"))
    T = lambda k: torch.tensor(o31[k].astype(np.float64))
    Cmc = T(f"C{l}"); K22mc = T(f"k22_{l}"); K22mc.fill_diagonal_(0.0)
    K31mc = T(f"k31_{l}"); K31mc.fill_diagonal_(0.0)
    Dmc = T(f"k21_{l}"); Dmc.fill_diagonal_(0.0)
    mumc, tmc, K4mc = (torch.tensor(mc[k][l]) for k in ("mean", "k3", "k4"))
    ref22 = o31[f"postk22_{l}"].astype(np.float64)[off]
    refC = o31[f"postC{l}"].astype(np.float64)[off]
    ref21 = o31[f"postk21_{l}"].astype(np.float64)[off]
    q = o31[f"q{l}"]
    print(f"\n=== layer {l}  (self-check vs chain.nonlin_slices: max|diff| {err:.1e})")
    print(f"MC pre: K22 pair-mean {float(K22mc.numpy()[off].mean()):.4e} | chain K22 {float(K22.numpy()[off].mean()):.4e}"
          f" | K4 diag mean MC {float(K4mc.mean()):.4e} chain {float(K4d.mean()):.4e}")
    print(f"MC pre: K31 vs 3*(K4d/var)*C/3 radial model: corr "
          f"{np.corrcoef(K31mc.numpy()[off], (K4mc / torch.diagonal(Cmc))[:, None].numpy().repeat(n, 1)[off] * Cmc.numpy()[off])[0, 1]:.3f}")
    print(f"common scale q: mean {q[0]:.5f} var {q[1]:.4e} kappa3 {q[2]:.4e} (kappa3/var^2 {q[2]/q[1]**2:.2f}; chi2-like = 2)")
    print(f"MC post: K22 pair-mean {ref22.mean():.4e}")

    def report(name, r):
        x22 = r["K22"].numpy()[off]; xc = r["K11"].numpy()[off]; x21 = r["K21"].numpy()[off]
        slope = lambda x, y: float(x @ y / (x @ x))
        print(f"  {name:34s} K22 ratio {x22.mean()/ref22.mean():.4f} corr {np.corrcoef(x22, ref22)[0,1]:.3f} | "
              f"K11 slope {slope(xc, refC):.4f} relerr {np.linalg.norm(xc-refC)/np.linalg.norm(refC):.3f} | "
              f"K21 slope {slope(x21, ref21):.4f} corr {np.corrcoef(x21, ref21)[0,1]:.3f}")

    base = dict(mu=mumc, C=Cmc, t=tmc, D=D, K4d=K4mc, K22=K22mc)
    report("chain inputs", step(mu, C, t, D, K4d, K22))
    report("MC mu,C,t,K4,K22 (chain D)", step(**base))
    b2 = dict(base, D=Dmc)
    report("  + MC D", step(**b2))
    report("  + MC D + marg", step(**b2, marg=True))
    report("  + MC D + K31", step(**b2, K31=K31mc))
    report("  + MC D + marg + K31", step(**b2, K31=K31mc, marg=True))
    report("  + MC D + K22^2", step(**b2, k22sq=True))
    report("  + MC D + C0 K22", step(**b2, c0k22=True))
    report("  + MC D + all", step(**b2, K31=K31mc, marg=True, k22sq=True, c0k22=True))
    report("chain inputs + marg", step(mu, C, t, D, K4d, K22, marg=True))
    report("  + MC D + tet4 (rho^3, rho^4)", step(**b2, tet=4))
    report("  + MC D + tet6 (rho^3..rho^6)", step(**b2, tet=6))
    report("  + MC D + C0*D", step(**b2, c0d=True))
    report("  + MC D + t*C0", step(**b2, tc0=True))
    report("  + MC D + kappa3^2 (D-blocks)", step(**b2, k3sq=True))
    report("  + MC D + tet6 + C0*D + t*C0 + k3sq", step(**b2, tet=6, c0d=True, tc0=True, k3sq=True))
    report("chain inputs + C0*D", step(mu, C, t, D, K4d, K22, c0d=True))
    report("MC inputs, chain D + C0*D", step(**base, c0d=True))
    print(f"  D.C0 pair-mean: chain {float((D * (C - torch.diag(torch.diagonal(C)))).numpy()[off].mean()):.4e}  MC {float((Dmc * (Cmc - torch.diag(torch.diagonal(Cmc)))).numpy()[off].mean()):.4e};  corr(chain D, MC D) {np.corrcoef(D.numpy()[off], Dmc.numpy()[off])[0, 1]:.3f}, slope {float(D.numpy()[off] @ Dmc.numpy()[off] / (D.numpy()[off] @ D.numpy()[off])):.3f}")
    report("  Gaussian only (MC mu,C; t=D=K4=K22=0)", step(mumc, Cmc, 0 * tmc, 0 * Dmc, 0 * K4mc, 0 * K22mc))
    report("  Gaussian only, tet6", step(mumc, Cmc, 0 * tmc, 0 * Dmc, 0 * K4mc, 0 * K22mc, tet=6))
