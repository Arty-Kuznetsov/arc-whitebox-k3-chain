"""Monte Carlo for RESULTS.md §1: per layer of a public 1024 x 16 network, under Gaussian or sphere input,
  pre-activation:  mu, var, k3 (per neuron), vpre_a = sum_b k3(z_a, z_b, z_b) = Cov(z_a, ||z - mu||^2)
  post-activation: m, var_h, v_a = sum_b k3(h_a, h_b, h_b) = Cov(h_a, ||h - m||^2), Cm = Cov(h, h) @ m
Two passes over the same random stream: pass 1 gives the exact sample means, pass 2 centres by them, so the partial traces
are plain third central moments E[y_a ||y||^2] (no Gram matrix, no cancellation). float32 GEMMs, float64 sums.
Input: gauss = N(0, I_n); sphere = the same draws normalised to radius sqrt(n) (removes the radial scale mode).
usage: python vm_mc.py <net> <gauss|sphere> <N> <seed> [relu|leaky|gelu|silu]   -> res/vm_<net>[_<act>]_<input>_s<seed>.npz
  (leaky: slope 0.2, weights rescaled to q = 2/(1 + 0.2^2); gelu, silu: q = 2)
  <net> = k (public 1024 x 16 network k) or s<n>x<L>-<k> (synthetic network k of ../2026-10-13_coherence-law/coh.py)"""
import sys, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
BAKED = HERE.parent / "2026-09-16_kprop-arc" / "baked"


def stream(n, N, seed, sphere, B):
    g = torch.Generator().manual_seed(seed); done = 0
    while done < N:
        b = min(B, N - done)
        x = torch.randn(b, n, generator=g, dtype=torch.float32)
        if sphere:
            x = x * (n ** 0.5 / x.norm(dim=1, keepdim=True))
        x = x.to(torch.float32)
        yield x; done += b


def load(net):
    if net.startswith("s"):
        sys.path.insert(0, str(HERE.parent / "2026-10-13_coherence-law"))
        torch.set_default_dtype(torch.float64)
        from coh import weights                                  # noqa: E402  (imports the chain; torch env)
        shp, k = net[1:].split("-"); n_, L_ = shp.split("x")
        return [w.to(torch.float32) for w in weights(int(n_), int(L_), int(k))]
    return [torch.from_numpy(np.array(w, dtype=np.float32)) for w in np.load(BAKED / f"weights_{net}.npy")]


ACT = {"relu": (torch.relu, 2.0), "leaky": (lambda z: torch.nn.functional.leaky_relu(z, 0.2), 2.0 / 1.04),
       "gelu": (torch.nn.functional.gelu, 2.0), "silu": (torch.nn.functional.silu, 2.0)}


def main(net, inp, N, seed, act="relu", B=1 << 15):
    W = load(net); f, q = ACT[act]; W = [w * (q / 2.0) ** 0.5 for w in W]
    L, n = len(W), W[0].shape[0]; sph = inp == "sphere"; t0 = time.time()
    S = torch.zeros(5, L, n, dtype=torch.float64)            # sum z, z^2, z^3, h, h^2
    for x in stream(n, N, seed, sph, B):
        y = x
        for l in range(L):
            z = y @ W[l]; zd = z.double(); z2 = zd * zd
            S[0, l] += zd.sum(0); S[1, l] += z2.sum(0); S[2, l] += (z2 * zd).sum(0)
            y = f(z); yd = y.double(); S[3, l] += yd.sum(0); S[4, l] += (yd * yd).sum(0)
    mu = S[0] / N; m = S[3] / N
    var = S[1] / N - mu * mu; k3 = S[2] / N - 3 * mu * S[1] / N + 2 * mu ** 3; varh = S[4] / N - m * m
    t1 = time.time()
    V = torch.zeros(3, L, n, dtype=torch.float64)             # vpre, v, Cm
    mhat = m / m.norm(dim=1, keepdim=True)
    A = torch.zeros(L, 4, dtype=torch.float64)               # sum of beta, beta^2, beta^3, Q*beta (beta = y.mhat)
    for x in stream(n, N, seed, sph, B):
        y = x
        for l in range(L):
            z = y @ W[l]; e = z.double() - mu[l]
            V[0, l] += (e * (e * e).sum(1, keepdim=True)).sum(0)
            y = f(z); c = y.double() - m[l]; Q = (c * c).sum(1, keepdim=True)
            V[1, l] += (c * Q).sum(0)
            beta = c @ mhat[l]
            V[2, l] += (c * beta[:, None]).sum(0)
            A[l, 0] += beta.sum(); A[l, 1] += (beta ** 2).sum(); A[l, 2] += (beta ** 3).sum(); A[l, 3] += (beta * Q[:, 0]).sum()
    V /= N; A /= N
    out = HERE / "res" / (f"vm_{net}_{inp}_s{seed}.npz" if act == "relu" else f"vm_{net}-{act}_{inp}_s{seed}.npz")
    np.savez(out, N=N, q=q, act=act, mu=mu.numpy(), var=var.numpy(), k3=k3.numpy(), m=m.numpy(), varh=varh.numpy(),
             vpre=V[0].numpy(), v=V[1].numpy(), Cmhat=V[2].numpy(), amp=A.numpy())
    print(f"{out.name}: pass 1 {t1 - t0:.0f} s, pass 2 {time.time() - t1:.0f} s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(float(sys.argv[3])), int(sys.argv[4]), *(sys.argv[5:6]))
