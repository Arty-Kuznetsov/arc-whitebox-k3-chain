"""Proposition 3 and Proposition 5's R^2 clause on the TRUE post-activation third-cumulant tensor (Monte Carlo),
not on the chain's tensor (../2026-10-13_prop3 measured it on the chain's own dense tensor).
Per selected layer l of a synthetic n x L network: G = kappa3(h_l) (n^3, two independent seeds A/B, each from two passes over
its stream: exact sample means, then the centred third moment), v = partial trace, energies as cross products of the seeds:
||G||^2 ~ <G_A, G_B>, ||G_AD||^2 (all-distinct entries), ||v||^2 ~ <v_A, v_B>.
  predicted share = sqrt(6 ||G_AD||^2 / (6 ||G||^2 + 9 ||v||^2));  measured share = ||AD|| / ||t'|| with the network's own
  W_{l+1} (AD = the all-distinct part of the contraction), energies again as cross products;
  predicted R^2(l+1) = cos^2(v, m) q_v / (1 + q_v), q_v = 9||v||^2/(6||G||^2), against the measured noise-corrected R^2 of
  kappa3 on mu at layer l+1 (pass-1 statistics of both seeds).
usage: python prop3_truth.py <n> <L> <net> <N per seed> <layers comma>   -> res/p3t_n<n>_L<L>_net<k>.json"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-13_coherence-law"))
torch.set_default_dtype(torch.float64)
from coh import weights                                          # noqa: E402
torch.set_default_dtype(torch.float32)


def stream(n, N, seed, B):
    g = torch.Generator().manual_seed(seed); done = 0
    while done < N:
        b = min(B, N - done); yield torch.randn(b, n, generator=g, dtype=torch.float32); done += b


def run_seed(W, N, seed, layers, B=None):
    L, n = len(W), W[0].shape[0]
    B = B or min(4096, 2 ** 27 // (n * n))                       # the outer-product batch stays <= 0.5 GB
    S = torch.zeros(4, L, n, dtype=torch.float64)                 # z, z^2, z^3 (pre), h (post)
    for x in stream(n, N, seed, B):
        y = x
        for l in range(L):
            z = y @ W[l]; zd = z.double(); z2 = zd * zd
            S[0, l] += zd.sum(0); S[1, l] += z2.sum(0); S[2, l] += (z2 * zd).sum(0)
            y = torch.relu(z); S[3, l] += y.double().sum(0)
    mu = S[0] / N; s2 = S[1] / N; k3 = S[2] / N - 3 * mu * s2 + 2 * mu ** 3; m = S[3] / N
    G = {l: torch.zeros(n * n, n, dtype=torch.float64) for l in layers}
    for x in stream(n, N, seed, B):
        y = x
        for l in range(L):
            y = torch.relu(y @ W[l])
            if l in G:
                c = (y.double() - m[l]).float(); b = c.shape[0]
                G[l] += ((c[:, :, None] * c[:, None, :]).reshape(b, n * n).T @ c).double()
    return dict(mu=mu.numpy(), k3=k3.numpy(), var=(s2 - mu * mu).numpy(), m=m.numpy(),
                G={l: (G[l] / N).reshape(n, n, n).numpy() for l in layers})


def main(n, L, net, N, layers):
    W = [w.to(torch.float32) for w in weights(n, L, net)]; t0 = time.time()
    A = run_seed(W, N, 101 + 1000 * net, layers); Bs = run_seed(W, N, 202 + 1000 * net, layers)
    idx = np.arange(n); out = {"n": n, "L": L, "net": net, "N": N, "layers": {}}
    dot = lambda a, b: float((a * b).sum())
    for l in layers:
        GA, GB = A["G"][l], Bs["G"][l]
        mask = np.ones((n, n, n), bool); mask[idx, idx, :] = mask[idx, :, idx] = mask[:, idx, idx] = False
        tot, ad = dot(GA, GB), dot(GA * mask, GB * mask)
        vA, vB = GA[:, idx, idx].sum(1), GB[:, idx, idx].sum(1); vv = dot(vA, vB)
        m = 0.5 * (A["m"][l] + Bs["m"][l]); v = 0.5 * (vA + vB)
        pred = (6 * ad / (6 * tot + 9 * vv)) ** 0.5; qv = 9 * vv / (6 * tot)
        Wn = np.asarray(W[l + 1], dtype=np.float64)
        con = lambda G: np.einsum("abc,ai,bi,ci->i", G, Wn, Wn, Wn)
        tA, tB, aA, aB = con(GA), con(GB), con(GA * mask), con(GB * mask)
        meas = (dot(aA, aB) / dot(tA, tB)) ** 0.5
        cosvm = float(v @ m / np.sqrt(vv * (m @ m)))
        # measured R^2 of kappa3 on mu at layer l+1, noise-corrected with the two seeds
        mu1 = 0.5 * (A["mu"][l + 1] + Bs["mu"][l + 1]); kA, kB = A["k3"][l + 1], Bs["k3"][l + 1]
        X = np.stack([np.ones(n), mu1], 1); bcoef = np.linalg.lstsq(X, 0.5 * (kA + kB), rcond=None)[0]; p = X @ bcoef
        Ssig = np.mean((kA - kA.mean()) * (kB - kB.mean())); R = np.mean((kA - p) * (kB - p)) - (kA.mean() - p.mean()) * (kB.mean() - p.mean())
        out["layers"][l] = dict(share_meas=meas, share_pred=pred, ratio=meas / pred, f_ad=ad / tot, q_v=qv, cos_v_m=cosvm,
                                R2_pred=cosvm ** 2 * qv / (1 + qv), R2_meas=1 - R / Ssig,
                                c_pred_n=6 * float(v @ m) / float(m @ m), c_meas_n=float(bcoef[1]) * n)
        r = out["layers"][l]
        print(f"n{n} L{L} net{net} layer {l}: share meas {meas:.3f} pred {pred:.3f} (x{meas/pred:.2f}) f_AD {ad/tot:.2f} q_v {qv:.2f} "
              f"cos(v,m) {cosvm:.3f} | R2 next pred {r['R2_pred']:.3f} meas {r['R2_meas']:.3f} | c*n pred {r['c_pred_n']:.2f} meas {r['c_meas_n']:.2f}", flush=True)
    out["seconds"] = time.time() - t0
    json.dump(out, open(HERE / "res" / f"p3t_n{n}_L{L}_net{net}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(float(sys.argv[4])), [int(x) for x in sys.argv[5].split(",")])
