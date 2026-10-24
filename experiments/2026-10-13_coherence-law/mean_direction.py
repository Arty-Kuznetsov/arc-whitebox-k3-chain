"""Mean-direction share of the error vs the coherent share, and where the learnable part lies (2026-10-24).

Asked by megan_park (forum 18249, post 3; her write-up 18282 §2.3): does error_anatomy's coherent share (projection of a
layer's post-mean error on the uniform direction 1/sqrt(n)) line up with the share along the mean direction m?
For every saved run of this folder (ARC SIMPLE k = 3 'c', k = 2 'c2', closures 'cs'/'cz'):
  1. share of the post-mean error along 1, along m-hat and in span{1, m}, by layer (1024 x 16: 1e9 truth; synthetic
     shapes: two MC seeds, energies as cross products so the MC noise drops out);
  2. at the last layer, the leave-one-network-out learnable share (coh_analyze's ridge on 8 smooth features) of the
     full error and of the error with its m-component removed (both normalised by the full energy), and the share of
     the learned correction that lies along m-hat.
usage: python3 mean_direction.py   (numpy only, seconds)
"""
import glob, importlib.util, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ca", HERE / "coh_analyze.py")
ca = importlib.util.module_from_spec(spec); sys.argv = sys.argv[:1]; spec.loader.exec_module(ca)


def mtrue(d, l):
    return d["gt_m"][l] if "gt_m" in d else (d["a_m"][l] + d["b_m"][l]) / 2


def shares(est, tag):
    ca.EST = est
    _, nets = ca.load(tag)
    L, n = nets[0]["d"]["c_m"].shape
    one = np.ones(n) / np.sqrt(n)
    s1, sm, ssp, cos = [], [], [], []
    for N in nets:
        r1, rm, rsp, rc = [], [], [], []
        for l in range(L):
            a, b = N["eA"]["m"][l], N["eB"]["m"][l]
            m = mtrue(N["d"], l); mh = m / np.linalg.norm(m)
            E = float(a @ b)
            q, _ = np.linalg.qr(np.stack([one, mh], 1))
            r1.append((a @ one) * (b @ one) / E); rm.append((a @ mh) * (b @ mh) / E)
            rsp.append(float((a @ q) @ (b @ q)) / E); rc.append(float(mh @ one))
        s1.append(r1); sm.append(rm); ssp.append(rsp); cos.append(rc)
    return [np.mean(x, 0) for x in (s1, sm, ssp, cos)]


def learnable(est, tag):
    ca.EST = est
    _, nets = ca.load(tag)
    l = nets[0]["d"]["c_m"].shape[0] - 1
    X = [ca.feats(N["d"], l) for N in nets]
    mh = [mtrue(N["d"], l) / np.linalg.norm(mtrue(N["d"], l)) for N in nets]
    perp = lambda v, u: v - (v @ u) * u
    full, prp, along = [], [], []
    for k in range(len(nets)):
        tr = [j for j in range(len(nets)) if j != k]
        a, b = nets[k]["eA"]["m"][l], nets[k]["eB"]["m"][l]
        S = float(np.mean(a * b))
        f = ca.ridge(np.vstack([X[j] for j in tr]), np.concatenate([nets[j]["eA"]["m"][l] for j in tr]))
        p = f(X[k]); full.append(1 - float(np.mean((a - p) * (b - p))) / S); along.append(float((p @ mh[k]) ** 2 / (p @ p)))
        g = ca.ridge(np.vstack([X[j] for j in tr]), np.concatenate([perp(nets[j]["eA"]["m"][l], mh[j]) for j in tr]))
        ap, bp, p = perp(a, mh[k]), perp(b, mh[k]), g(X[k])
        prp.append((float(np.mean(ap * bp)) - float(np.mean((ap - p) * (bp - p)))) / S)
    return np.mean(full), np.mean(prp), np.mean(along)


if __name__ == "__main__":
    for tag in ["n1024_L16", "n256_L8", "n512_L16", "n256_L16", "n256_L32"]:
        for est in ["c", "c2", "cs", "cz"]:
            try:
                s1, sm, ssp, cos = shares(est, tag)
            except Exception:
                continue
            fl, fp, al = learnable(est, tag)
            print(f"{tag} {est:>2}: last layer along 1 {100*s1[-1]:5.2f}%, along m {100*sm[-1]:5.2f}%, span {100*ssp[-1]:5.2f}%"
                  f" | learnable {100*fl:5.1f}%, after removing m {100*fp:5.1f}%, correction along m {100*al:3.0f}%"
                  f" | cos(1,m) last {cos[-1]:.3f}")
            if tag == "n1024_L16":
                print("   along 1 by layer:", np.round(100 * s1, 2).tolist())
                print("   along m by layer:", np.round(100 * sm, 2).tolist())
                print("   cos(1,m) by layer:", np.round(cos, 3).tolist())
