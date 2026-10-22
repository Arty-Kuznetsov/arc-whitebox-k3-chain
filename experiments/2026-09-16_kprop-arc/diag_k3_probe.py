"""Can the diagonal third cumulant be BORN and TRANSPORTED cheaply? (decisive go/no-go)

edgeworth_probe.py showed a vector-sized diagonal-kappa3 correction recovers most of the k=2
mean error -- but it took the diagonal from the k=3 oracle. To deploy it we must produce the
diagonal ourselves, in O(n^2) per layer. Two ingredients, tested here before anything is built:

  STEP 1  BIRTH. For z ~ N(mu, sigma^2), y = ReLU(z), a = mu/sigma:
            m1 = sigma*(a*Phi + phi)
            m2 = sigma^2*((a^2+1)*Phi + a*phi)
            m3 = sigma^3*((a^3+3a)*Phi + (a^2+2)*phi)
            kappa3 = m3 - 3*m1*m2 + 2*m1^3
          Checked two independent ways: (a) against Monte Carlo, (b) against the ARC reference
          chain at LAYER 0, where the pre-activations are exactly Gaussian so kappa3_pre = 0 and
          the reference's own post-ReLU diagonal must equal this closed form exactly.

  STEP 2  TRANSPORT. The exact map is
            kappa3_pre[i,i,i] = sum_{abc} W[a,i]W[b,i]W[c,i] * kappa3_post[a,b,c].
          The memoryless (diagonal-dominant) approximation keeps only a=b=c:
            kappa3_pre[i,i,i] ~= sum_a W[a,i]^3 * kappa3_post[a,a,a]   ->  (Ws.weight**3) @ k3d
          which is one O(n^2) matvec. We feed the TRUE post diagonal from the k=3 chain and
          compare the prediction against the TRUE pre diagonal one layer later. If the
          correlation is near zero the whole direction is dead and we stop here.

Run:  ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
      env PYTHONPATH="$ARC/src" uv run --project "$ARC" python diag_k3_probe.py [n_mlps]
"""
import sys
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False)
torch.set_default_dtype(torch.float64)
torch.set_num_threads(3)

from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind

BAKED = Path(__file__).resolve().parent / "baked"
n, depth = 1024, 16
SQRT2PI = float(np.sqrt(2.0 * np.pi))
nphi = lambda a: np.exp(-0.5 * a * a) / SQRT2PI
nPhi = lambda a: 0.5 * (1.0 + torch.erf(torch.as_tensor(a) / np.sqrt(2.0)).numpy())


def relu_gauss_cumulants(mu, sigma):
    """First three moments and the third cumulant of ReLU(N(mu, sigma^2))."""
    a = mu / sigma
    Ph, ph = nPhi(a), nphi(a)
    m1 = sigma * (a * Ph + ph)
    m2 = sigma ** 2 * ((a * a + 1.0) * Ph + a * ph)
    m3 = sigma ** 3 * ((a ** 3 + 3.0 * a) * Ph + (a * a + 2.0) * ph)
    return m1, m2, m3, m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3


# ---------------- STEP 1a: closed form vs Monte Carlo ----------------
rng = np.random.default_rng(0)
N = 20_000_000
print("STEP 1a  closed-form ReLU cumulants vs Monte Carlo (N = 20M)")
print(f"{'mu':>6} {'sigma':>6} | {'k3 closed':>12} {'k3 MC':>12} {'rel err':>9} | {'skew':>7}")
ok1a = True
for mu, sigma in [(0.0, 1.0), (0.5, 1.0), (-0.7, 1.0), (0.3, 0.6), (-0.2, 1.4), (1.5, 0.8)]:
    z = mu + sigma * rng.standard_normal(N)
    y = np.maximum(z, 0.0)
    c1 = y.mean(); c2 = ((y - c1) ** 2).mean(); c3_mc = ((y - c1) ** 3).mean()
    _, _, _, c3 = relu_gauss_cumulants(np.array(mu), np.array(sigma))
    rel = abs(float(c3) - c3_mc) / abs(c3_mc)
    ok1a &= rel < 5e-3
    print(f"{mu:>6.2f} {sigma:>6.2f} | {float(c3):>12.6f} {c3_mc:>12.6f} {rel:>8.2e} | "
          f"{c3_mc / c2 ** 1.5:>7.3f}")
print(f"  -> {'OK' if ok1a else 'MISMATCH'}  (MC noise floor ~ 1e-3 relative at this N)\n")

# ---------------- STEP 1b + STEP 2 on the real network ----------------
gt = np.load(BAKED / "gt.npy")
N_MLPS = min(int(sys.argv[1]) if len(sys.argv) > 1 else 2, gt.shape[0])
births, corrs, rels, mags = [], [], [], []
for i in range(N_MLPS):
    W = np.load(BAKED / f"weights_{i}.npy")
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1,
              nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = torch.tensor(W[l].T, dtype=torch.float64)
    Kb = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE,
                   factor=True, output_all=True, up_to_layer=f"act{depth-1}")
    pre_d = [np.asarray(Kb[f"pre{l}"][3].get_dslice((3,))) if 3 in Kb[f"pre{l}"] else np.zeros(n)
             for l in range(depth)]
    post_d = [np.asarray(Kb[f"act{l}"][3].get_dslice((3,))) if 3 in Kb[f"act{l}"] else np.zeros(n)
              for l in range(depth)]
    pre_mu = [np.asarray(Kb[f"pre{l}"][1].to_tensor()) for l in range(depth)]
    pre_sd = [np.sqrt(np.maximum(np.asarray(Kb[f"pre{l}"][2].to_tensor()).diagonal(), 1e-12))
              for l in range(depth)]
    Wcube = [(mlp.Ws[l].weight ** 3).numpy() for l in range(depth)]
    del Kb

    # STEP 1b: at layer 0 the pre-activations are exactly Gaussian -> birth formula must be exact
    _, _, _, k3_birth0 = relu_gauss_cumulants(pre_mu[0], pre_sd[0])
    denom = np.abs(post_d[0]).max()
    births.append(float(np.abs(k3_birth0 - post_d[0]).max() / denom))

    # STEP 2: memoryless transport, true post diagonal in -> predicted pre diagonal one layer on
    c, r, m = [], [], []
    for l in range(depth - 1):
        pred = Wcube[l + 1] @ post_d[l]
        true = pre_d[l + 1]
        c.append(float(np.corrcoef(pred, true)[0, 1]))
        r.append(float(np.sqrt(((pred - true) ** 2).mean() / (true ** 2).mean())))
        m.append(float(np.sqrt((pred ** 2).mean() / (true ** 2).mean())))
    corrs.append(c); rels.append(r); mags.append(m)
    print(f"  mlp {i} done", flush=True)

print(f"\nSTEP 1b  birth formula vs the ARC reference chain at layer 0 (must be exact):")
for i, b in enumerate(births):
    print(f"   mlp {i}: max rel deviation = {b:.3e}   {'OK' if b < 1e-10 else 'MISMATCH'}")

C, R, M = np.mean(corrs, axis=0), np.mean(rels, axis=0), np.mean(mags, axis=0)
print(f"\nSTEP 2  memoryless diagonal transport vs truth ({N_MLPS} MLPs, mean over MLPs)")
print("   predicted kappa3_pre[i,i,i] at layer l+1 from the TRUE post diagonal at layer l")
print(f"{'l+1':>4} | {'corr':>7} | {'rel RMS err':>12} | {'magnitude ratio':>16}")
for l in range(depth - 1):
    print(f"{l+1:>4} | {C[l]:>7.3f} | {R[l]:>12.3f} | {M[l]:>16.3f}")
print(f"\n   mean corr = {C.mean():.3f}   mean rel RMS err = {R.mean():.3f}   "
      f"mean magnitude ratio = {M.mean():.3f}")
print("\nReading: corr ~ 1 and rel err << 1 = the diagonal is self-propagating and the cheap")
print("scheme is viable. corr ~ 0 = the diagonal is made by OFF-diagonal cross-cumulants and")
print("cannot be propagated on its own; the direction is dead at this step.")
