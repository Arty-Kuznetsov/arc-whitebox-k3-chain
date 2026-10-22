"""Add a second estimator to the files of coh.py: ARC's SIMPLE k = 2 (covariance propagation with the Gaussian closure,
mlp_kprop, unfactored), so the same Monte-Carlo truth judges both truncation orders. Adds c2_m (post-means), c2_mu,
c2_var (pre-activation mean and variance) per layer; c2_t = 0 (k = 2 carries no third cumulant).
usage: python coh_k2.py <glob tag, e.g. n256_L32_net>"""
import sys, glob
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP                                   # noqa: E402
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind            # noqa: E402
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from coh import weights, BAKED                                  # noqa: E402


def k2(Ws):
    L, n = len(Ws), Ws[0].shape[0]
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=L + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(L):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=2, kind=Kind.SIMPLE, factor=False, output_all=True,
                    up_to_layer=f"act{L - 1}")
    m = np.stack([out[f"act{l}"][1].to_tensor().numpy() for l in range(L)])
    mu = np.stack([out[f"pre{l}"][1].to_tensor().numpy() for l in range(L)])
    var = np.stack([torch.diagonal(out[f"pre{l}"][2].to_tensor()).numpy() for l in range(L)])
    return m.astype(np.float64), mu.astype(np.float64), var.astype(np.float64)


for f in sorted(glob.glob(str(HERE / "res" / f"{sys.argv[1]}*.npz"))):
    d = dict(np.load(f))
    if "c2_m" in d: print("has k2:", Path(f).name, flush=True); continue
    name = Path(f).stem
    if "baked" in name:
        k = int(name.split("baked")[1]); Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(BAKED / f"weights_{k}.npy")]
    else:
        n_, L_, k = name.split("_"); Ws = weights(int(n_[1:]), int(L_[1:]), int(k[3:]))
    m, mu, var = k2(Ws)
    d.update(c2_m=m, c2_mu=mu, c2_var=var, c2_t=np.zeros_like(m))
    np.savez(f, **d)
    print(f"k2 added: {Path(f).name}; final MSE vs chain k3: {np.mean((m[-1] - d['c_m'][-1]) ** 2):.2e}", flush=True)
