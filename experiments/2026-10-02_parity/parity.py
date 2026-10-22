"""The parity rule of the random-projection theorem (write-up §6.4), on the per-neuron 1e7 moment oracles.
Layer l's pre-activation moments are projections of layer l-1's post-activation joint cumulants through the column w_i
of W_l, which is independent of them: var_i = w_i^T C w_i (quadratic), t_i = sum W_ai W_bi W_ci kappa3_abc (cubic).
The chain computes both the same way from its own post-activation state, so its errors are the same projections of
its joint errors dC, dK3. Given those, the mean over neurons is E_w[.]:
  * EVEN order (variance): a coherent level (2/n) tr dC, the same for every neuron;
  * ODD order (kappa3): none at all -- odd Gaussian moments vanish.
Test per layer: z = mean_i(e_i) / sqrt(mean_i(e_i^2) / n), ~N(0,1) without a coherent level, large with one.
(kappa4 is NOT a projection in the chain -- it is closed, 3 v var^2 -- so it is reported only for contrast.)
What a per-neuron feature can see of the kappa3 error (no intercept, noise-subtracted R2 = explained / signal energy):
  on the chain's own t (a cubic form of w_i against the chain's kappa3: the part of dK3 along kappa3_chain),
  on mu_pre (a linear form: the part along m_a = sum_b dK3_abb), and on both.
Oracles: experiments/2026-09-24_k3-chain/oracles/mc_oracle_<m>.npz and _s777 (1e7 samples each). Chain: B9.
    cd experiments/2026-10-02_parity && PYTHONPATH=$ARC/src uv run --project $ARC python parity.py <mlp>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                            # noqa: E402  (threads, chain.py on the path)
from chain import run_chain                          # noqa: E402

torch.set_grad_enabled(False); torch.set_num_threads(3)
m = int(sys.argv[1])
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
oA, oB = (np.load(ORA / f) for f in (f"mc_oracle_{m}.npz", f"mc_oracle_{m}_s777.npz"))
Ws = np.load(pb.BAKED / f"weights_{m}.npy")
_, info = run_chain([torch.tensor(w, dtype=torch.float64) for w in Ws], record=True,
                    **pb.matched(-2.5, satrow=2.5, satrow_k4="sep"))
L, n = len(info), info[0]["mu_pre"].shape[0]


def r2sig(F, eP, nu):
    """Noise-subtracted share of the error's signal energy explained by a no-intercept least-squares fit on F."""
    beta, *_ = np.linalg.lstsq(F, eP, rcond=None)
    fit = F @ beta
    S = float(eP @ eP - nu @ nu)
    return float(fit @ fit) / S if S > 0 else float("nan")


out = {"mlp": m, "layers": []}
print(f"MLP {m}: z = mean(e)/sqrt(mean(e^2)/n), two 1e7 oracles averaged; R2 = share of the kappa3 error's signal")
print(f"{'l':>3s} {'z var':>8s} {'z k3':>7s} {'z k4*':>8s} | {'k3: rel err':>11s} {'noise/sig':>9s} "
      f"{'R2 t':>6s} {'R2 mu':>6s} {'R2 both':>7s}")
for l in range(2, L):
    row = {"l": l}
    for ch, ck, ok in (("var", "var", "var"), ("k3", "t", "k3"), ("k4", "K4d", "k4")):
        x = info[l][ck].numpy().astype(np.float64)
        a, b = oA[ok][l].astype(np.float64), oB[ok][l].astype(np.float64)
        eP, nu = x - 0.5 * (a + b), 0.5 * (a - b)
        row[f"z_{ch}"] = float(eP.mean() / np.sqrt((eP @ eP) / n / n))
        if ch == "k3":
            t = x
            mu = info[l]["mu_pre"].numpy().astype(np.float64)
            row["k3_rel"] = float(np.sqrt(max(eP @ eP - nu @ nu, 0) / (0.25 * (a + b) @ (a + b))))
            row["k3_noise_over_signal"] = float((nu @ nu) / max(eP @ eP - nu @ nu, 1e-300))
            row["r2_t"] = r2sig(t[:, None], eP, nu)
            row["r2_mu"] = r2sig(mu[:, None], eP, nu)
            row["r2_both"] = r2sig(np.stack([t, mu], 1), eP, nu)
    out["layers"].append(row)
    print(f"{l:3d} {row['z_var']:8.1f} {row['z_k3']:7.2f} {row['z_k4']:8.1f} | {row['k3_rel']:11.3f} "
          f"{row['k3_noise_over_signal']:9.3f} {row['r2_t']:6.3f} {row['r2_mu']:6.3f} {row['r2_both']:7.3f}", flush=True)
(HERE / "res").mkdir(exist_ok=True)
json.dump(out, open(HERE / "res" / f"parity_{m}.json", "w"), indent=1)
