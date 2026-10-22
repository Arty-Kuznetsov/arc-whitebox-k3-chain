"""Chain side + comparison of the random-direction test (write-up §6.4, Propositions 1-2; cloud #12 task C).
For every target layer l: run the float64 chain (B9) on the real network (-> var_i, t_i, K4d_i of layer l for the real
columns), and once more with W_l replaced by the saved random matrix V_l (-> the same quantities read along random
directions independent of the network; layers > l are then meaningless and ignored). Against the two-seed MC of
randdir_mc.py (same samples for real columns and random directions): errors e = chain - MC, noise nu = (A - B)/2.
Check: (i) the mean error over real columns equals the mean over random directions (the coherent part: ~0 for kappa3,
a level for the variance); (ii) the noise-subtracted spread of the real columns' errors equals the random directions'
(i.i.d. with that variance); (iii) the standardised distributions match (quantiles, kurtosis).
    cd experiments/2026-10-07_randdir && PYTHONPATH=$ARC/src uv run --project $ARC python randdir.py 0"""
import json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                   # noqa: E402
from chain import run_chain                 # noqa: E402
torch.set_grad_enabled(False); torch.set_num_threads(3)
m = int(sys.argv[1])
A, B = (np.load(HERE / "res" / f"mc_{m}_s{s}.npz") for s in (1001, 1002))
targets = [int(x) for x in A["targets"]]
Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{m}.npy")]
CFG = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
_, info_real = run_chain(Ws, record=True, **CFG)
n = info_real[0]["mu_pre"].shape[0]
KEY = {"var": "var", "k3": "t", "k4": "K4d"}
def stats(e, nu):
    n_ = e.size
    mean = float(e.mean()); se = float(np.sqrt((e @ e) / n_ / n_))          # parity.py's standard error
    energy = float(e @ e / n_ - nu @ nu / n_)                                 # noise-subtracted mean square
    sd_sig = float(np.sqrt(max(energy - mean * mean, 0.0)))                   # noise-subtracted spread about the mean
    sd_raw = float(e.std()); zz = (e - mean) / sd_raw
    q = np.quantile(zz, [0.01, 0.10, 0.50, 0.90, 0.99]).tolist()
    kurt = float((zz ** 4).mean())
    return dict(mean=mean, z=mean / se, se=se, sd_sig=sd_sig, sd_raw=sd_raw, noise_sd=float(np.sqrt(nu @ nu / n_)), q=q, kurt=kurt)
out = {"mlp": m, "N": int(A["N"]), "layers": []}
print(f"MLP {m}, N = {int(A['N']):,} x 2 seeds; chain B9. For each layer and order: real columns vs random directions.")
print(f"{'l':>3s} {'ord':>3s} | {'mean real':>10s} {'z':>6s} | {'mean rand':>10s} {'z':>6s} | {'sd real':>9s} {'sd rand':>9s} {'ratio':>6s} | {'noise':>8s} | {'kurt r/v':>10s} | {'q99 r/v':>10s}")
for l in targets:
    Wl = list(Ws); Wl[l] = torch.tensor(A[f"V_{l}"], dtype=torch.float64)
    _, info_V = run_chain(Wl, record=True, **CFG)
    row = {"l": l}
    for ch in ("var", "k3", "k4"):
        xr = info_real[l][KEY[ch]].numpy().astype(np.float64); xv = info_V[l][KEY[ch]].numpy().astype(np.float64)
        ar, br = A[f"{ch}_real_{l}"], B[f"{ch}_real_{l}"]; av, bv = A[f"{ch}_rand_{l}"], B[f"{ch}_rand_{l}"]
        sr = stats(xr - 0.5 * (ar + br), 0.5 * (ar - br)); sv = stats(xv - 0.5 * (av + bv), 0.5 * (av - bv))
        dz = (sr["mean"] - sv["mean"]) / np.sqrt(sr["se"] ** 2 + sv["se"] ** 2)
        row[ch] = {"real": sr, "rand": sv, "mean_diff_z": float(dz), "sd_ratio": sr["sd_sig"] / sv["sd_sig"] if sv["sd_sig"] else float("nan"),
                   "scale": float(np.sqrt(np.mean((0.5 * (ar + br)) ** 2)))}
        print(f"{l:3d} {ch:>3s} | {sr['mean']:10.3e} {sr['z']:6.2f} | {sv['mean']:10.3e} {sv['z']:6.2f} | {sr['sd_sig']:9.3e} {sv['sd_sig']:9.3e} {row[ch]['sd_ratio']:6.3f} | {sr['noise_sd']:8.2e} | {sr['kurt']:4.1f}/{sv['kurt']:4.1f} | {sr['q'][4]:4.2f}/{sv['q'][4]:4.2f}", flush=True)
    out["layers"].append(row)
json.dump(out, open(HERE / "res" / f"randdir_{m}.json", "w"), indent=1)
print("saved res/randdir_%d.json" % m)
