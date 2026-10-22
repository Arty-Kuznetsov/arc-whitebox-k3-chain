"""Error bars for the oracle attribution: the same arms against two INDEPENDENT 1e7 oracles.

`RESULTS.md` §2-§3 showed that two MC oracles differing only by their own noise move the final MSE
by 48%, so every single-channel number in §1 is quoted without an error bar it badly needs. This
runs the arms twice — against runs/mc_oracle_0.npz (seed 12345+i) and runs/mc_oracle_0_s777.npz
(seed 777, same N=1e7) — and reports both plus their spread. Two draws give a crude but honest
interval: if an arm's two values straddle 1.0x, that arm measures nothing.

The pooled arm uses the mean of the two oracles (noise/sqrt(2)) and is the best estimate we have.

    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" \
        ../2026-09-26_accuracy/oracle_errorbar.py
"""
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

CHAIN = Path(__file__).resolve().parent.parent / "2026-09-24_k3-chain"
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
A = dict(np.load(CHAIN / "runs" / "mc_oracle_0.npz"))
B_PATH = CHAIN / "runs" / "mc_oracle_0_s777.npz"
if not B_PATH.exists():
    raise SystemExit(f"{B_PATH} is missing — run: python mc_oracle.py 0 1e7 777")
B = dict(np.load(B_PATH))
W = np.load(BAKED / "weights_0.npy"); gt = np.load(BAKED / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
L = len(Ws); allL = tuple(range(L))
K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True,
           k4feed="P2", marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1,
           pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0,
           k31mult=0.65)
REF = 2.4074e-08

for k in ("var", "k3", "k4"):
    d = np.abs(np.asarray(A[k]) - np.asarray(B[k])) / np.maximum(np.abs(np.asarray(A[k])), 1e-12)
    print(f"the two oracles differ on {k:4s} by {d.mean():.2e} relative (mean), {d.max():.2e} (max)")
P = {k: 0.5 * (np.asarray(A[k]) + np.asarray(B[k])) for k in ("var", "k3", "k4")}  # pooled

arms = {
    "variance":            {"var0_layers": allL},
    "k4 (diag)":           {"k4_layers": allL},
    "k3 (diag)":           {"k3_layers": allL},
    "k3 + k4":             {"k3_layers": allL, "k4_layers": allL},
    "variance + k3 + k4":  {"var0_layers": allL, "k3_layers": allL, "k4_layers": allL},
}
print(f"\n{'arm':22s} {'oracle A':>10s} {'oracle B':>10s} {'pooled':>10s}   {'spread A/B':>10s}")
for name, o in arms.items():
    out = {}
    for tag, src in (("A", A), ("B", B), ("P", P)):
        mu, _ = run_chain(Ws, oracle=dict(o, **{k: src[k] for k in ("var", "k3", "k4")}), **K3L)
        out[tag] = REF / float(((mu[-1].numpy() - gt[-1]) ** 2).mean())
    lo, hi = sorted((out["A"], out["B"]))
    print(f"{name:22s} {out['A']:9.2f}x {out['B']:9.2f}x {out['P']:9.2f}x   {hi/lo:9.2f}x", flush=True)
print("\n'spread A/B' is the ratio of the two draws: 1.0 = the arm is reproducible, >1.3 = the number "
      "is dominated by oracle noise and must not be quoted as an attribution.")
