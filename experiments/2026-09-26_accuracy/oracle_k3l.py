"""Where does k3l's remaining error live? Oracle substitutions on the CURRENT base (k3l), MLP 0.

`2026-09-24_k3-chain/oracle_subst3.py` answered this for k3f. Since then k3i (path births), k3j
(dead-row cut) and k3l (K31 x 0.65) landed and the raw MSE fell 2.80e-8 -> 2.31e-8, so the
attribution has to be redone before choosing the next accuracy lever: a channel that was worth
x2 on k3f may be worth x1.05 now.

Each arm replaces one pre-activation quantity by its MC-oracle value at every layer and reports
the final-layer MSE against the baked ground truth. Self-check: the no-oracle arm must print
2.4074e-08, k3l's logged value for MLP 0 (2026-09-24_k3-chain/logs/ablate_k3l_t_n10.txt).

Run it from the k3-chain folder (chain.py and runs/mc_oracle*.npz live there).

    # Mac: torch is in the ARC env, there is no .venv-torch here
    cd experiments/2026-09-24_k3-chain
    ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
    env PYTHONPATH="$ARC/src:." nice -n 10 "$ARC/.venv/bin/python" \
        ../2026-09-26_accuracy/oracle_k3l.py

    # a machine built by scripts/cloud_setup.sh, where the light torch env exists
    cd experiments/2026-09-24_k3-chain
    PYTHONPATH=. ../../.venv-torch/bin/python ../2026-09-26_accuracy/oracle_k3l.py
"""
from pathlib import Path

import numpy as np
import torch

torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from chain import run_chain

CHAIN = Path(__file__).resolve().parent.parent / "2026-09-24_k3-chain"
BAKED = Path(__file__).resolve().parent.parent / "2026-09-16_kprop-arc" / "baked"
mc = dict(np.load(CHAIN / "runs" / "mc_oracle_0.npz"))
ma = np.load(CHAIN / "runs" / "mc_oracle_all_0.npz")
W = np.load(BAKED / "weights_0.npy")
gt = np.load(BAKED / "gt.npy")[0]
Ws = [torch.tensor(w, dtype=torch.float64) for w in W]
L = len(Ws); allL = tuple(range(L))

# k3l = `k3l_t` in 2026-09-24_k3-chain/ablate.py (k3j + K31 x 0.65)
K3L = dict(k4="memo22", corr="split24rr", k4pat=True, k22pat=True, c0d=True, k4lvl=True,
           k4feed="P2", marg=True, k31pat=True, paths=("e3", "e3t", "k22", "cc"), e3age=1,
           pathD="merge", k4b=1, k4bparts=("sd",), pairx=("k22",), tet=3, deadcut=-3.0,
           k31mult=0.65)
SELF_CHECK = 2.4074e-08

base = {k: mc[k] for k in ("k3", "k4", "var")}
base["C"] = [ma[f"C{l}"] for l in range(L)]
base["D"] = [ma[f"D{l}"] for l in range(L)]

arms = {
    "k3l (no oracle)":            None,
    "var exact":                  {"var_layers": allL},
    "k3 (diag) exact":            {"k3_layers": allL},
    "k4 (diag) exact":            {"k4_layers": allL},
    "k3 + k4 exact":              {"k3_layers": allL, "k4_layers": allL},
    "var + k3 + k4 exact":        {"var_layers": allL, "k3_layers": allL, "k4_layers": allL},
    "D (k3 (2,1) slice) exact":   {"D_layers": allL},
    "C0 (off-diag) exact":        {"C0_layers": allL},
    "C (full) exact":             {"C_layers": allL},
    "C + D exact":                {"C_layers": allL, "D_layers": allL},
    "C + D + k3 + k4 exact":      {"C_layers": allL, "D_layers": allL, "k3_layers": allL, "k4_layers": allL},
}

ref = None
print(f"{'arm':28s} {'final MSE':>12s}  {'x vs k3l':>9s}")
for name, o in arms.items():
    mu, _ = run_chain(Ws, oracle=dict(o, **base) if o else None, **K3L)
    mse = float(((mu[-1].numpy() - gt[-1]) ** 2).mean())
    if ref is None:
        ref = mse
        ok = "OK" if abs(mse / SELF_CHECK - 1) < 2e-3 else f"MISMATCH, expected {SELF_CHECK:.4e}"
        print(f"{name:28s} {mse:12.4e}  {'1.00':>9s}   self-check {ok}")
        if "MISMATCH" in ok:
            raise SystemExit("the base configuration does not reproduce k3l on MLP 0 — stop, the "
                             "attribution below would be measured against the wrong chain")
    else:
        print(f"{name:28s} {mse:12.4e}  {ref / mse:8.2f}x", flush=True)
