"""Age-truncated k = 4 in ARC's own factored representation (write-up §7: the k = 4 gain lives in the kappa4 born in
the last two layers). ARC's SIMPLE k_max = 4 keeps kappa4 as Sym(sum_r A_r (x) B_r); every ReLU appends a TREE block
and a CORRECTION block of n columns each, and columns keep their place, so at the pre-activation of layer l the factor
has r = 2 n l columns and block b was born at layer b // 2 (age l - b // 2). Its cost per layer grows with r, so with
depth, as L^2 overall.
Variants, stepped as mlp_kprop does (../2026-10-04_r1/r1lib.k4_stats, which reproduces mlp_kprop's k4S exactly):
  full     : nothing dropped (gate: must equal k4S);
  age a    : after each linear step keep only blocks of age <= a (r <= 2 n a: cost linear in depth);
  age a+cl : the same, and the dropped blocks are replaced by ONE scale-mixture block v (C_ij C_kl + C_ik C_jl + C_il C_jk),
             v fitted on the dropped blocks' diagonal (pattern 3 v var^2), computed before they are dropped (O(n r)).
Truth: Monte-Carlo post-means at 1e7 samples (korder's cache, same seeds). Also k3S (ARC's SIMPLE k = 3) for scale.
usage: python k4age.py <n> <depth> <first seed> <count> [ages=1,2,3]
    ARC=$(cd ../../external/mlp_cumulant_propagation && pwd); env PYTHONPATH="$ARC/src:../2026-10-04_r1" \\
        nice -n 10 uv run --project "$ARC" python k4age.py 128 8 0 8
"""
import sys, json, time
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_num_threads(3); torch.set_default_dtype(torch.float64)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-10-04_r1"))
from r1lib import MLP, Kind, coerce_input, linear_kprop, nonlin_kprop, WICK_COEF_D, relu_wick, weights, mc  # noqa: E402
from mlp_kprop.factor_k4 import FactoredTensor4, _factored_get_dslice                                    # noqa: E402
from mlp_kprop.kprop_harmonic import mlp_kprop                                                           # noqa: E402


def k4_run(Ws, age=None, closure=False):
    n, depth = Ws[0].shape[0], len(Ws)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    K = coerce_input({1: torch.zeros(n), 2: torch.eye(n)}, k_max=4, kind=Kind.SIMPLE)
    mus, cols, vfit = [], [], []
    births = []                       # birth layer of every n-column block of the kappa4 factor; None = the closure block
    for l in range(depth):
        Wm = mlp.Ws[l]
        K = linear_kprop(K, Wm.weight, k_max=4, set_metric=mlp.init_scale[l], bias=Wm.bias)
        if 4 in K:
            A, B = K[4]._factors
            nb = A.shape[2] // n
            while len(births) < nb:
                births.append(l - 1)                                   # the two blocks the last ReLU appended
            assert len(births) == nb and A.shape[2] == nb * n, (len(births), nb, A.shape)
            if age is not None:
                keep = [i for i, m in enumerate(births) if m is not None and l - m <= age]
                drop = [i for i in range(nb) if i not in keep]
                if drop:
                    ci = lambda bs: torch.cat([torch.arange(b * n, (b + 1) * n) for b in bs])
                    Ak, Bk = (A[:, :, ci(keep)], B[:, :, ci(keep)]) if keep else (A[:, :, :0], B[:, :, :0])
                    if closure:                                        # one scale-mixture block for all that is dropped
                        var = torch.diagonal(K[2].to_tensor())
                        k4old = _factored_get_dslice(A[:, :, ci(drop)], B[:, :, ci(drop)], (4,))
                        P = 3.0 * var * var
                        v = float((k4old * P).sum() / (P * P).sum()); vfit.append((l, v))
                        C = K[2].to_tensor()
                        Ac = torch.zeros(n, n, n); Bc = torch.zeros(n, n, n)
                        Ac[:, :, 0] = C; Bc[:, :, 0] = 3.0 * v * C        # 3v Sym(C (x) C) = v (CC + CC + CC)
                        Ak = torch.cat([Ac, Ak], 2); Bk = torch.cat([Bc, Bk], 2)
                        births = [None] + [births[i] for i in keep]
                    else:
                        births = [births[i] for i in keep]
                    K[4] = FactoredTensor4(n, factors=(Ak, Bk), assume_symmetric=True)
        cols.append(int(K[4]._factors[0].shape[2]) if 4 in K else 0)
        if l == depth - 1:
            var = torch.diagonal(K[2].to_tensor())
            w, _, _ = relu_wick(K[1].to_tensor(), var)
            t = K[3].get_dslice((3,)); k4d = K[4].get_dslice((4,))
            mus.append(w[(0, 1)] + w[(3, 1)] * t / 6 + w[(4, 1)] * k4d / 24)
            break
        K = nonlin_kprop(K, nonlin_wick_coef=WICK_COEF_D[mlp.nonlin_names[l]], k_max=4, kind=Kind.SIMPLE, use_pK=True, factor=True)
        mus.append(K[1].to_tensor().clone())
    return torch.stack(mus).numpy(), cols, vfit


def k3(Ws):
    n, depth = Ws[0].shape[0], len(Ws)
    mlp = MLP(input_dim=n, hidden_dim=n, output_dim=n, num_layers=depth + 1, nonlin="relu", init_kind="manual", w_var=2.0)
    for l in range(depth):
        mlp.Ws[l].weight.data = Ws[l].T.clone()
    out = mlp_kprop(mlp, {1: torch.zeros(n), 2: torch.eye(n)}, k_max=3, kind=Kind.SIMPLE, factor=True, output_all=True,
                    up_to_layer=f"act{depth - 1}")
    return torch.stack([out[f"act{l}"][1].to_tensor() for l in range(depth)]).numpy()


if __name__ == "__main__":
    n, depth, first, count = (int(x) for x in sys.argv[1:5])
    ages = [int(x) for x in (sys.argv[5] if len(sys.argv) > 5 else "1,2,3").split(",")]
    N = int(1e7) if len(sys.argv) <= 6 else int(float(sys.argv[6]))
    rows = []
    for s in range(first, first + count):
        Ws = weights(n, depth, s)
        mu_mc, var_mc = mc(Ws, s, N)
        row = {"seed": s}
        t0 = time.time(); m3 = k3(Ws); row["k3S"] = {"final": float(np.mean((m3[-1] - mu_mc[-1]) ** 2)), "s": time.time() - t0}
        variants = [("full", None, False)] + [(f"age{a}", a, False) for a in ages] + [(f"age{a}+cl", a, True) for a in ages]
        for name, a, cl in variants:
            t0 = time.time(); m, cols, vfit = k4_run(Ws, a, cl); dt = time.time() - t0
            row[name] = {"final": float(np.mean((m[-1] - mu_mc[-1]) ** 2)), "s": dt, "cols_mean": float(np.mean(cols)),
                         "cols_max": int(max(cols)), "v_last": vfit[-1][1] if vfit else None}
        noise = float(np.mean(var_mc) / N)
        full = row["full"]["final"] - noise; k3f = row["k3S"]["final"] - noise
        line = f"n={n} L={depth} seed {s}: k3S {row['k3S']['final']:.3e}  k4 full {row['full']['final']:.3e} (cols {row['full']['cols_mean']:.0f}, {row['full']['s']:.1f}s)"
        for name, a, cl in variants[1:]:
            g = (k3f - (row[name]["final"] - noise)) / (k3f - full) if k3f != full else float("nan")
            row[name]["gain_kept"] = g
            line += f" | {name} {row[name]['final']:.3e} gain {100*g:5.1f}% cols {row[name]['cols_mean']:.0f} {row[name]['s']:.1f}s"
        row["mc_noise"] = noise
        print(line, flush=True); rows.append(row)
    json.dump(rows, open(HERE / "res" / f"k4age_n{n}_L{depth}_s{first}-{first+count-1}.json", "w"), indent=1)
