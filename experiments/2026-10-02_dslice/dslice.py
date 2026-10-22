"""Is the chain's (2,1) slice D = kappa3[i,i,j] what the step turns into covariance error? (2026-10-02, Mac)

Cloud #11: the step at layer l, fed the EXACT input covariance, still gives birth to 22-30% of dC0 at depth, while with
ALL inputs exact it is accurate to ~0.1% -- so the born error comes from the step's other inputs (mu, t, D, kappa4).
The largest dC entries sit on large-|D| pairs. Here, with Monte-Carlo oracles of D at layers 9 and 12 (d_oracle.py,
two seeds A/B per MLP, P = their mean) and the exact post covariance at layers 8, 9, 11, 12:
  1. how wrong the chain's D is (relative Frobenius error, noise-subtracted with the seed difference; slope; corr);
  2. BORN: the chain's step at l run on the exact C_pre(l), with the chain's D vs with the exact D -> how much of the
     born dC0(l) is D's doing (energies noise-subtracted);
  3. the end-to-end payoff of the exact D at l = 9, 12, and both (UPPER BOUNDS; P, A, B, Richardson 2P-(A+B)/2, and a
     noise control that adds (D_A - D_B)/2 to the chain's own D).
Base: B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep"), as in cloud #11 (gate: MLP 0 2.278922e-08, MLP 1 2.205732e-08).
    PYTHONPATH=$ARC/src:. uv run --project $ARC python dslice.py <mlp>
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb                                            # noqa: E402  (puts chain.py on the path)
from chain import run_chain                          # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(3)
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
GT = np.load(pb.BAKED / "gt.npy")
SEEDS = (3101, 3102)
LD = (9, 12)


def offd(X):
    X = np.array(X, dtype=np.float64, copy=True)
    np.fill_diagonal(X, 0.0)
    return X


def raw(pred, m):
    return float(((pred[-1] - torch.tensor(GT[m], dtype=torch.float64)[-1]) ** 2).mean())


def run(Ws, m, **kw):
    pred, info = run_chain(Ws, **B9, **kw)
    return raw(pred, m), info


def main(m):
    Ws = [torch.tensor(w, dtype=torch.float64) for w in np.load(pb.BAKED / f"weights_{m}.npy")]
    O = {s: np.load(HERE / "oracles" / f"d_{m}_s{s}.npz") for s in SEEDS}
    A, B = (O[s] for s in SEEDS)
    out = {"mlp": m}
    base, info = run(Ws, m, record="full")
    out["base"] = base
    print(f"=== MLP {m}: base B9 raw {base:.6e}", flush=True)

    # 1. accuracy of the chain's D
    acc = {}
    for l in LD:
        Dc = offd(info[l]["D"].numpy())
        DA, DB = offd(A[f"D_{l}"]), offd(B[f"D_{l}"])
        DP = 0.5 * (DA + DB)
        noise = float(((DA - DB) ** 2).sum()) / 4.0               # noise energy of the 2-seed mean
        err = float(((Dc - DP) ** 2).sum()) - noise
        sig = float((DP ** 2).sum()) - noise
        x, y = DP.ravel(), Dc.ravel()
        slope = float((x @ y) / (x @ x))
        acc[l] = dict(rel_err=float(np.sqrt(max(err, 0.0) / sig)), noise_share=noise / float((DP ** 2).sum()),
                      corr=float(np.corrcoef(x, y)[0, 1]), slope=slope)
        print(f"  D at l={l}: rel error {acc[l]['rel_err']:.3f} (noise-subtracted), corr {acc[l]['corr']:.4f}, "
              f"slope chain/oracle {slope:.3f}, oracle noise share {acc[l]['noise_share']:.3f}", flush=True)
    out["D_accuracy"] = acc

    # 2. BORN at l: the step on the exact C_pre(l) (exact C_post(l-1) through postc), chain's D vs exact D
    born = {}
    for l in LD:
        res = {}
        for lab, sub_d in (("chainD", False), ("exactD", True)):
            per = {}
            for s, Os in ((SEEDS[0], A), (SEEDS[1], B)):
                rec = {}
                Cin = torch.tensor(np.asarray(Os[f"C_post_{l - 1}"], dtype=np.float64))

                def postc(ll, C, Cin=Cin, rec=rec, l=l):
                    if ll == l - 1:
                        return 0.5 * (Cin + Cin.T)
                    if ll == l:
                        rec["C"] = C.numpy().copy()
                    return C
                orc = {"D": {l: np.asarray(Os[f"D_{l}"], dtype=np.float64)}, "D_layers": {l}} if sub_d else None
                run(Ws, m, postc=postc, oracle=orc)
                per[s] = offd(rec["C"]) - offd(Os[f"C_post_{l}"])     # same seed in and out: linear noise cancels
            dA, dB = per[SEEDS[0]], per[SEEDS[1]]
            e_mean = 0.5 * (dA + dB)
            noise = float(((dA - dB) ** 2).sum()) / 4.0
            res[lab] = dict(energy=float((e_mean ** 2).sum()) - noise, noise=noise)
        res["D_share"] = 1.0 - res["exactD"]["energy"] / res["chainD"]["energy"]
        born[l] = res
        print(f"  BORN dC0 at l={l}: energy with chain's D {res['chainD']['energy']:.3e}, with exact D "
              f"{res['exactD']['energy']:.3e}  -> D's share of the born error {res['D_share']:.1%} "
              f"(noise {res['chainD']['noise']:.1e})", flush=True)
    out["born"] = born

    # 3. end-to-end payoff of the exact D (upper bounds)
    pay = {}
    for lab, layers in (("l9", (9,)), ("l12", (12,)), ("both", LD)):
        r = {}
        for tag, src in (("P", None), ("A", A), ("B", B), ("noise", "noise")):
            Dsub = {}
            for l in layers:
                if src is None:
                    Dsub[l] = 0.5 * (np.asarray(A[f"D_{l}"], np.float64) + np.asarray(B[f"D_{l}"], np.float64))
                elif isinstance(src, str):   # noise control: the chain's own D + the P-level noise
                    Dsub[l] = info[l]["D"].numpy() + 0.5 * (np.asarray(A[f"D_{l}"], np.float64)
                                                           - np.asarray(B[f"D_{l}"], np.float64))
                else:
                    Dsub[l] = np.asarray(src[f"D_{l}"], np.float64)
            rr, _ = run(Ws, m, oracle={"D": Dsub, "D_layers": set(layers)})
            r[tag] = rr / base
        r["richardson"] = 2 * r["P"] - 0.5 * (r["A"] + r["B"])
        pay[lab] = r
        print(f"  PAYOFF exact D at {lab}: P x{r['P']:.4f}  A x{r['A']:.4f}  B x{r['B']:.4f}  "
              f"Richardson x{r['richardson']:.4f}  noise ctrl x{r['noise']:.4f}", flush=True)
    out["payoff"] = pay
    (HERE / "res").mkdir(exist_ok=True)
    json.dump(out, open(HERE / "res" / f"dslice_{m}.json", "w"), indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]))
