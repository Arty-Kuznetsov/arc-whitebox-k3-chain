"""Render the RESULTS.md tables of cloud #11 from the JSON results (res2/, res3/). Prints markdown to stdout.
    python tables.py t2      task 2: the energy split per layer and the payoff arms
    python tables.py t3      task 3: spectrum, rank-r / sparse arms, who, entrywise models"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R2, R3 = HERE / "res2", HERE / "res3"


def j(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def t2_split():
    for m in (0, 1):
        s = j(R2 / f"split_{m}.json")
        if s is None:
            continue
        print(f"\n**MLP {m}** — signal energies (noise-subtracted) of the next layer's variance error, summed over neurons; "
              f"`diag` = Σ_j W_ji² ΔC_jj, `off` = the rest; `2x` = twice the cross term; shares are of the total signal.\n")
        print("| layer l→l+1 | rel. rms e (all) | all: diag | all: off | all: 2x | all: total (4-seed var) | noise / ‖e‖² | "
              "core: diag | core: off | core: 2x |")
        print("|---|---|---|---|---|---|---|---|---|---|")
        for r in s["layers"]:
            a, c = r["all"], r["core"]
            St, Sc = a["S_tot"], c["S_tot"]
            f = lambda x, S: f"{x:.2e} ({100 * x / S:.0f}%)" if S > 0 else f"{x:.2e}"
            print(f"| {r['l']}→{r['l'] + 1} | {(a['E_tot'] / a['V']) ** 0.5:.1e} | {f(a['S_diag'], St)} | {f(a['S_off'], St)} | "
                  f"{f(2 * a['S_X'], St)} | {St:.2e} ({a['S_tot4']:.2e}) | {a['N_tot'] / a['E_tot']:.2f} | "
                  f"{f(c['S_diag'], Sc)} | {f(c['S_off'], Sc)} | {f(2 * c['S_X'], Sc)} |")
        for tag in ("all", "core"):
            t = s["sum_" + tag]
            S = t["S_tot"]
            print(f"| **Σ {tag}** | | {t['S_diag']:.2e} ({100 * t['S_diag'] / S:.0f}%) | {t['S_off']:.2e} "
                  f"({100 * t['S_off'] / S:.0f}%) | {2 * t['S_X']:+.2e} ({100 * 2 * t['S_X'] / S:.0f}%) | {S:.2e} "
                  f"({t['S_tot4']:.2e}) | {t['N_tot'] / t['E_tot']:.2f} | | | |")


def t2_arms():
    print("\n| MLP | layers | part | P (2e7) | A | B | 2P − (A+B)/2 | noise ctrl R (reset) | P / R | noise ctrl N (accumulating) |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    fx = lambda v: f"{v:.4f}" if v is not None else "—"
    for m in (0, 1):
        for set_ in ("s4", "all"):
            for part in ("off", "diag", "all", "diagr", "allr"):
                g = {v: j(R2 / f"arm_{m}_{set_}_{part}_{v}.json") for v in ("P", "N", "A", "B", "R")}
                x = {v: (g[v]["x"] if g[v] else None) for v in g}
                if all(v is None for v in x.values()):
                    continue
                ext = 2 * x["P"] - 0.5 * (x["A"] + x["B"]) if None not in (x["P"], x["A"], x["B"]) else None
                pr = x["P"] / x["R"] if None not in (x["P"], x["R"]) else None
                print(f"| {m} | {set_} | {part} | {fx(x['P'])} | {fx(x['A'])} | {fx(x['B'])} | {fx(ext)} | {fx(x['R'])} | "
                      f"{fx(pr)} | {fx(x['N'])} |")


if __name__ == "__main__":
    if sys.argv[1] == "t2":
        t2_split()
        t2_arms()
