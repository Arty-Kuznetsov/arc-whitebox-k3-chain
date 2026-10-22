"""Prompt B (notes/CLOUD-BRIEF.md): every STRUCTURAL accuracy switch of chain.py alone on the new base B9 (k3m BASE +
matched cut -2.5 + E2 satrow 2.5 sep), the fitted levels (k31mult, feedmult, k22c, sdmult, k4scale/k4mult, intermediate
k4k3) and the pure cost approximations (window, srank, dlr, d21_ages, ssp/shared_sub, tailx, stride, bandz, agecut) left
out. The tail restrictions (tailr) are cloud/tail-accuracy's and are not repeated. Tags = arm names; 'FULL+' prefix =
on #8's t4_final25a stack; '+' joins arms (their settings merged, later ones win)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-30_tail-accuracy"))
import ta  # noqa: E402

ta.RES = HERE / "res"
B9, FULL = ta.B9, ta.FULL
P4 = ("e3", "e3t", "k22", "cc")
ARMS = {
    # expansion terms closed on accuracy on the old base (NEGATIVE 65) or never re-measured on this one
    "aug": dict(aug=True), "edge2": dict(edge2=True), "q22x1": dict(q22x=1), "q22x2": dict(q22x=2),
    "tet4": dict(tet=4), "tet2": dict(tet=2),
    # kappa4 births and the kappa4 -> kappa3 channels (NEGATIVE 27, 40, 45, 47, 61)
    "e3": dict(e3=True), "k4b_full": dict(k4b=1, k4bparts=("pd", "p22", "sd", "s22")),
    "k4b_s22": dict(k4bparts=("sd", "s22")), "k4b_pd": dict(k4bparts=("pd", "sd")), "k4b2": dict(k4b=2),
    "k4b3": dict(k4b=3), "k4b0": dict(k4b=0), "k4star_off": dict(k4star=False), "k4k3_1": dict(k4k3=1.0),
    "feed_off": dict(k4feed=None), "feed_P1P2": dict(k4feed="P1P2"),
    # path births (NEGATIVE 61: pathD full, allages)
    "pathD_full": dict(pathD="full"), "allages": dict(e3age=None), "e3age2": dict(e3age=2), "e3age3": dict(e3age=3),
    "paths_no_e3": dict(paths=("e3t", "k22", "cc")), "paths_no_e3t": dict(paths=("e3", "k22", "cc")),
    "paths_no_k22": dict(paths=("e3", "e3t", "cc")), "paths_no_cc": dict(paths=("e3", "e3t", "k22")),
    "paths_off": dict(paths=None), "pathsub_off": dict(pathsub=False), "pathfold": dict(pathfold=True),
    # pair terms
    "c0d_off": dict(c0d=False), "c0d_22": dict(c0d="22"), "c0d_11_22": dict(c0d="11 22"), "c0d_21_22": dict(c0d="21 22"),
    "pairx_off": dict(pairx=None), "k31pat_off": dict(k31pat=False),
    # marginal coefficients per power cumulant (cf. marg2 in cloud/tail-accuracy: k-selective, not available here)
    "marg_off": dict(marg=False), "marg_11": dict(marg="11"), "marg_21": dict(marg="21"), "marg_22": dict(marg="22"),
    "marg_11_21": dict(marg="11 21"), "marg_11_22": dict(marg="11 22"), "marg_21_22": dict(marg="21 22"),
    "margleg": dict(margleg=True),
    # the kappa4 closure
    "k22pat_off": dict(k22pat=False), "k4pat_off": dict(k4pat=False), "k4lvl_off": dict(k4lvl=False),
    # the correction leg's thin rank (accuracy probe; a cost lever when raised)
    "thin48": dict(corr="split48rr"), "thin96": dict(corr="split96rr"), "thin192": dict(corr="split192rr"),
    "thin384": dict(corr="split384rr"),
    # the cuts themselves
    "cuts_off": dict(deadcut=None, colcut=None), "colcut_off": dict(colcut=None), "deadC": dict(deadC=True),
    "e2_restore": dict(satrow_k4="restore"),
    # cloud/tail-accuracy's confirmed lever (Gaussian k = 2 pair coefficient outside the dead tail), for the stacks
    "marg2": dict(tailr=(("marg2", "sat", None, -2.0, "gauss"),)),
}


def cfg_of(tag):
    base = B9
    if tag.startswith("FULL+"):
        base, tag = FULL, tag[5:]
    cfg = dict(base)
    for part in tag.split("+"):
        if part.startswith("g_"):  # task 3 grid on FULL: g_<cut>_<E2 theta>_<recv>
            t, s_, r = (float(x) for x in part[2:].split("_"))
            cfg.update(deadcut=t, colcut=(t, 1e9), satrow=s_, recv=(r, "b"))
        elif part == "band6_384":
            cfg["bandz"] = dict(cfg["bandz"], a0=6, r=384)
        else:
            cfg.update(ARMS[part])
    return cfg


GRID3 = [f"FULL+g_{t}_{s_}_{r}" for t in (-2.25, -2.5, -2.75) for s_ in (2.25, 2.5, 2.75) for r in (-3.0, -3.25)
         if (t, s_, r) != (-2.5, 2.5, -3.0)] + ["FULL+band6_384"]


if __name__ == "__main__":
    for t in ARMS:
        cfg_of(t)
    print(len(ARMS), "arms")
