"""The exhaustive tail-restriction grid (Prompt A task 2): every term the chain computes per neuron or per pair x tail
(dead alpha < -theta / saturated alpha > theta; rows and columns separately where a term has two) x theta in
{2.0, 2.5, 3.0}. Each arm = B9 + one tailr rule (chain.py `_tailr_rules`). Skipped as exact no-ops on B9: dead-side D
terms and dead-side births at theta >= 2.5 (deadcut/colcut -2.5 already zero them), saturated D rows at theta >= 2.5
(E2 already cuts them) and the state K31 cuts (identical to qK31, selfcheck_wired.py). E2's own curve (Drow sat) is #8's.
Tags: <label>_<side>_<theta>; '+' joins arms (their rules are applied together); a 'FULL+' prefix puts them on FULL."""
import ta

TH = (2.5, 3.0, 2.0)          # queue order: theta = 2.5 first
D20 = (2.0,)
SPEC = [  # (label, term, axis, mode, dead thetas, saturated thetas)
    # state level: the tail part of a slice is not used by the step at all (pair terms, readout and births)
    ("Dcol", "Dcol", None, "zero", D20, TH), ("Drow", "Drow", None, "zero", D20, ()),
    ("t", "t", None, "zero", TH, TH), ("K4d", "K4d", None, "zero", TH, TH), ("K22", "K22", None, "zero", TH, TH),
    # pair terms of the power cumulants Q(p,q) (rows/cols = repeated/single index of the slice; symmetric terms: either)
    ("qC0", "qC0", None, "zero", TH, TH), ("qC0sq", "qC0sq", None, "zero", TH, TH),
    ("qK22", "qK22", None, "zero", TH, TH), ("qC0K22", "qC0K22", None, "zero", TH, TH),
    ("qC0cube", "qC0^3", None, "zero", TH, TH),
    ("qD-dbl", "qD", "dbl", "zero", D20, D20), ("qD-sgl", "qD", "sgl", "zero", D20, TH),
    ("qC0D-dbl", "qC0D", "dbl", "zero", D20, D20), ("qC0D-sgl", "qC0D", "sgl", "zero", D20, TH),
    ("qK31-tri", "qK31", "dbl", "zero", TH, TH), ("qK31-sgl", "qK31", "sgl", "zero", TH, TH),
    # readout: the Edgeworth correction of the power moments (P1 = the mean itself, P2, P3/P4 -> K3, K4)
    ("P1", "P1", None, "zero", TH, TH), ("P2", "P2", None, "zero", TH, TH), ("P34", "P34", None, "zero", TH, TH),
    # marginal Edgeworth correction of the pair Wick coefficients coef(k, p), k = 1, 2 -> Gaussian value in the tail
    ("marg1", "marg1", None, "gauss", TH, TH), ("marg2", "marg2", None, "gauss", TH, TH),
    # births: columns = hub p at the birth layer, rows = the legs' other index (birth layer), star_row = K4d rows
    ("tree_col", "tree_col", None, "zero", D20, TH), ("corr_col", "corr_col", None, "zero", D20, TH),
    ("path_col", "path_col", None, "zero", D20, TH), ("star_col", "star_col", None, "zero", D20, TH),
    ("feed_col", "feed_col", None, "zero", D20, TH), ("B_row", "B_row", None, "zero", D20, TH),
    ("corr_row", "corr_row", None, "zero", TH, TH), ("path_row", "path_row", None, "zero", TH, TH),
    ("star_row", "star_row", None, "zero", TH, TH),
]
ARMS, ORDER = {}, []
for label, term, axis, mode, dth, sth in SPEC:
    for side, ths in (("dead", dth), ("sat", sth)):
        for th in ths:
            tag = f"{label}_{side}_{th}"
            ARMS[tag] = dict(ta.B9, tailr=((term, side, axis, th, mode),))
            ORDER.append((TH.index(th), len(ORDER), tag))
ORDER = [t for _, _, t in sorted(ORDER)]
# follow-ups beyond the exhaustive grid (thresholds only, no levels): the best survivor's curve continued downwards
EXTRA = []
for th, lab in ((1.5, "1.5"), (1.0, "1.0"), (0.0, "0.0"), (-1e9, "all")):
    EXTRA.append(f"marg2_sat_{lab}")
    ARMS[EXTRA[-1]] = dict(ta.B9, tailr=(("marg2", "sat", None, th, "gauss"),))
for th, lab in ((-1.0, "m1.0"), (-2.0, "m2.0")):   # keep the correction only in the dead tail alpha <= -1 / -2
    EXTRA.append(f"marg2_sat_{lab}")
    ARMS[EXTRA[-1]] = dict(ta.B9, tailr=(("marg2", "sat", None, th, "gauss"),))
for q in ("11", "21", "22"):                       # which power cumulant's k=2 correction hurts
    EXTRA.append(f"marg2_all_q{q}")
    ARMS[EXTRA[-1]] = dict(ta.B9, tailr=(("marg2", "sat", None, -1e9, "gauss", q),))


def cfg_of(tag):
    base = ta.B9
    if tag.startswith("FULL+"):
        base, tag = ta.FULL, tag[5:]
    rules = []
    for part in tag.split("+"):
        rules += list(ARMS[part]["tailr"])
    return dict(base, tailr=tuple(rules))


if __name__ == "__main__":
    print(len(ARMS), "arms")
    print("\n".join(ORDER))
