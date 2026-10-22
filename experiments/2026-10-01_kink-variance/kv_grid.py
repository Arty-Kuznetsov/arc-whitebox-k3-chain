"""Arms of cloud #10 (filled task by task). Tags: arm names; 'FULL+' = on the full stack; '+' joins arms."""
import kv

ARMS = {}
# task 2: the next-order pair diagrams absent from the expansion (chain.py `pairnew`), and marg3
PN = ("c0k31", "d2", "ddt", "k22sq", "dk22", "c0sqd", "c0sqk22")
for t_ in PN:
    ARMS[t_] = dict(pairnew=(t_,))
ARMS["marg3"] = dict(marg3=True)
ARMS["pairnew_all"] = dict(pairnew=PN)
ARMS["pairnew_all_marg3"] = dict(pairnew=PN, marg3=True)
T2 = list(PN) + ["marg3", "pairnew_all", "pairnew_all_marg3"]
ARMS["marg2"] = dict(tailr=(("marg2", "sat", None, -2.0, "gauss"),))   # #9A's lever, shipped by the Mac as k3r
ARMS["c0k31_d2"] = dict(pairnew=("c0k31", "d2"))                          # the two task-2 survivors together
# task 3: the consistency audit (one object entering twice, forced to agree)
ARMS["cons_k4d_from_k22"] = dict(k4cons="k4d_from_k22")   # kappa4 diagonal level := the (2,2) slice's level
ARMS["cons_k22_from_k4d"] = dict(k4cons="k22_from_k4d")   # the (2,2) slice's level := the diagonal's
ARMS["cons_v_from_k22"] = dict(k4cons="v_from_k22")       # K31 and the feed read v from K22, not from K4d
ARMS["cons_k2pair"] = dict(k2pair=True)                   # next variance = the pair expansion's own diagonal
ARMS["cons_marg1_q22"] = dict(tailr=(("marg1", "sat", None, -1e9, "gauss", "22"),))   # k=1 correction off in Q22 only
ARMS["cons_marg1_all"] = dict(tailr=(("marg1", "sat", None, -1e9, "gauss"),))          # k=1 off everywhere, consistently
T3 = ["cons_k4d_from_k22", "cons_k22_from_k4d", "cons_v_from_k22", "cons_k2pair", "cons_marg1_q22", "cons_marg1_all"]


def cfg_of(tag):
    base = kv.B9
    if tag.startswith("FULL+"):
        base, tag = kv.FULL, tag[5:]
    cfg = dict(base)
    for part in tag.split("+"):
        cfg.update(ARMS[part])
    return cfg
