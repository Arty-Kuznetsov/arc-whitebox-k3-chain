"""Prompt B pricing: the switches that the port-battery cost model (cost_pb.py) does not know, as EXTRA dense n x n x n
products added to its Strassen-billed units (BILL_S = 0.636), with the measured alive fractions (cost_pb.dead):
    e3age = a (path legs live a layers; None = all ages = 'allages'): +1 leg transport per source of age 2..min(a, S),
            on alive input rows (pathD 'merge' folds the path into the existing pairings: no extra pairing);
    pathD = 'full': +1 pairing product (A g B) @ B2^T per source that carries a path leg, alive x alive;
    k4k3 > 0: ~4 products per source per layer (NEGATIVE 61);
    k4bparts containing 'pd' or 'p22': +1 product X @ C0 per source in the k4b window (+1 more for 'p22').
Every other catalog switch is O(n^2) or a coefficient change and is priced at zero. The result is a ratio to the
model's own B9, applied to B9's meter-anchored utilisation (22.69%, see ../2026-09-30_tail-accuracy/frontier.py)."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "2026-09-30_tail-accuracy"))
import ta  # noqa: E402,F401  (path to the port-battery folder)
import cost_pb  # noqa: E402

L = 16
B9_KW = dict(t_row=-2.5, t_col=-2.5, satrow=2.5)
U_B9_METER = 0.23792 * (0.1973 / 0.2069)


def extra_units(cfg):
    f = [1.0 - d for d in cost_pb.dead(cfg.get("deadcut", -2.5))]   # alive fractions (pooled k3m alpha table)
    a = cfg.get("e3age", 1)
    paths = cfg.get("paths")
    full = cfg.get("pathD") == "full"
    k4k3 = cfg.get("k4k3", 0.0)
    parts = cfg.get("k4bparts") or ()
    k4b = cfg.get("k4b", 0)
    ex = 0.0
    for l in range(L):
        S = l
        last = l == L - 1
        fin = f[l - 1] if l > 0 else 1.0
        fo = f[l]
        if S == 0:
            continue
        npath = (S if a is None else min(a, S)) if paths else 0
        ex += max(npath - 1, 0) * fin
        if full and not last:
            ex += npath * fo * fo
        if k4k3:
            ex += 4.0 * S
        if k4b and ("pd" in parts or "p22" in parts):
            ex += min(k4b, S) * (1 + ("p22" in parts))
    return ex


def thin_points(cfg):
    """The correction leg's thin rank k (corr='split{k}rr'; base 24): NEGATIVE 61 measured thin384 at ~+5 points of B,
    taken linear in k - 24 (an estimate, not a meter reading)."""
    c = cfg.get("corr", "split24rr")
    k = int(c[5:].replace("rr", "")) if c.startswith("split") else 24
    return 0.05 * (k - 24) / 360.0


def cut_kw(cfg):
    """The port-battery model's own knobs from a chain config: dead-row cut, dead-side column cut, E2 rows."""
    cc = cfg.get("colcut")
    return dict(t_row=cfg.get("deadcut"), t_col=(cc[0] if isinstance(cc, tuple) else cc), satrow=cfg.get("satrow"))


def util(cfg):
    st, pl = cost_pb.units(**cut_kw(cfg))
    base = cost_pb.util_model(**B9_KW)
    arm = (cost_pb.BILL_S * (st + extra_units(cfg) - extra_units(ta.B9)) + cost_pb.BILL_P * pl) / cost_pb.B_UNITS
    return U_B9_METER * arm / base + thin_points(cfg)


if __name__ == "__main__":
    import cat_grid
    for t in ("allages", "k4k3_1", "e3age3", "e3age2", "pathD_full", "k4b_pd", "q22x2", "k4b_full"):
        print(f"{t:12s} util {100 * util(cat_grid.cfg_of(t)):.2f}%  (B9 {100 * util(ta.B9):.2f}%)")
