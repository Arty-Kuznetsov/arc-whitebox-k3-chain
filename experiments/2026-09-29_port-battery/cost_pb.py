"""Honest FLOP-cost model for the port battery (units: one dense n x n x n product = 2 n^3 flops; B = 1024 units).

Extends experiments/2026-09-26_cost-frontier/cost.py (same structure, same two billing factors: BILL_S = 0.636 for
everything that rides the Strassen stacks, BILL_P = 1.0 for plain products and the linalg calls) PER SOURCE, so the
column cut, the band, and the row compactions can all be priced in one pass, and it prices the band's CONSTRUCTION
explicitly (NEGATIVE 54-59: the construction is what killed five ideas):

  dense source s (born b, age a = l - b) at layer l:
      transport   (a == 1 ? 1 : 2) * fin * c(b) * g_s        B only at age 1 (A = Wt is free); contraction over the
                                                             alive rows of l-1 (fin); c(b) = alive-parent column
                                                             fraction (colcut); g_s = output-row fraction (E3)
      pairing     2 * rowsD * fo * c(b)                      D = sum_s L1 B^T + L2 A^T over alive x alive (satrow
                                                             removes saturated rows: rowsD = fo - sat)
  band member (age >= a0, coefficient form legs = P Z):
      materialise 2 * q * c(b) * g                           A, B = P Z  (q = r / n)
      pairing     2 * q * rowsD * c(b)                       (L1 Z_B^T + L2 Z_A^T), then one shared  q * rowsD * fo
  band, shared per layer:
      P transport q * fin * g (Strassen) | QR (plain, flopscope qr_cost reduced = 2 (2 n r^2 - 2 r^3/3)) and the
      coefficient rotation 2 q^2 c(b) per member (plain) ONLY on QR layers | the entrant's projection:
        ne: G = P^T P (q^2, plain) + P^T X (2 q c, Strassen) + LU solve (q^3/3 + 2 q^2 c, plain; flopscope solve_cost)
        qr: QR of P (plain) + Q^T X (2 q c, Strassen) + solve with R (q^3/3 + 2 q^2 c, plain)
      entry="post" (E1 semantics, measured in task 1): the entrant is still transported densely at age a0 (2 fin c g)
      entry="pre": projected at the end of the previous layer, so that transport is saved
  base products: C_pre (1 + 1, the fused Wt C and the second product; 1 at layer 0; the last layer needs only the
      fused one) and the path leg (1, full rows) -- as cost.py.
The thin correction legs (n x kk) are ignored, as in cost.py (absorbed by the calibration).

Utilisation is reported as a RATIO to k3m in this same model and scaled to the graded 24.835% (submission 332403).
"""
import json
import math
from pathlib import Path

N, L = 1024, 16
B_UNITS = 1024.0
BILL_S = 0.636
BILL_P = 1.0
UTIL_K3M = 0.24835
L5_METER = 0.9165           # Strassen level 5, measured on the meter (NEGATIVE 56)
HERE = Path(__file__).resolve().parent
_FR = None


def _fr():
    global _FR
    if _FR is None:
        _FR = json.loads((HERE / "alpha_fractions_k3m.json").read_text())
    return _FR


def _interp(tab, t, lo_first):
    keys = sorted(float(k) for k in tab)
    if t is None:
        return [0.0] * L
    if t <= keys[0]:
        return list(tab[f"{keys[0]:.2f}"]) if not lo_first else [0.0] * L
    if t >= keys[-1]:
        return list(tab[f"{keys[-1]:.2f}"])
    hi = min(k for k in keys if k >= t)
    lo = max(k for k in keys if k <= t)
    if hi == lo:
        return list(tab[f"{lo:.2f}"])
    w = (t - lo) / (hi - lo)
    a, b = tab[f"{lo:.2f}"], tab[f"{hi:.2f}"]
    return [a[i] * (1 - w) + b[i] * w for i in range(L)]


def dead(t):
    """Per-layer fraction with alpha_l < t (pooled over 10 MLPs, k3m chain)."""
    if t is None:
        return [0.0] * L
    return _interp(_fr()["dead"], t, lo_first=True)


def sat(th):
    """Per-layer fraction with alpha_l > th."""
    if th is None:
        return [0.0] * L
    return _interp(_fr()["sat"], th, lo_first=False)


def qr_units(r):
    q = r / N
    return 2.0 * (q * q - q ** 3 / 3.0)          # flopscope qr_cost(n, r, "reduced") / (2 n^3)


def solve_units(r, nrhs_cols):
    q = r / N
    return q ** 3 / 3.0 + q * q * nrhs_cols      # flopscope solve_cost(r, nrhs) / (2 n^3), nrhs = cols * n


def units(t_row=-2.5, t_col=None, band=None, satrow=None, e3=None, e4=None, detail=False):
    """band: dict(a0, r, k (None = never re-orthonormalise), proj ("ne" | "qr"), entry ("post" | "pre")).
    e3: dict(theta, arm ("a" | "b")); e4: dict(a_min, t_old) -- old legs (age >= a_min) cut at t_old (rows and,
    from age a_min on, columns). Returns (strassen_units, plain_units[, breakdown])."""
    dr = dead(t_row)
    alive = [1.0 - x for x in dr]
    dcol = dead(t_col) if t_col is not None else [0.0] * L
    ccol = [1.0 - x for x in dcol]
    s_sat = sat(satrow)
    g_out = [1.0 - x for x in dead(e3["theta"])] if e3 is not None else [1.0] * L
    if e4 is not None:
        alive_old = [1.0 - x for x in dead(e4["t_old"])]
        cold = [1.0 - x for x in dead(e4["t_old"])]
    bd = {k: 0.0 for k in ("base", "path", "transport", "pairing", "band_mat", "band_pair", "band_P",
                           "band_entrant_transport", "band_proj_S", "band_proj_P", "band_qr", "band_rot",
                           "e3b_rep", "band_init_S")}
    q = band["r"] / N if band else 0.0
    l0 = None
    for l in range(L):
        last = l == L - 1
        fin = alive[l - 1] if l > 0 else 1.0
        fo = alive[l]
        rowsD = max(fo - s_sat[l], 0.0)
        g = g_out[l]
        S = l
        # base products (as cost.py): layer 0 one product; l >= 1 the fused Wt C + the path leg; C_pre second
        if S > 0:
            bd["base"] += 1.0
            bd["path"] += 1.0
        bd["base"] += 0.0 if last else 1.0
        members = []
        entrant = None
        for b in range(S):
            a = l - b
            old = e4 is not None and a >= e4["a_min"]
            c = ccol[b] if not old else min(ccol[b], cold[b])
            fin_s = (alive_old[l - 1] if l > 0 else 1.0) if old else fin   # rows cut at the gating of l-1
            fo_s = fo if not old else alive_old[l]
            rows_s = rowsD if not old else max(alive_old[l] - s_sat[l], 0.0)
            in_band = band is not None and a >= band["a0"]
            if in_band and a == band["a0"]:
                entrant = (b, c)
                if band.get("entry", "post") == "post":
                    bd["band_entrant_transport"] += 2.0 * fin_s * c * g
            if in_band:
                members.append((b, c))
                bd["band_mat"] += 2.0 * q * c * g
                if not last:
                    bd["band_pair"] += 2.0 * q * rows_s * c
                continue
            g_s = 1.0 if (e3 is not None and e3.get("arm", "a") in ("a", "b") and a == 1) else g
            bd["transport"] += (1.0 if a == 1 else 2.0) * fin_s * c * g_s
            if not last:
                bd["pairing"] += 2.0 * rows_s * fo_s * c
        if e3 is not None and e3.get("arm") == "b" and S > 1:
            bd["e3b_rep"] += (1.0 - g) * 1.0      # t_rep for the dropped rows: (M o M)_rows D M^T (plain)
        if band is not None and members:
            if l0 is None:
                l0 = l if band.get("entry", "post") == "post" else l - 1
                if band.get("init", "cols") == "rr":   # one-time range finder on the first entrant's [A B]:
                    bd["band_init_S"] += 6.0 * q        # [A B] Om, [A B]^T Y, [A B] (.) : 3 products of 2q (Strassen)
                    bd["band_qr"] += qr_units(band["r"])
            old_b = e4 is not None and band["a0"] >= e4["a_min"]     # members are old: P gated with the harsher cut
            fin_b = (alive_old[l - 1] if l > 0 else 1.0) if old_b else fin
            fo_b = alive_old[l] if old_b else fo
            rows_b = max(fo_b - s_sat[l], 0.0)
            if l > l0:
                bd["band_P"] += q * fin_b * g
            if not last:
                bd["band_pair"] += q * rows_b * fo_b
            k = band.get("k")
            if l > l0 and k is not None and (l - l0) % int(k) == 0:
                bd["band_qr"] += qr_units(band["r"])
                bd["band_rot"] += sum(2.0 * q * q * c for (b, c) in members if entrant is None or b != entrant[0])
            if entrant is not None:
                ce = entrant[1]
                bd["band_proj_S"] += 2.0 * q * ce
                if band.get("proj", "ne") == "ne":
                    bd["band_proj_P"] += q * q + solve_units(band["r"], 2.0 * ce)
                else:
                    bd["band_proj_P"] += qr_units(band["r"]) + solve_units(band["r"], 2.0 * ce)
    st = sum(v for k, v in bd.items() if k not in ("band_proj_P", "band_qr", "band_rot", "e3b_rep"))
    pl = bd["band_proj_P"] + bd["band_qr"] + bd["band_rot"] + bd["e3b_rep"]
    return (st, pl, bd) if detail else (st, pl)


def util_model(**kw):
    st, pl = units(**kw)
    return (BILL_S * st + BILL_P * pl) / B_UNITS


def util(**kw):
    """Projected production utilisation: the model's ratio to k3m (t_row=-2.5, nothing else), times 24.835%."""
    return UTIL_K3M * util_model(**kw) / util_model(t_row=-2.5)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(HERE.parent / "2026-09-26_cost-frontier"))
    import cost as cf
    print(f"model k3m (t_row -2.5): {100 * util_model(t_row=-2.5):.2f}%   cost.py util(deadcut=-2.5): "
          f"{100 * cf.util(deadcut=-2.5):.2f}%   (graded 24.835%)")
    for kw in [dict(t_row=-2.25, t_col=-2.25), dict(t_row=-2.5, t_col=-2.5),
               dict(t_row=-2.5, band=dict(a0=6, r=448, k=None, proj="qr")),
               dict(t_row=-2.5, band=dict(a0=6, r=384, k=None, proj="qr")),
               dict(t_row=-2.5, band=dict(a0=6, r=448, k=1, proj="ne")),
               dict(t_row=-2.5, band=dict(a0=6, r=448, k=3, proj="ne")),
               dict(t_row=-2.5, band=dict(a0=6, r=448, k=None, proj="qr", entry="pre"))]:
        st, pl, bd = units(detail=True, **kw)
        print(f"{str(kw):95s} util {100 * util(**kw):6.2f}%  x{util(**kw) / UTIL_K3M:.4f}")
