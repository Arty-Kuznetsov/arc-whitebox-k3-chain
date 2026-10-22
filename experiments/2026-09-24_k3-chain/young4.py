"""2026-10-04 (R2/R3): the young kappa4 trees on LEGS -- the same object as chain.py k4young impl="arc" (ARC's
FactoredTensor4, small widths only), rewritten with dense n x w legs so it runs at full width.

A generation born at the ReLU of layer m holds (ARC convention: full C as the edge, full kappa3):
  T   the transport since birth (identity at birth; gated by d1 and multiplied by W^T like any leg),
  U~  = T D1 C (C the pre-activation covariance at m, with its diagonal), carried like a leg,
  G   = 12 D2 C D2 (the path's hub coupling, fixed), d2, d3 (gates at m, fixed),
  the kappa3 of layer m as CP terms (X, Y, Z; w): X frozen at m (it sits on the 3+2's shared vertex), Y and Z
  carried (the generation's own copies, so a leg the chain later drops -- the path leg B2 -- is kept).
With x = U~ o T, the trees' factored form (factor_k4.py) is A'(a,b,r) = 1/2 (U~_ar T_br + T_ar U~_br) and
  path  B'(a,b,r) = 1/2 sum_q G_rq (U~_aq T_bq + T_aq U~_bq)
  star  B'(a,b,r) = 4 d3_r U~_ar U~_br
  3+2   B'(a,b,r) = 12 d2_r k3x(r; a, b),  k3x(r; a, b) = sum_terms sum_p w_p X_m(r,p) Y(a,p) Z(b,p)
and the slices read are K4d(a) = sum_r A'(a,a,r) B'(a,a,r), K31(a,b) = 1/2 sum_r [A'(a,a,r) B'(a,b,r) + A'(a,b,r) B'(a,a,r)]:
  path  K4d = rowsum(x o y), y = x G;   K31 = 1/2 [(y o T) U~^T + (y o U~) T^T]
  star  K4d = rowsum(s4 x o U~^2);  K31 = 3/4 (s4 T o U~^2) U~^T + 1/4 (s4 U~^3) T^T,  s4 = 4 d3 (columns)
  3+2   Dx(a, r) = k3x(r; a, a) = sum_terms ((Y o Z) w) X_m^T;   K4d = 12 rowsum(x o Dx D2);
        K31 = 6 sum_terms [((x D2 X_m) o Y) w] Z^T + 3 [((Dx D2) o U~) T^T + ((Dx D2) o T) U~^T]
(the K31 diagonal is zeroed afterwards, as ARC's dslices). test: experiments/2026-10-04_r1/test_young_legs.py.
"""
import itertools

import torch


def _src_terms(s, eleg):
    """The chain's kappa3 source s as CP terms over its hub columns: [(X, Y, Z, w)] with leg NAMES, plus the legs."""
    legs = {"A": s["A"], "B": s["B"]}
    terms = [("A", "B", "B", s["g"]), ("B", "A", "B", s["g"]), ("B", "B", "A", s["g"])]
    E = eleg(s)
    if E is not None:
        legs["E"] = E
        third = torch.full_like(s["g"], 1.0 / 3.0)
        terms += [("E", "A", "A", third), ("A", "E", "A", third), ("A", "A", "E", third)]
    if s.get("B2") is not None:
        legs["B2"] = s["B2"]
        w = 0.5 * s["g3"]
        terms += [p + (w,) for p in itertools.permutations(("A", "B", "B2"))]
    if s.get("h") is not None:
        legs["Bd"] = s["B"] * s["d1b"][None, :]
        legs["M1"] = (s["A"] @ s["h"])[:, None].expand(-1, s["A"].shape[1]).clone()
        half = torch.full_like(s["g"], 0.5)
        terms += [("M1", "Bd", "A", half), ("M1", "A", "Bd", half), ("Bd", "M1", "A", half), ("A", "M1", "Bd", half),
                  ("Bd", "A", "M1", half), ("A", "Bd", "M1", half)]
    return legs, terms


class Young4:
    def __init__(self, ages=2, types=("path", "star", "32"), couple=None, k31_32="full"):
        """k31_32: the 3+2's K31 reader -- "full"; "cheap" (only its O(n^3) part given Dx, without the
        6 sum_terms [((x D2 X_m) o Y) w] Z^T term, which costs ~12 D pairings); "none"."""
        self.ages, self.types, self.couple, self.k31_32 = int(ages), tuple(types), couple, k31_32
        self.gens = []

    def transport(self, Wt):
        for g in self.gens:
            g["T"] = Wt.clone() if g["T"] is None else Wt @ g["T"]
            g["U"] = Wt @ g["U"]
            for src in g["k3"]:
                src["cur"] = {k: Wt @ v for k, v in src["cur"].items()}

    def gate_and_retire(self, d1, l):
        self.gens = [g for g in self.gens if l - g["born"] < self.ages]
        for g in self.gens:
            g["T"] = d1[:, None] * g["T"]
            g["U"] = d1[:, None] * g["U"]
            for src in g["k3"]:
                src["cur"] = {k: d1[:, None] * v for k, v in src["cur"].items()}

    def birth(self, l, C, d1, d2, d3, sources, eleg):
        """At the ReLU of layer l: C the pre-activation covariance (with its diagonal), d1 the gate the chain's legs
        get (dead rows zeroed), d2, d3 the Wick gates; sources: the chain's kappa3 sources at pre(l), not yet gated."""
        G = 12.0 * d2[:, None] * C * d2[None, :]
        if "pathdiag" in self.types:      # piece study: the path with its two hubs coinciding (G's diagonal) only
            G = torch.diag(torch.diagonal(G))
        elif "pathoff" in self.types:     # ... or with distinct hubs only
            G = G - torch.diag(torch.diagonal(G))
        g = {"born": l, "T": None, "U": d1[:, None] * C, "G": G, "d2": d2.clone(), "d3": d3.clone(), "k3": []}
        if "32" in self.types:
            for s in sources:
                if s.get("A") is None or (self.couple is not None and l - s["born"] > self.couple):
                    continue
                legs, terms = _src_terms(s, eleg)
                g["k3"].append({"m": {k: v.clone() for k, v in legs.items()},       # frozen at the birth layer
                                "cur": {k: d1[:, None] * v for k, v in legs.items()},  # carried (gated now, post(l))
                                "terms": terms})
        self.gens.append(g)

    def read(self, want=("K4d", "K31")):
        n = self.gens[0]["U"].shape[0] if self.gens else 0
        K4 = torch.zeros(n)
        K31 = torch.zeros(n, n) if "K31" in want else None
        for g in self.gens:
            T, U = g["T"], g["U"]
            if T is None:
                continue
            x = U * T
            if "path" in self.types or "pathdiag" in self.types or "pathoff" in self.types:
                y = x @ g["G"]
                K4 = K4 + (x * y).sum(1)
                if K31 is not None:
                    K31 = K31 + 0.5 * ((y * T) @ U.T + (y * U) @ T.T)
            if "star" in self.types:
                s4 = 4.0 * g["d3"]
                K4 = K4 + (x * U * U * s4[None, :]).sum(1)
                if K31 is not None:
                    K31 = K31 + 0.75 * ((U * U * T * s4[None, :]) @ U.T) + 0.25 * ((U * U * U * s4[None, :]) @ T.T)
            if "32" in self.types and g["k3"]:
                Dx = torch.zeros(n, n)
                for src in g["k3"]:      # grouped by the frozen leg X: one (n x w)(w x n) product per distinct X
                    Xm, Yc = src["m"], src["cur"]
                    acc = {}
                    for X, Yn, Zn, w in src["terms"]:
                        acc[X] = acc.get(X, 0.0) + (Yc[Yn] * Yc[Zn]) * w[None, :]
                    for X, a_ in acc.items():
                        Dx = Dx + a_ @ Xm[X].T
                Dd = Dx * g["d2"][None, :]
                K4 = K4 + 12.0 * (x * Dd).sum(1)
                if K31 is not None and self.k31_32 == "full":
                    xd = x * g["d2"][None, :]
                    for src in g["k3"]:  # (x D2 X_m) once per distinct X, then grouped by the last leg Z
                        Xm, Yc = src["m"], src["cur"]
                        xX = {X: xd @ Xm[X] for X in {t_[0] for t_ in src["terms"]}}
                        acc = {}
                        for X, Yn, Zn, w in src["terms"]:
                            acc[Zn] = acc.get(Zn, 0.0) + (xX[X] * Yc[Yn]) * w[None, :]
                        for Zn, a_ in acc.items():
                            K31 = K31 + 6.0 * (a_ @ Yc[Zn].T)
                if K31 is not None and self.k31_32 in ("full", "cheap"):
                    K31 = K31 + 3.0 * ((Dd * U) @ T.T + (Dd * T) @ U.T)
        if K31 is not None:
            K31 = K31 - torch.diag(torch.diagonal(K31))
        return K4, K31


class Young4Chain:
    """2026-10-04 (R3): the young-kappa4 K4d reader in the form the production estimator can run -- nothing carried that
    the chain does not already hold:
      T, U~  the legs of the chain's OWN source born at m: T = its A, U~ = its B + A diag(cm d1 var) (its column cut
             cm applied to the diagonal part too), so the path / star hubs are that source's hubs;
      Dx     the chain's own D pairing (read_slices' terms: tree L1 B^T + L2 A^T, correction (2/3)(E o A) A^T +
             (1/3)(A o A) E^T, path (A g3 B2) B^T + (B B2 g3) A^T while B2 lives, feed 2 m Bd A^T + q m^T) with the
             SINGLE legs (B, A, E, m) taken at the birth layer m instead of now -- the restructured pairing of
             experiments/2026-10-04_r3, which yields D(l) = Dx V for one product per age;
    and reads the K4d of the path's coinciding-hub piece (O(n^2); "path" = the full quadratic form, O(n^3)), the star
    (O(n^2)) and the 3+2 (O(n^2) given Dx). K4d only."""

    def __init__(self, ages=2, types=("pathdiag", "star", "32"), ages32=None, last32=True, only_last32=False,
                 dx="full"):
        """ages32: the 3+2 (the one reader that costs a product -- D = Dx V -- per age) only for generations of age
        <= ages32 (default: all ages); the O(n^2) path / star readers keep `ages`."""
        self.ages, self.types = int(ages), tuple(types)
        self.ages32 = self.ages if ages32 is None else int(ages32)
        self.last32 = bool(last32)   # False: no 3+2 at the LAST layer (production computes no D pairing there)
        self.only_last32 = bool(only_last32)   # True: the 3+2 at the last layer only (no D = Dx V products upstream)
        # dx="prod" (2026-10-06, k4a port check): Dx exactly as the production partners give it -- the correction
        # term through the snapshot's A dK3 + 3 B alpha only (no thin remainder ER at the snapshot), no q m^T
        self.dx = str(dx)
        self.rec = None   # DEBUG: a list -> (l, born, pathdiag, star, 32) per generation read
        self.gens = []

    def gate_and_retire(self, l):
        self.gens = [g for g in self.gens if l - g["born"] < self.ages]

    def birth(self, l, C, d1, d2, d3, sources, eleg):
        var = torch.diagonal(C)
        g = {"born": l, "d1v": d1 * var, "var": var.clone(), "d2": d2.clone(), "d3": d3.clone(), "snap": []}
        if "path" in self.types:
            g["G"] = 12.0 * d2[:, None] * C * d2[None, :]
        if "32" in self.types:
            for s in sources:
                if s.get("A") is None:
                    continue
                snap = {"ref": s, "A": s["A"].clone(), "B": s["B"].clone()}
                E = eleg(s)
                if E is not None:
                    snap["E"] = E.clone()
                if s.get("h") is not None:
                    snap["m"] = s["A"] @ s["h"]
                g["snap"].append(snap)
        self.gens.append(g)

    def read(self, sources, eleg, l=None, last=False, rowmask=None):
        """rowmask (k4a port check): 1/0 per neuron -- the 3+2 is zeroed where 0 (production's row cuts)."""
        n = sources[0]["A"].shape[0] if sources and sources[0].get("A") is not None else 0
        K4 = torch.zeros(n)
        for g in self.gens:
            use32 = ((l is None or l - g["born"] <= self.ages32) and (self.last32 or not last)
                     and (last or not self.only_last32))
            own = [s for s in sources if s.get("born") == g["born"]]
            if not own or own[0].get("A") is None:
                continue
            s0 = own[0]
            T = s0["A"]
            cm = s0.get("colmask")
            dd = g["d1v"] if cm is None else g["d1v"] * cm
            U = s0["B"] + T * dd[None, :]
            x = U * T
            pd_ = st_ = t32_ = None
            if "pathdiag" in self.types:      # the path with its two hubs coinciding: G_rr = 12 d2_r^2 var_r
                pd_ = (x * x * (12.0 * g["d2"] * g["d2"] * g["var"])[None, :]).sum(1)
                K4 = K4 + pd_
            if "path" in self.types:
                K4 = K4 + (x * (x @ g["G"])).sum(1)
            if "star" in self.types:
                st_ = (x * U * U * (4.0 * g["d3"])[None, :]).sum(1)
                K4 = K4 + st_
            if "32" in self.types and g["snap"] and use32:
                Dx = torch.zeros(n, n)
                for sn in g["snap"]:
                    s = sn["ref"]
                    A, B, gg = s["A"], s["B"], s["g"]
                    Dx = Dx + (2.0 * A * B * gg[None, :]) @ sn["B"].T + (B * B * gg[None, :]) @ sn["A"].T
                    E = eleg(s)
                    if E is not None and "E" in sn:
                        if self.dx == "prod" and s.get("ER") is not None:
                            snE = sn["A"] * s["dK3"][None, :] + 3.0 * sn["B"] * s["alpha"][None, :]
                        else:
                            snE = sn["E"]
                        Dx = Dx + (2.0 / 3.0) * (E * A) @ sn["A"].T + (1.0 / 3.0) * (A * A) @ snE.T
                    if s.get("B2") is not None:
                        g3 = s["g3"]
                        Dx = Dx + (A * g3[None, :] * s["B2"]) @ sn["B"].T + (B * s["B2"] * g3[None, :]) @ sn["A"].T
                    if s.get("h") is not None and "m" in sn:
                        ml = A @ s["h"]
                        Bd = B * s["d1b"][None, :]
                        q = (Bd * A).sum(1)
                        Dx = Dx + (2.0 * ml[:, None] * Bd) @ sn["A"].T
                        if self.dx != "prod":
                            Dx = Dx + q[:, None] * sn["m"][None, :]
                t32_ = 12.0 * (x * Dx * g["d2"][None, :]).sum(1)
                if rowmask is not None:
                    t32_ = t32_ * rowmask
                K4 = K4 + t32_
            if self.rec is not None:
                self.rec.append((l, g["born"], None if pd_ is None else pd_.clone(), None if st_ is None else st_.clone(),
                                 None if t32_ is None else t32_.clone()))
        return K4
