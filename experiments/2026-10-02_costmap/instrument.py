"""Build estimator_k3y_prof.py = src/estimator_k3y.py with flopscope namespaces around every call site of the budget
(enter/exit lines inserted around anchored regions, no re-indentation, so the arithmetic is untouched: the profile
copy must be BITWISE k3y with equal FLOPs -- checked by costmap.py)."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
NAME = sys.argv[1] if len(sys.argv) > 1 else "k3y"          # k3z: the same anchors (2026-10-02)
src = (HERE.parents[1] / "src" / f"estimator_{NAME}.py").read_text().split("\n")


def find(pat, start=0, exact=False):
    for i in range(start, len(src)):
        if (src[i].strip() == pat) if exact else (pat in src[i]):
            return i
    raise ValueError(pat)


def indent(i):
    return src[i][:len(src[i]) - len(src[i].lstrip())]


regions = []   # (name, first line, last line) -- inclusive, by index into src (computed before any insertion)
i = find("zero_rows(CY, Rc, Rcp)"); j = find("P2 = CYo[:n]", i)
regions.append(("ctransport", i + 1, j - 1))
i = find("if self.csym and sw is not None and n % (2 * quantum) == 0:"); j = find("C_pre = BO[0]", i)
regions.append(("cpre", i, j - 1))
i = find("zero_rows(GX, R, Rp)"); j = find("fnp.take(w, Pn, axis=0, out=XN[R:R + wn])", i)
regions.append(("transport", i + 1, j - 1))
i = find("fnp.matmul(Wt, Hb[cur][m:S], out=Hb[nxt][m:S])")
regions.append(("thin_transport", i, i))
i = find("fnp.multiply(ZBT[:om], 1.0, out=ZS[:om])"); j = find("fnp.matmul(fnp.swapaxes(PT, 0, 1), ZH[:m], out=Hb[nxt][:m])", i)
regions.append(("band_members", i, j))
i = find("t = region_readout(BTn, ATn, om, WB"); regions.append(("readout", i, i))
i = find("t = t + region_readout(BTm, ATm, 0, om"); regions.append(("readout", i, i))
i = find("region_pairs(BTn, ATn, om, WB"); regions.append(("pairprep", i, i))
i = find("region_pairs(BTm, ATm, 0, om"); regions.append(("pairprep", i, i))
# the E2 branch (production): non-members' dense pairing, then the members' compressed pairing
i = find('Dv = ws["Dext"][:(n_r + 1) * (npc + 1)].reshape((n_r + 1, npc + 1))')
j = find("sw.cap = self._cap0", i)
regions.append(("dpair", i + 1, j))
i2 = find("Rm2 = 2 * om", j); i2 -= 1                       # the `if m:` line
j2 = find("fnp.add(Dv[:n_r, :npc], ws[\"Dtc\"][:n_r * npc].reshape((n_r, npc)), out=Dv[:n_r, :npc])", i2)
regions.append(("mpair", i2, j2))
i = find("K4d = K22 = None"); j = find("K22 = pat * ((Ssum - trM) / fnp.sum(pat))", i)
regions.append(("k4levels", i, j))
i = find("if ((self.k4corr or self._k4rec is not None) and not last and K4d is not None and D is not None")
j = find("K4d = K4d - (self.k4s * fnp.mean(fnp.abs(K4d))) * (pe + cb0)", i)
regions.append(("k4corr", i, j))
i = find("o = _nonlin(mu_pre, var, C0, t, D, K4d, K22"); j = find("c0k22=self.c0k22, marg2_keep=m2keep)", i)
regions.append(("nonlin", i, j))
i = find("if band is not None and S > 0 and 0 <= l + 1 - a0b == m and m < S:"); j = find("m += 1", i)
regions.append(("band_entry", i, j))
i = find("U = d1[:, None] * C0"); j = find("fnp.multiply(R_.T @ Qr, 1.0, out=Hb[cur][S])", i)
regions.append(("births", i, j))

ins = {}   # line index -> list of (before|after, text)
for k, (name, a, b) in enumerate(regions):
    ind = indent(a)
    var = f"_nsp{k}"
    ins.setdefault(a, []).append(("before", f'{ind}{var} = flops.namespace("{name}"); {var}.__enter__()'))
    ins.setdefault(b, []).append(("after", f"{ind}{var}.__exit__(None, None, None)"))
out = []
for idx, line in enumerate(src):
    for where, text in ins.get(idx, []):
        if where == "before":
            out.append(text)
    out.append(line)
    for where, text in reversed(ins.get(idx, [])):
        if where == "after":
            out.append(text)
(HERE / f"estimator_{NAME}_prof.py").write_text("\n".join(out))
for name, a, b in regions:
    print(f"{name:15s} lines {a + 1}-{b + 1}: {src[a].strip()[:70]!r} .. {src[b].strip()[:50]!r}")
