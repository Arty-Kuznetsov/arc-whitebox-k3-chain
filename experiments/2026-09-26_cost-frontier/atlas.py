"""Cost atlas of k3l: where the 26.27% of B are, by component and by layer, and the ceiling of each lever.
Model only (cost.py), calibrated on the three production flopscope runs k3i / k3j / k3k."""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import cost

N, L, BS = cost.N, cost.L, cost.BILL_S
MEAS = 0.2627          # k3j/k3l measured utilisation (100-MLP flopscope run)


def breakdown(deadcut=-3.0, window=None, paths=True, pathrank=None):
    f = cost.alive(deadcut)
    qp = 1.0 if pathrank is None else pathrank / N
    comp = {k: [0.0] * L for k in ("legs", "pair", "pathleg", "Cfused", "Cpre")}
    for l in range(L):
        S = l if window is None else min(l, window)
        last = l == L - 1
        fin = f[l - 1] if l > 0 else 1.0
        fo = f[l]
        if S > 0:
            comp["legs"][l] = (2 * S - 1) * fin
            comp["Cfused"][l] = 1.0
            if paths:
                comp["pathleg"][l] = qp
            if not last:
                comp["pair"][l] = 2 * S * fo * fo
        if not last:
            comp["Cpre"][l] = 1.0
    return comp


LABEL = {"legs": "source leg transports (A, B)", "pair": "D pairings (alive x alive)",
         "pathleg": "path leg transport", "Cfused": "fused Wt.C (rides the leg stack)",
         "Cpre": "C_pre second product"}

c = breakdown()
tot = sum(sum(v) for v in c.values())
scale = MEAS / (BS * tot / cost.B_UNITS)      # rescale the model onto the measured 26.27%
print(f"ATLAS of k3l  (model {100*BS*tot/cost.B_UNITS:.2f}% of B, rescaled onto the measured {100*MEAS:.2f}%)\n")
print(f"{'component':38s} {'units':>8s} {'% of B':>8s} {'share':>7s}")
for k, v in sorted(c.items(), key=lambda kv: -sum(kv[1])):
    u = sum(v)
    print(f"  {LABEL[k]:36s} {u:8.1f} {100*scale*BS*u/cost.B_UNITS:7.2f}% {100*u/tot:6.1f}%")
print(f"  {'TOTAL':36s} {tot:8.1f} {100*MEAS:7.2f}% {100.0:6.1f}%")

print("\nby layer (units of one n^3 product; 'cum%' = cumulative share of the bill)")
print(f"{'l':>3s} {'S':>3s} {'alive':>6s} {'legs':>7s} {'pairs':>7s} {'base':>6s} {'total':>7s} {'% of B':>7s} {'cum%':>6s}")
cum = 0.0
for l in range(L):
    tl = sum(c[k][l] for k in c)
    cum += tl
    print(f"{l:3d} {l:3d} {cost.alive(-3.0)[l]:6.3f} {c['legs'][l]:7.1f} {c['pair'][l]:7.1f}"
          f" {c['Cfused'][l]+c['Cpre'][l]+c['pathleg'][l]:6.1f} {tl:7.1f}"
          f" {100*scale*BS*tl/cost.B_UNITS:6.2f}% {100*cum/tot:5.1f}%")

print("\nceiling of each lever: utilisation if that component were FREE (accuracy ignored)")
base = 100 * MEAS
for k in ("legs", "pair", "pathleg", "Cfused", "Cpre"):
    u = tot - sum(c[k])
    print(f"  drop {LABEL[k]:36s} -> {100*scale*BS*u/cost.B_UNITS:6.2f}%  (x{base/(100*scale*BS*u/cost.B_UNITS):.2f})")
print(f"  {'the 10% floor needs':36s} -> {10.0:6.2f}%  (x{base/10:.2f}), i.e. {10/100/scale/BS*cost.B_UNITS:.0f} of {tot:.0f} units")

print("\nwhat the floor means structurally: units per layer if only the youngest k sources stay dense")
for k in (1, 2, 3, 4):
    st, pl = cost.units(window=k)
    print(f"  window {k}: {100*scale*100*cost.util(window=k)/100:6.2f}%   ({st:.0f} units)")
print("\nStrassen depth (billing 0.636 -> L5 would be ~0.560 on the stacked products; residual-bound, Mac call)")
for b, nm in ((0.636, "L4 (current, measured)"), (0.560, "L5 (hypothetical)"), (1.0, "no Strassen")):
    print(f"  billing {b:.3f} {nm:26s} -> {100*scale*b*tot/cost.B_UNITS:6.2f}%")

print("\nwhere the bill sits by SOURCE AGE (legs + D pairings only; 412 units total, 26.27% of B)")
f = cost.alive(-3.0)
by = {}
for l in range(L):
    for a in range(1, l + 1):
        u = (1 if a == 1 else 2) * (f[l-1] if l else 1.0) + (0 if l == L-1 else 2 * f[l] * f[l])
        by[a] = by.get(a, 0.0) + u
tot_age = sum(by.values())
cum = 0.0
print(f"{'age':>4s} {'units':>7s} {'% of B':>7s} {'cum % of B':>11s}   {'bill if all ages >= a were FREE':>32s}")
for a in sorted(by):
    cum += by[a]
    rest = tot + 0.0 - sum(by[x] for x in by if x >= a)
    print(f"{a:4d} {by[a]:7.1f} {100*scale*BS*by[a]/cost.B_UNITS:6.2f}% {100*scale*BS*cum/cost.B_UNITS:10.2f}%"
          f"   {100*scale*BS*rest/cost.B_UNITS:6.2f}% of B")
print("\nleaders' telemetry for scale: raw 1.56-1.67e-8 at 12.7-15.5% of B = 200-243 units;")
print("504aldo 2.13e-8 at ~25.4% = our cost.  10% = 157 units.")
