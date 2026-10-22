import csv, math, statistics as st
import os
D = os.path.dirname(os.path.abspath(__file__))
names = ["ours","j2w","marius","luna","mliston","meatproxy"]
label = {"ours":"ours k3p","j2w":"J2W #1","marius":"marius #2","luna":"Luna #3","mliston":"mliston #5","meatproxy":"MeatProxy #6"}
flops = {"ours":454.2,"j2w":240.6,"marius":331.1,"luna":285.8,"mliston":275.2,"meatproxy":349.1}
data = {}
for n in names:
    d = {}
    for row in csv.reader(open(f"{D}/{n}.csv")):
        d[row[0]] = (float(row[1]), float(row[2]))
    data[n] = d
mlps = list(data["ours"].keys())
assert all(set(data[n].keys())==set(mlps) for n in names)

def pear(a,b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x-ma)*(y-mb) for x,y in zip(a,b))
    den = math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))
    return num/den
def spear(a,b):
    ra = {v:i for i,v in enumerate(sorted(a))}
    rb = {v:i for i,v in enumerate(sorted(b))}
    return pear([ra[x] for x in a],[rb[x] for x in b])

print("=== means (raw final MSE, 1e-9 units; util = flops/2200) ===")
for n in names:
    raws = [data[n][m][0] for m in mlps]; adjs=[data[n][m][1] for m in mlps]
    print(f"{label[n]:14s} raw {st.mean(raws):6.2f}  adj {st.mean(adjs):6.3f}  flops {flops[n]:6.1f}G  util {flops[n]/2200*100:5.2f}%  raw/ours {st.mean(raws)/st.mean([data['ours'][m][0] for m in mlps]):.3f}")

print()
print("=== correlation of per-MLP RAW error with ours (is the difficulty ordering the same?) ===")
ours = [data["ours"][m][0] for m in mlps]
for n in names[1:]:
    o = [data[n][m][0] for m in mlps]
    print(f"{label[n]:14s} pearson {pear(ours,o):+.3f}   spearman {spear(ours,o):+.3f}")

print()
print("=== is their advantage a CONSTANT factor or MLP-dependent? (ratio theirs/ours per MLP) ===")
for n in names[1:]:
    r = [data[n][m][0]/data["ours"][m][0] for m in mlps]
    cv = st.stdev(r)/st.mean(r)
    print(f"{label[n]:14s} ratio mean {st.mean(r):.3f}  sd {st.stdev(r):.3f}  CV {cv*100:4.1f}%  min {min(r):.3f} ({mlps[r.index(min(r))]})  max {max(r):.3f} ({mlps[r.index(max(r))]})")
ourcv = st.stdev(ours)/st.mean(ours)
print(f"{'(our own raw)':14s} CV across MLPs {ourcv*100:4.1f}%")

print()
print("=== residual after removing the constant factor: correlation of log-ratio with our own error level ===")
for n in names[1:]:
    lr = [math.log(data[n][m][0]/data["ours"][m][0]) for m in mlps]
    print(f"{label[n]:14s} corr(log ratio, our log raw) {pear(lr,[math.log(x) for x in ours]):+.3f}   corr(log ratio, THEIR log raw) {pear(lr,[math.log(data[n][m][0]) for m in mlps]):+.3f}")

print()
print("=== the 6 MLPs where we are relatively WORST vs #2 marius, and the 6 where we are relatively best ===")
r = sorted(mlps, key=lambda m: data["marius"][m][0]/data["ours"][m][0])
for m in r[:6]:
    print(f"  best-for-us  {m:20s} ours {data['ours'][m][0]:6.2f}  marius {data['marius'][m][0]:6.2f}  ratio {data['marius'][m][0]/data['ours'][m][0]:.3f}")
for m in r[-6:]:
    print(f"  worst-for-us {m:20s} ours {data['ours'][m][0]:6.2f}  marius {data['marius'][m][0]:6.2f}  ratio {data['marius'][m][0]/data['ours'][m][0]:.3f}")

print()
print("=== cross-leader correlation (do the leaders agree with each other more than with us?) ===")
for i,a in enumerate(names):
    row = []
    for b in names:
        row.append(f"{pear([data[a][m][0] for m in mlps],[data[b][m][0] for m in mlps]):+.2f}")
    print(f"{label[a]:14s} " + " ".join(f"{x:>6s}" for x in row))
print(" " * 14 + " " + " ".join(f"{label[b][:6]:>6s}" for b in names))
