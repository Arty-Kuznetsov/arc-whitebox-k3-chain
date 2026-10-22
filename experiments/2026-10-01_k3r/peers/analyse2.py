import csv, math, statistics as st, os
D = os.path.dirname(os.path.abspath(__file__))
names = ["ours","j2w","marius","luna","mliston","meatproxy"]
label = {"ours":"ours k3p","j2w":"J2W #1","marius":"marius #2","luna":"Luna #3","mliston":"mliston #5","meatproxy":"MeatProxy #6"}
flops = {"ours":454.2,"j2w":240.6,"marius":331.1,"luna":285.8,"mliston":275.2,"meatproxy":349.1}
data = {n: {r[0]:(float(r[1]),float(r[2])) for r in csv.reader(open(f"{D}/{n}.csv"))} for n in names}
mlps = list(data["ours"].keys())
def pear(a,b):
    ma,mb=st.mean(a),st.mean(b)
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/math.sqrt(sum((x-ma)**2 for x in a)*sum((y-mb)**2 for y in b))

print("=== spread of raw across the 50 public MLPs (is their error MORE uniform than ours?) ===")
print(f"{'who':14s} {'mean':>7s} {'sd':>6s} {'CV%':>6s} {'min':>7s} {'max':>7s} {'max/min':>8s}")
for n in names:
    r=[data[n][m][0] for m in mlps]
    print(f"{label[n]:14s} {st.mean(r):7.2f} {st.stdev(r):6.2f} {st.stdev(r)/st.mean(r)*100:6.1f} {min(r):7.2f} {max(r):7.2f} {max(r)/min(r):8.3f}")

print()
print("=== decomposition: our raw = (common difficulty) x (our extra). Regress log(ours) on log(marius) ===")
lo=[math.log(data["ours"][m][0]) for m in mlps]; lm=[math.log(data["marius"][m][0]) for m in mlps]
mo,mm=st.mean(lo),st.mean(lm)
beta=sum((x-mm)*(y-mo) for x,y in zip(lm,lo))/sum((x-mm)**2 for x in lm)
resid=[y-(mo+beta*(x-mm)) for x,y in zip(lm,lo)]
print(f"slope beta = {beta:.3f} (1.0 = pure common factor), R2 = {pear(lo,lm)**2:.3f}")
print(f"residual sd of log(ours) after marius explains it: {st.stdev(resid):.4f}  (= {st.stdev(resid)*100:.1f}% MLP-specific error of OURS that marius does not share)")
print(f"total sd of log(ours): {st.stdev(lo):.4f}")
print("top-5 MLPs by OUR unexplained excess:")
for m,_ in sorted(zip(mlps,resid), key=lambda t:-t[1])[:5]:
    i=mlps.index(m); print(f"   {m:20s} excess x{math.exp(resid[i]):.3f}  ours {data['ours'][m][0]:6.2f}  marius {data['marius'][m][0]:6.2f}  J2W {data['j2w'][m][0]:6.2f}")
print("bottom-5 (we are relatively fine):")
for m,_ in sorted(zip(mlps,resid), key=lambda t:t[1])[:5]:
    i=mlps.index(m); print(f"   {m:20s} excess x{math.exp(resid[i]):.3f}  ours {data['ours'][m][0]:6.2f}  marius {data['marius'][m][0]:6.2f}  J2W {data['j2w'][m][0]:6.2f}")

print()
print("=== what product (raw x util) each needs: adj = raw[1e-9] x util ===")
for n in names:
    r=st.mean([data[n][m][0] for m in mlps]); u=flops[n]/2200
    print(f"{label[n]:14s} raw {r:6.2f} x util {u*100:5.2f}% = {r*u:6.3f}e-9   (board {st.mean([data[n][m][1] for m in mlps]):.3f})")
print()
print("=== what we need for 2.00e-9 from our k3q point (raw 21.48, util 19.735%) ===")
for u in (19.735, 18.0, 16.0, 14.0, 12.0, 10.94):
    print(f"   at util {u:5.2f}%  ->  raw must be {2.00/(u/100):6.2f}e-9   (we have 21.48; marius has 11.36, the board's best raw)")
