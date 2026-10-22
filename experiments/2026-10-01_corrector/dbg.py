import sys, numpy as np, torch
from pathlib import Path
HERE = Path('.').resolve()
sys.path.insert(0, str(HERE.parent / "2026-09-29_port-battery"))
import pb
from chain import run_chain
torch.set_grad_enabled(False); torch.set_num_threads(3)
B9 = pb.matched(-2.5, satrow=2.5, satrow_k4="sep")
Ws = np.load(pb.BAKED / "weights_0.npy")
Wt = [torch.tensor(w, dtype=torch.float64) for w in Ws]
pred, info = run_chain(Wt, record=True, **B9)
ORA = HERE.parent / "2026-09-24_k3-chain" / "oracles"
oa = np.load(ORA / "mc_oracle_0.npz"); ob = np.load(ORA / "mc_oracle_0_s777.npz")
n = 1024; L = 16
for l in (8, 12, 15):
    mu = info[l]["mu_pre"].numpy(); var = np.maximum(info[l]["var"].numpy(), 1e-12); sd = np.sqrt(var); a = mu/sd
    for ck, ok in (("var","var"),("t","k3"),("K4d","k4")):
        x = info[l][ck].numpy().astype(np.float64)
        ea = x - oa[ok][l]; eb = x - ob[ok][l]
        print(f"L{l:2d} {ck:4s}: |x| rms {np.sqrt((x**2).mean()):.4e}  |ea| rms {np.sqrt((ea**2).mean()):.4e}  "
              f"ea.std {ea.std():.4e}  corr(eA,eB) {np.corrcoef(ea,eb)[0,1]:.3f}  "
              f"corr(ea, a) {np.corrcoef(ea,a)[0,1]:+.3f}  corr(ea, x) {np.corrcoef(ea,x)[0,1]:+.3f}")
    print()

print("=== ridge debug, layer 15 var ===")
l = 15
mu = info[l]["mu_pre"].numpy(); var = np.maximum(info[l]["var"].numpy(), 1e-12); sd = np.sqrt(var); a = mu/sd
from math import erf
phi = np.exp(-0.5*a*a)/np.sqrt(2*np.pi)
Ph = 0.5*(1.0+np.array([erf(z/np.sqrt(2)) for z in a]))
t = info[l]["t"].numpy(); k4 = info[l]["K4d"].numpy()
u = (Ws[min(l+1,len(Ws)-1)].astype(np.float64)**2).sum(1); wcol = (Ws[l].astype(np.float64)**2).sum(0)
rank = np.argsort(np.argsort(-a))/n
F = np.stack([np.ones(n), a, a*a, a**3, np.clip(a,-6,6)**4, phi, Ph, a*phi, (a*a-1)*phi,
              sd, var, t/(sd**3+1e-30), k4/(var*var+1e-30), u, wcol, rank], 1)
x = info[l]["var"].numpy().astype(np.float64); ea = x - oa["var"][l]; y = ea - ea.mean()
s = F.std(0); s[s==0]=1.0; Fz = F/s
print("cond(Fz'Fz) =", np.linalg.cond(Fz.T@Fz))
A = Fz.T@Fz + 1e-6*np.trace(Fz.T@Fz)/Fz.shape[1]*np.eye(Fz.shape[1])
beta = np.linalg.solve(A, Fz.T@y)
print("beta[:5] =", beta[:5], " |beta| =", np.abs(beta).max())
res = y - Fz@beta
print("R2 =", 1 - res@res/(y@y))
beta2, *_ = np.linalg.lstsq(Fz, y, rcond=None)
res2 = y - Fz@beta2
print("lstsq R2 =", 1 - res2@res2/(y@y))
