import torch, numpy as np
from pathlib import Path
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
import mlp_kprop.factor_k3 as fk3
n=1024; depth=16; B=2**41
gt=np.load("baked/gt.npy"); W=np.load("baked/weights_0.npy"); g=gt[0]
def mk():
    m=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
    for l in range(depth): m.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
    return m
def run():
    Kb=mlp_kprop(mk(),{1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=Kind.SIMPLE,factor=True,output_all=True,up_to_layer=f"act{depth-1}")
    return np.asarray(Kb[f"act{depth-1}"][1].to_tensor())
orig=fk3.FactoredTensor.contract_W
def make_capped(Rmax):
    def capped(self,Wm):
        F=self._factors
        R=F[0].shape[1]
        if R>Rmax:
            w=torch.ones(R,dtype=F[0].dtype)
            for f in F: w=w*f.norm(dim=0)
            idx=torch.topk(w,Rmax).indices
            F=[f[:,idx] for f in F]
        nf=tuple(Wm@f for f in F)
        return fk3.FactoredTensor(n=self.n,d=self.d,factors=nf,device=self.device,dtype=self.dtype)
    return capped
# full reference
fk3.FactoredTensor.contract_W=orig
full=run(); fin_full=float(((full-g[-1])**2).mean())
print(f"full k=3 (no cap): final_mse={fin_full:.3e}  (max R reached ~ {2*n*depth})")
for Rmax in [128,256,512,1024,2048,4096]:
    fk3.FactoredTensor.contract_W=make_capped(Rmax)
    p=run(); fin=float(((p-g[-1])**2).mean())
    print(f"  R_max={Rmax:5d}: final_mse={fin:.3e}   (x{fin/fin_full:.1f} vs full)")
fk3.FactoredTensor.contract_W=orig
