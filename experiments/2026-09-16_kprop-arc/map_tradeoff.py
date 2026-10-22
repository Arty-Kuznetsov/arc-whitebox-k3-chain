import torch, numpy as np
from pathlib import Path
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.flop_utils import NamedFlopCounter
BAKED=Path(__file__).resolve().parent/"baked"; gt=np.load(BAKED/"gt.npy"); B=2**41; n=1024; depth=16
def mk():
    m=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
    return m
def inject(m,W):
    for l in range(depth): m.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
    return m
def run(W,k,kind,factor):
    m=inject(mk(),W)
    K_by=mlp_kprop(m,{1:torch.zeros(n),2:torch.eye(n)},k_max=k,kind=kind,factor=factor,output_all=True,up_to_layer=f"act{depth-1}")
    return np.stack([np.asarray(K_by[f"act{l}"][1].to_tensor()) for l in range(depth)])
def flops(k,kind,factor):
    with NamedFlopCounter(strict=False) as c:
        mlp_kprop(mk(),{1:torch.zeros(n),2:torch.eye(n)},k_max=k,kind=kind,factor=factor,output_all=True,up_to_layer=f"act{depth-1}")
    return c.total()
configs=[("k2",2,Kind.SIMPLE,False),("k3-BASE",3,Kind.BASE,True),("k3-SIMPLE",3,Kind.SIMPLE,True)]
K=3
print(f"{'config':12s} {'final_mse':>11s} {'%budget':>9s} {'mult':>6s} {'adjusted':>11s}  note")
for name,k,kind,factor in configs:
    fl=flops(k,kind,factor); over=fl>B; mult=max(0.1,fl/B)
    fins=[float(((run(np.load(BAKED/f'weights_{i}.npy'),k,kind,factor)[-1]-gt[i][-1])**2).mean()) for i in range(K)]
    fin=np.mean(fins)
    note="OVER BUDGET -> FAIL (zeroed)" if over else "ok"
    print(f"{name:12s} {fin:>11.3e} {100*fl/B:>8.1f}% {mult:>6.3f} {fin*mult:>11.3e}  {note}")
