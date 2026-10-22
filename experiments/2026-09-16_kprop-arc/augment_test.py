import torch, numpy as np
from pathlib import Path
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64); torch.set_num_threads(3)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.flop_utils import NamedFlopCounter
BAKED=Path("baked"); gt=np.load(BAKED/"gt.npy"); n=1024; depth=16; B=2**41
def mk(): return MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
def inj(m,W):
    for l in range(depth): m.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
    return m
def run(W,kind):
    Kb=mlp_kprop(inj(mk(),W),{1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=kind,factor=True,output_all=True,up_to_layer=f"act{depth-1}")
    return np.stack([np.asarray(Kb[f"act{l}"][1].to_tensor()) for l in range(depth)])
def fl(kind):
    with NamedFlopCounter(strict=False) as c:
        mlp_kprop(mk(),{1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=kind,factor=True,output_all=True,up_to_layer=f"act{depth-1}")
    return c.total()
for name,kind in [("k3-SIMPLE",Kind.SIMPLE),("k3-AUGMENT",Kind.AUGMENT)]:
    f=fl(kind); fins=[float(((run(np.load(BAKED/f'weights_{i}.npy'),kind)[-1]-gt[i][-1])**2).mean()) for i in range(3)]
    fin=np.mean(fins); mult=max(0.1,f/B)
    print(f"{name}: final_mse={fin:.3e}  flops={f:.3e} ({100*f/B:.1f}%)  mult={mult:.3f}  adjusted~{fin*mult:.2e} {'[OVER BUDGET]' if f>B else ''}")
