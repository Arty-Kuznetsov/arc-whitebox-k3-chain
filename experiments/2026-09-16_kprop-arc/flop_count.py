import torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.flop_utils import NamedFlopCounter
n=1024; depth=16; B=2**41
mlp = MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
print("our fnp covariance baseline (measured by flopscope): 5.17e10 = 2.35% of B")
for k in [2,3]:
    with NamedFlopCounter(strict=False) as c:
        mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=k,kind=Kind.SIMPLE,factor=(k>=3),output_all=True,up_to_layer=f"act{depth-1}")
    tot=c.total(); raw=c.raw_total()
    mult=max(0.1, tot/B)
    print(f"k={k}: sym-opt total={tot:.3e} ({100*tot/B:.2f}% of B, mult={mult:.3f})  raw={raw:.3e} ({100*raw/B:.2f}%)")
