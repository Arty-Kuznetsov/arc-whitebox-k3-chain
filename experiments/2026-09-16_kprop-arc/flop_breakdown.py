import torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.flop_utils import NamedFlopCounter
n=1024; depth=16; B=2**41
mlp=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
for k,factor in [(2,False),(3,True)]:
    with NamedFlopCounter(strict=False, aggregate=True) as c:
        mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=k,kind=Kind.SIMPLE,factor=factor,output_all=True,up_to_layer=f"act{depth-1}")
    d=c.flop_dict(); tot=sum(d.values())
    print(f"\n=== k={k} total={tot:.3e} ({100*tot/B:.1f}% of B) — top regions ===")
    for name,v in sorted(d.items(), key=lambda x:-x[1])[:10]:
        print(f"  {name:28s} {v:.3e}  {100*v/tot:5.1f}%")
