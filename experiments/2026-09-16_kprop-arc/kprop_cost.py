import torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.flop_utils import NamedFlopCounter
n=1024; B=2**41
def cost(depth,k,factor,breakdown=False):
    mlp=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
    with NamedFlopCounter(strict=False, aggregate=True) as c:
        mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=k,kind=Kind.SIMPLE,factor=factor,output_all=True,up_to_layer=f"act{depth-1}")
    d=c.flop_dict(); tot=sum(d.values())
    if breakdown:
        for name,v in sorted(d.items(),key=lambda x:-x[1])[:8]:
            print(f"    {name:28s} {v:.3e}  {100*v/tot:5.1f}%")
    return tot
print("=== k=3 per-region breakdown (depth 16) ===")
cost(16,3,True,breakdown=True)
print("\n=== depth scaling of k=3 total (paper predicts factorized ~ O(L^2)) ===")
prev=None
for depth in [2,4,8,16]:
    t=cost(depth,3,True); r=f"  (x{t/prev:.2f} per 2x depth)" if prev else ""
    print(f"  depth={depth:2d}: {t:.3e}  {100*t/B:6.1f}% of B{r}"); prev=t
