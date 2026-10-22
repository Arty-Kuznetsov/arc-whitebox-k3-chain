import torch, numpy as np
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind, linear_kprop, relu_kprop, coerce_input
from mlp_kprop.flop_utils import NamedFlopCounter
n=1024; depth=16; B=2**41
gt=np.load("baked/gt.npy"); W=np.load("baked/weights_0.npy"); g=gt[0]
def mk():
    m=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
    for l in range(depth): m.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
    return m
def adaptive(mlp, m):  # k=3 for layers 0..m-1, then k=2 for m..depth-1
    means=[]
    if m>0:
        Kb=mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=Kind.SIMPLE,factor=True,output_all=True,up_to_layer=f"act{m-1}")
        for l in range(m): means.append(np.asarray(Kb[f"act{l}"][1].to_tensor()))
        end=Kb[f"act{m-1}"]; mean=end[1].to_tensor(); cov=end[2].to_tensor()
    else:
        mean=torch.zeros(n); cov=torch.eye(n)
    K2=coerce_input({1:mean,2:cov},k_max=2,kind=Kind.SIMPLE)
    for l in range(m,depth):
        K2=linear_kprop(K2,mlp.Ws[l].weight,k_max=2,set_metric=mlp.init_scale[l])
        K2=relu_kprop(K2,k_max=2,kind=Kind.SIMPLE)
        means.append(np.asarray(K2[1].to_tensor()))
    return np.stack(means)
# references for self-check
ref3=np.asarray(mlp_kprop(mk(),{1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=Kind.SIMPLE,factor=True,output_all=True,up_to_layer=f"act{depth-1}")[f"act{depth-1}"][1].to_tensor())
ref2=np.asarray(mlp_kprop(mk(),{1:torch.zeros(n),2:torch.eye(n)},k_max=2,kind=Kind.SIMPLE,output_all=True,up_to_layer=f"act{depth-1}")[f"act{depth-1}"][1].to_tensor())
print(f"SELF-CHECK: full k=3 final_mse={float(((ref3-g[-1])**2).mean()):.3e} ; full k=2={float(((ref2-g[-1])**2).mean()):.3e}")
print("m (k3 layers) | final_mse | %budget | mult | adjusted | vs k2")
mk2=float(((ref2-g[-1])**2).mean())
for m in [0,1,2,3,4,6,8,12,16]:
    with NamedFlopCounter(strict=False) as c:
        pred=adaptive(mk(),m)
    fl=c.total(); fin=float(((pred[-1]-g[-1])**2).mean()); mult=max(0.1,fl/B)
    tag=" [OVER]" if fl>B else ""
    print(f"  m={m:2d}        | {fin:.3e} | {100*fl/B:6.1f}% | {mult:.3f} | {fin*mult:.2e} | x{mk2/fin:.1f}{tag}")
