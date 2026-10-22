import torch, numpy as np
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind, linear_kprop, nonlin_kprop, coerce_input
from mlp_kprop.wick import relu_wick_coef
from mlp_kprop.flop_utils import NamedFlopCounter
n=1024; depth=16; B=2**41
gt=np.load("baked/gt.npy"); W=np.load("baked/weights_0.npy"); g=gt[0]
def mk():
    m=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
    for l in range(depth): m.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
    return m
def reverse(mlp, j):  # k=2 for first depth-j layers, then k=3 (from Gaussian) for last j
    m2=depth-j; means=[]
    if m2>0:
        Kb=mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=2,kind=Kind.SIMPLE,output_all=True,up_to_layer=f"act{m2-1}")
        for l in range(m2): means.append(np.asarray(Kb[f"act{l}"][1].to_tensor()))
        end=Kb[f"act{m2-1}"]; mean=end[1].to_tensor(); cov=end[2].to_tensor()
    else:
        mean=torch.zeros(n); cov=torch.eye(n)
    K3=coerce_input({1:mean,2:cov},k_max=3,kind=Kind.SIMPLE)
    for l in range(m2,depth):
        K3=linear_kprop(K3,mlp.Ws[l].weight,k_max=3,set_metric=mlp.init_scale[l])
        K3=nonlin_kprop(K3,nonlin_wick_coef=relu_wick_coef,k_max=3,kind=Kind.SIMPLE,factor=True)
        means.append(np.asarray(K3[1].to_tensor()))
    return np.stack(means)
mk2=float(((reverse(mk(),0)[-1]-g[-1])**2).mean())
print(f"SELF-CHECK: j=0 (all k2)={mk2:.3e} (expect 4.50e-6) ; j=16 (all k3)={float(((reverse(mk(),16)[-1]-g[-1])**2).mean()):.3e} (expect 3.7e-8)")
print("j (last-layers k3) | final_mse | %budget | mult | adjusted | vs k2")
for j in [0,1,2,3,4,6,8,16]:
    with NamedFlopCounter(strict=False) as c:
        pred=reverse(mk(),j)
    fl=c.total(); fin=float(((pred[-1]-g[-1])**2).mean()); mult=max(0.1,fl/B)
    print(f"  j={j:2d}             | {fin:.3e} | {100*fl/B:6.1f}% | {mult:.3f} | {fin*mult:.2e} | x{mk2/fin:.1f}{' [OVER]' if fl>B else ''}")
