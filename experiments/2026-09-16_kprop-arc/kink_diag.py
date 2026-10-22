import torch, numpy as np
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
n=1024; depth=16
W=np.load("baked/weights_0.npy")
mlp=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
for l in range(depth): mlp.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
Kb=mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=2,kind=Kind.SIMPLE,output_all=True,up_to_layer=f"act{depth-1}")
print("Activation regime per layer (alpha = mu_pre/sigma_pre). 'kink' = needs nonlinear treatment.")
print("layer | frac |a|<1 (hard kink) | |a|<2 | |a|<3 | frac dead(a<-3) | on(a>3)")
kink2=[]
for l in range(depth):
    pre=Kb.get(f"pre{l}")
    if pre is None: continue
    mu=np.asarray(pre[1].to_tensor()); cov=np.asarray(pre[2].to_tensor())
    var=np.diag(cov) if cov.ndim==2 else cov
    a=mu/np.sqrt(np.maximum(var,1e-12))
    f1=np.mean(np.abs(a)<1); f2=np.mean(np.abs(a)<2); f3=np.mean(np.abs(a)<3)
    dead=np.mean(a<-3); on=np.mean(a>3)
    kink2.append(f2)
    print(f"{l:2d} |   {f1:.3f}           | {f2:.3f} | {f3:.3f} |  {dead:.3f}        | {on:.3f}")
print(f"\nmean frac |alpha|<2 across layers = {np.mean(kink2):.3f}  -> if large, on/off/kink is a weak lever")
