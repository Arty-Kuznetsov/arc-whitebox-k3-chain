import torch, numpy as np
from pathlib import Path
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from mlp_kprop.mlp import MLP
from mlp_kprop.kprop_harmonic import mlp_kprop, Kind
from mlp_kprop.factor_k3 import FactoredTensor
n=1024; depth=16
W=np.load("baked/weights_0.npy")
mlp=MLP(input_dim=n,hidden_dim=n,output_dim=n,num_layers=depth+1,nonlin="relu",init_kind="manual",w_var=2.0)
for l in range(depth): mlp.Ws[l].weight.data=torch.tensor(W[l].T,dtype=torch.float64)
Kb=mlp_kprop(mlp,{1:torch.zeros(n),2:torch.eye(n)},k_max=3,kind=Kind.SIMPLE,factor=True,output_all=True,up_to_layer=f"act{depth-1}")
print("Is kappa_3 low-rank? per-layer effective rank of factor[0] (n x R)")
print("layer | type        | R    | eff-rank(>1e-3,>1e-6) | energy in top 8 / 32 / 128 sv")
for l in range(depth):
    K3=Kb[f"act{l}"].get(3,None)
    if K3 is None: print(f"{l:2d} | (no deg-3)"); continue
    t=type(K3).__name__
    if isinstance(K3,FactoredTensor):
        A=K3._factors[0]
        s=torch.linalg.svdvals(A)
        s2=s*s; e=float(s2.sum())
        f=lambda k:float(s2[:k].sum())/e
        r3=int((s>1e-3*s[0]).sum()); r6=int((s>1e-6*s[0]).sum())
        print(f"{l:2d} | FactoredT   | {A.shape[1]:4d} | {r3:5d}, {r6:5d}          | {f(8):.4f} / {f(32):.4f} / {f(128):.4f}")
    else:
        print(f"{l:2d} | {t}")
