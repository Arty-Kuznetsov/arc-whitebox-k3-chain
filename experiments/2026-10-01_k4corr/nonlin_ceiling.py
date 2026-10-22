"""Is the kappa4 corrector limited by its LINEAR form? Ridge on the 39 features reaches held-out R2 ~0.353 with 10 or 30
training MLPs alike, so data is not the bottleneck. Here: a small MLP (39 -> H -> 1, tanh) -- cheap in flopscope (two
matmuls and one activation per layer) -- under 5-fold cross-validation by WHOLE NETWORKS (6 held out per fold).
If its held-out R2 is not clearly above ridge's, the linear corrector is the end of this road."""
import numpy as np, torch
from pathlib import Path
torch.manual_seed(0); torch.set_num_threads(3)
HERE = Path(__file__).resolve().parent
R = np.load(HERE / "recorded.npy", allow_pickle=True).item()
R.update(np.load(HERE / "recorded_10_29.npy", allow_pickle=True).item())
ids = sorted(R)
X, Y, G = [], [], []
for i in ids:
    for (l, F, k4, ora) in R[i]:
        sc = np.abs(k4).mean() + 1e-30
        X.append(F); Y.append((k4 - ora) / sc); G.append(np.full(len(k4), i))
X, Y, G = np.concatenate(X), np.concatenate(Y), np.concatenate(G)
print(f"samples {len(Y)}, features {X.shape[1]}, networks {len(ids)}")
folds = [ids[k::5] for k in range(5)]


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def ridge(Xtr, ytr, Xte, lam=1e-4):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    A = (Xtr - mu) / sd
    b = np.linalg.solve(A.T @ A + lam * len(ytr) * np.eye(A.shape[1]), A.T @ (ytr - ytr.mean()))
    return ((Xte - mu) / sd) @ b + ytr.mean()


def mlp(Xtr, ytr, Xte, H, epochs=60, wd=1e-4):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd == 0] = 1
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32); yt = torch.tensor(ytr, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], H), torch.nn.Tanh(), torch.nn.Linear(H, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=wd)
    n = len(yt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for k in range(0, n, 4096):
            idx = perm[k:k + 4096]
            opt.zero_grad()
            loss = ((net(xt[idx]).squeeze(1) - yt[idx]) ** 2).mean()
            loss.backward(); opt.step()
    with torch.no_grad():
        return net(torch.tensor((Xte - mu) / sd, dtype=torch.float32)).squeeze(1).numpy()


res = {"ridge": [], "mlp16": [], "mlp32": []}
for f in folds:
    te = np.isin(G, f); tr = ~te
    res["ridge"].append(r2(Y[te], ridge(X[tr], Y[tr], X[te])))
    res["mlp16"].append(r2(Y[te], mlp(X[tr], Y[tr], X[te], 16)))
    res["mlp32"].append(r2(Y[te], mlp(X[tr], Y[tr], X[te], 32)))
    print("  fold held-out", f, "  ".join(f"{k} {v[-1]:.3f}" for k, v in res.items()), flush=True)
print("mean held-out R2:", "  ".join(f"{k} {np.mean(v):.3f}" for k, v in res.items()))
