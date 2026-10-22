import sys
from pathlib import Path
import numpy as np, torch
torch.set_grad_enabled(False); torch.set_default_dtype(torch.float64)
from chain import run_chain
HERE = Path(__file__).resolve().parent
W = np.load(HERE / "runs" / "fs_test_W.npy"); out = np.load(HERE / "runs" / "fs_test_out.npy")
ref, _ = run_chain([torch.tensor(w, dtype=torch.float64) for w in W], k4="memo22", corr="split24")
for l in range(W.shape[0]):
    e = np.abs(out[l] - ref[l].numpy()).max() / np.abs(ref[l].numpy()).max()
    print(f"layer {l}: max rel diff flopscope vs torch chain = {e:.2e}")
