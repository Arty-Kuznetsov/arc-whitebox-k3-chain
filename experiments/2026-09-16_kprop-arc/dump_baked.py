"""Dump baked-GT weights + ground-truth per-layer means for the first K mini MLPs to ./baked/*.npy
so the ARC-venv kprop script (torch) can consume them without whestbench. Run in the whest venv."""
import sys
from pathlib import Path
import numpy as np
import whestbench as wb

K = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = Path(__file__).resolve().parent / "baked"
OUT.mkdir(exist_ok=True)

ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
if not hasattr(ds, "column_names"):
    ds = ds["mini"]
print("columns:", ds.column_names)
gt_name = next((c for c in ["all_layer_means", "all_means", "layer_means", "final_means", "means"]
                if c in ds.column_names), None)
print("GT column:", gt_name)

# weights per MLP (challenge orientation (in,out)), via iter_mlps
for i, mlp in enumerate(wb.iter_mlps(ds)):
    if i >= K:
        break
    W = np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights])  # (depth, in, out)
    np.save(OUT / f"weights_{i}.npy", W)
    if i == 0:
        print("per-MLP weights shape:", W.shape)

gt = np.asarray(ds.with_format("numpy")[gt_name][:K], dtype=np.float32)
np.save(OUT / "gt.npy", gt)
print(f"dumped {K} MLPs; GT '{gt_name}' shape {gt.shape}  (min {gt.min():.4f}, max {gt.max():.4f})")
