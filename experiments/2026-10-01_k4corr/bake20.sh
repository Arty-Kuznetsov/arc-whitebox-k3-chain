#!/bin/bash
# oracles for mini MLPs 10-29 (2e6, one seed) so the corrector can be trained on 30 networks and validated on 70.
# The baked weights/GT set only covers 0-9: dump 10-29 first (weights only are needed for the oracle).
cd "$(dirname "$0")/../.."
nice -n 12 uv run --project external/whest-starterkit python - <<'PY'
import numpy as np, whestbench as wb
from pathlib import Path
OUT = Path("experiments/2026-09-16_kprop-arc/baked")
ds = wb.load_dataset("aicrowd/arc-whestbench-public-2026", revision="v2-phase2", split="mini")
if not hasattr(ds, "column_names"):
    ds = ds["mini"]
for i, mlp in enumerate(wb.iter_mlps(ds)):
    if i >= 30:
        break
    if i >= 10:
        np.save(OUT / f"weights_{i}.npy", np.stack([np.asarray(w, dtype=np.float32) for w in mlp.weights]))
print("dumped weights 10-29")
PY
cd experiments/2026-09-24_k3-chain
ARC="$(cd ../../external/mlp_cumulant_propagation && pwd)"
for i in $(seq 10 29); do
  echo "=== oracle $i $(date +%H:%M:%S)"
  env PYTHONPATH="$ARC/src:." nice -n 15 uv run --project "$ARC" python mc_oracle.py $i 2e6 > logs/mc_oracle_${i}_2e6.txt 2>&1
  echo "    exit=$? $(date +%H:%M:%S)"
done
echo BAKE20DONE
