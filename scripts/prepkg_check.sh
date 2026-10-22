#!/bin/bash
# Static pre-package check of an estimator file against the grader's prohibited list (SPEC "Allowed code"):
# only the standard library, flopscope and whestbench may be imported; no numpy/scipy/torch, no ctypes/cffi,
# no threads/subprocess/async/multiprocessing. The grader rejects the WHOLE submission ("Prohibited file in
# submission") on a static match, even inside a branch that never runs -- 333367 died of exactly that.
f="$1"
bad=$(grep -nE '^\s*(import|from)\s+(numpy|scipy|torch|jax|numba|ctypes|cffi|threading|multiprocessing|subprocess|asyncio|concurrent)\b' "$f")
if [ -n "$bad" ]; then echo "PROHIBITED IMPORTS in $f:"; echo "$bad"; exit 1; fi
echo "static import scan clean: $f"
