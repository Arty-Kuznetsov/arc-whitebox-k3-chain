"""Rebuild experiments/2026-09-16_kprop-arc/baked/*.npy on a fresh machine.

`dump_baked.py` produces the same files, but only after `wb.load_dataset` has
materialised the entire 100-MLP `mini` split: ~7 GB downloaded and ~14 GB on disk.
This script downloads only the parquet shards that actually hold the first K MLPs
(16 MLPs per shard, ~1.12 GB each), so the default K=10 costs one shard.

Output (identical to dump_baked.py — verified bit-for-bit against the Mac's baked/
on 2026-09-25, weights_0 / weights_9 / gt all matching by sha256):
    weights_<i>.npy   (16, 1024, 1024) float32   challenge orientation (in, out)
    gt.npy            (K, 16, 1024)    float32   `all_layer_means`, the target

The dataset row stores `weights` exactly as `MLP.from_row` hands it to an
estimator (it casts to float32 and does not transpose), so reading the column
directly is the same data the grader feeds `predict()`.

    python scripts/fetch_baked.py                 # first 10 MLPs
    python scripts/fetch_baked.py --n-mlps 16     # the whole first shard, same download
Run it inside the whest env (it needs huggingface_hub + pyarrow):
    uv run --project external/whest-starterkit python scripts/fetch_baked.py
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO_ID = "aicrowd/arc-whestbench-public-2026"
SHARDS = 7  # mini split: data/mini-000NN-of-00007.parquet, 16 MLPs each
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "experiments" / "2026-09-16_kprop-arc" / "baked"
# Known-good sha256 prefixes of the Mac's baked/ arrays (bytes, not the .npy file).
KNOWN = {"weights_0": "d524e662eca3d117", "weights_9": "3af7229b6cc196af", "gt_10": "557e6fcc891474bb"}


def digest(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-mlps", type=int, default=10, help="how many MLPs to bake (default 10)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--revision", default="v2-phase2", help="HF revision (tag or SHA)")
    ap.add_argument("--cache", type=Path, default=None, help="where to put the parquet shards")
    ap.add_argument("--keep-parquet", action="store_true", help="do not delete the shards afterwards")
    ap.add_argument("--force", action="store_true", help="re-bake even if the outputs exist")
    args = ap.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    want = [out / f"weights_{i}.npy" for i in range(args.n_mlps)] + [out / "gt.npy"]
    if not args.force and all(p.exists() for p in want):
        print(f"baked/ already holds {args.n_mlps} MLPs at {out} — nothing to do (--force to re-bake)")
        return

    cache = args.cache or (out.parent / "_parquet")
    cache.mkdir(parents=True, exist_ok=True)

    gts: list[np.ndarray] = []
    done = 0
    for shard in range(SHARDS):
        if done >= args.n_mlps:
            break
        name = f"data/mini-{shard:05d}-of-{SHARDS:05d}.parquet"
        print(f"downloading {name} (~1.1 GB) ...", flush=True)
        path = Path(hf_hub_download(REPO_ID, name, repo_type="dataset",
                                    revision=args.revision, local_dir=cache))
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            if done >= args.n_mlps:
                break
            tbl = pf.read_row_group(rg, columns=["mlp_id", "weights", "all_layer_means"])
            n = tbl.num_rows
            # Array3D/Array2D land as nested list columns; flatten() honours the
            # list offsets (unlike .values), so this is an exact, cheap reshape.
            w = np.asarray(tbl.column("weights").combine_chunks().flatten().flatten().flatten(),
                           dtype=np.float32).reshape(n, 16, 1024, 1024)
            g = np.asarray(tbl.column("all_layer_means").combine_chunks().flatten().flatten(),
                           dtype=np.float32).reshape(n, 16, 1024)
            for i in range(n):
                if done >= args.n_mlps:
                    break
                np.save(out / f"weights_{done}.npy", w[i])
                gts.append(g[i])
                done += 1
            del tbl, w, g
        del pf
        if not args.keep_parquet:
            path.unlink()
            print(f"    removed {path.name}")

    if done < args.n_mlps:
        raise SystemExit(f"only {done} MLPs available in the mini split, asked for {args.n_mlps}")

    gt = np.stack(gts)
    np.save(out / "gt.npy", gt)
    print(f"baked {done} MLPs -> {out}  (gt {gt.shape}, min {gt.min():.4f}, max {gt.max():.4f})")

    # Self-check: these arrays must be bit-identical to the ones the results in
    # notes/ were computed from. A mismatch means the dataset revision moved.
    checks = {"weights_0": digest(np.load(out / "weights_0.npy"))}
    if done >= 10:
        checks["weights_9"] = digest(np.load(out / "weights_9.npy"))
        checks["gt_10"] = digest(gt[:10])
    bad = [k for k, v in checks.items() if KNOWN[k] != v]
    for k, v in checks.items():
        print(f"    sha256[{k}] = {v}  {'OK' if KNOWN[k] == v else 'MISMATCH, expected ' + KNOWN[k]}")
    if bad:
        raise SystemExit("baked data does not match the reference — do NOT compare any number "
                         "produced from it against notes/ until this is understood")


if __name__ == "__main__":
    main()
