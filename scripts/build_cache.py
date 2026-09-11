#!/usr/bin/env python3
"""Precompute the prepared-frame cache.

    python scripts/build_cache.py                     # every train volume
    python scripts/build_cache.py --limit 3           # smoke subset
    python scripts/build_cache.py --dir <test_dir>    # test volumes

~35 ms/frame, so a full 199-volume train cache is ~12 min and 5.2 GB. Run this
once at the start of a Kaggle session; without it each epoch re-does the same
decompression (~7 min/epoch).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cell_tracking.cache import build_volume_cache, cache_path
from cell_tracking.config import get_cache_dir, get_train_dir


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None, help="Directory of .zarr volumes (default: train dir).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N volumes.")
    parser.add_argument("--volume", action="append", dest="volumes", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dtype", choices=("uint8", "float16"), default="uint8", help="Storage dtype (see cache.py).")
    args = parser.parse_args()

    src_dir = args.dir or get_train_dir()
    cache_dir = args.cache_dir or get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)

    names = args.volumes or sorted(p.stem for p in Path(src_dir).glob("*.zarr"))
    if args.limit:
        names = names[: args.limit]
    if not names:
        raise SystemExit(f"No .zarr volumes found under {src_dir}")

    print(f"Caching {len(names)} volume(s) from {src_dir} -> {cache_dir}")
    started = time.time()
    total_bytes = 0
    for i, name in enumerate(names, 1):
        out = cache_path(cache_dir, name)
        t0 = time.time()
        build_volume_cache(Path(src_dir) / f"{name}.zarr", out, overwrite=args.overwrite, dtype=args.dtype)
        total_bytes += out.stat().st_size
        print(f"[{i}/{len(names)}] {name}  {time.time() - t0:5.1f}s  {out.stat().st_size / 1e6:6.1f} MB")

    elapsed = time.time() - started
    print(f"\nDone in {elapsed / 60:.1f} min, {total_bytes / 1e9:.2f} GB total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
