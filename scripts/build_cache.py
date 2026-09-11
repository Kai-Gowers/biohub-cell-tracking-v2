#!/usr/bin/env python3
"""Precompute the decimated-frame cache the pack pipeline trains from.

    python scripts/build_cache.py                 # every train volume
    python scripts/build_cache.py --limit 3        # smoke subset
    python scripts/build_cache.py --dir <test_dir> # test volumes

One uint16 .npy per volume (raw values decimated (1,4,4); normalisation is
applied on read). ~10 GB for all 199 train volumes; put CACHE_DIR on /projects.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cell_tracking.cache import build_volume_cache, cache_path
from cell_tracking.config import get_cache_dir, get_train_dir
from cell_tracking.io_zarr import list_zarr_datasets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None, help="Directory of .zarr volumes (default: train dir).")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--volume", action="append", dest="volumes", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    data_dir = args.dir or get_train_dir()
    cache_dir = args.cache_dir or get_cache_dir()
    names = args.volumes or list_zarr_datasets(data_dir)
    if args.limit:
        names = names[: args.limit]
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"caching {len(names)} volume(s) from {data_dir} -> {cache_dir}")
    started = time.time()
    for i, name in enumerate(names, 1):
        out = build_volume_cache(data_dir / f"{name}.zarr", cache_path(cache_dir, name), overwrite=args.overwrite)
        print(f"[{i}/{len(names)}] {name} -> {out.name} ({(time.time() - started) / 60:.1f} min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
