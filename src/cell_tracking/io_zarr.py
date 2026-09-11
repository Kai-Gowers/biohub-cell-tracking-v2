"""Read blosc2-compressed zarr volumes from competition layout."""

from __future__ import annotations

import json
import os
from pathlib import Path

import blosc2
import numpy as np


def list_zarr_datasets(test_dir: Path | str) -> list[str]:
    test_dir = Path(test_dir)
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: {test_dir}. "
            "Set TEST_DIR / COMP_DATA_DIR, place data under data/, "
            "or run on Kaggle with competition data attached."
        )
    return sorted(
        d.replace(".zarr", "")
        for d in os.listdir(test_dir)
        if d.endswith(".zarr")
    )


def read_attrs(zarr_path: Path | str) -> dict:
    """Root ``attributes`` of an OME-zarr v3 volume (multiscales, image_statistics)."""
    zarr_path = Path(zarr_path)
    meta_path = zarr_path / "zarr.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f).get("attributes", {})
    zattrs = zarr_path / ".zattrs"
    if zattrs.exists():
        with open(zattrs) as f:
            return json.load(f)
    return {}


def read_scale(zarr_path: Path | str) -> tuple[float, float, float]:
    """(z, y, x) voxel scale in µm from the multiscales transform, else the default."""
    attrs = read_attrs(zarr_path)
    try:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform["type"] == "scale":
            return tuple(float(v) for v in transform["scale"][-3:])
    except (KeyError, IndexError, TypeError):
        pass
    return (1.625, 0.40625, 0.40625)


def read_array_meta(zarr_path: Path | str) -> tuple[tuple[int, ...], np.dtype]:
    zarr_path = Path(zarr_path)
    with open(zarr_path / "0" / "zarr.json") as f:
        arr_meta = json.load(f)
    shape = tuple(arr_meta["shape"])  # (T, Z, Y, X)
    dtype = np.dtype(arr_meta["data_type"])
    return shape, dtype


def read_volume(
    zarr_path: Path | str,
    t: int,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> np.ndarray:
    """Load one timepoint volume as (Z, Y, X)."""
    zarr_path = Path(zarr_path)
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    with open(chunk_path, "rb") as f:
        compressed = f.read()
    decompressed = blosc2.decompress(compressed)
    return np.frombuffer(decompressed, dtype=dtype).reshape(shape[1:])
