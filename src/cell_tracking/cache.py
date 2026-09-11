"""On-disk cache of prepared frames.

Going from raw zarr to a model-ready frame costs ~35 ms (10 ms blosc2
decompress + 25 ms normalize/downsample) and the result never changes. At
~12,000 frame-preps per epoch that is ~7 min/epoch of identical work, ~6 h
across a 50-epoch run. Precomputing all 19,900 frames takes ~12 min and 5.2 GB;
a cached read is ~1 ms.

Layout: `<cache_dir>/<name>.npy`, uint8, shape (T, Z', Y', X') on the model
grid. Read back memory-mapped so a 5.2 GB cache never has to fit in RAM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cell_tracking.io_zarr import read_array_meta, read_volume
from cell_tracking.preprocess import from_uint8, prepare_frame, to_uint8


def cache_path(cache_dir: Path | str, name: str) -> Path:
    return Path(cache_dir) / f"{name}.npy"


def build_volume_cache(
    zarr_path: Path | str,
    out_path: Path | str,
    *,
    overwrite: bool = False,
    dtype: str = "uint8",
) -> Path:
    """Prepare every frame of one volume and write it as one array.

    `dtype="uint8"` quantises the [0, 1] frame to 256 levels (the original
    5 GB cache, a Kaggle-era disk/RAM trade-off); `"float16"` keeps ~3
    significant digits at twice the size and removes the question of whether
    dim nuclei lose detail to quantisation. `VolumeFrames` reads either.
    """
    if dtype not in ("uint8", "float16"):
        raise ValueError(f"dtype must be 'uint8' or 'float16', got {dtype!r}")
    to_store = to_uint8 if dtype == "uint8" else (lambda f: f.astype(np.float16))
    zarr_path, out_path = Path(zarr_path), Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path

    shape, dtype_raw = read_array_meta(zarr_path)
    n_t = int(shape[0])
    first = prepare_frame(read_volume(zarr_path, 0, shape, dtype_raw))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npy.tmp")
    # Write through a memmap so peak RAM stays at one frame, and via a temp
    # file so an interrupted build can't leave a half-written cache that later
    # reads would silently accept.
    arr = np.lib.format.open_memmap(
        tmp, mode="w+", dtype=np.dtype(dtype), shape=(n_t, *first.shape)
    )
    arr[0] = to_store(first)
    for t in range(1, n_t):
        arr[t] = to_store(prepare_frame(read_volume(zarr_path, t, shape, dtype_raw)))
    arr.flush()
    del arr
    tmp.replace(out_path)
    return out_path


class VolumeFrames:
    """Prepared frames for one volume, from cache when available.

    Falls back to reading and preparing raw zarr on the fly, so nothing
    *requires* a cache -- it is purely a speed optimization.
    """

    def __init__(self, zarr_path: Path | str, cache_file: Path | str | None = None) -> None:
        self.zarr_path = Path(zarr_path)
        self._cached: np.ndarray | None = None
        if cache_file is not None and Path(cache_file).exists():
            self._cached = np.load(cache_file, mmap_mode="r")
            self.n_t = int(self._cached.shape[0])
            self.shape = tuple(int(s) for s in self._cached.shape[1:])
        else:
            raw_shape, self._dtype = read_array_meta(self.zarr_path)
            self._raw_shape = raw_shape
            self.n_t = int(raw_shape[0])
            self.shape = prepare_frame(
                read_volume(self.zarr_path, 0, raw_shape, self._dtype)
            ).shape

    @property
    def cached(self) -> bool:
        return self._cached is not None

    def frame(self, t: int) -> np.ndarray:
        """Prepared float32 frame at timepoint `t`."""
        if self._cached is not None:
            if self._cached.dtype == np.uint8:
                return from_uint8(np.asarray(self._cached[t]))
            return np.asarray(self._cached[t], dtype=np.float32)
        return prepare_frame(read_volume(self.zarr_path, t, self._raw_shape, self._dtype))

    def window(self, t0: int, size: int) -> np.ndarray:
        """`size` consecutive frames starting at `t0`, clamped at the edges."""
        idx = [min(max(t0 + i, 0), self.n_t - 1) for i in range(size)]
        return np.stack([self.frame(i) for i in idx], axis=0)
