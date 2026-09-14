"""On-disk cache of decimated raw frames for the pack pipeline.

The pack reads each frame from zarr and decimates it (``raw[::1, ::4, ::4]``)
at load time. Decompressing a 64x256x256 uint16 chunk costs ~10 ms and the
training loop touches ~40k frames per epoch, so the decimated frames are
cached once as one memory-mapped uint16 ``.npy`` per volume
(``(T, 64, 64, 64)``, 0.5 MB per frame, ~10 GB for all 199 volumes).

The cache stores the *raw* decimated values, not normalised ones: the pack's
per-video quantile normalisation (``preprocess.pack_normalize``) is applied on
read with the quantiles from the zarr attrs, so the cache is exactly equivalent
to reading the zarr (``VolumeFrames`` falls back to that when no cache exists).
Layout: ``<cache_dir>/<name>.npy``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cell_tracking.io_zarr import read_array_meta, read_attrs, read_volume
from cell_tracking.preprocess import decimate, pack_normalize, video_quantiles


def cache_path(cache_dir: Path | str, name: str) -> Path:
    return Path(cache_dir) / f"{name}.npy"


def build_volume_cache(
    zarr_path: Path | str,
    out_path: Path | str,
    *,
    overwrite: bool = False,
) -> Path:
    """Decimate every frame of one volume and write it as one uint16 array."""
    zarr_path, out_path = Path(zarr_path), Path(out_path)
    if out_path.exists() and not overwrite:
        return out_path
    shape, dtype_raw = read_array_meta(zarr_path)
    n_t = int(shape[0])
    first = decimate(read_volume(zarr_path, 0, shape, dtype_raw))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".npy.tmp")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=dtype_raw, shape=(n_t, *first.shape))
    arr[0] = first
    for t in range(1, n_t):
        arr[t] = decimate(read_volume(zarr_path, t, shape, dtype_raw))
    arr.flush()
    del arr
    tmp.replace(out_path)
    return out_path


class VolumeFrames:
    """Normalised model-grid frames for one volume, from cache when available."""

    def __init__(self, zarr_path: Path | str, cache_file: Path | str | None = None) -> None:
        self.zarr_path = Path(zarr_path)
        self.name = self.zarr_path.stem
        self.raw_shape, self._dtype = read_array_meta(self.zarr_path)
        self.n_t = int(self.raw_shape[0])
        self.q_low, self.q_high = video_quantiles(read_attrs(self.zarr_path))
        self._cached: np.ndarray | None = None
        self._cache_file = Path(cache_file) if cache_file is not None and Path(cache_file).exists() else None
        if self._cache_file is not None:
            self._cached = np.load(self._cache_file, mmap_mode="r")
            if int(self._cached.shape[0]) != self.n_t:
                raise ValueError(f"{cache_file}: {self._cached.shape[0]} frames, zarr has {self.n_t}")
            self.shape = tuple(int(s) for s in self._cached.shape[1:])
        else:
            self.shape = decimate(read_volume(self.zarr_path, 0, self.raw_shape, self._dtype)).shape

    def __getstate__(self) -> dict:
        # never pickle the memory map itself (it would serialise the whole volume); reopen it instead
        state = dict(self.__dict__)
        state["_cached"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        if self._cache_file is not None:
            self._cached = np.load(self._cache_file, mmap_mode="r")

    @property
    def cached(self) -> bool:
        return self._cached is not None

    def raw_decimated(self, t: int) -> np.ndarray:
        if self._cached is not None:
            return np.asarray(self._cached[t])
        return decimate(read_volume(self.zarr_path, t, self.raw_shape, self._dtype))

    def frame(self, t: int) -> np.ndarray:
        """Normalised float32 model-grid frame at timepoint ``t``."""
        return pack_normalize(self.raw_decimated(t), self.q_low, self.q_high)

    def window(self, t0: int, size: int) -> np.ndarray:
        """``size`` consecutive frames starting at ``t0`` (no clamping; callers keep windows in range)."""
        return np.stack([self.frame(t0 + i) for i in range(size)], axis=0)
