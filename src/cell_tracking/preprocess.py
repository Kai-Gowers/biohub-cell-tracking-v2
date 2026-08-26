"""Raw volume -> model-grid frames, plus the on-disk frame cache.

`prepare_frame` is the ONLY path from raw voxels to model input. Training and
inference must both go through it -- if normalization differs between them the
detector's probabilities shift and `TAU` silently stops meaning what it meant
during training.
"""

from __future__ import annotations

import numpy as np

from cell_tracking.config import DOWNSAMPLE

NORM_LO_PCT = 1.0
NORM_HI_PCT = 99.9


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Percentile-normalize one raw (Z,Y,X) volume to [0, 1].

    Per-frame percentiles rather than global ones: image dynamic range varies
    ~24x across volumes (p99.9/median of 24 on the best-detected volume vs 2-3
    on the worst), so a shared scale leaves the low-contrast volumes with
    almost no usable range.
    """
    v = vol.astype(np.float32)
    lo, hi = np.percentile(v, (NORM_LO_PCT, NORM_HI_PCT))
    return np.clip((v - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)


def downsample_volume(vol: np.ndarray, factors: tuple[int, int, int] = DOWNSAMPLE) -> np.ndarray:
    """Mean-pool by `factors` per axis.

    Mean-pool, not striding: striding aliases, throwing away 15/16 of the
    photons in xy and making single-voxel cell centres flicker between frames.
    """
    fz, fy, fx = factors
    z, y, x = vol.shape
    zc, yc, xc = (z // fz) * fz, (y // fy) * fy, (x // fx) * fx
    v = vol[:zc, :yc, :xc]
    return v.reshape(zc // fz, fz, yc // fy, fy, xc // fx, fx).mean(axis=(1, 3, 5))


def prepare_frame(vol: np.ndarray) -> np.ndarray:
    """Raw (Z,Y,X) uint16 volume -> float32 model grid in [0, 1]."""
    return downsample_volume(normalize_volume(vol)).astype(np.float32)


def to_uint8(frame: np.ndarray) -> np.ndarray:
    """Quantize a prepared frame for cache storage (32x smaller than raw)."""
    return np.clip(np.rint(frame * 255.0), 0, 255).astype(np.uint8)


def from_uint8(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0


def grid_shape(raw_shape_zyx: tuple[int, int, int]) -> tuple[int, int, int]:
    fz, fy, fx = DOWNSAMPLE
    z, y, x = raw_shape_zyx
    return (z // fz, y // fy, x // fx)
