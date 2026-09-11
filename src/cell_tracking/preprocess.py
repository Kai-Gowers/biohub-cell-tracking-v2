"""Raw voxels -> model input, the pack's way.

The pilkwang pack (``predict_unet_transformer.py`` / ``train_unet_transformer.py``)
normalises per *video*, not per frame: it reads the ``image_statistics.quantiles``
that the competition zarr attrs carry (keys ``"0.001"`` and ``"0.999"``) and
applies ``((raw - q_low) / (q_high - q_low + 1e-6)).clamp(min=0)``. Downsampling
is strided decimation ``raw[::1, ::4, ::4]`` (not mean-pooling), giving the same
isotropic 1.625 µm grid as before. Training and inference must both go through
these two functions -- if normalisation differs the detector's probabilities
shift and the 0.965 threshold stops meaning what it meant during training.
"""

from __future__ import annotations

import numpy as np

from cell_tracking.config import DOWNSAMPLE

QUANTILE_LOW_KEY = "0.001"
QUANTILE_HIGH_KEY = "0.999"


def video_quantiles(attrs: dict) -> tuple[float, float]:
    """``(q_low, q_high)`` from a zarr root ``attributes`` dict; raises if missing."""
    q = attrs.get("image_statistics", {}).get("quantiles", {})
    if QUANTILE_LOW_KEY not in q or QUANTILE_HIGH_KEY not in q:
        raise ValueError(
            "zarr attrs missing image_statistics.quantiles "
            f"{QUANTILE_LOW_KEY!r}/{QUANTILE_HIGH_KEY!r}; the pack pipeline needs them"
        )
    return float(q[QUANTILE_LOW_KEY]), float(q[QUANTILE_HIGH_KEY])


def decimate(vol: np.ndarray, factors: tuple[int, int, int] = DOWNSAMPLE) -> np.ndarray:
    """Strided (z, y, x) decimation of a raw (Z, Y, X) frame; no averaging."""
    dz, dy, dx = factors
    return vol[::dz, ::dy, ::dx]


def pack_normalize(vol: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Per-video quantile normalisation, clamped at zero (no upper clip)."""
    out = (vol.astype(np.float32) - np.float32(q_low)) / np.float32(q_high - q_low + 1e-6)
    return np.maximum(out, 0.0, out=out)


def prepare_frame(vol: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    """Raw (Z, Y, X) frame -> normalised float32 model-grid frame."""
    return pack_normalize(decimate(vol), q_low, q_high)


def grid_shape(raw_shape_zyx: tuple[int, ...]) -> tuple[int, ...]:
    """Model-grid shape for a raw (Z, Y, X) shape under strided decimation (ceil)."""
    return tuple(-(-s // d) for s, d in zip(raw_shape_zyx, DOWNSAMPLE))
