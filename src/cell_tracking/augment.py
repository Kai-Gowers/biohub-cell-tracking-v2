"""Training-time augmentation: random y/x flip + brightness jitter.

z is excluded from flipping, matching the TTA rationale in `config.py`:
`DOWNSAMPLE` already makes z ~4x coarser than y/x, so a z-flip does not test
the same invariance a y/x flip does. One flip/brightness draw is shared
across every frame in a training window (and the target frame's GT grid), so
the sample stays geometrically and photometrically consistent rather than
flickering between frames.

Added per reports/2026-08-25-sample-solution-0.90-comparison.md item 3 --
neither this repo nor the v1 sibling repo augmented at all; the 0.90 sample
solution used random brightness + random flip every training step.
"""

from __future__ import annotations

import random

import numpy as np

from cell_tracking.config import AUGMENT_BRIGHTNESS_BIAS, AUGMENT_BRIGHTNESS_GAIN, AUGMENT_FLIP_PROB


def augment_window(
    window: np.ndarray, node_grid: np.ndarray, rng: random.Random
) -> tuple[np.ndarray, np.ndarray]:
    """`window`: (T, Z, Y, X) float32 in [0, 1]. `node_grid`: (N, 3) int zyx grid
    coords of the target (last) frame. Returns the augmented pair."""
    _, z, y, x = window.shape
    grid = node_grid.copy()

    if rng.random() < AUGMENT_FLIP_PROB:
        window = window[:, :, ::-1, :]
        if len(grid):
            grid[:, 1] = y - 1 - grid[:, 1]
    if rng.random() < AUGMENT_FLIP_PROB:
        window = window[:, :, :, ::-1]
        if len(grid):
            grid[:, 2] = x - 1 - grid[:, 2]

    gain = rng.uniform(*AUGMENT_BRIGHTNESS_GAIN)
    bias = rng.uniform(*AUGMENT_BRIGHTNESS_BIAS)
    window = np.clip(window * gain + bias, 0.0, 1.0).astype(np.float32)
    return np.ascontiguousarray(window), grid
