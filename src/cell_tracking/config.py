"""Paths and every algorithm hyperparameter.

This is the v2 baseline: a single-frame 3D U-Net detector followed by
distance-gated bipartite linking (scipy `linear_sum_assignment`, no learned
edge model, no repair passes). Deliberately narrow -- see README.md for the
philosophy. Add a constant here, with a comment explaining why, only once
`reports/` has evidence that it helps.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

COMPETITION_SLUG = "biohub-cell-tracking-during-development"
KAGGLE_TEST_DIR = f"/kaggle/input/competitions/{COMPETITION_SLUG}/test"
KAGGLE_TRAIN_DIR = f"/kaggle/input/competitions/{COMPETITION_SLUG}/train"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCAL_DATA = REPO_ROOT / "data" / COMPETITION_SLUG
DEFAULT_LOCAL_TRAIN_DIR = DEFAULT_LOCAL_DATA / "train"
DEFAULT_LOCAL_TEST_DIR = DEFAULT_LOCAL_DATA / "test"
DEFAULT_MODEL_DIR = REPO_ROOT / "dist" / "models"
DEFAULT_CACHE_DIR = REPO_ROOT / "dist" / "cache"

CHECKPOINT_NAME = "detector.pt"          # last epoch, so a run stays resumable
BEST_CHECKPOINT_NAME = "detector_best.pt"  # best validation epoch -- what inference wants

# --- Geometry -------------------------------------------------------------
# d_um = sqrt((1.625 dz)^2 + (0.40625 dy)^2 + (0.40625 dx)^2)
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # µm per voxel, (z,y,x)

# Downsampling xy only gives an ISOTROPIC 1.625 µm grid: a uniform 4x
# downsample would make z voxels 6.5 µm, eating ~49% of the metric's 7 µm
# match tolerance before the model does anything.
DOWNSAMPLE = (1, 4, 4)  # (z, y, x) factors applied to the raw volume
GRID_SPACING = SCALE * np.array(DOWNSAMPLE, dtype=np.float64)  # (1.625,)*3 µm

# --- Detection ------------------------------------------------------------
TAU = 0.985  # detections are local maxima above this probability
# NMS radius in grid voxels, applied as a max-pool neighbourhood. A binary
# training target saturates the sigmoid, so a cell is a REGION at exactly
# 1.0 -- this also has to break ties, see peaks.py.
PEAK_NMS_RADIUS_VX = 2
MAX_DETECTIONS_PER_FRAME = 1500

# --- Model ----------------------------------------------------------------
# Single frame in, single frame out -- no temporal window. Add one back only
# if a held-out score shows single-frame detection is the bottleneck.
UNET_BASE_CHANNELS = 16
UNET_DEPTH = 3

# --- Detection loss ---------------------------------------------------------
# w+(b) = 1/N+(b), w-(b) = alpha/N-(b), alpha = 0.01.
# Only ~2.8% of real cells are annotated, so the negative pool is mostly
# unlabelled true cells; alpha is what stops them dominating. Total positive
# weight is 1.0 vs 0.01 negative -- a deliberate 100:1 bias toward recall, and
# what makes TAU=0.985 select peaks rather than nothing (a soft target never
# saturates the sigmoid).
CENTER_NEG_ALPHA = 0.01

# --- Linking ---------------------------------------------------------------
# The entire "edge model" in this baseline: gate candidate links to this
# radius, then solve an exact minimum-distance bipartite assignment (scipy) with
# in/out-degree <= 1 -- no division support, no learned score, no repair pass.
# A link is preferred over leaving both nodes unmatched only if it is closer
# than this radius, which doubles as the birth/death cost.
LINK_RADIUS_UM = 15.0

# --- Training -------------------------------------------------------------
VAL_FRAC = 0.1
VAL_SEED = 0


def on_kaggle() -> bool:
    return Path("/kaggle/input").exists()


def get_test_dir() -> Path:
    """Resolve test data directory for Kaggle or local runs."""
    if test_dir := os.environ.get("TEST_DIR"):
        return Path(test_dir)
    if comp_dir := os.environ.get("COMP_DATA_DIR"):
        return Path(comp_dir) / "test"
    if on_kaggle():
        return Path(KAGGLE_TEST_DIR)
    return DEFAULT_LOCAL_TEST_DIR


def get_train_dir() -> Path:
    if train_dir := os.environ.get("TRAIN_DIR"):
        return Path(train_dir)
    if comp_dir := os.environ.get("COMP_DATA_DIR"):
        return Path(comp_dir) / "train"
    if on_kaggle():
        return Path(KAGGLE_TRAIN_DIR)
    return DEFAULT_LOCAL_TRAIN_DIR


def get_cache_dir() -> Path:
    if cache_dir := os.environ.get("CACHE_DIR"):
        return Path(cache_dir)
    if on_kaggle():
        return Path("/kaggle/working/cache")
    return DEFAULT_CACHE_DIR


def _find_checkpoint_dir(root: Path, max_depth: int = 5) -> Path | None:
    """Breadth-first search for a directory holding `CHECKPOINT_NAME`.

    Kaggle does not always mount a Dataset at `/kaggle/input/<slug>`, and a
    one-level scan misses the alternative layout. Bounded and pruned rather
    than a plain `rglob`: the competition data is ~90GB of `.zarr`/`.geff`
    chunk trees with millions of tiny files, and walking into those costs
    minutes.
    """
    frontier = [root]
    for _ in range(max_depth):
        nxt: list[Path] = []
        for d in frontier:
            if (d / CHECKPOINT_NAME).exists() or (d / BEST_CHECKPOINT_NAME).exists():
                return d
            try:
                children = sorted(p for p in d.iterdir() if p.is_dir())
            except (PermissionError, OSError):
                continue
            nxt.extend(c for c in children if c.suffix not in (".zarr", ".geff"))
        if not nxt:
            break
        frontier = nxt
    return None


def get_model_dir() -> Path:
    if model_dir := os.environ.get("MODEL_DIR"):
        return Path(model_dir)
    if on_kaggle():
        input_root = Path("/kaggle/input")
        if found := _find_checkpoint_dir(input_root):
            return found
        return input_root
    return DEFAULT_MODEL_DIR


def get_checkpoint() -> Path | None:
    """Locate the weights to run inference on, or None -- callers treat None as fatal.

    Prefers `detector_best.pt` (best validation epoch) over `detector.pt`
    (last epoch, kept so a run stays resumable). `DETECTOR_CHECKPOINT`
    overrides both and is taken literally.

    There is deliberately no untrained fallback: a missing checkpoint should
    raise, not silently run classical blob detection.
    """
    if path := os.environ.get("DETECTOR_CHECKPOINT"):
        p = Path(path)
        return p if p.exists() else None
    model_dir = get_model_dir()
    for parent in (model_dir, model_dir / "models"):
        for name in (BEST_CHECKPOINT_NAME, CHECKPOINT_NAME):
            candidate = parent / name
            if candidate.exists():
                return candidate
    return None


def voxels_to_um(coords: np.ndarray, scale: np.ndarray = SCALE) -> np.ndarray:
    """Convert (..., 3) voxel zyx coords to µm."""
    return np.asarray(coords, dtype=np.float64) * scale


def um_to_voxels(coords_um: np.ndarray, scale: np.ndarray = SCALE) -> np.ndarray:
    return np.asarray(coords_um, dtype=np.float64) / scale


def grid_to_voxels(coords_grid: np.ndarray) -> np.ndarray:
    """Map model-grid (z,y,x) indices back to raw-volume voxel coordinates.

    A grid cell covers `DOWNSAMPLE` raw voxels, so its centre sits at
    `d * i + (d - 1) / 2`.
    """
    d = np.asarray(DOWNSAMPLE, dtype=np.float64)
    return np.asarray(coords_grid, dtype=np.float64) * d + (d - 1.0) / 2.0


def voxels_to_grid(coords_vx: np.ndarray) -> np.ndarray:
    """Inverse of `grid_to_voxels`, unrounded."""
    d = np.asarray(DOWNSAMPLE, dtype=np.float64)
    return (np.asarray(coords_vx, dtype=np.float64) - (d - 1.0) / 2.0) / d
