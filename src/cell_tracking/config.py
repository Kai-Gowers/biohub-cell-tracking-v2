"""Paths and every algorithm hyperparameter.

v2 has grown past its original single-frame/no-edge-model baseline: a
2-frame windowed 3D U-Net with cross-frame attention, a learned edge scorer
trained via detect-and-match, augmentation + TTA, and greedy (not exact)
linking -- see reports/2026-08-25-sample-solution-0.90-comparison.md and
reports/2026-08-25-four-sample-solution-features.md for why. There is still
no repair pass and no divisions. Add a constant here, with a comment
explaining why, only once `reports/` has evidence that it helps.
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
# 2-frame temporal window with multi-head self-attention across the window at
# every encoder stage except full-res, reintroduced from the v2 baseline's
# single-frame default per reports/2026-08-25-sample-solution-0.90-comparison.md
# item 4 (the 0.90 sample solution attends at every encoder stage; the v1
# sibling repo only mixed frames at the bottleneck).
UNET_BASE_CHANNELS = 16
UNET_DEPTH = 3
WINDOW_SIZE = 2       # frames per model input; the LAST frame is the prediction target
ATTN_HEADS = 4        # must evenly divide every encoder stage's channel count

# Dropout in the coarser stages only (not stage 0/full-res, not the finest
# decoder stage that feeds out_proj and the edge scorer) -- added after a
# clean, low-noise training run (val_frames_per_volume=16, an actually
# annealing LR) still showed train loss improving smoothly to epoch 21 while
# val_loss peaked at epoch 6 and never recovered: a real train/val gap, not
# a measurement artifact. Full-res dropout risks the sub-voxel localization
# precision this task depends on, so it stays concentrated where the model
# is learning more abstract, more overfit-prone representations.
UNET_DROPOUT = 0.1
EDGE_DROPOUT = 0.1

# --- Edge model -------------------------------------------------------------
# A learned edge scorer (report item 1/2), trained via detect-and-match in
# train.py/edge_train.py: candidate pairs and their positive/negative labels
# come from the detector's OWN live peaks matched to ground truth, never a
# synthetic GT-node + nearest-peak-decoy set. Node features are sampled
# (trilinear) from the decoder's finest feature map, which has
# UNET_BASE_CHANNELS channels at full grid resolution.
EDGE_FEATURE_DIM = UNET_BASE_CHANNELS
EDGE_HIDDEN_DIM = 64
EDGE_MATCH_RADIUS_UM = 5.0   # GT-match radius for building detect-and-match labels
EDGE_LOSS_WEIGHT = 1.0
EDGE_NEG_ALPHA = 0.05        # candidate pairs are mostly non-edges; mirrors CENTER_NEG_ALPHA

# --- Augmentation -----------------------------------------------------------
# y/x flip + brightness jitter, shared across every frame in a training
# window (and the target's GT grid) so the sample stays geometrically and
# photometrically consistent. z is excluded from flipping for the same
# anisotropy reason TTA excludes it below.
AUGMENT_FLIP_PROB = 0.5
AUGMENT_BRIGHTNESS_GAIN = (0.85, 1.15)
AUGMENT_BRIGHTNESS_BIAS = (-0.05, 0.05)

# --- Test-time augmentation --------------------------------------------------
# Average detection LOGITS over identity + 3 flips. z is excluded: DOWNSAMPLE
# already makes z ~4x coarser than y/x, so a z-flip is not the same kind of
# invariance a y/x flip tests. Each tuple is the `dims` argument to
# `torch.flip` on a (..., Z, Y, X) tensor.
TTA_FLIPS: tuple[tuple[int, ...], ...] = ((), (-1,), (-2,), (-2, -1))

# --- Detection loss ---------------------------------------------------------
# w+(b) = 1/N+(b), w-(b) = alpha/N-(b), alpha = 0.01.
# Only ~2.8% of real cells are annotated, so the negative pool is mostly
# unlabelled true cells; alpha is what stops them dominating. Total positive
# weight is 1.0 vs 0.01 negative -- a deliberate 100:1 bias toward recall, and
# what makes TAU=0.985 select peaks rather than nothing (a soft target never
# saturates the sigmoid).
CENTER_NEG_ALPHA = 0.01

# --- Linking ---------------------------------------------------------------
# Candidates are still gated to this radius, but selection is now plain
# greedy score-sorted thresholding, not an exact bipartite assignment --
# report item 5: even the 0.90 sample solution uses greedy-by-default (its
# ILP solver ships but is OFF), so this is not where points are being left on
# the table. In/out-degree is still <= 1 by construction (no divisions).
LINK_RADIUS_UM = 15.0
# Score threshold when a learned edge model is available (its sigmoid output
# is a probability); with no edge model, link.py falls back to negative
# distance and this threshold is unused.
LINK_SCORE_THRESHOLD = 0.5

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
