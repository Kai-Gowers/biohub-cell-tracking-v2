"""Paths, geometry, and the split seed.

Algorithm hyperparameters live with the stage they belong to, as dataclasses
whose defaults are the 0.942 notebook's committed values:
``pack_predict.PredictConfig`` (detection / edge scoring / TTA / dual seed),
``ilp.ILPConfig`` (graph selection), ``postprocess.PostprocessConfig``
(graph repair), and ``pack_train.TrainConfig`` (the pack's training recipe).
This module keeps only what every stage shares.
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

# --- Geometry -------------------------------------------------------------
# d_um = sqrt((1.625 dz)^2 + (0.40625 dy)^2 + (0.40625 dx)^2)
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)  # µm per voxel, (z,y,x)

# Downsampling xy only gives an ISOTROPIC 1.625 µm grid: a uniform 4x
# downsample would make z voxels 6.5 µm, eating ~49% of the metric's 7 µm
# match tolerance before the model does anything. The pack decimates
# (`raw[::1, ::4, ::4]`) rather than mean-pooling; see preprocess.py.
DOWNSAMPLE = (1, 4, 4)  # (z, y, x) factors applied to the raw volume

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
