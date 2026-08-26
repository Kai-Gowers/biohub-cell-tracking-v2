"""Inference-time detection: run the model on one frame, threshold, extract peaks.

There is deliberately no classical-blob fallback. A missing checkpoint is
fatal, and zero detections get reported rather than papered over.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cell_tracking.config import (
    DOWNSAMPLE,
    MAX_DETECTIONS_PER_FRAME,
    SCALE,
    TAU,
    get_checkpoint,
    grid_to_voxels,
)
from cell_tracking.models.detector import UNet3D
from cell_tracking.peaks import local_maxima, subvoxel_offset


@dataclass
class FrameDetections:
    """Detections for one timepoint."""

    t: int
    grid: np.ndarray  # (N, 3) int, model-grid coordinates -- the INDEX
    zyx: np.ndarray  # (N, 3) float, raw-volume voxel coordinates, sub-voxel refined
    score: np.ndarray  # (N,)
    max_prob: float  # frame-wide max probability, for TAU calibration checks

    @property
    def um(self) -> np.ndarray:
        return self.zyx * SCALE

    def __len__(self) -> int:
        return len(self.grid)


def load_model(
    checkpoint: Path | str | None = None, device: torch.device | None = None
) -> tuple[UNet3D, torch.device]:
    path = Path(checkpoint) if checkpoint else get_checkpoint()
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "No detector.pt found. Set DETECTOR_CHECKPOINT or MODEL_DIR, or train one "
            "with scripts/train.py. This pipeline has no untrained fallback."
        )
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    print(
        f"loaded {Path(path).name} (epoch {ckpt.get('epoch')})"
        f"{' [best: ' + str(ckpt['selected_by']) + ']' if 'selected_by' in ckpt else ' [LAST epoch, not best]'}"
    )
    model = UNet3D(**ckpt.get("config", {})).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, device


def extract_detections(
    prob: torch.Tensor,
    t: int,
    *,
    logits: torch.Tensor | None = None,
    tau: float = TAU,
    max_per_frame: int = MAX_DETECTIONS_PER_FRAME,
    subvoxel: bool = True,
) -> FrameDetections:
    """Local maxima of one frame's probability volume above `tau`.

    With `subvoxel` (the default) the returned `zyx` carries the *continuous*
    peak position: `local_maxima` picks a voxel and `peaks.subvoxel_offset`
    says where inside it the field actually peaks. The model grid is 1.625 µm
    isotropic, so an integer readout throws away up to +-0.81 µm per axis
    before any downstream stage runs, against a 7 µm match tolerance.

    `grid` stays the integer INDEX regardless.

    `logits` is required when `subvoxel` is on: refining on the saturated
    sigmoid instead would silently return near-zero offsets rather than fail.
    """
    if subvoxel and logits is None:
        raise ValueError(
            "extract_detections(subvoxel=True) needs `logits`. The sigmoid is "
            "saturated at the peaks, so refining on `prob` silently yields "
            "nothing; pass the pre-sigmoid logits or set subvoxel=False."
        )

    idx, score = local_maxima(prob, threshold=tau, max_peaks=max_per_frame)
    grid = idx.detach().cpu().numpy().astype(np.int64)
    if not len(grid):
        empty = np.zeros((0, 3), dtype=np.float64)
        return FrameDetections(t=t, grid=grid, zyx=empty, score=np.zeros(0), max_prob=float(prob.max()))

    if subvoxel:
        offset = subvoxel_offset(logits, idx).detach().cpu().numpy().astype(np.float64)
    else:
        offset = np.zeros_like(grid, dtype=np.float64)

    zyx = grid_to_voxels(grid + offset)
    # A boundary peak can be nudged outside the volume; keep it in it.
    grid_extent = np.asarray(prob.shape, dtype=np.float64) * np.asarray(DOWNSAMPLE, dtype=np.float64)
    zyx = np.clip(zyx, 0.0, grid_extent - 1.0)

    return FrameDetections(
        t=t,
        grid=grid,
        zyx=zyx,
        score=score.detach().cpu().numpy().astype(np.float64),
        max_prob=float(prob.max()),
    )


def detection_report(dets: list[FrameDetections]) -> str:
    """One-line calibration summary.

    If TAU yields ~0 detections, suspect the training target before the
    threshold: a binary target saturates the sigmoid, a soft Gaussian one
    never does. `p_max/frame` is the number that distinguishes the two.
    """
    counts = np.array([len(d) for d in dets]) if dets else np.array([0])
    maxima = np.array([d.max_prob for d in dets]) if dets else np.array([0.0])
    return (
        f"frames={len(dets)}  detections/frame mean={counts.mean():.1f} "
        f"min={counts.min()} max={counts.max()}  "
        f"p_max/frame mean={maxima.mean():.4f} min={maxima.min():.4f} max={maxima.max():.4f}"
    )
