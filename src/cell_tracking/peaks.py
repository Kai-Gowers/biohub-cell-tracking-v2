"""Local-maximum extraction and candidate-pair gating.

Shared by training (the detection target's inverse -- finding what the model
currently predicts) and inference, so both paths agree on what a "detection"
is.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from cell_tracking.config import PEAK_NMS_RADIUS_VX

# Below this curvature a 3-point neighbourhood is not a strict maximum along
# the axis, so the parabola through it has no usable vertex.
_SUBVOXEL_MIN_CURVATURE = 1e-6


def local_maxima(
    prob: torch.Tensor,
    *,
    threshold: float,
    radius: int = PEAK_NMS_RADIUS_VX,
    max_peaks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-max-suppressed peaks of a (Z, Y, X) probability volume.

    Returns (idx, score), idx being (N, 3) int64 grid coordinates sorted by
    descending score.

    Suppression is two max-pools, both fully vectorized. An explicit greedy
    NMS loop is the obvious alternative and is dramatically slower per step.

    1. `prob >= maxpool(prob)` finds neighbourhood maxima.
    2. A second max-pool over a position index breaks *ties*. This matters
       because the binary training target saturates the sigmoid, so a
       detected cell is a region of voxels at exactly 1.0 rather than a
       single peak. Every tied voxel passes step 1 -- a 3x3x3 saturated blob
       yields 27 detections instead of 1. Ranking ties by position keeps one
       per neighbourhood.

    The index must be pooled separately rather than folded into `prob`:
    adding a small ramp does NOT work, because float32 has only ~1.2e-7 of
    resolution at 1.0 and any ramp fine enough to be safe rounds away
    entirely. Indices below 2^24 are exact in float32.
    """
    if prob.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X), got {tuple(prob.shape)}")

    k = 2 * radius + 1
    pooled = F.max_pool3d(prob[None, None], kernel_size=k, stride=1, padding=radius)[0, 0]
    is_max = (prob >= pooled) & (prob > threshold)
    if not bool(is_max.any()):
        return torch.zeros((0, 3), dtype=torch.long, device=prob.device), prob.new_zeros(0)

    ramp = torch.arange(prob.numel(), device=prob.device, dtype=torch.float32).reshape(prob.shape)
    keyed = torch.where(is_max, ramp, torch.full_like(ramp, -1.0))
    pooled_idx = F.max_pool3d(keyed[None, None], kernel_size=k, stride=1, padding=radius)[0, 0]
    keep = is_max & (keyed >= pooled_idx)

    idx = torch.nonzero(keep, as_tuple=False)
    if idx.numel() == 0:
        return idx.reshape(0, 3), prob.new_zeros(0)

    scores = prob[idx[:, 0], idx[:, 1], idx[:, 2]]
    order = torch.argsort(scores, descending=True)
    idx, scores = idx[order], scores[order]
    if max_peaks is not None and idx.shape[0] > max_peaks:
        idx, scores = idx[:max_peaks], scores[:max_peaks]
    return idx, scores


def subvoxel_offset(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Continuous position of each peak within its own voxel, in grid units.

    Returns (N, 3) float in [-0.5, 0.5], to be added to the integer peak
    index before converting to voxels or µm.

    Three-point parabolic interpolation per axis. For logits `Lm, L0, Lp` at
    `i-1, i, i+1` the parabola through them has its vertex at

        delta = (Lm - Lp) / (2 * (Lm - 2 L0 + Lp))

    **`logits`, not probabilities.** The detection target is binary and
    `CENTER_NEG_ALPHA` biases 100:1 toward recall, so the sigmoid saturates:
    peak probabilities sit in the last four decimals where float32 has
    ~1.2e-7 of resolution, and a fit there is dominated by rounding. The
    logit is unbounded and keeps full resolution at exactly the peaks that
    matter.

    Returns 0 for an axis when the neighbourhood is not a strict maximum
    along it, or when the peak sits on a volume boundary with no neighbour on
    one side -- a deliberate no-op rather than an extrapolation.
    """
    n = idx.shape[0]
    out = torch.zeros((n, 3), dtype=torch.float32, device=logits.device)
    if n == 0:
        return out
    if logits.ndim != 3:
        raise ValueError(f"Expected (Z, Y, X) logits, got {tuple(logits.shape)}")

    lg = logits.float()
    centre = lg[idx[:, 0], idx[:, 1], idx[:, 2]]
    for axis in range(3):
        i = idx[:, axis]
        limit = lg.shape[axis] - 1
        interior = (i > 0) & (i < limit)

        lo = idx.clone()
        lo[:, axis] = (i - 1).clamp(0, limit)
        hi = idx.clone()
        hi[:, axis] = (i + 1).clamp(0, limit)
        left = lg[lo[:, 0], lo[:, 1], lo[:, 2]]
        right = lg[hi[:, 0], hi[:, 1], hi[:, 2]]

        denom = left - 2.0 * centre + right
        usable = interior & (denom < -_SUBVOXEL_MIN_CURVATURE)
        # Guard the division itself: `torch.where` still evaluates both branches.
        safe = torch.where(usable, denom, torch.ones_like(denom))
        delta = torch.where(usable, 0.5 * (left - right) / safe, torch.zeros_like(denom))
        out[:, axis] = delta.clamp(-0.5, 0.5)
    return out


def pair_within_radius(
    pos_src_um: np.ndarray,
    pos_dst_um: np.ndarray,
    radius_um: float,
) -> np.ndarray:
    """Candidate (src, dst) index pairs closer than `radius_um`."""
    if len(pos_src_um) == 0 or len(pos_dst_um) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    d = np.linalg.norm(pos_src_um[:, None, :] - pos_dst_um[None, :, :], axis=-1)
    si, di = np.nonzero(d <= radius_um)
    if len(si) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    return np.stack([si, di], axis=1).astype(np.int64)


def match_to_reference(pos_um: np.ndarray, ref_um: np.ndarray, radius_um: float) -> np.ndarray:
    """Optimal one-to-one bipartite match of each `pos_um` row to `ref_um`, within radius.

    Same shape of matching as `metric.match_nodes_per_frame` (exact
    assignment via a distance-or-huge cost matrix), but reusable with an
    arbitrary radius -- `metric.py` is copied verbatim from the sibling repo
    and is the competition's metric, not a pipeline choice, so this stays a
    separate helper rather than parameterizing that one.

    Returns an (N,) int64 array of row indices into `ref_um`, or -1 where a
    `pos_um` row has no match within `radius_um`.
    """
    n, m = len(pos_um), len(ref_um)
    out = np.full(n, -1, dtype=np.int64)
    if n == 0 or m == 0:
        return out
    dist = np.linalg.norm(pos_um[:, None, :] - ref_um[None, :, :], axis=-1)
    cost = np.where(dist <= radius_um, dist, 1e6)
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        if cost[r, c] <= radius_um:
            out[r] = c
    return out
