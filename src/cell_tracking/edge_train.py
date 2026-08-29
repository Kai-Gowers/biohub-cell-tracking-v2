"""Edge-model training data: detect-and-match, not GT+decoys.

Every call runs real peak detection (identical NMS/threshold to inference) on
the detector's OWN live logits for two consecutive frames, matches each
detected peak to ground truth within `EDGE_MATCH_RADIUS_UM`, and derives
candidate-edge labels from those matched identities plus the volume's real GT
edge set. The edge scorer never sees a synthetic GT-node + nearest-peak-decoy
candidate set -- see
reports/2026-08-25-sample-solution-0.90-comparison.md item 2.

A consequence worth knowing before reading a training curve: early in
training the detector clears `TAU` on almost nothing, so `sample_edge_pair`
returns `None` far more often than not and the edge loss contributes little.
That is not a bug -- edge supervision is bootstrapped by detection quality
and only gets dense once the detector is doing its job, same as it would for
any detect-and-match design.
"""

from __future__ import annotations

import numpy as np
import torch

from cell_tracking.config import (
    EDGE_MATCH_RADIUS_UM,
    LINK_RADIUS_UM,
    MAX_DETECTIONS_PER_FRAME,
    TAU,
    WINDOW_SIZE,
    grid_to_voxels,
    voxels_to_um,
)
from cell_tracking.models.edge_model import sample_node_features
from cell_tracking.peaks import local_maxima, match_to_reference, pair_within_radius, subvoxel_offset


def _detect_frame(
    logits_2d: torch.Tensor, tau: float, max_per_frame: int
) -> tuple[torch.Tensor, np.ndarray]:
    """Local maxima of one frame's logits -> (continuous grid coords, raw voxel zyx)."""
    prob = torch.sigmoid(logits_2d)
    idx, _score = local_maxima(prob, threshold=tau, max_peaks=max_per_frame)
    if idx.shape[0] == 0:
        return idx.new_zeros((0, 3)).float(), np.zeros((0, 3), dtype=np.float64)
    offset = subvoxel_offset(logits_2d, idx)
    grid_f = idx.float() + offset
    zyx = grid_to_voxels(grid_f.detach().cpu().numpy().astype(np.float64))
    return grid_f, zyx


def sample_edge_pair(
    model,
    sampler,
    t: int,
    device: torch.device,
    *,
    tau: float = TAU,
    max_per_frame: int = MAX_DETECTIONS_PER_FRAME,
    link_radius_um: float = LINK_RADIUS_UM,
    match_radius_um: float = EDGE_MATCH_RADIUS_UM,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """One forward pass over frames (t, t+1) of `sampler`'s volume.

    Returns `(feat_src, feat_dst, rel_um, labels)` for every candidate pair
    within `link_radius_um`, or `None` when either frame detects nothing, or
    there are no candidate pairs at all (both are common early in training).
    """
    n_t = sampler.frames.n_t
    if n_t < 2:
        return None
    t = min(max(t, 0), n_t - 2)

    win_t = sampler.frames.window(t - (WINDOW_SIZE - 1), WINDOW_SIZE)
    win_t1 = sampler.frames.window(t + 1 - (WINDOW_SIZE - 1), WINDOW_SIZE)
    x = torch.from_numpy(np.stack([win_t, win_t1])).to(device)
    logits, feats = model(x, return_features=True)

    grid0, zyx0 = _detect_frame(logits[0, 0], tau, max_per_frame)
    grid1, zyx1 = _detect_frame(logits[1, 0], tau, max_per_frame)
    if grid0.shape[0] == 0 or grid1.shape[0] == 0:
        return None

    um0, um1 = voxels_to_um(zyx0), voxels_to_um(zyx1)
    pairs = pair_within_radius(um0, um1, link_radius_um)
    if len(pairs) == 0:
        return None

    gt_um0 = sampler.gt_um.get(t, np.zeros((0, 3)))
    gt_um1 = sampler.gt_um.get(t + 1, np.zeros((0, 3)))
    gt_ids0 = sampler.gt_ids.get(t, np.zeros((0,), dtype=np.int64))
    gt_ids1 = sampler.gt_ids.get(t + 1, np.zeros((0,), dtype=np.int64))

    match0 = match_to_reference(um0, gt_um0, match_radius_um)
    match1 = match_to_reference(um1, gt_um1, match_radius_um)
    # `np.where` evaluates both branches eagerly, so indexing `gt_idsN` at a
    # clipped-to-0 position still runs even when `gt_idsN` is empty and that
    # branch is never selected (match is always -1 when there is no GT at
    # all) -- guard the indexing itself, same rationale as the
    # torch.where-both-branches note in peaks.subvoxel_offset.
    matched_gt0 = (
        np.where(match0 >= 0, gt_ids0[np.clip(match0, 0, len(gt_ids0) - 1)], -1)
        if len(gt_ids0)
        else np.full(len(match0), -1, dtype=np.int64)
    )
    matched_gt1 = (
        np.where(match1 >= 0, gt_ids1[np.clip(match1, 0, len(gt_ids1) - 1)], -1)
        if len(gt_ids1)
        else np.full(len(match1), -1, dtype=np.int64)
    )

    labels = np.array(
        [
            1.0
            if (
                matched_gt0[i] >= 0
                and matched_gt1[j] >= 0
                and (int(matched_gt0[i]), int(matched_gt1[j])) in sampler.gt_edges
            )
            else 0.0
            for i, j in pairs
        ],
        dtype=np.float32,
    )

    feat0 = sample_node_features(feats[0], grid0)
    feat1 = sample_node_features(feats[1], grid1)
    src_idx = torch.from_numpy(pairs[:, 0]).to(device)
    dst_idx = torch.from_numpy(pairs[:, 1]).to(device)
    feat_src, feat_dst = feat0[src_idx], feat1[dst_idx]
    rel_um = torch.from_numpy(um1[pairs[:, 1]] - um0[pairs[:, 0]]).float().to(device)
    y = torch.from_numpy(labels).to(device)
    return feat_src, feat_dst, rel_um, y
