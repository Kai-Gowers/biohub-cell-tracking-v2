"""Learned edge scorer: is (src, dst) the same lineage across one frame step?

Trained via detect-and-match (`cell_tracking.edge_train`): candidate pairs and
their positive/negative labels come from the detector's OWN live detections
matched to ground truth, never a synthetic GT-node + nearest-peak-decoy set.
See reports/2026-08-25-sample-solution-0.90-comparison.md item 2 -- this is
the design choice that most directly targets crowded-region rank-1 edge
accuracy, which the v1 sibling repo measured collapsing from 0.98 to 0.58
under local crowding when trained on a synthetic candidate set instead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from cell_tracking.config import EDGE_FEATURE_DIM, EDGE_HIDDEN_DIM


def sample_node_features(feature_map: torch.Tensor, grid_coords: torch.Tensor) -> torch.Tensor:
    """Trilinearly sample `feature_map` (C, Z, Y, X) at continuous grid coords (N, 3) zyx.

    `grid_coords` must be in the same (unrounded) grid-index space as the
    model input -- e.g. `config.voxels_to_grid(detection.zyx)`, or the
    integer peak index plus `peaks.subvoxel_offset` -- not raw voxels or µm.
    """
    if grid_coords.shape[0] == 0:
        return grid_coords.new_zeros((0, feature_map.shape[0]))
    c, z, y, x = feature_map.shape
    size = grid_coords.new_tensor([z - 1, y - 1, x - 1]).clamp_min(1.0)
    norm = grid_coords / size * 2.0 - 1.0
    # grid_sample expects grid[..., :] in (x, y, z) order for a 5D input.
    grid = norm.flip(-1).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        feature_map.unsqueeze(0), grid, mode="bilinear", align_corners=True, padding_mode="border"
    )
    return sampled.view(c, -1).t()


class EdgeScorer(nn.Module):
    """Scores one candidate (src, dst) link from sampled node features + relative position."""

    def __init__(self, feature_dim: int = EDGE_FEATURE_DIM, hidden_dim: int = EDGE_HIDDEN_DIM) -> None:
        super().__init__()
        in_dim = feature_dim * 2 + 4  # src feat, dst feat, rel zyx (um), distance (um)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, feat_src: torch.Tensor, feat_dst: torch.Tensor, rel_um: torch.Tensor) -> torch.Tensor:
        """Returns per-pair LOGITS, shape (N,)."""
        dist = rel_um.norm(dim=-1, keepdim=True)
        h = torch.cat([feat_src, feat_dst, rel_um, dist], dim=-1)
        return self.mlp(h).squeeze(-1)
