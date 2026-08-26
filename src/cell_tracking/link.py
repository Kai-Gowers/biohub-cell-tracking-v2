"""Link consecutive frames: distance-gated, exact bipartite assignment.

The entire "graph selection" stage of this baseline. No learned edge model,
no ILP costs, no divisions, no repair -- each target has at most one parent,
each source at most one child, and a link is only worth making if it is
closer than `LINK_RADIUS_UM`.

Solved exactly with scipy, not greedy nearest-neighbour: with in/out-degree
<= 1 this is a minimum-cost bipartite matching, and `linear_sum_assignment`
gets the exact optimum for the price of padding in dummy rows/columns that
encode "stay unmatched" at cost `LINK_RADIUS_UM` (a link beyond the gate is
never worth taking over leaving both nodes unmatched).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from cell_tracking.config import LINK_RADIUS_UM
from cell_tracking.peaks import pair_within_radius


def link_frames(
    pos_src_um: np.ndarray,
    pos_dst_um: np.ndarray,
    *,
    radius_um: float = LINK_RADIUS_UM,
) -> np.ndarray:
    """Return selected (src_idx, dst_idx) pairs between two consecutive frames."""
    n_src, n_dst = len(pos_src_um), len(pos_dst_um)
    if n_src == 0 or n_dst == 0:
        return np.zeros((0, 2), dtype=np.int64)

    pairs = pair_within_radius(pos_src_um, pos_dst_um, radius_um)
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    dist = np.linalg.norm(
        pos_src_um[pairs[:, 0]] - pos_dst_um[pairs[:, 1]], axis=-1
    )

    # Square (n_src+n_dst) x (n_dst+n_src) cost matrix in four blocks:
    #   [real src  , real dst ]  n_src x n_dst : dist if gated, else BIG (disallowed)
    #   [real src  , dummy dst]  n_src x n_src : radius_um -- "this source is unmatched"
    #   [dummy src , real dst ]  n_dst x n_dst : radius_um -- "this target is unmatched"
    #   [dummy src , dummy dst]  n_dst x n_src : 0         -- leftover pairing, free
    # A real pair is worth taking only when it beats the radius_um cost of
    # leaving both ends unmatched, and BIG keeps an out-of-gate pair from
    # ever being chosen just to balance the assignment.
    BIG = 1e6
    real = np.full((n_src, n_dst), BIG, dtype=np.float64)
    real[pairs[:, 0], pairs[:, 1]] = dist
    top = np.concatenate([real, np.full((n_src, n_src), radius_um)], axis=1)
    bottom = np.concatenate(
        [np.full((n_dst, n_dst), radius_um), np.zeros((n_dst, n_src))], axis=1
    )
    cost = np.concatenate([top, bottom], axis=0)

    rows, cols = linear_sum_assignment(cost)
    selected = [(r, c) for r, c in zip(rows, cols) if r < n_src and c < n_dst]
    if not selected:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(selected, dtype=np.int64)
