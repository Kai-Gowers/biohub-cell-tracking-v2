"""Link consecutive frames: distance-gated, greedy score-sorted selection.

Candidates are gated to `LINK_RADIUS_UM`, sorted by score descending, and
accepted greedily while both endpoints are still free -- in/out-degree stay
<= 1 by construction, so there is still no division support and no repair
pass. This replaces the v2 baseline's exact bipartite assignment (scipy
`linear_sum_assignment`) with plain greedy thresholding, per
reports/2026-08-25-sample-solution-0.90-comparison.md item 5: even the 0.90
sample solution uses greedy-by-default selection (its global ILP solver
ships but is off by default), so the exact-assignment optimality this
baseline previously bought was not where points were being left on the
table.

With a learned edge model, `edge_scores` are sigmoid probabilities and
candidates below `LINK_SCORE_THRESHOLD` are dropped before the greedy pass.
Without one, the score falls back to negative distance -- `radius_um` alone
gates what counts as a candidate, so nothing extra needs thresholding.
"""

from __future__ import annotations

import numpy as np

from cell_tracking.config import LINK_RADIUS_UM, LINK_SCORE_THRESHOLD
from cell_tracking.peaks import pair_within_radius


def link_frames(
    pos_src_um: np.ndarray,
    pos_dst_um: np.ndarray,
    *,
    radius_um: float = LINK_RADIUS_UM,
    pairs: np.ndarray | None = None,
    edge_scores: np.ndarray | None = None,
    score_threshold: float = LINK_SCORE_THRESHOLD,
) -> np.ndarray:
    """Return selected (src_idx, dst_idx) pairs between two consecutive frames.

    `pairs`/`edge_scores` let a caller reuse a candidate set and learned
    scores it already computed (see `predict.predict_volume`) instead of
    recomputing distance gating here.
    """
    n_src, n_dst = len(pos_src_um), len(pos_dst_um)
    if n_src == 0 or n_dst == 0:
        return np.zeros((0, 2), dtype=np.int64)

    if pairs is None:
        pairs = pair_within_radius(pos_src_um, pos_dst_um, radius_um)
    if len(pairs) == 0:
        return np.zeros((0, 2), dtype=np.int64)

    if edge_scores is not None:
        keep = edge_scores >= score_threshold
        pairs, order_score = pairs[keep], edge_scores[keep]
        if len(pairs) == 0:
            return np.zeros((0, 2), dtype=np.int64)
    else:
        dist = np.linalg.norm(pos_src_um[pairs[:, 0]] - pos_dst_um[pairs[:, 1]], axis=-1)
        order_score = -dist  # closer is better; no extra threshold beyond the radius gate

    order = np.argsort(-order_score, kind="stable")
    used_src: set[int] = set()
    used_dst: set[int] = set()
    selected: list[tuple[int, int]] = []
    for k in order:
        s, d = int(pairs[k, 0]), int(pairs[k, 1])
        if s in used_src or d in used_dst:
            continue
        used_src.add(s)
        used_dst.add(d)
        selected.append((s, d))

    if not selected:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(selected, dtype=np.int64)
