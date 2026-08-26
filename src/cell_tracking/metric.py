"""Local implementation of the competition metric (see `context/metrics.md`).

    score = adjusted_edge_jaccard + 0.1 * division_jaccard

Read the caveats before trusting a number out of this module:

- **Cross-checkpoint comparisons have been anti-correlated with the real
  leaderboard** (r = -0.799 over five submissions). Same checkpoint, different
  config, is the one axis this has tracked reliably.
- Ground truth is sparse (2.8% of true cells). ~98.5% of predicted edges touch
  no annotated node and are *ignored* rather than counted, so over-linking is
  close to invisible here.
- Per-sample weights `w_i = TP+FP+FN` track how densely a volume is annotated,
  not how big it is; local annotation density varies 69x across volumes.

The reliable use is a held-out set the model never trained on, which is what
`scripts/score_local.py` defaults to.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from cell_tracking.config import SCALE

MAX_MATCH_DIST_UM = 7.0
NODE_PENALTY_A = 0.1
DIVISION_WEIGHT = 0.1


def match_nodes_per_frame(
    gt_nodes_by_t: dict[int, list], pred_nodes_by_t: dict[int, list]
) -> dict[int, int]:
    """Optimal bipartite node matching per timepoint, within 7 µm."""
    pred_to_gt: dict[int, int] = {}
    for t in sorted(set(gt_nodes_by_t) | set(pred_nodes_by_t)):
        gt_list, pred_list = gt_nodes_by_t.get(t, []), pred_nodes_by_t.get(t, [])
        if not gt_list or not pred_list:
            continue
        gt_ids = [nid for nid, _ in gt_list]
        pred_ids = [nid for nid, _ in pred_list]
        gt_xyz = np.array([c for _, c in gt_list], dtype=np.float64) * SCALE
        pred_xyz = np.array([c for _, c in pred_list], dtype=np.float64) * SCALE

        dist = np.linalg.norm(gt_xyz[:, None, :] - pred_xyz[None, :, :], axis=-1)
        cost = np.where(dist <= MAX_MATCH_DIST_UM, dist, 1e6)
        rows, cols = linear_sum_assignment(cost)
        for r, c in zip(rows, cols):
            if cost[r, c] <= MAX_MATCH_DIST_UM:
                pred_to_gt[pred_ids[c]] = gt_ids[r]
    return pred_to_gt


@dataclass
class EdgeResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignored: int = 0

    @property
    def jaccard(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else 0.0

    @property
    def weight(self) -> int:
        return self.tp + self.fp + self.fn


def score_edges(
    gt_edges: set[tuple[int, int]],
    pred_edges: list[tuple[int, int]],
    pred_to_gt: dict[int, int],
) -> EdgeResult:
    """TP/FP/FN per `metrics.md`.

    A predicted edge is a TP when both endpoints match GT nodes joined by a GT
    edge. A non-TP edge is an FP only when a matched endpoint's true
    connectivity is actively contradicted -- an edge touching an unmatched
    (real but unannotated) node is ignored, not penalized.
    """
    gt_has_child = {u for u, _ in gt_edges}
    gt_has_parent = {v for _, v in gt_edges}

    result = EdgeResult()
    matched_gt_edges: set[tuple[int, int]] = set()
    for s, t in pred_edges:
        gs, gt_ = pred_to_gt.get(s), pred_to_gt.get(t)
        if gs is not None and gt_ is not None and (gs, gt_) in gt_edges:
            matched_gt_edges.add((gs, gt_))
            continue
        contradicted = (gt_ is not None and gt_ in gt_has_parent) or (
            gs is not None and gs in gt_has_child
        )
        if contradicted:
            result.fp += 1
        else:
            result.ignored += 1

    result.tp = len(matched_gt_edges)
    result.fn = len(gt_edges) - result.tp
    return result


@dataclass
class DivisionResult:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    candidates: int = 0

    @property
    def jaccard(self) -> float:
        d = self.tp + self.fp + self.fn
        return self.tp / d if d else 0.0


@dataclass
class _Topology:
    succ: dict[int, list[int]] = field(default_factory=dict)
    pred: dict[int, list[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, edges) -> "_Topology":
        topo = cls()
        for u, v in edges:
            topo.succ.setdefault(u, []).append(v)
            topo.pred.setdefault(v, []).append(u)
        return topo


def score_divisions(
    gt_edges: set[tuple[int, int]],
    pred_edges: list[tuple[int, int]],
    pred_to_gt: dict[int, int],
) -> DivisionResult:
    """Windowed division matching per `metrics.md`.

    A GT division is recovered when a predicted fork satisfies all of:

    - **Local parent anchor** -- a prediction matches the dividing parent or
      its immediate predecessor, and the fork is that node or its successor.
    - **Two distinct daughter branches** -- the two GT daughter lineages
      (child + its children) match two *different* predicted branches under a
      bipartite assignment, possibly at different timepoints in the window.
    - **Unmerged branches** -- a child used by the fork has that fork as its
      sole parent.

    Unmatched but structurally valid forks with no local GT evidence are
    ignored rather than counted as false positives, mirroring the sparse-GT
    treatment on the edge side.
    """
    gt = _Topology.build(gt_edges)
    pred = _Topology.build(pred_edges)
    gt_to_pred: dict[int, int] = {}
    for p, g in pred_to_gt.items():
        gt_to_pred.setdefault(g, p)

    gt_divisions = [u for u, kids in gt.succ.items() if len(kids) >= 2]
    pred_forks = {u for u, kids in pred.succ.items() if len(kids) >= 2}

    def gt_lineage(child: int) -> set[int]:
        return {child} | set(gt.succ.get(child, []))

    def pred_branch(fork: int, child: int) -> set[int]:
        # A branch is only usable if the fork is the child's sole parent;
        # a locally merged branch cannot represent a distinct daughter path.
        if pred.pred.get(child, []) != [fork]:
            return set()
        return {child} | set(pred.succ.get(child, []))

    # Which forks are locally evaluable at all -- i.e. the GT annotates this
    # part of the lineage, so silence is informative.
    evaluable: set[int] = set()
    for fork in pred_forks:
        g = pred_to_gt.get(fork)
        if g is not None and g in gt.succ:
            evaluable.add(fork)

    pairings: list[tuple[int, int]] = []
    for parent in gt_divisions:
        anchors = {parent} | set(gt.pred.get(parent, []))
        pred_anchors = {gt_to_pred[a] for a in anchors if a in gt_to_pred}
        # The fork must be a matched parent-side node or an immediate successor
        # of one -- "downstream of the anchor", not merely in the same
        # connected component.
        local_forks = {
            f
            for f in pred_forks
            if f in pred_anchors or any(f in pred.succ.get(a, []) for a in pred_anchors)
        }
        if not local_forks:
            continue

        daughters = gt.succ[parent][:2]
        for fork in local_forks:
            evaluable.add(fork)
            branches = [pred_branch(fork, c) for c in pred.succ.get(fork, [])]
            branches = [b for b in branches if b]
            if len(branches) < 2:
                continue
            # Bipartite: each GT daughter lineage must land on a *different*
            # predicted branch.
            support = np.zeros((2, len(branches)), dtype=np.int64)
            for di, d in enumerate(daughters):
                lineage = gt_lineage(d)
                for bi, b in enumerate(branches):
                    support[di, bi] = sum(
                        1 for n in b if pred_to_gt.get(n) in lineage
                    )
            if (support.max(axis=1) == 0).any():
                continue
            rows, cols = linear_sum_assignment(-support)
            if all(support[r, c] > 0 for r, c in zip(rows, cols)):
                pairings.append((parent, fork))

    # One fork recovers at most one GT division and vice versa.
    matched_gt: set[int] = set()
    matched_fork: set[int] = set()
    for parent, fork in pairings:
        if parent in matched_gt or fork in matched_fork:
            continue
        matched_gt.add(parent)
        matched_fork.add(fork)

    return DivisionResult(
        tp=len(matched_gt),
        fp=len(evaluable - matched_fork),
        fn=len(gt_divisions) - len(matched_gt),
        candidates=len(pred_forks),
    )


def detection_recall(
    gt_nodes_by_t: dict[int, list], pred_nodes_by_t: dict[int, list]
) -> dict:
    """Unconstrained nearest-neighbour recall: is the cell detected *at all*?

    For each GT node, the distance to the closest prediction in the same frame,
    with no bipartite competition. This is the most generous possible view of
    detection, and it is the single most diagnostic number in this project:
    the previous pipeline missed 80-95% of GT nodes on 3 of 4 volumes even
    under this measure, which caps every downstream stage. No amount of linking
    or repair can recover a cell that was never detected.

    Unlike the headline score, this is comparable ACROSS checkpoints -- it does
    not depend on annotation density, edge weighting, or the node-count
    penalty, all of which are what make cross-checkpoint score comparison
    unreliable here.
    """
    dists: list[float] = []
    n_gt = 0
    for t, gt_list in gt_nodes_by_t.items():
        n_gt += len(gt_list)
        pred_list = pred_nodes_by_t.get(t, [])
        if not pred_list:
            continue
        pred_xyz = np.array([c for _, c in pred_list], dtype=np.float64) * SCALE
        for _, c in gt_list:
            g = np.array(c, dtype=np.float64) * SCALE
            dists.append(float(np.linalg.norm(pred_xyz - g[None, :], axis=-1).min()))

    if not dists:
        return {"n_gt": n_gt, "with_pred": 0, "within_7um": 0, "recall": 0.0,
                "median_um": float("nan"), "mean_um": float("nan")}
    d = np.array(dists)
    return {
        "n_gt": n_gt,
        "with_pred": len(d),
        "within_7um": int((d <= MAX_MATCH_DIST_UM).sum()),
        "recall": float((d <= MAX_MATCH_DIST_UM).sum() / max(n_gt, 1)),
        "median_um": float(np.median(d)),
        "mean_um": float(d.mean()),
    }


def adjusted_jaccard(jaccard: float, n_pred_nodes: int, t_true: int | None) -> float:
    """`max(0, jaccard * (1 - a * (T_pred - T_true) / T_true))`, a = 0.1."""
    if not t_true:
        return jaccard
    return max(0.0, jaccard * (1 - NODE_PENALTY_A * (n_pred_nodes - t_true) / t_true))
