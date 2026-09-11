"""Score a predicted graph against a volume's ground truth, in memory.

One implementation shared by `scripts/score_local.py` (graphs read back from
`.geff`) and `train.eval_tracking` (graphs straight out of `predict_volume`,
every few epochs, so checkpoints can be selected by the competition metric
instead of the detection-only `val_loss` -- which picked a worse checkpoint
than the last epoch in 14 of 17 cluster runs).
"""

from __future__ import annotations

import numpy as np

from cell_tracking.graph import TrackGraph
from cell_tracking.io_geff import GeffGraph, embryo_of
from cell_tracking.metric import (
    DIVISION_WEIGHT,
    adjusted_jaccard,
    detection_recall,
    match_nodes_per_frame,
    score_divisions,
    score_edges,
)


def track_graph_to_geff(graph: TrackGraph, name: str = "") -> GeffGraph:
    """`TrackGraph` (prediction in memory) -> `GeffGraph` (what the metric reads).

    Coordinates are rounded to integer voxels here exactly as `write_geff` +
    `read_geff` would round-trip them, so an in-memory score equals the
    on-disk one.
    """
    a = graph.to_arrays()
    g = GeffGraph(
        node_ids=a["node_ids"],
        t=a["t"],
        z=np.rint(a["z"]).astype(np.int64),
        y=np.rint(a["y"]).astype(np.int64),
        x=np.rint(a["x"]).astype(np.int64),
        edges=a["edges"],
    )
    g.name = name  # type: ignore[attr-defined]
    return g


def score_volume_graphs(name: str, gt: GeffGraph, pred: GeffGraph) -> dict:
    """Per-volume metric breakdown (the row format `score_local.py` prints and saves)."""
    gt_edges = {(int(u), int(v)) for u, v in gt.edges}
    pred_edges = [(int(u), int(v)) for u, v in pred.edges]
    pred_to_gt = match_nodes_per_frame(gt.nodes_by_t(), pred.nodes_by_t())

    edge = score_edges(gt_edges, pred_edges, pred_to_gt)
    div = score_divisions(gt_edges, pred_edges, pred_to_gt)
    rec = detection_recall(gt.nodes_by_t(), pred.nodes_by_t())

    n_pred = len(pred.node_ids)
    t_true = gt.estimated_number_of_nodes
    return {
        "name": name,
        "embryo": embryo_of(name),
        "n_gt_nodes": len(gt.node_ids),
        "n_pred_nodes": n_pred,
        "t_true": t_true,
        "node_ratio": (n_pred / t_true) if t_true else float("nan"),
        "n_gt_edges": len(gt_edges),
        "n_pred_edges": len(pred_edges),
        "matched_nodes": len(pred_to_gt),
        "tp": edge.tp,
        "fp": edge.fp,
        "fn": edge.fn,
        "ignored": edge.ignored,
        "weight": edge.weight,
        "edge_jaccard": edge.jaccard,
        "adjusted_edge_jaccard": adjusted_jaccard(edge.jaccard, n_pred, t_true),
        "n_gt_divisions": len(gt.divisions()),
        "pred_forks": div.candidates,
        "div_tp": div.tp,
        "div_fp": div.fp,
        "div_fn": div.fn,
        "det_recall": rec["recall"],
        "det_median_um": rec["median_um"],
        "det_within_7um": rec["within_7um"],
    }


def summarize(results: list[dict]) -> dict:
    """Weighted adjusted edge Jaccard + micro division Jaccard over a set of volumes.

    Same arithmetic as the official aggregation (per-sample adjusted Jaccard
    weighted by `TP+FP+FN`; divisions micro-averaged), so a subset's summary
    (one embryo) is directly comparable to the full-split total.
    """
    total_w = sum(r["weight"] for r in results)
    adj = sum(r["adjusted_edge_jaccard"] * r["weight"] for r in results) / total_w if total_w else 0.0
    tp, fp, fn = (sum(r[k] for r in results) for k in ("tp", "fp", "fn"))
    dtp, dfp, dfn = (sum(r[k] for r in results) for k in ("div_tp", "div_fp", "div_fn"))
    div_j = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else 0.0
    gt = sum(r["n_gt_nodes"] for r in results)
    hit = sum(r["det_within_7um"] for r in results)
    t_true = sum(r["t_true"] for r in results if r["t_true"])
    pred = sum(r["n_pred_nodes"] for r in results if r["t_true"])
    return {
        "n": len(results),
        "adj": adj,
        "div": div_j,
        "score": adj + DIVISION_WEIGHT * div_j,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "weight": total_w,
        "det_recall": hit / gt if gt else float("nan"),
        "node_ratio": pred / t_true if t_true else float("nan"),
    }


def summarize_by_embryo(results: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r.get("embryo") or embryo_of(r["name"]), []).append(r)
    return {emb: summarize(rs) for emb, rs in sorted(by.items())}
