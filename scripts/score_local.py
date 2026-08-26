#!/usr/bin/env python3
"""Score predicted `.geff` graphs against local ground truth.

    # primary signal: volumes the model never trained on
    python scripts/score_local.py --geff-dir dist/preds_val --held-out dist/models/detector.pt

    # the 4 competition volumes (trained on -- optimistic, reported separately)
    python scripts/score_local.py --geff-dir dist/preds --competition

Read `src/cell_tracking/metric.py` for what this number can and cannot tell
you: trust same-checkpoint comparisons, distrust cross-checkpoint ones, and
always read the node ratio alongside the score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cell_tracking.config import get_train_dir
from cell_tracking.io_geff import read_geff
from cell_tracking.metric import (
    DIVISION_WEIGHT,
    adjusted_jaccard,
    detection_recall,
    match_nodes_per_frame,
    score_divisions,
    score_edges,
)

COMPETITION_VOLUMES = [
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
]


def score_volume(name: str, train_dir: Path, geff_dir: Path) -> dict:
    gt = read_geff(train_dir / f"{name}.geff")
    pred = read_geff(geff_dir / f"{name}.geff")

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


def held_out_names(checkpoint: Path) -> list[str]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    names = ckpt.get("val_names")
    if not names:
        raise SystemExit(
            f"{checkpoint} has no 'val_names'; it predates held-out tracking or was "
            "trained on every volume."
        )
    return list(names)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geff-dir", type=Path, default=Path("dist/preds"))
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--volume", action="append", dest="volumes", default=None)
    parser.add_argument(
        "--held-out",
        type=Path,
        default=None,
        help="Checkpoint whose held-out volume list to score (the honest signal).",
    )
    parser.add_argument("--competition", action="store_true", help="Score the 4 test volumes.")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    train_dir = args.train_dir or get_train_dir()
    if args.volumes:
        names, label = args.volumes, "requested"
    elif args.held_out:
        names, label = held_out_names(args.held_out), "HELD OUT (never trained on)"
    elif args.competition:
        names, label = COMPETITION_VOLUMES, "COMPETITION volumes (trained on -- optimistic)"
    else:
        names, label = COMPETITION_VOLUMES, "COMPETITION volumes (trained on -- optimistic)"

    names = [n for n in names if (args.geff_dir / f"{n}.geff").exists()]
    if not names:
        raise SystemExit(f"No predicted graphs found in {args.geff_dir}")

    results = [score_volume(n, train_dir, args.geff_dir) for n in names]

    print(f"\n=== per-volume: {label} ===")
    for r in results:
        print(
            f"{r['name']:16s} gt_nodes={r['n_gt_nodes']:5d} pred={r['n_pred_nodes']:6d} "
            f"ratio={r['node_ratio']:5.3f}x  gt_edges={r['n_gt_edges']:5d} "
            f"TP={r['tp']:5d} FP={r['fp']:5d} FN={r['fn']:5d} ign={r['ignored']:6d}  "
            f"jac={r['edge_jaccard']:.4f} adj={r['adjusted_edge_jaccard']:.4f}  "
            f"div {r['div_tp']}/{r['div_fp']}/{r['div_fn']} (tp/fp/fn)"
        )

    print(f"\n=== detection recall (nearest prediction to each GT node, no competition) ===")
    for r in results:
        print(
            f"{r['name']:16s} within 7um: {r['det_within_7um']:5d}/{r['n_gt_nodes']:5d} "
            f"({100*r['det_recall']:5.1f}%)   median NN distance {r['det_median_um']:6.2f} um"
        )
    tot_gt = sum(r["n_gt_nodes"] for r in results)
    tot_hit = sum(r["det_within_7um"] for r in results)
    print(f"{'OVERALL':16s} within 7um: {tot_hit:5d}/{tot_gt:5d} ({100*tot_hit/max(tot_gt,1):5.1f}%)")
    print("  This number is comparable across checkpoints; the score below is not.")

    total_w = sum(r["weight"] for r in results)
    adj = (
        sum(r["adjusted_edge_jaccard"] * r["weight"] for r in results) / total_w
        if total_w
        else 0.0
    )
    tp, fp, fn = (sum(r[k] for r in results) for k in ("tp", "fp", "fn"))
    raw = tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    dtp, dfp, dfn = (sum(r[k] for r in results) for k in ("div_tp", "div_fp", "div_fn"))
    div_j = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else 0.0

    t_true_total = sum(r["t_true"] for r in results if r["t_true"])
    pred_total = sum(r["n_pred_nodes"] for r in results if r["t_true"])

    print(f"\n=== totals ({label}) ===")
    print(f"edge_jaccard (un-adjusted)      = {raw:.4f}   (TP={tp} FP={fp} FN={fn})")
    print(f"adjusted_edge_jaccard           = {adj:.4f}")
    print(f"division_jaccard (micro)        = {div_j:.4f}   (TP={dtp} FP={dfp} FN={dfn})")
    print(f"SCORE = adj + {DIVISION_WEIGHT}*div        = {adj + DIVISION_WEIGHT * div_j:.4f}")
    if t_true_total:
        print(f"\nglobal node ratio               = {pred_total / t_true_total:.3f}x")
        print("  Diagnostic, NOT a target -- v1's history refuted driving this directly.")
    if "COMPETITION" in label:
        print(
            "\nNOTE: these volumes are in the training set, so this is an upper bound.\n"
            "      Use --held-out for a generalization estimate."
        )

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=1))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
