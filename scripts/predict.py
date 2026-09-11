#!/usr/bin/env python3
"""Predict one `.geff` tracking graph per test video.

    python scripts/predict.py --out-dir dist/preds
    python scripts/predict.py --split split.json --out-dir dist/preds

Kept separate from CSV generation so the predicted graphs stay inspectable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from cell_tracking.config import (
    LINK_RADIUS_UM,
    LINK_SCORE_THRESHOLD,
    TTA_FLIPS,
    TTA_FLIPS_WITH_Z,
    get_cache_dir,
    get_test_dir,
)
from cell_tracking.detect import load_model
from cell_tracking.predict import predict_volume, write_prediction


def build_split(test_dir: Path, out_path: Path) -> list[str]:
    names = sorted(p.stem for p in test_dir.glob("*.zarr"))
    out_path.write_text(json.dumps({"datasets": names}, indent=1))
    return names


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("dist/preds"))
    parser.add_argument("--split", type=Path, default=None, help="JSON with {'datasets': [...]}.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Checkpoint to load. Repeat to ensemble several detectors (logits averaged; "
        "the edge scorer comes from the first).",
    )
    parser.add_argument("--volume", action="append", dest="volumes", default=None)
    parser.add_argument("--t-max", type=int, default=None, help="Debug: only the first N frames.")
    parser.add_argument("--no-subvoxel", action="store_true", help="Emit integer-grid detection coordinates.")
    parser.add_argument("--no-tta", action="store_true", help="Skip flip test-time augmentation.")
    parser.add_argument("--tta-z", action="store_true", help="8-way TTA: also reflect z.")
    parser.add_argument(
        "--dump-candidates",
        type=Path,
        default=None,
        help="Directory for per-volume .npz candidate/score dumps (for scripts/analyze_errors.py).",
    )
    parser.add_argument(
        "--link-radius-um", type=float, default=LINK_RADIUS_UM, help="Candidate-link gate in um."
    )
    parser.add_argument(
        "--no-edge-model", action="store_true", help="Ignore a checkpoint's edge scorer; link by distance."
    )
    parser.add_argument(
        "--link-score-threshold",
        type=float,
        default=LINK_SCORE_THRESHOLD,
        help="Minimum edge-scorer probability to keep a candidate link.",
    )
    args = parser.parse_args()

    test_dir = args.test_dir or get_test_dir()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.volumes:
        names = args.volumes
    elif args.split and args.split.exists():
        names = json.loads(args.split.read_text())["datasets"]
    else:
        split_path = args.split or (args.out_dir / "split.json")
        names = build_split(test_dir, split_path)
        print(f"wrote split: {split_path}")

    checkpoints = args.checkpoint or [None]
    model, edge_scorer, device = load_model(checkpoints[0])
    models = [model]
    for extra in checkpoints[1:]:
        m, _, _ = load_model(extra, device=device)
        models.append(m)
    if args.no_edge_model:
        edge_scorer = None
    print(
        f"model on {device}; {len(names)} volume(s) from {test_dir}"
        + (f"; ensemble of {len(models)} detectors" if len(models) > 1 else "")
    )

    started = time.time()
    for i, name in enumerate(names, 1):
        graph, stats = predict_volume(
            models if len(models) > 1 else model,
            device,
            Path(test_dir) / f"{name}.zarr",
            cache_dir=args.cache_dir or get_cache_dir(),
            t_max=args.t_max,
            subvoxel=not args.no_subvoxel,
            edge_scorer=edge_scorer,
            tta=not args.no_tta,
            tta_flips=TTA_FLIPS_WITH_Z if args.tta_z else TTA_FLIPS,
            link_radius_um=args.link_radius_um,
            link_score_threshold=args.link_score_threshold,
            dump_candidates=args.dump_candidates,
        )
        write_prediction(graph, args.out_dir, name)
        print(
            f"[{i}/{len(names)}] {name}: {stats['nodes']} nodes, {stats['edges']} edges, "
            f"{stats['detections_per_frame']:.1f} det/frame, "
            f"p_max={stats['max_prob_mean']:.4f}, {stats['infer_seconds']:.1f}s"
        )

    print(f"\nWrote {len(names)} graph(s) to {args.out_dir} in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
