#!/usr/bin/env python3
"""Predict one `.geff` tracking graph per video with the ported 0.942 pipeline.

    # full pipeline on the held-out split with the public weights
    python scripts/predict.py --checkpoint context/pack_primary/weights/unet_transformer/split_0/edge_predictor_best.pth \
        --secondary-checkpoint context/pack_secondary/weights/unet_transformer/split_0/edge_predictor_best.pth \
        --deepcenter context/pack_deepcenter/weights/full_frame_center/best.pt \
        --test-dir data/biohub-cell-tracking-during-development/train --split dist/heldout_split.json \
        --out-dir dist/preds_val_pack

    # competition test folder, default paths
    python scripts/predict.py --checkpoint <ckpt> --out-dir dist/preds

Kept separate from CSV generation so the predicted graphs stay inspectable.
`--stage ilp` writes the pack-equivalent graph (before post-processing) with
`edge_prob` as an edge property; `--stage raw` skips the ILP as well.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

from cell_tracking.config import get_test_dir
from cell_tracking.ilp import ILPConfig
from cell_tracking.pack_predict import PredictConfig
from cell_tracking.pipeline import STAGES, load_models, run_volume, write_result
from cell_tracking.postprocess import PostprocessConfig


def build_split(test_dir: Path, out_path: Path) -> list[str]:
    names = sorted(p.stem for p in test_dir.glob("*.zarr"))
    out_path.write_text(json.dumps({"datasets": names}, indent=1))
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("dist/preds"))
    parser.add_argument("--split", type=Path, default=None, help="JSON with {'datasets': [...]}.")
    parser.add_argument("--volume", action="append", dest="volumes", default=None)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Primary weights (.pth state dict or repo checkpoint).")
    parser.add_argument("--secondary-checkpoint", type=Path, default=None, help="Second seed for the dual-seed blend.")
    parser.add_argument("--deepcenter", type=Path, default=None, help="DeepCenter best.pt for the gap-closing veto.")
    parser.add_argument("--no-deepcenter-epoch-check", action="store_true")
    parser.add_argument("--stage", choices=STAGES, default="full")
    parser.add_argument("--t-max", type=int, default=None, help="Debug: only the first N frames.")
    parser.add_argument("--no-tta", action="store_true", help="Single view instead of 8-view D4 TTA.")
    parser.add_argument("--tta-views", choices=["d4", "flip"], default="d4",
                        help="d4 = notebook's 8 planar views; flip = the pack's original 4-view flip TTA.")
    parser.add_argument("--no-edge-tta", action="store_true", help="Disable edge-feature TTA.")
    parser.add_argument("--no-bidir", action="store_true", help="Disable bidirectional harmonic fusion.")
    parser.add_argument("--no-ilp", action="store_true", help="Skip the ILP (greedy 1-parent/2-children candidates).")
    parser.add_argument("--no-postprocess", action="store_true", help="Same as --stage ilp.")
    parser.add_argument("--det-threshold", type=float, default=None)
    parser.add_argument("--edge-threshold", type=float, default=None, help="Override the 0.48/0.5 candidate threshold.")
    parser.add_argument("--preset", choices=["tuned", "notebook"], default="tuned",
                        help="'notebook' = the 0.942 notebook's committed post-processing (PostprocessConfig defaults); "
                             "'tuned' (default) = notebook + our held-out-validated deviations, currently only "
                             "output_motion_relink=False (+0.011 blend / +0.009 single seed / +0.018 shipped weights on the "
                             "clean 20-volume split, reports/2026-09-17-error-analysis-0942.md). --post-off/--post-set apply on top.")
    parser.add_argument("--post-off", action="append", default=[], metavar="FLAG",
                        help="PostprocessConfig boolean to switch off, e.g. output_gap2_recovery (repeatable).")
    parser.add_argument("--post-set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override any PostprocessConfig field, e.g. gap_close_um=5.0 or deepcenter_tta=1 (repeatable).")
    parser.add_argument("--ilp-set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override any ILPConfig field, e.g. disappearance_weight=1.0 (repeatable).")
    parser.add_argument("--predict-set", action="append", default=[], metavar="FIELD=VALUE",
                        help="Override any PredictConfig field, e.g. secondary_edge_weight=0.15 (repeatable).")
    parser.add_argument("--dump-stats", type=Path, default=None, help="Write per-volume stats JSON here.")
    parser.add_argument("--dump-det-dir", type=Path, default=None,
                        help="Diagnostics: write each frame's final detection probability map (float16 npy) under this dir.")
    parser.add_argument("--verbose", action="store_true")
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

    stage = "ilp" if args.no_postprocess and args.stage == "full" else args.stage
    pcfg_kwargs: dict = {"use_ilp": not args.no_ilp}
    if args.no_tta:
        pcfg_kwargs["det_tta"] = False
    pcfg_kwargs["det_tta_views"] = args.tta_views
    if args.no_edge_tta:
        pcfg_kwargs["edge_feature_tta"] = False
        pcfg_kwargs["secondary_edge_feature_tta"] = False
    if args.no_bidir:
        pcfg_kwargs["bidirectional_edge_weight"] = 0.0
    if args.det_threshold is not None:
        pcfg_kwargs["det_threshold"] = args.det_threshold
    if args.edge_threshold is not None:
        pcfg_kwargs["edge_threshold_single"] = args.edge_threshold
        pcfg_kwargs["edge_threshold_dual"] = args.edge_threshold
    def _typed_overrides(cls, items: list[str]) -> dict:
        fields = {f.name: f for f in dataclasses.fields(cls)}
        out: dict = {}
        for item in items:
            if "=" not in item:
                raise SystemExit(f"expected FIELD=VALUE, got {item!r}")
            name, raw = item.split("=", 1)
            if name not in fields:
                raise SystemExit(f"unknown {cls.__name__} field {name!r}")
            typ = fields[name].type if not isinstance(fields[name].type, str) else fields[name].type
            tname = typ if isinstance(typ, str) else getattr(typ, "__name__", str(typ))
            if tname == "bool":
                out[name] = raw.strip().lower() in ("1", "true", "yes", "on")
            elif tname == "int":
                out[name] = int(raw)
            elif tname == "float":
                out[name] = float(raw)
            else:
                out[name] = raw
        return out

    pcfg_kwargs.update(_typed_overrides(PredictConfig, args.predict_set))
    predict_cfg = PredictConfig(**pcfg_kwargs)
    ilp_cfg = ILPConfig(**_typed_overrides(ILPConfig, args.ilp_set))
    PRESETS = {"notebook": {}, "tuned": {"output_motion_relink": False}}
    post_kwargs = dict(PRESETS[args.preset])
    valid = {f.name for f in dataclasses.fields(PostprocessConfig)}
    for flag in args.post_off:
        if flag not in valid:
            raise SystemExit(f"unknown PostprocessConfig field {flag!r}")
        post_kwargs[flag] = False
    post_kwargs.update(_typed_overrides(PostprocessConfig, args.post_set))
    post_cfg = PostprocessConfig(**post_kwargs)
    print(f"post-processing preset '{args.preset}': {PRESETS[args.preset]}"
          + (f"; overrides: predict={_typed_overrides(PredictConfig, args.predict_set)} post={_typed_overrides(PostprocessConfig, args.post_set)}"
             if (args.post_set or args.predict_set) else "")
          + (f"; post-off={args.post_off}" if args.post_off else "")
          + (f"; ilp={_typed_overrides(ILPConfig, args.ilp_set)}" if args.ilp_set else ""), flush=True)

    models = load_models(
        args.checkpoint, args.secondary_checkpoint, args.deepcenter,
        deepcenter_expected_epoch=None if args.no_deepcenter_epoch_check else 2,
    )
    print(
        f"models on {models.device}; dual_seed={models.secondary is not None}; "
        f"deepcenter={models.deepcenter is not None}; stage={stage}; ilp={not args.no_ilp}; "
        f"{len(names)} volume(s) from {test_dir}"
    )

    all_stats = []
    started = time.time()
    for i, name in enumerate(names, 1):
        result = run_volume(
            models, Path(test_dir) / f"{name}.zarr",
            predict_cfg=predict_cfg, ilp_cfg=ilp_cfg, post_cfg=post_cfg,
            stage=stage, use_ilp=not args.no_ilp, max_frames=args.t_max, verbose=args.verbose,
            dump_det_dir=args.dump_det_dir,
        )
        write_result(result, args.out_dir, name)
        s = result.stats
        all_stats.append(s)
        print(
            f"[{i}/{len(names)}] {name}: {s['nodes']} nodes, {s['edges']} edges, {s['divisions']} divisions, "
            f"det={s['predict']['detections']} cand={s['predict']['candidate_edges']} "
            f"retention_fallback={s['predict']['retention_fallback_frames']} {s['seconds']:.0f}s",
            flush=True,
        )
    if args.dump_stats:
        args.dump_stats.write_text(json.dumps(all_stats, indent=1, default=str))
    print(f"\nWrote {len(names)} graph(s) to {args.out_dir} in {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
