#!/usr/bin/env python3
"""Train the DeepCenter center-heatmap veto model with the artifact's recipe.

    python scripts/train_deepcenter.py --out-dir dist/models/deepcenter --epochs 50

Validation = this repo's held-out split (the artifact's embryo-prefix split put
128 volumes in val and 71 in train). Everything else follows
context/pack_deepcenter/source_scripts/train_full_frame_center_detector.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_tracking.config import get_train_dir
from cell_tracking.deepcenter_train import DeepCenterTrainConfig, train
from cell_tracking.io_geff import list_geff_datasets


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("dist/models/deepcenter"))
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--val-embryo", type=str, default=None)
    p.add_argument("--val-batches", type=int, default=24)
    p.add_argument("--limit-volumes", type=int, default=None)
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    train_dir = args.train_dir or get_train_dir()
    names = list_geff_datasets(train_dir)
    if args.limit_volumes:
        names = names[: args.limit_volumes]
    cfg = DeepCenterTrainConfig(seed=args.seed, batch_size=args.batch_size, epochs=args.epochs, num_workers=args.num_workers,
                                learning_rate=args.lr, val_prefix=args.val_embryo, val_batches=args.val_batches)
    train(train_dir, args.out_dir, cfg, names=names, resume=not args.no_resume, max_hours=args.max_hours)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
