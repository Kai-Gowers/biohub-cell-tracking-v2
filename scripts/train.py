#!/usr/bin/env python3
"""Train the detector.

    python scripts/train.py --epochs 20 --out dist/models/detector.pt

Checkpoints every epoch and resumes by default, so a session that hits a
wall-clock limit can be restarted with the same command.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from cell_tracking.config import CHECKPOINT_NAME, get_cache_dir, get_train_dir
from cell_tracking.train import History, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("dist/models") / CHECKPOINT_NAME)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--frames-per-volume", type=int, default=20)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--limit-volumes", type=int, default=None, help="Smoke mode: first N volumes only.")
    parser.add_argument("--select-by", default="val_loss", help="Metric the `_best` checkpoint tracks.")
    parser.add_argument("--patience", type=int, default=None, help="Stop after N epochs with no --select-by gain.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--history-json", type=Path, default=None)
    args = parser.parse_args()

    train_dir = args.train_dir or get_train_dir()
    names = None
    if args.limit_volumes:
        from cell_tracking.io_geff import list_geff_datasets

        names = list_geff_datasets(train_dir)[: args.limit_volumes]
        print(f"smoke mode: {names}")

    history = History()
    out = train(
        train_dir,
        args.out,
        epochs=args.epochs,
        lr=args.lr,
        frames_per_volume=args.frames_per_volume,
        grad_accum=args.grad_accum,
        cache_dir=args.cache_dir or get_cache_dir(),
        names=names,
        resume=not args.no_resume,
        history=history,
        select_by=args.select_by,
        patience=args.patience,
    )
    if args.history_json:
        history.to_json(args.history_json)
    print(f"checkpoint: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
