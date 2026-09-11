#!/usr/bin/env python3
"""Train the ported pack model (TemporalUNet3D + node transformer) on the held-out split.

    # the pack recipe (seed-314159 training_config.json), our 20-volume held-out split
    python scripts/train.py --epochs 400 --lr 1e-4 --batch-size 8 --seed 0 --out dist/models/pack_s0.pt

    # smoke
    python scripts/train.py --limit-volumes 8 --epochs 2 --eval-tracking-every 1 --out dist/models/smoke.pt

Writes `<out>` (last epoch, resumable), `<out>_best.pt` (best --select-by, default
val_score = competition metric on the held-out volumes), `<out>_epochN.pt` every
--save-every, and `history_<out>.json`. `--max-hours` stops cleanly for SLURM chains.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cell_tracking.config import get_cache_dir, get_train_dir
from cell_tracking.io_geff import list_geff_datasets
from cell_tracking.pack_train import TrainConfig, train


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-dir", type=Path, default=None)
    p.add_argument("--cache-dir", type=Path, default=None, help="Decimated-frame cache (default: dist/cache_pack).")
    p.add_argument("--no-cache", action="store_true", help="Read frames from zarr instead of the cache.")
    p.add_argument("--out", type=Path, default=Path("dist/models/pack.pt"))
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--window-size", type=int, default=2)
    p.add_argument("--unet-layers", type=str, default="32,64,128")
    p.add_argument("--unet-out-channels", type=int, default=32)
    p.add_argument("--det-loss-weight", type=float, default=1.0)
    p.add_argument("--det-neg-weight", type=float, default=0.01)
    p.add_argument("--pool-kernel-um", type=float, default=5.0)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-embryo", type=str, default=None, help="Hold out every crop of this embryo (44b6|6bba).")
    p.add_argument("--eval-tracking-every", type=int, default=5)
    p.add_argument("--eval-max-batches", type=int, default=200)
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--select-by", type=str, default="val_score")
    p.add_argument("--max-hours", type=float, default=None)
    p.add_argument("--single-gpu", action="store_true", help="Disable DataParallel on the U-Net.")
    p.add_argument("--limit-volumes", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--log-every", type=int, default=200)
    args = p.parse_args()

    train_dir = args.train_dir or get_train_dir()
    cache_dir = None if args.no_cache else (args.cache_dir or Path("dist/cache_pack"))
    if cache_dir is not None and not cache_dir.exists():
        cache_dir = get_cache_dir() if get_cache_dir().exists() else None
    names = list_geff_datasets(train_dir)
    if args.limit_volumes:
        names = names[: args.limit_volumes]

    cfg = TrainConfig(
        epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, num_workers=args.num_workers,
        window_size=args.window_size, unet_out_channels=args.unet_out_channels,
        unet_layers=tuple(int(x) for x in args.unet_layers.split(",")),
        det_loss_weight=args.det_loss_weight, det_neg_weight=args.det_neg_weight,
        pool_kernel_um=args.pool_kernel_um, augment=not args.no_augment, seed=args.seed,
        eval_tracking_every=args.eval_tracking_every, eval_max_batches=args.eval_max_batches,
        save_every=args.save_every, select_by=args.select_by, max_hours=args.max_hours,
        data_parallel=not args.single_gpu, val_prefix=args.val_embryo,
    )
    print(f"config: {cfg.to_dict()}", flush=True)
    print(f"cache: {cache_dir}", flush=True)
    train(train_dir, args.out, cfg, cache_dir=cache_dir, names=names, resume=not args.no_resume,
          started_at=time.time(), log_every=args.log_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
