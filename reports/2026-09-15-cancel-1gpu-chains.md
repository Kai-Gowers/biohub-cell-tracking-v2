# 2026-09-15 -- 1-GPU training chains cancelled; DDP runs continue

## Decision

Cancelled the pending continuations of the two 1-GPU self-chaining runs
(`ct-pack-s0` job 3006595, `ct-pack-s314159` job 3007452) at 16:45.
The 4-GPU DDP runs (`ct-pack-s0-ddp` 3003485, `ct-pack-s314159-ddp` 3003486,
`long` partition, started 2026-09-14) continue and are projected to reach
epoch 400 on 2026-09-16 mid afternoon.

Checkpoints from the 1-GPU chains are kept: `dist/models/pack_s{0,314159}.pt`
(last, epochs 162 / 161), `_best.pt`, `_epochN.pt` every 25, and
`history_pack_s*.json`.

## State at cancellation

| run | epoch | best val_score (epoch) | 44b6 / 6bba at best | s / epoch |
|---|---|---|---|---|
| ct-pack-s0 (1 GPU) | 162 | 0.8731 (145) | 0.7612 / 0.8869 | ~1430 |
| ct-pack-s314159 (1 GPU) | 161 | 0.8705 (155) | 0.7429 / 0.8863 | ~1450 |
| ct-pack-s0-ddp (4 GPU) | 215 | 0.8623 (195) | 0.7545 / 0.8756 | ~360 (+270 s val every 5) |
| ct-pack-s314159-ddp (4 GPU) | 209 | 0.8700 (205) | 0.7664 / 0.8826 | ~360 (+270 s val every 5) |

`val_score` = competition metric via `pipeline.run_volume(stage="ilp")`, no TTA,
on the 20-volume `dist/heldout_split.json`.

## Why

1. **Not independent seeds.** Same seed, split and recipe as the DDP twin;
   DDP uses `SyncBatchNorm` so the global batch of 8 is normalised as on one
   GPU. Only data order and float rounding differ. As blend members they add
   less diversity than a fresh seed, which costs ~2 days on DDP vs ~9 here.
2. **Cannot finish in useful time.** 238 epochs left x ~24 min = ~95 GPU-h =
   eight more 12 h `short` slots, with 12-20 h queue waits between slots
   (3006595 pending since 09-14 21:08, est. start 09-16 09:46; 3007452 since
   09-15 00:29, est. start 09-16 17:21) and cluster maintenance 09-18 08-16 h.
   Realistic finish ~09-24; the DDP runs finish 09-16.
3. **They would compete for GPUs** with the checkpoint scoring sweep
   (8-view TTA, dual-seed blend) that follows the DDP runs.

## Open observation (not acted on)

Over epochs 100-160 the 1-GPU seed-0 run averages ~0.015 higher `val_score`
than DDP seed 0 (e.g. 0.8731 vs 0.8474 at epoch 145); the seed-314159 pair is
indistinguishable. Adjacent-eval jitter is +-0.01 total and +-0.03 on 44b6, so
this is within noise. If it matters, compare `pack_s0_epoch150.pt` against
`pack_s0_ddp_epoch150.pt` with the full pipeline (both exist); no further
1-GPU training is needed for that.

## Next

- 09-16: score the four DDP checkpoints (`_best` and epoch 400, both seeds)
  and the two 1-GPU `_best` with `scripts/score_one.sh` / `slurm_score_sweep.sbatch`,
  per embryo; try the dual-seed blend with every pairing.
- Then the post-processing experiments from
  `reports/2026-09-14-jump-frames-postprocessing.md`.
