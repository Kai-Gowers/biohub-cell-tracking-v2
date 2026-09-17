# 2026-09-14 — 4-GPU DDP runs started from scratch (fork did not happen); status of all four chains

Checked 2026-09-14 16:10.

## What happened

The two 4-GPU `--ddp` jobs submitted 02:27 (`ct-pack-s0-ddp` 3003485, `ct-pack-s314159-ddp` 3003486,
`long`, 3-day limit, `--max-hours 70`) started at 13:47 and 14:29 on g012 / g014. They were meant to
fork from the 1-GPU chains' checkpoints (`CT_FORK_FROM=dist/models/pack_s{0,314159}.pt`, epoch ~140).
Neither log contains the `forking ...` line and both begin at `epoch 1/400`, so the fork condition was
false. The batch script SLURM captured at submit time does contain the fork block (`scontrol write
batch_script`), and `--out` did not exist beforehand (the checkpoint files were created at 16:04 /
16:07 today), so the only remaining explanation is that `CT_FORK_FROM` was not in the job's
environment at submit time (the header has no `--export`, so sbatch would have propagated it had it
been set in the submitting shell). The exact shell mistake cannot be recovered after the fact.

## Is a from-scratch DDP run still worth the allocation? Yes — kept running.

DDP tracks the 1-GPU recipe closely at equal epochs (same held-out 20 volumes, ILP stage, no TTA):

| epoch | s0 1-GPU | s0 DDP | s314159 1-GPU | s314159 DDP |
|---|---|---|---|---|
| 5  | 0.8034 | 0.7662 | 0.7764 | 0.8080 |
| 10 | 0.8119 | 0.8049 | 0.7997 | 0.8097 |
| 15 | 0.8232 | 0.8256 | 0.8090 | — |

Differences are within the seed noise seen between the two 1-GPU chains themselves. Speed: 376-385 s
train per epoch (vs 1420-1470 s on one L40S/A100), 280-300 s for the 5-epoch `val_score`, i.e.
~7.5 min/epoch wall including evals; GPU utilisation 74-98 % on all four GPUs of each node.

Projection: 382 remaining epochs x 7.5 min = ~48 h -> both DDP runs reach epoch 400 around
2026-09-16 evening, inside `--max-hours 70` and the 3-day limit (job end 09-17 13:47), and before
the 2026-09-18 08:00-16:00 maintenance reservation. The 1-GPU chains (s0 epoch 149, s314159 epoch 141
at 16:10) need ~251 x 25 min = ~4.4 days -> ~2026-09-19, and their 12-h resubmits will be held
back by the maintenance window. The DDP runs overtake the 1-GPU chains around epoch ~205, roughly
2026-09-15 15:00.

Cancelling and resubmitting with a correct fork would have cost the allocation (the jobs waited 11 h
for a whole L40S node) for a ~5 h gain at best, so nothing was cancelled.

## Current chains (16:10)

| job | id | node | epoch | best val_score (epoch) | 44b6 / 6bba at best |
|---|---|---|---|---|---|
| ct-pack-s0 (1 GPU, short) | 3003711 | g017 | 149 | 0.8731 (145) | 0.7612 / 0.8869 |
| ct-pack-s314159 (1 GPU, short) | 3004176 | g006 | 141 | 0.8645 (130) | — |
| ct-pack-s0-ddp (4 GPU, long) | 3003485 | g012 | 18 | 0.8256 (15) | 0.6849 / 0.8431 |
| ct-pack-s314159-ddp (4 GPU, long) | 3003486 | g014 | 13 | 0.8097 (10) | 0.6577 / 0.8289 |

Checkpoints: `dist/models/pack_s0{,_best,_epochN}.pt`, `pack_s314159*.pt`, `pack_s0_ddp*.pt`,
`pack_s314159_ddp*.pt`; histories `dist/models/history_pack_*.json`.

## Decision left for the user

The 1-GPU chains are no longer redundant with the DDP runs (different trajectories, and they hold the
best checkpoints so far). Options: (a) keep all four until the DDP runs pass epoch ~150 tomorrow
afternoon and their curves are confirmed at or above the 1-GPU ones, then `scancel -n ct-pack-s0`
/ `-n ct-pack-s314159`; (b) keep all four to the end and treat them as four seeds for blending.
Default taken here: (a)'s first half — nothing cancelled.

## Fixes

- `scripts/slurm_train.sbatch` now echoes `CT_FORK_FROM`, `CT_CANCEL_JOBNAME`, `CT_CHAIN` and whether
  `--out` exists at start; if `CT_FORK_FROM` is set it exits 1 when the source is missing or `--out`
  is absent, and prints `fork skipped: ... already exists` when it resumes instead of forking.
- `pack_train._train_impl` prints `resumed from <out> at epoch N` on rank 0 (it used to resume
  silently, so a fork could not be confirmed from the log).
