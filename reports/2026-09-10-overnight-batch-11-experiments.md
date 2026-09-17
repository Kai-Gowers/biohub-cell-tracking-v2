# Overnight batch: 11 single-variable runs to explain the 0.7574 cluster regression

## Setup

Follow-up to `reports/2026-09-10-cluster-longer-training.md`, where the
first cluster run (`--epochs 100 --batch-size 16 --grad-accum 1
--val-frames-per-volume 16 --patience 20`, SLURM job 2988722) trained
stably but scored **0.7574** held-out, at the bottom of this repo's
0.7486-0.8074 range. That report named batch size as the prime suspect and
the halved optimizer-step count (224 steps/epoch vs. history's ~448) as the
mechanism. Eleven runs were launched to test that and the alternatives,
all via `scripts/slurm_train.sbatch` on `short` (L40S, bf16 autocast), all
with `--val-frames-per-volume 16 --no-resume`, each writing
`dist/models/detector_<name>.pt` (last epoch) and `detector_<name>_best.pt`
(lowest `val_loss`), plus `history_<name>.json`.

| name | job | differs from the 0.7574 run by |
|---|---|---|
| `ctrl_bs4` | 2988750 | batch 4, grad_accum 2 (history's effective batch 8, with accumulation) |
| `bs8` | 2988752 | batch 8, grad_accum 1 |
| `bs32` | 2988756 | batch 32 |
| `bs16_ep200` | 2988751 | epochs 200, patience 40 (restore total step budget) |
| `bs16_pat40` | 2988755 | patience 40 |
| `fpv40` | 2988765 | `--frames-per-volume 40` (448 steps/epoch, schedule untouched) |
| `bs16_lr2e3` | 2988754 | lr 2e-3 (linear scaling rule) |
| `bs16_noedge` | 2988753 | `--no-edge-model` |
| `edgeevery4` | 2988766 | `--edge-every 4` |
| `edgew03` | 2988767 | `--edge-loss-weight 0.3` |
| `replicate_tmax30` | 2988768 | **epochs 30, lr 5e-4, patience 15, batch 4 x accum 2** -- the exact config of the historical best (`git show 385a600`, `notebooks/kaggle_train.ipynb`) |

All 11 completed (exit 0, no NaN, `skipped_steps=0` throughout) and every
one ended via patience, not by reaching its epoch limit. Runtimes 25-90 min.

## Scoring

Every checkpoint -- **both** `_best` and last-epoch, per the gotcha in
`reports/2026-08-31-tmax30-held-out-score.md` -- was predicted and scored on
the same 20-volume held-out split (`VAL_SEED=0`; `val_names` verified
identical across all 22 checkpoints and equal to `dist/heldout_split.json`)
with the new `scripts/slurm_score.sbatch` (`predict.py` +
`score_local.py --held-out`, ~1 min per checkpoint on a GPU node from the
frame cache). Outputs: `dist/preds_val_<name>_{last,best}/` and
`dist/score_<name>_{last,best}.json`.

Two historical checkpoints were also re-predicted with the current code on
this cluster as a control on the inference side:

| checkpoint | historical score | re-scored here |
|---|---|---|
| `detector_tmax30_epoch23.pt` | 0.8074 | 0.8073 |
| `detector_best_0841.pt` | 0.8031 | 0.8031 |

Inference on the cluster reproduces the old numbers. Whatever differs is on
the training side.

## Results

`SCORE = adjusted edge Jaccard + 0.1 * division Jaccard`, 20 held-out
volumes. `steps` = optimizer updates (batches / grad_accum). `rec` =
detection recall within 7 um; `ratio` = predicted / estimated true nodes.

| run | stopped at | best ep | val_loss best | **SCORE last** | **SCORE best** | rec last | ratio last | FP last |
|---|---|---|---|---|---|---|---|---|
| **`replicate_tmax30`** | 22/30 | 7 | 0.01803 | **0.8056** | 0.7730 | 0.953 | 1.043 | 1104 |
| `bs16_noedge` | 35/100 | 15 | 0.01360 | 0.7702 | 0.7452 | 0.960 | 1.140 | 1646 |
| `ctrl_bs4` | 33/100 | 13 | 0.02035 | 0.7696 | 0.7622 | 0.938 | 0.990 | 1440 |
| `bs16_ep200` | 79/200 | 39 | 0.01774 | 0.7640 | 0.7416 | 0.945 | 1.050 | 1419 |
| `bs8` | 30/100 | 10 | 0.02039 | 0.7632 | 0.7356 | 0.948 | 1.047 | 1450 |
| `bs16_lr2e3` | 55/100 | 35 | 0.02392 | 0.7619 | 0.6822 | 0.936 | 0.952 | 1433 |
| `bs16_pat40` | 91/100 | 51 | 0.01982 | 0.7612 | 0.7601 | 0.942 | 1.001 | 1387 |
| `edgew03` | 40/100 | 20 | 0.01679 | 0.7599 | 0.7410 | 0.957 | 1.063 | 1604 |
| `fpv40` | 26/100 | 6 | 0.02084 | 0.7493 | 0.7423 | 0.942 | 0.987 | 1580 |
| `bs32` | 31/100 | 11 | 0.02040 | 0.7446 | 0.7609 | 0.922 | 0.927 | 1545 |
| `edgeevery4` | 31/100 | 11 | 0.02218 | 0.7237 | 0.7244 | 0.939 | 1.074 | 1827 |
| *reference: 0.7574 run (`100ep`)* | 50/100 | 30 | 0.01923 | -- | 0.7574 | 0.924 | 0.958 | 1328 |
| *reference: tmax30_epoch23 (Kaggle)* | 23/30 | 8 | 0.01716 | 0.8074 | 0.7972 | 0.952 | 1.055 | 1052 |

## What this says

1. **The historical config reproduces on the cluster: 0.8056 vs. 0.8074.**
   Same hardware, same bf16 autocast, same forced-MATH attention as the
   0.7574 run. So neither the L40S, nor bf16-vs-fp16, nor the SDPA backend
   change explains the regression on its own -- and the 0.8074 was not a
   lucky Kaggle draw.

2. **The batch-size hypothesis is refuted.** `ctrl_bs4` (0.7696), `bs8`
   (0.7632), `bs16_ep200`/`bs16_pat40` (0.764/0.761) and `bs32` (0.7446)
   are all within a few hundredths of each other and of the 0.7574 run.
   Batch size 4x2, 8, 16, 32 under the 100-epoch/lr 1e-3 schedule all land
   at ~0.76.

3. **The step-count hypothesis is refuted too.** `fpv40` restored 448
   steps/epoch exactly and scored 0.7493, the third-worst run; `bs16_ep200`
   ran 17.7k optimizer steps (more than any historical run) and scored
   0.7640.

4. **The variable that actually changed was the learning-rate schedule,
   and it was changed by accident.** The 0.7574 run used `--epochs 100` and
   left `--lr` at the CLI default `1e-3`. The historical best used
   `epochs=30, lr=5e-4` (set explicitly in the notebook, not the default).
   Every one of the ten ~0.76 runs shares `lr=1e-3, T_max=100`; the one
   0.80 run is the one with `lr=5e-4, T_max=30`. Whether it is the peak LR,
   the schedule length, or their interaction is not separated by this
   batch -- `bs16_pat40` did run to 91/100 and anneal fully
   (final lr 2e-5) and still scored 0.7612, so annealing alone is not the
   fix; the peak LR and/or the number of epochs spent near it matters.
   `|W|_conv` at the end is 46.6 for `replicate_tmax30` vs. 80-144 for the
   lr 1e-3 runs -- consistent with the higher LR inflating weights, but
   that is a correlation across configs, not a controlled measurement.

5. **`val_loss` checkpoint selection is the wrong selector, systematically.**
   The last-epoch checkpoint beat the `val_loss`-selected `_best` in 9 of
   11 runs, by +0.001 to +0.080 (median about +0.02); the two exceptions
   (`bs32`, `edgeevery4`) were within 0.016 and 0.001. `_best` picks
   early epochs (6-15 in eight runs) when the LR is still near its peak,
   and `val_loss` has no view of edge-linking quality. This is the
   `2026-08-31` gotcha again, now on 11 independent runs rather than one.
   Every `_best` file in this batch and the `-- use this one` message in
   `train.py`'s output are misleading for this pipeline.

6. **The edge ablations did not help.** Removing the edge scorer
   (`bs16_noedge`, distance-only linking) was the best of the lr 1e-3
   family at 0.7702 with the highest detection recall (0.960), but it
   over-detects (ratio 1.14, FP 1646) and still sits far below 0.80.
   Shrinking the edge loss (`edgew03`, 0.7599) or sampling it less
   (`edgeevery4`, 0.7237) both scored below the unablated controls. The
   edge scorer's gradient sharing the clip budget with conv is real but
   not the cause of the regression.

7. **`lr=2e-3` was actively harmful** (`_best` 0.6822, det recall 0.898),
   and `bs32` had the lowest detection recall of the batch (0.922). Larger
   batch or LR moves this model in the wrong direction on this data.

8. **FP edges track the config.** All lr 1e-3 runs produce 1387-1827 FP
   edges on the held-out split vs. 1052-1104 for the two 0.80 checkpoints.
   The regression is mostly precision on linked edges, not detection
   recall (0.94-0.96 for most runs vs. 0.95 for the best).

## Decisions

- **Canonical training config on the cluster is the historical one:**
  `--epochs 30 --lr 5e-4 --val-frames-per-volume 16 --patience 15`
  (batch 4, grad_accum 2, frames_per_volume 20 defaults). Do not use the
  `--lr` default of `1e-3` with a long schedule again without a held-out
  number to justify it.
- **Use the last-epoch checkpoint for inference, not `_best`,** until a
  selector that actually tracks the held-out edge score exists. The
  `CLAUDE.md` instruction to load `_best` was written when `_best` and
  last were assumed to agree; on this pipeline they do not.
- `dist/models/detector_replicate_tmax30.pt` (epoch 22, 0.8056) is the
  best cluster-trained checkpoint. `detector.pt` / `detector_best.pt`
  (the 0.7574 run) remain in place, unchanged, and should not be used for a
  submission.

## Open questions / next single-variable runs

**Follow-up 2026-09-10 afternoon:** `reports/2026-09-10-followup-lr-schedule-noise-floor.md` -- the run-to-run noise floor turned out to be ~0.05 (identical config: 0.8056 / 0.8007 / 0.7513), which softens points 2-4 above; the peak-LR finding (5e-4 vs 1e-3) survives it, the batch-size and step-count refutations should be read as "no effect detectable above noise."

- Separate peak LR from schedule length: `lr 5e-4, T_max 100` and
  `lr 1e-3, T_max 30`, one run each, same everything else.
- Both 0.80 checkpoints stopped via patience at epoch 22-23 of 30 with the
  `_best` at epoch 7-8 -- the schedule has never actually finished. Run the
  30-epoch config with patience disabled (or patience on a last-N-epoch
  window) and score epoch 30.
- Given (5), add a `--select-by` option that keeps the last checkpoint, or
  saves every Nth epoch, so held-out scoring can pick instead of `val_loss`.
- bf16 vs. fp16: the replicate run reproduced 0.8074 under bf16, so this
  is not urgent, but it has never been a controlled comparison.
- Whether `batch 16, lr 5e-4, T_max 30` keeps the 0.80 while also keeping
  the stability benefit the 0.7574 run showed. That is the run that would
  actually deliver what the first cluster run was for.
