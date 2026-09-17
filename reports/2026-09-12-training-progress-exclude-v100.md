# 2026-09-12 -- Pack-recipe training at epoch ~50: progress check, V100 nodes excluded from the chain

Follow-up to `reports/2026-09-11-pack-port-parity-and-heldout.md` section 5
(training launched 2026-09-11 15:08). Two seeds of the pack recipe
(`scripts/train.py --epochs 400 --lr 1e-4 --batch-size 8 --eval-tracking-every 5
--save-every 25 --max-hours 11.4`, 179 train / 20 held-out volumes) are
running as self-chaining 12 h jobs on `short`. No hyperparameter changed; the
only edit in this iteration is a scheduling constraint.

## 1. State at 21:30 on 2026-09-12 (third chunk of each chain)

| seed | jobs so far | current node / GPU | epoch | s / epoch | best `val_score` (epoch) | `44b6` / `6bba` at best |
|---|---|---|---|---|---|---|
| 0 | 2998081 (g012 L40S) -> 2998574 (g002 V100) -> 2998940 (g003 A100) | g003 A100-40GB | 55 / 400 | ~1520 | **0.8522** (55) | 0.7218 / 0.8682 |
| 314159 | 2998082 (g013 L40S) -> 2998581 (g002 V100) -> 2998951 (g002 V100) | g002 V100-16GB | 48 / 400 | ~3020 | 0.8257 (45) | 0.6782 / 0.8438 |

Every chunk resumed from `<out>.pt` at the right epoch; each chain resubmitted
itself with `afterany` as designed. Losses fall smoothly (seed 0: `det`
0.0126 -> 0.0057, `edge` 0.0004 -> 0.0003 between epochs 5 and 55), clipped
gradient norms show no sustained spikes, and the pack's own `acc x recall`
validation sits at 0.92-0.94 for both seeds.

`val_score` is the competition metric via `pipeline.run_volume(stage="ilp")`
on the 20 held-out volumes: single seed, no TTA, no post-processing. It is a
lower bound on what the full pipeline would score with the same weights
(on the public weights the ILP-stage-only number was ~0.06 below the full
pipeline). Every 5 epochs, seed 0:

| epoch | 5 | 10 | 15 | 20 | 25 | 30 | 35 | 40 | 45 | 50 | 55 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| total | 0.803 | 0.812 | 0.823 | 0.818 | 0.836 | 0.842 | 0.833 | 0.840 | 0.846 | 0.842 | 0.852 |
| `44b6` | 0.682 | 0.687 | 0.682 | 0.678 | 0.673 | 0.706 | 0.700 | 0.717 | 0.740 | 0.713 | 0.722 |
| `6bba` | 0.819 | 0.827 | 0.841 | 0.836 | 0.856 | 0.859 | 0.850 | 0.855 | 0.859 | 0.858 | 0.868 |
| det recall | 0.960 | 0.950 | 0.956 | 0.947 | 0.958 | 0.960 | 0.953 | 0.953 | 0.966 | 0.959 | 0.958 |
| node ratio | 1.23 | 1.16 | 1.14 | 1.11 | 1.13 | 1.13 | 1.11 | 1.07 | 1.13 | 1.12 | 1.05 |

Seed 314159 follows the same shape about 0.025 lower (0.776 at epoch 5, 0.819
at 25, 0.826 at 45). Both curves are still rising at ~+0.01 per 25 epochs;
the pack's own history peaked at epoch 381, so the epoch-100 decision point
from the previous report stands. Note that seed 0 at epoch 55, ILP stage
only, is already above `main`'s best full-pipeline held-out number (0.8056),
though that comparison crosses checkpoints and architectures and is not to
be trusted beyond "the recipe trains".

## 2. The problem: heterogeneous GPUs on `short`

`sinfo` shows the `short`/`medium`/`long` pool is mixed: g001-g002 V100 x4
(16 GB), g003-g009 A100 x4, g010-g018 L40S x4, gb001 H200 x8. The sbatch
script asked for a generic `--gres=gpu:1`, so every resubmission was a
lottery. Measured throughput for this recipe (batch 8, `--num-workers 6`):

| GPU | s / epoch | epochs per 11.4 h chunk | 400 epochs |
|---|---|---|---|
| L40S (g012, g013) | ~1500 | ~27 | ~6.3 days |
| A100-40GB (g003) | ~1520 | ~27 | ~6.3 days |
| V100-16GB (g002) | ~3020 | ~13 | ~12.5 days |

Seed 314159 drew g002 twice in a row (chunks 2 and 3) and seed 0 once
(chunk 2, epochs 26-40), which is why the two seeds are 7 epochs apart after
identical wall time and why seed 314159 is on track for ~12 days instead of
~6. The `val_score` evaluation is also 1.7x slower there (~500 s vs ~300 s,
every 5 epochs).

## 3. Change: exclude the V100 nodes

`scripts/slurm_train.sbatch` now carries `#SBATCH --exclude=g001,g002`
(g001 is down anyway). The running jobs execute SLURM's stored copy of the
script and are unaffected; the self-resubmit line reads the script from disk,
so the next chunk of each chain (due 2026-09-13 ~02:00 and ~02:30) will be
the first to honour it. Expected effect: both seeds finish in ~6 days from
launch (around 2026-09-17/18) instead of ~6 and ~12. This is a scheduling
change only; it does not alter numerics, seeds, or the resume path.

Not done, deliberately: pinning a GPU type (`--gres=gpu:a100:1` or
`gpu:l40s:1`) would narrow the pool to 3 or 9 nodes and risk queueing behind
the large `medium` backlog visible in `squeue`; exclusion keeps every fast
node eligible.

## 4. Next

- 2026-09-13: confirm the chunk-4 jobs landed off g002 (`squeue -o "%N"`).
- Epoch 100 (~2026-09-14 for seed 0): run the epoch-25/50/75/100 snapshots
  through the full pipeline with `scripts/score_one.sh` per embryo, as the
  previous report planned, and decide whether to let the chain run to 400.
