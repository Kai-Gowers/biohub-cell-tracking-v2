# biohub-cell-tracking-v2

Local development for the [Biohub -- Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
Kaggle competition: detect cell centres in 3D+time light-sheet volumes and link them into lineage
tracks, submitted as a CSV of nodes and edges.

**`CLAUDE.md` is the full command reference and the architecture description** (also read by Claude
Code). This file is the short orientation.

## What this is

A faithful, attributed port of a public 0.942-leaderboard pipeline, not a home-grown architecture:

- model code and primary weights from the `biohub-tracking-support-pack` by pilkwang (RoyerLab baseline
  code), a second-seed pack, and a DeepCenter veto model;
- inference and post-processing from the public notebook `biohub-0-942-lb-proxy-score-0-9417`
  (tracked at `context/biohub-0-942-lb-proxy-score-0-9417.ipynb`; the packs themselves are ~750 MB and
  gitignored under `context/pack_*`).

Pipeline: **TemporalUNet3D + node transformer** (8-view TTA, dual-seed blend) -> **tracksdata ILP** ->
**notebook post-processing** (motion relink, gap closing, safe divisions, short-track filter, line fit)
-> one `.geff` per volume -> `submission.csv`. Every hyperparameter is a dataclass default carrying the
notebook's committed value (`PredictConfig`, `ILPConfig`, `PostprocessConfig`, `TrainConfig`); changing
one is an experiment, not a fix. The port is bit-exact against the pack's own script on two held-out
volumes (`scripts/parity/`).

The repo's earlier architecture (2-frame U-Net detector + pair MLP + greedy linker, leaderboard 0.842)
was retired on 2026-09-11; its error-analysis scripts are kept in `scripts/archive_0842_analysis/`.

## Leaderboard history

| date | submission | public LB |
|---|---|---|
| 2026-08-31 | old architecture, `detector_tmax30_epoch23` | 0.842 |
| 2026-09-17 | this pipeline, our own weights (`pack_s314159_ddp_best` + `pack_s0_ddp_best`, DeepCenter ep 25) | **0.927** |
| reference | the public notebook with its shipped weights (trained on all 199 volumes) | 0.942 |

## Setup

```bash
# cluster: the conda env at /projects/twist2d/gowers/miniconda3/envs/cell-tracking already has everything
pip install -e ".[dev]"
# ILP stack, offline from the pack's wheels (do NOT install the bundled geff wheel; it is older than the env's)
pip install --no-index --find-links context/pack_primary/wheels tracksdata ilpy pyscipopt polars polars_runtime_32 rustworkx sqlalchemy dask imagecodecs pyarrow
export POLARS_PREFER_PKG=32
```

## Workflow

```bash
python scripts/build_cache.py --cache-dir dist/cache_pack                     # decimated uint16 frame cache (~10 GB)
python scripts/train.py --epochs 400 --lr 1e-4 --batch-size 8 --seed 0 --out dist/models/pack_s0.pt   # or --ddp on 4 GPUs
python scripts/predict.py --checkpoint <ckpt> --secondary-checkpoint <ckpt2> --deepcenter <dc> \
    --test-dir data/biohub-cell-tracking-during-development/train --split dist/heldout_split.json --out-dir dist/preds_val_x
python scripts/score_local.py --geff-dir dist/preds_val_x --split dist/heldout_split.json   # read the per-embryo lines
python scripts/make_submission.py --geff-dir dist/preds --out submission.csv
python scripts/package_for_kaggle.py --primary <ckpt> --secondary <ckpt2> --deepcenter <dc>  # zip for the Kaggle Dataset
```

On Kaggle, `notebooks/kaggle_run.ipynb` (internet off, GPU) installs the ILP stack from the wheels
Dataset, verifies the model sha256s, runs predict + make_submission and audits the CSV.

## Reports

`reports/YYYY-MM-DD-topic.md`, one per tuning or investigation iteration, is the durable record of *why*
something was added, rejected or measured as inert. `git log` shows what changed; the reports hold the
evidence. Start with `reports/2026-09-17-leaderboard-0.927-and-repo-cleanup.md` (current state and next
levers), `reports/2026-09-16-ddp-score-sweep.md` (our checkpoints scored) and
`reports/2026-09-11-pack-port-parity-and-heldout.md` (the port).

## Data (~90GB, gitignored)

```text
data/biohub-cell-tracking-during-development/
  train/   # *.zarr + *.geff   (199 volumes, 100 frames of 64x256x256 uint16)
  test/    # *.zarr            (4 volumes, all present in train/ -- placeholders, not the hidden test)
```

`data` is a symlink to `/projects/twist2d/gowers/biohub-cell-tracking-v2-data/data`; `TRAIN_DIR` /
`TEST_DIR` / `COMP_DATA_DIR` / `CACHE_DIR` override the paths. All 199 training crops come from two
embryos (`44b6`: 71 crops, `6bba`: 128); the hidden test embryos are different ones, so report
per-embryo numbers and treat the leaderboard as the only true cross-embryo signal. Our held-out split
is `dist/heldout_split.json` (5 x `44b6`, 15 x `6bba`).

## Submission format

CSV with index `id` and columns
`dataset`, `row_type` (`node`/`edge`), `node_id`, `t`, `z`, `y`, `x`, `source_id`, `target_id`.
Node ids restart at 1 per dataset; unused fields are `-1`.
