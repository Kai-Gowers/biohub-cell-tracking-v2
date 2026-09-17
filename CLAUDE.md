# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local development environment for the [Biohub -- Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) Kaggle competition: detect cell centers in 3D+time microscopy volumes and link them into lineage tracks, output as a submission CSV of nodes/edges.

**Branch `replicate-0942` is a faithful port of a public 0.942-LB pipeline**, not a home-grown architecture. `main` (held-out ~0.80) carried this repo's own 2-frame U-Net + pair-MLP + greedy linker; that was dropped here on 2026-09-11 in favour of replicating, near-verbatim and with attribution, two public artifacts the user copied into `context/` (gitignored, ~750 MB):

| folder | what it is |
|---|---|
| `context/pack_primary/` | `biohub-tracking-support-pack-400ep-snapshot-v1` (pilkwang; RoyerLab baseline code): `repo/` source, primary weights `weights/unet_transformer/split_0/edge_predictor_best.pth` + `config.json`, offline `wheels/` |
| `context/pack_secondary/` | `biohub-temporal-unet3d-seed314159-v1`: identical code, second-seed weights, `training_config.json` (the recipe), 400-epoch `history.csv` |
| `context/pack_deepcenter/` | `biohub-deepcenter-unet3d-center-prior-v1`: DeepCenter veto model `best.pt` (epoch 2) + its trainer |
| `context/biohub-0-942-lb-proxy-score-0-9417.ipynb` | the notebook (tracked): cell 8 = env vars, cell 12 = constants, cell 16 = six inference patches applied to the pack script, cell 18 = post-processing |

The 0.908 "clean approach" markdown in `context/` documents a *different* notebook; its numbers (0.96875, 1.575, 6.0/9.5 µm, 4.66/8.5 µm) are not used anywhere here.

**Pipeline: TemporalUNet3D + node transformer (`pack_predict.py`: 8-view TTA, edge-feature TTA, dual-seed blend + retention guard, bidirectional harmonic fusion, low-margin consensus) -> tracksdata ILP (`ilp.py`) -> notebook post-processing (`postprocess.py`: motion relink, gap-1/gap-2 closing, safe divisions, short-track filter + rescue, line-fit) -> one `.geff` per volume -> `submission.csv`.**

Philosophy on this branch: replicate exactly first, measure per embryo, then deviate only on held-out evidence written to `reports/`. Every hyperparameter is a dataclass default whose value is the notebook's committed value: `pack_predict.PredictConfig`, `ilp.ILPConfig`, `postprocess.PostprocessConfig`, `pack_train.TrainConfig`, `deepcenter_train.DeepCenterTrainConfig`. If you change one, it is an experiment, not a fix.

**Parity is the contract.** `scripts/parity/build_ref_oracle.py` reruns the pack's own script (unpatched = "vanilla", or with the notebook's cell-16 patches = "patched") on two held-out volumes into `dist/ref_pack/`; `scripts/parity/parity_check.py` compares our `--stage ilp` output and our post-processing against it. On 2026-09-11 all three were bit-exact (identical node sets, edge sets and `edge_prob`). Re-run this after touching anything in `pack_predict.py`, `ilp.py` or `postprocess.py`.

## Reports

Write a report for any tuning/investigation iteration (smoke test results, config changes, a newly added pipeline stage) to `reports/`, one markdown file per iteration, named `reports/YYYY-MM-DD-short-topic.md`. This is the durable record of *why* something was added or rejected -- `git log`/`git blame` show *what* changed, not the evidence behind it. Before adding complexity (a learned edge score, divisions, any repair pass), write down the held-out score with and without it, even if the answer is "no measurable effect" -- that negative result is exactly what keeps this repo from re-accumulating the previous one's dead weight.

## Setup

```bash
cd /path/to/biohub-cell-tracking-v2
# cluster: the conda env at /projects/twist2d/gowers/miniconda3/envs/cell-tracking already has everything
pip install -e ".[dev]"
# ILP stack, offline from the pack's wheels (do NOT install the bundled geff wheel; it is older than the env's)
pip install --no-index --find-links context/pack_primary/wheels tracksdata ilpy pyscipopt polars polars_runtime_32 rustworkx sqlalchemy dask imagecodecs pyarrow
export POLARS_PREFER_PKG=32
```

## Common commands

```bash
# Decimated-frame cache for training (uint16, ~10 GB; dist/cache_pack -> /projects)
python scripts/build_cache.py --cache-dir dist/cache_pack

# Predict with the public weights (full pipeline) on the held-out split, then score per embryo
P=context/pack_primary/weights/unet_transformer/split_0/edge_predictor_best.pth
S=context/pack_secondary/weights/unet_transformer/split_0/edge_predictor_best.pth
DC=context/pack_deepcenter/weights/full_frame_center/best.pt
python scripts/predict.py --checkpoint $P --secondary-checkpoint $S --deepcenter $DC \
    --test-dir data/biohub-cell-tracking-during-development/train --split dist/heldout_split.json --out-dir dist/preds_val_x
python scripts/score_local.py --geff-dir dist/preds_val_x --split dist/heldout_split.json
# ablations: --stage ilp (no post-processing) | --no-ilp | --no-bidir | --no-edge-tta | --tta-views flip |
#            --post-off output_gap2_recovery (any PostprocessConfig boolean) | drop --secondary-checkpoint / --deepcenter

# Train the pack recipe on our held-out split (resumable; val_score selects _best)
python scripts/train.py --epochs 400 --lr 1e-4 --batch-size 8 --seed 0 --out dist/models/pack_s0.pt
sbatch --partition=medium --time=2-00:00:00 scripts/slurm_train.sbatch --epochs 400 --seed 0 --max-hours 47 --out dist/models/pack_s0.pt   # self-chains
python scripts/train_deepcenter.py --out-dir dist/models/deepcenter --epochs 50

# Score checkpoints (our checkpoints carry val_names; extra args go to predict.py)
scripts/score_one.sh <tag> <checkpoint> --secondary-checkpoint <ckpt2> --deepcenter <dc>
sbatch scripts/slurm_score_sweep.sbatch pack_s0 --deepcenter dist/models/deepcenter/best.pt
srun --partition=interactivegpu --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --time=01:00:00 scripts/slurm_gpu_batch.sh <taskfile>

# Parity against the reference pack (GPU)
python scripts/parity/build_ref_oracle.py --mode patched --stems 44b6_1574802b 6bba_2312ac41 --work-dir /projects/.../scratch/oracle --out-dir dist/ref_pack/patched
python scripts/predict.py --checkpoint $P --secondary-checkpoint $S --stage ilp --volume 44b6_1574802b --volume 6bba_2312ac41 --test-dir .../train --out-dir dist/parity/ilp_patched
python scripts/parity/parity_check.py --ours dist/parity/ilp_patched --oracle dist/ref_pack/patched
python scripts/parity/parity_check.py --mode postprocess --oracle dist/ref_pack/patched --notebook-module scripts/parity/notebook_postprocess_oracle.py --deepcenter $DC

# Submission: convert, then package code + models for a Kaggle Dataset
python scripts/make_submission.py --geff-dir dist/preds --out submission.csv
python scripts/package_for_kaggle.py --primary <ckpt> --secondary <ckpt2> --deepcenter <dc>   # defaults: the public weights
```

On Kaggle, `notebooks/kaggle_run.ipynb` (internet off, GPU) installs the ILP stack from an attached `biohub-tracking-wheels` Dataset (= `context/pack_primary/wheels/`), verifies every model file's sha256 against `ARTIFACT_MANIFEST.json`, runs `scripts/predict.py` -> `make_submission.py`, and audits the CSV (contiguous ids, dt=1, in-degree <= 1, out-degree <= 2, no negative coordinates).

## Cluster notes

- Compute nodes cannot see the session's `/tmp`; put job scripts under `dist/` or `/projects/...`, and use `srun --ntasks=1` on `interactivegpu` (1 h, one running job per user, L40S x4). `short`/`medium`/`long` (12 h / 2 d / 5 d) take `sbatch`; `slurm_train.sbatch` resubmits itself with `afterany` until `--epochs` is reached, so pass `--max-hours` below the wall limit.
- The pack's per-voxel temporal attention flattens B*Z*Y*X rows (~262k) into the batch; fused SDPA kernels crash on L40S/torch 2.6, so `models/temporal_unet.py` forces the MATH backend (reports/2026-09-09-sdpa-math-backend-l40s.md). The reference oracle runs under the same backend.
- Installing the wheels downgraded numpy 2.5.3 -> 2.4.6 (numba's pin). torch 2.6 is fine with it.

## Data layout

Competition data (~90GB, gitignored) at `data/biohub-cell-tracking-during-development/{train,test}` (symlink to `/projects/twist2d/gowers/biohub-cell-tracking-v2-data/data/...`). `train/` has 199 `<name>.zarr` + `<name>.geff`; `test/` has 4 `.zarr` that are also in `train/` (placeholders, not the hidden test). `config.get_train_dir()` / `get_test_dir()` / `get_cache_dir()` resolve env vars (`TRAIN_DIR` / `TEST_DIR` / `COMP_DATA_DIR` / `CACHE_DIR`), then Kaggle, then the local paths.

The zarr root attrs carry `image_statistics.quantiles`; the pack normalises each *video* with its own `0.001` / `0.999` quantiles (`preprocess.pack_normalize`) and decimates `raw[::1, ::4, ::4]` -- no mean-pooling, no per-frame percentiles. Both training and inference go through `preprocess.prepare_frame`; `cache.py` stores the decimated raw uint16 frames and normalises on read, so cache and zarr are interchangeable.

Zarr volumes are read frame by frame as raw blosc2 chunks (`io_zarr.py`). GEFF graphs (`io_geff.py`) round-trip an optional `edge_prob` edge property, which the post-processing needs for its motion-relink bonus.

## Architecture (what the port contains)

- `models/temporal_unet.py` -- `TemporalUNet3D(in=1, out=32, layers=(32,64,128))`: Conv3d/BatchNorm/ReLU blocks, per-voxel multi-head attention across the 2-frame window at the two coarser stages, 32 feature channels for both frames.
- `models/node_transformer.py` -- `SimpleNodeTransformer`: 4 alternating cross-attention blocks (t->t+1, t+1->t), hidden 128, pair MLP on `[h_src, h_tgt, (coord_src - coord_tgt)/100]` in original-voxel units -> logit matrix.
- `models/unet_node_transformer.py` -- `UNetNodeTransformer` (`encode`, `_index_features` = integer gather, `predict_edges`), sinusoidal position features, `load_pack_model` (accepts the pack's bare state dicts and our training checkpoints).
- `models/deepcenter.py` -- DeepCenter GroupNorm/SiLU U-Net on the 4x mean-pooled frame; used only as an add-only gate on synthetic gap nodes with span >= 8.5 µm (threshold 0.25).
- `pack_predict.py` -- `predict_video`: windows of stride 1, `sigmoid > 0.965` peaks with a (3,3,3) max-pool on the 1.625 µm grid, softmax over *sources* per target, candidates above 0.48 (dual seed) / 0.5 (single). Coordinates are `grid * 4` (no half-cell offset), faithful to the pack.
- `ilp.py` -- `tracksdata.solvers.ILPSolver(edge_weight=-1.0*edge_prob, appearance=0.0, disappearance=2.0, division=1.2)`. Its edges are replaced by the motion relink downstream; what survives is the node set and the `edge_prob` table.
- `postprocess.py` -- cell 18 in the notebook's order; see the module docstring for every constant.
- `pipeline.py` -- `run_volume(models, zarr, stage="raw"|"ilp"|"full")`, all in memory, `TrackGraph.from_arrays` keeps post-processing's node ids.
- `pack_train.py` -- the recipe: AdamW 1e-4 (default weight decay), no scheduler, clip 1.0, batch 8 full-frame windows, brightness shift +-0.1 and independent z/y/x flips, `loss = edge + 1.0 * det`; detection loss = one-hot weighted BCE (`w_pos = 1/n_pos`, `w_neg = 0.01/n_neg`); edge targets from detect-and-match on the live logits (logit > 0.3, 5 µm match), focal softmax-over-sources BCE on annotated rows/columns. Additions: resume, `--seed`, `--save-every`, `--max-hours`, and `val_score` (competition metric via `pipeline.run_volume(stage="ilp")`, no TTA) every `--eval-tracking-every` epochs for `_best` selection. The pack's `acc x recall` validation is logged as `pack_val_*` only.

## Training protocol

Train on 179 volumes, hold out `dist/heldout_split.json` (5 x `44b6`, 15 x `6bba`); the pack's own "all 199 train, validate on 40 of them" recipe is deliberately not used. Checkpoints: `<out>.pt` (last, resumable), `<out>_best.pt` (best `val_score`), `<out>_epochN.pt`. Their history: 1200 s/epoch on an RTX 5090, best epoch 381 of 400; the shipped weights were trained on all 199 volumes, so any held-out number computed with them is a train-set number (0.9175 total / 0.8314 `44b6` / 0.9305 `6bba` on 2026-09-11, vs 0.8056 for `main`'s best).

## Reading local scores

`metric.py` implements the competition metric (`edge Jaccard + 0.1 * division Jaccard`, 7 µm node matching) and is copied verbatim from the sibling repo -- it is the competition's metric, not a pipeline choice. Read its module docstring before trusting a number out of it:

- Ground truth is sparse, so ~98.5% of predicted edges touch no annotated node and are *ignored* rather than counted -- over-linking is close to invisible locally.
- Per-volume weights track annotation density, not volume size.
- **Cross-checkpoint score comparisons were anti-correlated with the leaderboard in the sibling repo** (r = -0.799 over five submissions). Trust same-checkpoint, different-config comparisons; distrust cross-checkpoint ones, including comparisons back to that repo's numbers.

The reliable use is a held-out set the model never trained on: `scripts/score_local.py --held-out <checkpoint>`. It applies the official node-count adjustment (`metric.adjusted_jaccard`) and the official per-sample weighting, and prints **per-embryo subtotals** -- read those, not just the total. `scripts/analyze_errors.py` (same inputs) splits every counted edge error by cause (endpoint undetected vs. wrong link vs. unlinked), by embryo and by crowding, and with a `--dump-candidates` prediction also reports true-partner rank and greedy-vs-exact-assignment regret.

## Embryos

Dataset names are `{embryo_id}_{field_of_view}`; Kaggle states train and test are **embryo-disjoint**. All 199 training crops come from just two embryos (`44b6`: 71 crops, `6bba`: 128), and every checkpoint scored so far does about 0.2 worse on `44b6` (≈0.62) than on `6bba` (≈0.83); the aggregate ≈0.80 is 88% weighted toward `6bba`. `44b6` is also the more crowded embryo, and association recall collapses in crowded regions (75% at <7 µm nearest-neighbour vs 99% at >14 µm), so the embryo gap and the association problem are largely the same problem. The visible `test/` folder holds four `44b6`/`6bba` crops that are also in `train/` -- placeholders for notebook debugging, not the hidden test embryos. Report per-embryo numbers always; use `--val-embryo` for a cross-embryo estimate; treat the leaderboard as the only true cross-embryo signal.
