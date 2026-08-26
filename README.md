# biohub-cell-tracking-v2

Local development for the [Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
Kaggle competition.

This started as a **from-scratch, deliberately simple baseline** (a rewrite
of [biohub-cell-tracking](../biohub-cell-tracking)): a single-frame 3D U-Net
detector, followed by distance-gated frame-to-frame linking solved exactly
with a scipy bipartite assignment.

It has since grown a learned edge model, a 2-frame temporal window with
cross-frame attention, augmentation + TTA, and greedy (not exact) linking --
see `reports/2026-08-25-sample-solution-0.90-comparison.md` and
`reports/2026-08-25-four-sample-solution-features.md`. There is still no ILP,
no divisions, and no repair passes.

**detector** (windowed U-Net + attention) → **edge scorer** → **greedy link**
→ `.geff` → `submission.csv`.

## Philosophy

Start as narrow as a real, submittable pipeline can be, then add complexity
against a held-out score (or a clear qualitative failure), not from memory of
what worked elsewhere. Every addition gets a report in `reports/` recording
the before/after evidence, so the history of *why* something was added
survives even if it later gets reverted -- the four features above were added
on direct request rather than in that order, and their reports say so; a
held-out before/after comparison for each is still open, see that report's
"Status" section.

The previous repo (`biohub-cell-tracking`) accumulated a joint detector+edge
transformer, an ILP-flavoured selector, and five deterministic repair passes
-- and several of the additions along the way (a hardest-negative margin
loss, gap2 repair, division recovery) measured as inert or actively harmful.
This repo started from the other end, and re-additions here are meant to be
evidence-driven even when the *decision* to try them wasn't.

## Workflow

1. Build the prepared-frame cache once: `python scripts/build_cache.py`.
2. Train: `python scripts/train.py --epochs 20` -- or
   [`notebooks/kaggle_train.ipynb`](notebooks/kaggle_train.ipynb) on a GPU.
   Checkpoints every epoch and resumes, so a session timeout is not fatal.
3. Package: `python scripts/package_for_kaggle.py` (code + `detector.pt` +
   `ARTIFACT_MANIFEST.json` recording the checkpoint's sha256).
4. Predict + submit, via [`notebooks/kaggle_run.ipynb`](notebooks/kaggle_run.ipynb) or:
   ```bash
   python scripts/predict.py --out-dir dist/preds
   python scripts/make_submission.py --geff-dir dist/preds --out submission.csv
   ```

There is **no untrained fallback**: without `detector.pt`, inference raises.

## Setup

```bash
cd /path/to/biohub-cell-tracking-v2
source cell-tracking-env/bin/activate
pip install -e ".[dev]"
```

## Data (~90GB)

```text
data/biohub-cell-tracking-during-development/
  train/   # *.zarr + *.geff   (199 volumes, 100 frames of 64x256x256 uint16)
  test/    # *.zarr            (4 volumes, all present in train/)
```

Same competition data as `biohub-cell-tracking` -- point `TRAIN_DIR` /
`TEST_DIR` / `COMP_DATA_DIR` at that repo's `data/` (or symlink it in) rather
than re-downloading 90GB. `CACHE_DIR` / `MODEL_DIR` likewise; the prepared-frame
cache format is identical, so it can be reused directly.

## Package layout

| Path | Role |
|------|------|
| `config.py` | Paths + every hyperparameter |
| `io_zarr.py` / `io_geff.py` | Raw volumes / GEFF graphs (read + write) |
| `preprocess.py` / `cache.py` | Raw → model grid, and the prepared-frame cache |
| `models/detector.py` | 2-frame windowed 3D U-Net, cross-frame attention per encoder stage |
| `models/edge_model.py` | Learned edge scorer over sampled node features |
| `augment.py` | Training-time y/x flip + brightness jitter |
| `losses.py` | Weighted detection BCE + weighted edge BCE |
| `train.py` / `edge_train.py` | Joint detector + edge-scorer training loop, one checkpoint |
| `peaks.py` / `detect.py` | Local maxima; τ = 0.985 detections; flip TTA |
| `link.py` | Distance-gated, greedy score-sorted linking between consecutive frames |
| `graph.py` | `TrackGraph` + the dt=1 and in-degree invariants |
| `predict.py` / `submit.py` | Per-volume `.geff`, then streamed CSV |
| `metric.py` | Local implementation of the competition metric |

## Key design points

Geometry is µm everywhere after prediction (`SCALE` = 1.625 / 0.40625 /
0.40625 µm per voxel, z vs xy differ ~4x). The model grid downsamples **xy
only**, giving an isotropic 1.625 µm grid -- a uniform 4× downsample would
make z voxels 6.5 µm and consume half the metric's 7 µm match tolerance.

Detection targets are **binary** (GT node voxels positive, everything else a
weak negative at α = 0.01). Only ~2.8% of real cells are annotated, so the
negative pool is mostly unlabelled true cells; α is what keeps them from
dominating, and it is what makes the sigmoid saturate so τ = 0.985 selects
peaks rather than nothing.

Linking uses a **learned edge score** when a checkpoint has one (candidate
pairs gated at `LINK_RADIUS_UM`, scored by `EdgeScorer`, kept above
`LINK_SCORE_THRESHOLD`), falling back to negative distance otherwise.
Selection is plain greedy score-sorted thresholding, not an exact
assignment -- the 0.90 sample solution also defaults to greedy (its ILP
solver ships but is off). In-degree and out-degree are both ≤ 1 -- no
division support yet.

There is **no repair stage**. Gap-bridging, motion relinking, and
coordinate smoothing all measurably changed the score in the previous repo
(for better and for worse) -- add any of them back only against a held-out
number here, not from memory of what worked before.

## Scoring locally

```bash
python scripts/score_local.py --geff-dir dist/preds --held-out dist/models/detector.pt
```

Read the caveats at the top of `src/cell_tracking/metric.py` first: ground
truth is sparse, so over-linking is close to invisible locally, and
cross-checkpoint score comparisons are not reliable -- trust same-checkpoint,
different-config comparisons instead.

## Submission format

CSV with index `id` and columns
`dataset`, `row_type` (`node`/`edge`), `node_id`, `t`, `z`, `y`, `x`, `source_id`, `target_id`.
Node ids restart at 1 per dataset; unused fields are `-1`.
