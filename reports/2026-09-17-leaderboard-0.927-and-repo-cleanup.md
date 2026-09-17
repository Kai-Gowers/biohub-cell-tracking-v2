# 2026-09-17 -- Leaderboard 0.927 with our own weights; repo cleanup and merge into `main`

## The submission

Kaggle notebook `kaigow/kaggle-run-0942` v1 (submission 56288258, run on 2026-09-16 22:31, hidden-test
rerun finished overnight): the full pipeline of this branch with our own checkpoints,

| role | file | note |
|---|---|---|
| primary | `dist/models/pack_s314159_ddp_best.pt` | 4-GPU DDP run of the pack recipe, 179 train volumes, best `val_score` @ epoch 385 |
| secondary | `dist/models/pack_s0_ddp_best.pt` | second seed, best @ epoch 345 |
| DeepCenter | `dist/models/deepcenter/best.pt` | ours, epoch 25 (`--no-deepcenter-epoch-check`) |

Package: `dist/cell_tracking_0942_ddp_best.zip` (81 MB), notebook `notebooks/kaggle_run.ipynb` as committed
in 6210a54. Runtime on Kaggle: 18.8 min for the 4 placeholder volumes, i.e. ~3-8 min per volume.

**Public leaderboard: 0.927.**

| | score |
|---|---|
| previous best, old architecture (2026-08-31) | 0.842 |
| this submission | **0.927** |
| the public notebook with its shipped weights (trained on all 199 volumes) | 0.942 |
| same blend on our clean 20-volume held-out (`reports/2026-09-16-ddp-score-sweep.md`) | 0.8952 (44b6 0.793 / 6bba 0.909) |

## Reading it

1. **Leaderboard > local by 0.03.** The hidden embryos are kinder than our 44b6-weighted held-out set;
   the local number was conservative, not misleading. Same direction as every LB/local pair so far.
2. **The 0.015 gap to the reference is the one deliberate difference**: the shipped weights trained on all
   199 volumes, ours on 179. Locally the whole gap to the shipped weights was detection recall
   (96.7 vs 98.3 %), not linking.
3. Blending two of our seeds added nothing locally (single s314159 0.8976 >= blend 0.8952); it was
   submitted because the notebook does it and it is cheap.

## Next levers, in order

1. Train on all 199 volumes with the identical recipe, two seeds; pick a fixed late epoch (our sweep:
   `_best` beat `_last` by ~0.01 and ep 150 == ep 385, so the plateau is flat). Two days of DDP per seed.
2. Detection-threshold sweep on the held-out set (the gap is missed cells).
3. Motion-aware relink for the whole-frame z-jumps (`reports/2026-09-14-jump-frames-postprocessing.md`;
   -0.078 on `44b6_d754aa59` locally).

## Repo cleanup (same day)

Decided with the user: `replicate-0942` is the only pipeline we iterate on, so `main` was fast-forwarded to
it (it was a strict ancestor: no conflicts, no code change) and the branch and the `main` worktree on
`/projects` were removed. Now tracked, so teammates can read them: `reports/` (with figures), `CLAUDE.md`,
`dist/heldout_split.json`, `docs/pipeline-0942.html`.

Removed from the tree (all verified to have zero importers; none is in the import closure of
`scripts/predict.py` or `scripts/make_submission.py`, which is exactly what the Kaggle notebook runs):

- files: `context/biohub_cell_tracking_deep_critique.md`, `context/clean_approach_lightweight_local_cv_context.md`
  (both describe the abandoned architecture / a different notebook), `scripts/download_data.py`,
  `notebooks/kaggle_build_wheels.ipynb`, `requirements.txt`, the old-architecture `docs/pipeline-diagram.html`;
- code: the `detector.pt` checkpoint-discovery block and unused unit helpers in `config.py`, three unused
  `GeffGraph` accessors in `io_geff.py`, `TrackGraph.nodes_at`; `pandas` dropped from `pyproject.toml`
  (imported nowhere), `polars_runtime_32` added to the `ilp` extra.

Not touched: `pack_predict.py`, `ilp.py`, `postprocess.py`, `pipeline.py`, `preprocess.py`, `submit.py`,
`io_zarr.py`, `models/*`. `analysis.py`'s `--candidates` path stays (no producer yet, but interleaved with the
live error decomposition). The old pipeline's error-analysis scripts are archived in
`scripts/archive_0842_analysis/` with a README; the three reports that cite them are unchanged.

Regression gate before the fast-forward (results appended below): the submitted configuration rerun on one
held-out volume must match the 2026-09-16 output bit-exactly, and a rebuilt Kaggle package must differ from
the submitted zip only in the removed / archived scripts.

Disk: `dist/` on `/home` (92 % full) lost the old-architecture checkpoints (`detector_*`, `smoke_*`) and
prediction folders; list in `dist/logs/cleanup-2026-09-17-deleted.txt`.

## Regression gate results

(filled in below once run)
