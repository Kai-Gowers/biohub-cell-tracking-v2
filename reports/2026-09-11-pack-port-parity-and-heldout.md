# 2026-09-11 -- Branch `replicate-0942`: porting the public 0.942 pipeline, parity, first held-out numbers

Follows the approved plan `~/.claude/plans/i-want-you-to-serene-hare.md`. The user
asked to drop this repo's own architecture and replicate, as faithfully as
possible, the pipeline behind `context/biohub-0-942-lb-proxy-score-0-9417.ipynb`
(public LB 0.942). This report records what was ported, how faithfulness was
verified, and the first per-embryo numbers on our 20-volume held-out split.

## 1. What the 0.942 pipeline actually is

The notebook is a driver. Its model code lives in an attached Kaggle artifact
(pilkwang's `biohub-tracking-support-pack`, a repackaging of the RoyerLab
baseline) that the user copied into `context/pack_primary` (primary weights +
source + offline wheels), `context/pack_secondary` (a second seed, 314159,
with the training recipe and its 400-epoch history) and
`context/pack_deepcenter` (the DeepCenter veto model, epoch 2 of 500).

| stage | where it came from | key values |
|---|---|---|
| preprocessing | pack | per-*video* quantiles 0.001/0.999 from the zarr attrs, clamp at 0; strided decimation (1,4,4) |
| detector + edge model | pack | `TemporalUNet3D(32,64,128)` with per-voxel temporal attention; `SimpleNodeTransformer` (4 cross-attention blocks, hidden 128); integer feature gather + 32-d sinusoidal position |
| detection | pack + notebook patch 1 | sigmoid > 0.965, (3,3,3) max-pool NMS, **8-view planar D4 TTA** (z never flipped) |
| dual seed | notebook patches 2, 3, 6 | secondary logits moment-matched, blend 0.2/0.8, retention guard 0.90; secondary edge-feature TTA 0.75 |
| edge fusion | notebook patches 4, 5 | edge-feature TTA (mean of 8 views); bidirectional harmonic fusion w=0.15; low-margin consensus 0.20 / 0.35; candidates softmax-over-sources > 0.48 |
| graph selection | pack `--use-ilp` | tracksdata ILP (SCIP via ilpy): edge -1.0*prob, appearance 0.0, disappearance 2.0, division 1.2 |
| post-processing | notebook cell 18 | motion relink (tight 5.5 / relaxed 10 µm, learned bonus 1.0), gap-1 (5.8 µm, density-adaptive, intensity-refined synthetic node, DeepCenter gate at span >= 8.5 µm / 0.25), gap-2 (10.2 / 4.4 µm), safe divisions (9.0 / 14.0 µm, divergence 2.25, symmetry 0.6), short-track filter (6) + rescue (len 4-5, prob >= 0.88, dist <= 3.0), line-fit (window 2, weight 0.8) |

Two facts that matter for reading numbers: (a) the ILP's *edges* are discarded
by the motion relink; the ILP contributes the node set and the `edge_prob`
lookup only. (b) The shipped weights were trained on **all 199** training
volumes (`split_manifest.train` contains every held-out stem), so every number
computed with them on our held-out split is a train-set number.

The 0.908 "clean approach" markdown in `context/` describes a different
notebook; its thresholds (0.96875, 1.575, 6.0/9.5 µm, 4.66/8.5 µm) were not used.

## 2. The port

`src/cell_tracking/`: `models/{temporal_unet,node_transformer,unet_node_transformer,deepcenter}.py`,
`pack_predict.py`, `ilp.py`, `postprocess.py`, `pipeline.py`, `pack_train.py`,
`deepcenter_train.py`, plus `preprocess.py`/`cache.py` rewritten for the
pack's normalisation. Removed: the old detector, edge MLP, greedy linker,
trainer. Every notebook constant is a dataclass default (`PredictConfig`,
`ILPConfig`, `PostprocessConfig`, `TrainConfig`, `DeepCenterTrainConfig`).

Environment: tracksdata 0.1.0rc6, ilpy 0.6.0, pyscipopt 6.2.1, polars 1.42,
rustworkx 0.18 installed offline from the pack's wheels into the cluster conda
env. numpy was downgraded 2.5.3 -> 2.4.6 by numba's pin; torch 2.6 unaffected.
The bundled geff 1.2 wheel was skipped (env has 1.3.1; tracksdata round-trips fine).

## 3. Parity against the reference code (the fidelity check)

`scripts/parity/build_ref_oracle.py` copies the pack's `repo/`, optionally
applies the notebook's six cell-16 `str.replace` patches *verbatim* (executing
the cell's own patch code), sets the cell-8/14 environment, and runs the pack's
`predict_unet_transformer.py` on two held-out volumes (`44b6_1574802b`,
`6bba_2312ac41`) under the MATH SDPA backend (the fused kernels crash on L40S
at the attention's 262k-row batch, as in `reports/2026-09-09-sdpa-math-backend-l40s.md`).
`scripts/parity/notebook_postprocess_oracle.py` executes cells 8, 12 and 18
as-is and runs the notebook's `filter_output_graph`.

| comparison | 44b6_1574802b | 6bba_2312ac41 |
|---|---|---|
| our `--stage ilp` (single seed, 4-view flip TTA) vs pack unpatched | 19310 nodes identical, 18176/18176 edges, edge_prob max diff 0.0 | 12376 identical, 11871/11871, 0.0 |
| our `--stage ilp` (dual seed, D4, bidir, edge TTA) vs pack + notebook patches | 19790 identical, 18778/18778, 0.0 | 12538 identical, 12084/12084, 0.0 |
| our `postprocess.filter_output_graph` vs notebook cell 18 (with DeepCenter) | 19645 identical, 19016/19016, 0.0 | 12383 identical, 12055/12055, 0.0 |

**Bit-exact on all three.** The port is the notebook's pipeline, not an
approximation of it. Runtime on one L40S: ~25 s per 100-frame volume for
dual-seed inference + ILP, ~35 s with post-processing (the pack's 2-GPU
Kaggle run took 32 min for 4 volumes).

## 4. Held-out numbers with the shipped weights (train-set numbers; see §1)

20-volume split `dist/heldout_split.json` (5 x `44b6`, 15 x `6bba`),
`scripts/score_local.py --split`. For reference, `main`'s best checkpoint
(`detector_replicate_tmax30`, own architecture, honest held-out) scored
0.8056 total / 0.6365 `44b6` / 0.8278 `6bba`.

| configuration | total | `44b6` (5) | `6bba` (15) | edge J (raw) | TP / FP / FN | div TP/FP/FN | node ratio |
|---|---|---|---|---|---|---|---|
| **full pipeline** (dual seed, D4 + edge TTA, bidir, ILP, post-processing incl. DeepCenter) | **0.9175** | 0.8314 | 0.9305 | 0.9096 | 13053 / 659 / 639 | 3 / 18 / 16 | 0.910 |
| stop after ILP (no post-processing) | 0.9182 | 0.8404 | 0.9276 | 0.9201 | 13087 / 532 / 605 | 0 / 0 / 19 | 0.927 |
| single seed (primary only, full post-processing) | 0.9075 | 0.8241 | 0.9198 | 0.9018 | 12991 / 713 / 701 | 2 / 13 / 17 | 0.920 |
| no ILP (greedy 1-parent/2-children candidates, full post-processing) | 0.8837 | 0.7919 | 0.8964 | 0.8876 | 12979 / 931 / 713 | 3 / 26 / 16 | 1.018 |
| `main`'s best own checkpoint (honest held-out, for scale) | 0.8056 | 0.6365 | 0.8278 | | | | 1.043 |

Error decomposition of the full pipeline (`scripts/analyze_errors.py --split`): detection
recall 98.2 %, association recall 97.7 %; FN split 49.5 % detector / 48.7 % linker;
`44b6` is still the linker-limited embryo (58.7 % of its FN are wrong links, assoc
recall 91.7 % below 7 µm vs 99.6 % above 14 µm), `6bba` the detector-limited one.
Full output in `dist/errors_pack_full.json`.

Readings:

- **The shipped model is ~0.11 above `main`'s best on our split, on both embryos**
  (+0.19 on `44b6`). Train-set number, but the gap is far outside the 0.05 seed noise
  floor measured for `main`.
- **Post-processing is neutral-to-negative on this metric** (0.9175 vs 0.9182 stopping
  after the ILP): motion relink + repairs add 127 FP and 34 FN edges and buy +0.008 from
  three recovered divisions. The notebook authors saw the same sign flip between their
  sparse proxy and the dense leaderboard ("Proxy НЕ коррелирует с LB", cell 2), so this
  is not evidence to drop it; it is evidence that our local metric cannot rank
  post-processing variants. Keep the full chain for submissions; rank models by the
  ILP-stage number.
- **Dual seed is worth +0.010** total (+0.007 `44b6`, +0.011 `6bba`) -- consistent with
  the notebook lineage's own +0.017 from the dual-seed step.
- The node ratio is **below 1** (0.74 on `44b6`): the ILP's disappearance cost 2.0 and the
  min-track-length filter drop many detections. The metric's node-count penalty only
  bites when the ratio exceeds 1, so this is free here; it would matter for a denser
  annotation.
- 8 of 20 volumes triggered the retention guard at least once (`stats.json`), all on
  `44b6_e57ff5c6` (20 frames) and small `6bba` volumes -- the dual-seed blend is
  occasionally losing peaks, as in the notebook's own run (65/400 frames).

## 5. Decisions

- The port is accepted as the branch's pipeline; `scripts/parity/parity_check.py`
  is the regression test for any change to `pack_predict.py`, `ilp.py`,
  `postprocess.py`.
- Training (Phase C) launched 2026-09-11 15:20: `ct-pack-s0` (job 2998081) and
  `ct-pack-s314159` (2998082), one L40S each on `short`, 400 epochs, `--max-hours 11.4`
  with the self-resubmitting chain; DeepCenter `ct-deepcenter` (2998077), 50 epochs.
  Measured throughput: 0.7 s/step, 16788 windows/epoch -> ~25 min/epoch -> ~7 days per
  seed (the pack's own history: ~20 min/epoch on an RTX 5090, 9.3 days wall clock).
  4x L40S DataParallel measured only 1.6x (CPU-side per-sample loops dominate) and was
  rejected; the H200 node was not allocatable.
- **Decision point at epoch 100 (~1.7 days):** sweep the epoch-25/50/75/100 snapshots
  through the full pipeline per embryo. If the in-training `val_score` is flat from
  ~epoch 75, cancel the chain and use `_best`; else continue. The pack's own curve was
  still (slowly) improving at 381/400 on its detector-recall metric, which is the only
  evidence for 400.
- Not done (user chose to let the runs proceed as-is): profiling/vectorising the
  Python loops in the training step (a same-numerics speedup worth ~1.5-2x if they
  dominate), and moving the chains to `long` to avoid 12-hour requeues.
