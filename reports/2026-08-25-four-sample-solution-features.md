# Implementing four sample-solution features (edge model, augmentation/TTA, deeper temporal attention, greedy linking)

Follows `reports/2026-08-25-sample-solution-0.90-comparison.md`. That report's
own recommendation was narrower -- train the existing single-frame detector
longer first, to isolate whether the detector is the bottleneck before
touching anything else. This iteration does not do that: it was a direct
request to implement four of that report's items in one pass (its #1/#2, #3,
#4, and #5), ahead of the recommended order and without a held-out score in
hand yet. Recording that explicitly here, since it's a deviation from the
"one piece at a time, only on evidence" rule this repo otherwise holds to.

## What changed

1. **Edge model trained on its own detections, not GT+decoys** (report item
   1/2). `models/edge_model.py` adds `EdgeScorer`, a small MLP over sampled
   node features + relative position, scoring one candidate (src, dst) pair.
   `edge_train.py`'s `sample_edge_pair` does detect-and-match: it runs real
   peak detection (identical NMS/`TAU` to inference) on the detector's live
   logits for two consecutive frames, matches detected peaks to GT within
   `EDGE_MATCH_RADIUS_UM` (5 µm), and derives candidate-pair labels from those
   matched identities plus the volume's real GT edge set. No synthetic
   GT-node + nearest-peak-decoy set anywhere.
2. **Augmentation + TTA** (item 3). `augment.py` adds random y/x flip +
   brightness/bias jitter, shared across every frame in a training window and
   applied to the GT grid too. `detect.predict_logits_tta` averages detection
   *logits* over identity + 3 flips (y, x, xy) at inference, mirroring the
   report's finding that the sample solution excludes z for the same
   anisotropy reason this repo's `TAU`/NMS work already documents.
3. **Richer temporal modeling: full multi-head attention at every encoder
   stage** (item 4). `models/detector.py`'s `UNet3D` now takes a
   `WINDOW_SIZE=2` window, encodes each frame independently (shared weights)
   per stage, and applies `TemporalAttention3d` (self-attention across the
   window, per spatial location) at every encoder stage except full
   resolution, plus the bottleneck -- matching the sample solution's stated
   design more closely than the v1 sibling repo's bottleneck-only mixing.
   The decoder predicts only the last (target) frame, using that frame's own
   post-attention features as skip connections.
4. **Plain greedy thresholding for linking** (item 5). `link.py` no longer
   solves an exact bipartite assignment; it sorts distance-gated candidates
   by score (edge-model probability when available, else negative distance)
   and accepts greedily while both endpoints are free. In/out-degree stay
   <= 1 by construction -- no divisions were added, since the report's
   comparison table shows the sample solution's *default* selector is also
   degree <= 1 in effect for the un-forked case, and divisions were not part
   of this request.

Also updated to keep working: `detect.load_model` now returns
`(model, edge_scorer, device)` (edge_scorer is `None` for a checkpoint
without one); `predict.predict_volume` builds windows, runs TTA, samples node
features for edge scoring, and passes edge scores into `link_frames`;
`train.py` trains the edge scorer jointly with the detector (one backward
pass every `edge_every`-th step, default every step); checkpoints now carry
an `edge_state_dict` key alongside `state_dict`. `scripts/train.py` and
`scripts/predict.py` got `--no-augment`, `--no-edge-model`, `--edge-every`,
`--no-tta` flags. `README.md`, `config.py`'s docstring, and
`package_for_kaggle.py`'s manifest string were updated to stop describing the
old single-frame/no-edge-model pipeline. `notebooks/local_smoke.ipynb`'s
`load_model` call was patched for the new 3-tuple return; the Kaggle
notebooks only shell out to the CLI scripts and needed no changes.

## Design choices not dictated by the request

- **Edge-model architecture is an MLP scorer, not a transformer.** The
  report's item 2 emphasized *what it trains on* (live detections, not
  decoys) as the load-bearing design choice, not the sample solution's
  specific cross-attention architecture. An MLP over sampled node features +
  relative position is simpler to get right and is architecturally
  equivalent for a 1:1 candidate-scoring task; a true cross-attention
  "compete over candidates sharing a source" layer would be a reasonable
  follow-up but wasn't built here.
- **Edge-scorer gradients flow into the detector.** Joint training, one
  optimizer, one backward pass per step that includes an edge sample. This
  is simpler than freezing the detector for edge training and is closer to
  "trains on its own detections" in spirit, but it does mean the detector's
  detection loss is no longer the only thing shaping its features -- worth
  watching if `val_loss` behaves differently than the pre-edge-model
  baseline.
- **No divisions, no ILP.** Out of scope for this request; degree stays
  <= 1/<= 1 as before.
- **Detect-and-match self-limits early in training.** `TAU = 0.985` means an
  early, barely-trained detector clears almost nothing, so
  `sample_edge_pair` returns `None` most of the time until detection quality
  improves -- edge loss is near-zero for a while, then ramps up. See the
  smoke-test numbers below; this is expected, not a bug.

## Correctness bug found and fixed during smoke testing

`edge_train.py`'s original `matched_gt0/1 = np.where(match >= 0, gt_ids[...], -1)`
crashed with an `IndexError` whenever a frame had zero annotated GT nodes:
`np.where` evaluates both branches eagerly, so `gt_ids[np.clip(match, 0, None)]`
still runs (indexing an empty array at position 0) even though that branch is
never selected. Same class of bug the existing `peaks.subvoxel_offset`
docstring already warns about for `torch.where`. Fixed by guarding the
indexing when `gt_ids` is empty.

## Verification performed

No unit tests exist in this repo (none were added; testing infra wasn't part
of the request). Verified instead by:

- A synthetic-tensor smoke script exercising `UNet3D` forward + backward
  (including `return_features`), `EdgeScorer`, `sample_node_features`,
  `augment_window`, `link_frames` (both distance-fallback and edge-score
  paths), and `match_to_reference` -- all passed.
- A real end-to-end run: `scripts/train.py --limit-volumes 4 --epochs 15
  --frames-per-volume 15` on cached local data (this is where the
  `edge_train.py` bug above surfaced and was fixed), followed by
  `scripts/predict.py` and `scripts/score_local.py` on the resulting
  checkpoint. Pipeline runs cleanly end to end: detection loss drops
  (0.39 -> 0.01), edge loss activates once detections clear `TAU` and reaches
  ~83% pair-accuracy on this tiny run, and `score_local.py` returns a
  well-formed (if poor -- 5.8% recall) score.
- **This is a plumbing check, not a result.** 4 volumes / 15 epochs / 15
  frames-per-volume is far below anything that could show whether these
  features help. No held-out before/after comparison exists yet.

## Status: held-out and official scores now in hand

Three checkpoints, all trained on the full 179-volume training split, all
scored against the same 20-volume held-out split (never trained on) and via
real Kaggle leaderboard submissions:

| Checkpoint | Epoch | Held-out `SCORE` | Official leaderboard |
|---|---|---|---|
| Baseline (pre-this-report, single-frame/no-edge-model/exact-assignment) | 6 | 0.6432 | 0.683 |
| This report's 4 features, no dropout | 6 (patience=5 stop) | 0.7597 | 0.820 |
| + dropout (see below) | 23 (stopped by a NaN divergence, not patience) | not separately measured | **0.841** |

Local held-out and official numbers agreed in direction and rough magnitude
both times a comparison was made (+0.116 local / +0.137 official for the
first jump; the dropout run's held-out `val_loss` improving from 0.02287 to
0.01879 also correctly predicted the +0.021 official gain) -- reassuring
given this repo's own warning that cross-checkpoint local-metric comparisons
were anti-correlated with the real leaderboard in the sibling repo's
history. Small sample (two comparisons), but no sign of that failure mode
here so far.

**What it took to get a trainable-to-convergence checkpoint, briefly:**
the first few full-budget training attempts weren't limited by the training
budget itself (the original hypothesis from
`2026-08-25-sample-solution-0.90-comparison.md` item 1) but by three
compounding measurement/stability bugs, fixed in order:
1. `val_frames_per_volume=4` (80 held-out frames total) made `val_loss` too
   noisy for `patience` to mean anything, and the LR schedule's `T_max` was
   left at a nominal `epochs=500` while `max_hours` actually governed when
   the run stopped -- so the LR never meaningfully annealed either. Fixed:
   `val_frames_per_volume=16`, `epochs` set to a real target matching
   measured throughput.
2. Once those were fixed, training diverged to NaN under AMP (`RuntimeError:
   Efficient attention cannot produce valid seed and offset outputs when the
   batch size exceeds 65535`) -- a real bug in `TemporalAttention3d`, not a
   dataset or hyperparameter issue; see `models/detector.py`'s docstring.
   `train.py` also lost ~50 minutes of GPU time grinding through 13 further
   epochs before `patience` noticed the NaN, since `nan < best_value` is
   always `False`; fixed with `TrainingDivergedError`, which now stops the
   instant a training-branch loss goes non-finite.
3. With the noise/schedule fixed, the val curve showed a clean train/val gap
   (train loss still dropping smoothly to epoch 21, val loss peaking at
   epoch 6 and never recovering) -- genuine overfitting, not an artifact.
   Added `UNET_DROPOUT=0.1` (`Dropout3d`, concentrated in coarser stages,
   skipping full-res and the finest decoder stage that feeds the edge
   scorer) and `EDGE_DROPOUT=0.1`. Result: new val-loss bests kept appearing
   through epoch 23 instead of stalling at epoch 6, and the official score
   improved again (0.820 -> 0.841).

**Still open**: the NaN divergence itself was never fixed, only delayed --
it recurred at epoch 26 even with dropout and a halved LR (`5e-4`), and the
held-out val curve was still finding new bests when it was cut off, so
there is a live, unresolved question of how much more headroom exists if
that instability were actually root-caused rather than out-run. Also still
true from before: no ablation isolating which of the four original features
(edge model, augmentation/TTA, attention, greedy linking) is carrying the
official-score gain -- the numbers above bundle all four plus dropout
together.
