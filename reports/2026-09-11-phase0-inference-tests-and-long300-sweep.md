# 2026-09-11 -- Phase 0 wrap-up: ten inference-side tests, error decomposition, 300-epoch sweep

Follow-up to `reports/2026-09-10-followup-lr-schedule-noise-floor.md` and the
approved redesign plan. Everything here is on the 20-volume random held-out
split (5 x `44b6`, 15 x `6bba`), scored with `scripts/score_local.py --held-out`,
per-embryo numbers from `evaluate.summarize_by_embryo`. Noise floor from the
previous report: identical configs differ by up to ~0.05 total; single
differences below ~0.03 are not results.

## 0. Plumbing that this report depends on

- **`val_score` selection works.** A 2-epoch, 8-volume smoke run
  (`--eval-tracking-every 1 --select-by val_score`) logged
  `val_score=0.5728 (val_score_44b6=...)` on each epoch line and wrote
  `smoke_valscore_best.pt` at `new best val_score`. The tracking eval on the
  smoke split took ~1 s/epoch. Phase 1 can select by the competition metric.
- **Cluster GPU backlog.** Every `short`-partition GPU job submitted at 17:24
  and 20:24 yesterday was still `PENDING (Priority)` 17 h later (157 GPU jobs
  from other users queued). The `interactivegpu` node (4x L40S) was idle but
  only accepts `srun`, caps at 1 h and one running job per user. Added
  `scripts/slurm_gpu_batch.sh` (runs a task file 4-wide, one task per GPU,
  under one `srun`) and `scripts/score_one.sh` (the predict+score body).
  The whole batch below -- 10 inference tests, the smoke run, and 60 sweep
  checkpoints -- ran in ~17 minutes at ~48 s per checkpoint.
- The original `ct-sweep-long300_s{0,1}` jobs did start on `short` at 01:17
  and ran 7.5 min before being cancelled in favour of the batch; the
  interactive batch re-scored every snapshot except epochs 10/20, which were
  scored yesterday. Both paths produce identical numbers (same code, TTA is
  deterministic).

## 1. Ten inference-side tests on the best checkpoint

Baseline: `detector_replicate_tmax30.pt` last epoch (lr 5e-4, 30 epochs).
All tests change only inference; the checkpoint is fixed, so these are
same-checkpoint comparisons -- the kind the local scorer is trusted for.

| tag | change | total | 44b6 | 6bba | det recall | node ratio |
|---|---|---|---|---|---|---|
| replicate_tmax30_last | baseline | **0.8056** | 0.6365 | 0.8278 | 95.3% | 1.043 |
| replicate_tmax30_dump | baseline re-run (+ candidate dump) | 0.8056 | 0.6365 | 0.8278 | 95.3% | 1.043 |
| replicate_tmax30_ttaz | + z-flip TTA (8-way) | 0.7992 | 0.6431 | 0.8189 | 94.4% | 1.034 |
| tmax30_nopatience_ttaz | + z-flip TTA, 2nd ckpt (base 0.8007 / 0.6031 / 0.8267) | 0.7930 | 0.5987 | 0.8184 | 94.7% | 1.047 |
| lr5e4_ep100_ttaz | + z-flip TTA, 3rd ckpt (base 0.7996 / 0.6121 / 0.8241) | 0.7942 | 0.6033 | 0.8190 | 94.3% | 1.017 |
| replicate_tmax30_r10 | link gate 15 -> 10 um | 0.8052 | 0.6381 | 0.8270 | 95.3% | 1.043 |
| replicate_tmax30_r12 | link gate 15 -> 12 um | 0.8054 | 0.6365 | 0.8275 | 95.3% | 1.043 |
| replicate_tmax30_thr04 | link threshold 0.5 -> 0.4 | 0.8055 | 0.6350 | 0.8279 | 95.3% | 1.043 |
| replicate_tmax30_thr06 | link threshold 0.5 -> 0.6 | 0.8054 | 0.6383 | 0.8272 | 95.3% | 1.043 |
| ens3_tmax30 | logit ensemble of the 3 checkpoints above | 0.8059 | 0.6165 | 0.8309 | 95.0% | 1.036 |
| ens3_tmax30_ttaz | 3-ensemble + z-flip TTA | 0.8058 | 0.6283 | 0.8289 | 94.7% | 1.029 |

Readings:

- **z-flip TTA is a consistent negative**: -0.006, -0.008, -0.005 on three
  independent checkpoints, and detection recall drops ~0.5-1 pt each time.
  This is the same direction on every checkpoint, so it is a real effect,
  not noise. Consistent with the `DOWNSAMPLE` argument in CLAUDE.md: z is
  already ~4x coarser than xy and the network's z-asymmetry is not a
  symmetry of the data. Keep `TTA_FLIPS` y/x only. **Rejected.**
- **Link gate and link threshold are inert** (all within 0.0004 of baseline;
  detections identical). The edge scorer's probabilities are already
  well-separated around 0.5 and the true partner is almost never 12-15 um
  away. No change.
- **3-checkpoint logit ensemble is inert on the total** (+0.0003) and
  *worse on 44b6* (-0.020). Averaging three detectors that each score
  0.60-0.64 on 44b6 does not recover crowded-region detections. Not worth
  3x inference cost. **Rejected.**

Inference-side knobs are exhausted at this checkpoint. Whatever moves the
score next has to come from the model or the training split.

## 2. Error decomposition with candidate ranks (baseline checkpoint)

`scripts/analyze_errors.py --candidates` on the `_dump` prediction.

| | all 20 | 44b6 (5) | 6bba (15) |
|---|---|---|---|
| detection recall | 95.3% | 94.0% | 95.4% |
| association recall (both endpoints detected) | 94.1% | 86.4% | 95.0% |
| FN attributable to detector | 54.7% | 45.0% | 57.0% |
| FN attributable to linker | 44.3% | 54.1% | 42.0% |
| true partner ranked 1st among gated candidates | 96.1% | 89.7% | 96.8% |
| true partner ranked 2nd | 2.8% | 6.2% | 2.4% |
| true partner not a candidate (outside 15 um gate) | 0.1% | 0.1% | 0.1% |
| FP where the true partner was detected but linker chose another | 76.7% | 81.2% | 75.1% |
| assoc recall, source NN < 7 um | 75.3% | 79.4% | 72.8% |
| assoc recall, source NN 7-10 um | 91.1% | 87.5% | 92.0% |
| assoc recall, source NN >= 14 um | 98.7% | (n=6) | 98.7% |

- **Greedy vs. exact assignment on the same scores: TP -153, FP +474.**
  Exact bipartite assignment would be *worse* than greedy. The ILP-flavoured
  selector the sibling repo carried is not a lever here; the scorer's
  ordering, not the matching algorithm, decides these edges. Do not add an
  ILP.
- **On 44b6 the problem is the linker**: 54% of FNs and 81% of FPs are
  "both endpoints detected, wrong link", and the true partner is only ranked
  first 89.7% of the time (vs 96.8% on 6bba). Roughly 1 in 10 crowded
  associations is a scorer ranking failure. That is the target for a richer
  edge scorer (plan step 3), and the reason the MLP's pairwise, context-free
  scoring is suspect: it never sees the competing candidates.
- **On 6bba the problem is the detector**: 57% of FNs trace to a missed
  endpoint, and there the scorer already ranks the true partner first 96.8%
  of the time.
- The 15 um gate is not losing anything (0.1% not-a-candidate), confirming
  section 1.

## 3. 300-epoch runs, two seeds, scored every 10 epochs

Config: lr 5e-4 cosine over 300 epochs, no patience, batch 4x2, seeds 0 and
1, `--save-every 10`. Training took 6.5 h each (1.3 min/epoch on a V100).
Training loss went 0.036 -> 0.0093 (both seeds); `val_loss` bottomed at
epoch 7 (s0, 0.0196) / epoch 147 (s1, 0.0197) and then drifted to 0.028-0.029,
i.e. `val_loss` is useless as a selector here, as before. Train edge
accuracy climbed monotonically 0.55 -> 0.92/0.93.

Score of every 10th-epoch snapshot on the 20-volume held-out split (last
row = the `_best` by `val_loss`: s0 epoch 7 -> 0.7566, s1 epoch 147 -> 0.7339):

| epoch | s0 total | s0 44b6 | s0 6bba | s1 total | s1 44b6 | s1 6bba | mean total | mean det recall |
|---|---|---|---|---|---|---|---|---|
| 10 | 0.7802 | 0.6051 | 0.8028 | 0.7121 | 0.5622 | 0.7314 | 0.7461 | 92.7% |
| 20 | 0.7829 | 0.5710 | 0.8108 | 0.7872 | 0.5825 | 0.8139 | 0.7851 | 93.1% |
| 30 | 0.7867 | 0.5865 | 0.8128 | 0.7654 | 0.5940 | 0.7871 | 0.7760 | 94.2% |
| 40 | 0.7920 | 0.5746 | 0.8211 | 0.7768 | 0.5747 | 0.8034 | 0.7844 | 94.7% |
| 50 | 0.7869 | 0.5826 | 0.8135 | 0.7781 | 0.5492 | 0.8086 | 0.7825 | 93.4% |
| 60 | 0.7807 | 0.5870 | 0.8062 | 0.7804 | 0.5666 | 0.8090 | 0.7805 | 94.3% |
| 70 | 0.7919 | 0.5716 | 0.8210 | 0.7767 | 0.5749 | 0.8035 | 0.7843 | 94.2% |
| 80 | 0.7947 | 0.5853 | 0.8222 | 0.7818 | 0.5804 | 0.8085 | 0.7882 | 94.7% |
| 90 | 0.7797 | 0.5501 | 0.8096 | 0.7861 | 0.5676 | 0.8150 | 0.7829 | 94.2% |
| 100 | 0.7826 | 0.5609 | 0.8121 | 0.7868 | 0.5897 | 0.8124 | 0.7847 | 94.8% |
| 110 | 0.7873 | 0.5812 | 0.8141 | 0.7851 | 0.5727 | 0.8132 | 0.7862 | 95.0% |
| 120 | 0.7757 | 0.5399 | 0.8070 | 0.7603 | 0.5356 | 0.7897 | 0.7680 | 95.2% |
| 130 | 0.7600 | 0.5174 | 0.7919 | 0.7844 | 0.5943 | 0.8092 | 0.7722 | 95.4% |
| 140 | 0.7799 | 0.5536 | 0.8096 | 0.7454 | 0.5218 | 0.7751 | 0.7627 | 95.4% |
| 150 | 0.7437 | 0.5121 | 0.7732 | 0.7519 | 0.5286 | 0.7808 | 0.7478 | 95.4% |
| 160 | 0.7478 | 0.5138 | 0.7781 | 0.7560 | 0.5101 | 0.7880 | 0.7519 | 95.6% |
| 170 | 0.7340 | 0.5054 | 0.7628 | 0.7390 | 0.4717 | 0.7737 | 0.7365 | 95.6% |
| 180 | 0.7507 | 0.5295 | 0.7794 | 0.7514 | 0.5407 | 0.7789 | 0.7511 | 95.8% |
| 190 | 0.7224 | 0.4995 | 0.7505 | 0.7062 | 0.4244 | 0.7425 | 0.7143 | 95.7% |
| 200 | 0.7105 | 0.4749 | 0.7401 | 0.7212 | 0.4532 | 0.7558 | 0.7158 | 95.6% |
| 210 | 0.6766 | 0.3951 | 0.7111 | 0.7040 | 0.4436 | 0.7375 | 0.6903 | 95.4% |
| 220 | 0.6836 | 0.3931 | 0.7199 | 0.7000 | 0.4316 | 0.7344 | 0.6918 | 95.4% |
| 230 | 0.6781 | 0.3913 | 0.7138 | 0.6835 | 0.4142 | 0.7177 | 0.6808 | 95.4% |
| 240 | 0.6766 | 0.3792 | 0.7139 | 0.6683 | 0.3948 | 0.7030 | 0.6725 | 95.4% |
| 250 | 0.6671 | 0.3585 | 0.7055 | 0.6542 | 0.3733 | 0.6904 | 0.6606 | 95.5% |
| 260 | 0.6580 | 0.3461 | 0.6968 | 0.6542 | 0.3694 | 0.6904 | 0.6561 | 95.5% |
| 270 | 0.6546 | 0.3434 | 0.6933 | 0.6421 | 0.3611 | 0.6779 | 0.6484 | 95.5% |
| 280 | 0.6534 | 0.3506 | 0.6908 | 0.6455 | 0.3606 | 0.6819 | 0.6495 | 95.5% |
| 290 | 0.6522 | 0.3433 | 0.6905 | 0.6412 | 0.3567 | 0.6773 | 0.6467 | 95.5% |
| 300 | 0.6523 | 0.3404 | 0.6909 | 0.6421 | 0.3587 | 0.6781 | 0.6472 | 95.6% |

**Longer training does not help; past ~epoch 110 it actively destroys the
score.** Both seeds peak in the 0.78-0.79 band somewhere in epochs 40-110
(seed means: 0.784 at 40, 0.788 at 80, 0.786 at 110), which is inside the
noise band of the 30-epoch runs (0.8056 / 0.8007 / 0.7513). Then both fall
monotonically to 0.65 at epoch 300, and `44b6` falls from ~0.58 to ~0.35.

What degrades is the **edge scorer, not the detector**:

| epoch (s0) | det recall | node ratio | TP | FP | FN | pred edges / pred node | val_edge_loss | val_edge_acc |
|---|---|---|---|---|---|---|---|---|
| 20 | 93.1% | 0.953 | 11590 | 1135 | 2102 | 0.850 | 0.23 | 0.755 |
| 80 | 94.7% | 1.047 | 11832 | 1099 | 1860 | 0.836 | 0.20 | 0.808 |
| 150 | 95.4% | 1.065 | 10951 | 912 | 2741 | 0.595 | 0.53 | 0.873 |
| 200 | 95.9% | 1.114 | 10572 | 959 | 3120 | 0.550 | 0.94 | 0.896 |
| 300 | 95.7% | 1.086 | 9542 | 786 | 4150 | 0.396 | 1.28 | 0.925 |

(seed 1 is the same story: edges/node 0.88 -> 0.43, val_edge_loss 0.08 -> 1.18.)

- Detection *improves* throughout: recall 92.4% -> 95.7%, node ratio stays
  ~1.05-1.1. The detector is not overfitting in any way the metric sees.
- The number of accepted links per detected node halves (0.85 -> 0.40): the
  scorer's probability on true pairs drifts below the 0.5 threshold, so FN
  doubles (2100 -> 4150) while FP *falls* (1135 -> 786). The scorer becomes
  under-confident on positives / miscalibrated, not wrong-ranked -- the
  classic overfitting signature of a small MLP trained for 300 epochs on
  roughly **2 positive pairs per step** (`val_edge_pos` ~1.9 of ~500
  candidate pairs; sparse GT means detect-and-match yields very few matched
  edges per window).
- `val_edge_loss` is the honest indicator: 0.09 -> 1.28 while
  `val_edge_acc` climbs 0.67 -> 0.93. **`edge_acc` is a useless metric** --
  it is thresholded accuracy over a 250:1 negative-dominated pair set and
  goes *up* as the scorer learns to say "no" to everything. Stop reading it.
- `val_loss` (detection) picked epoch 7 (s0) / 147 (s1) as `_best`: 0.7566
  and 0.7339, both below the epoch-80 snapshot. Confirms (again) that
  `val_loss` cannot select; `val_score` would have picked ~epoch 80.

Attribution test: the same snapshots re-scored with `--no-edge-model`
(distance-only greedy linking; detections identical):

| snapshot | with edge scorer | distance-only | edge scorer delta | 44b6 with / without |
|---|---|---|---|---|
| s0 epoch 80 | 0.7947 | 0.7798 | **+0.015** | 0.585 / 0.555 |
| s1 epoch 80 | 0.7818 | 0.7731 | **+0.009** | 0.580 / 0.556 |
| s0 epoch 150 | 0.7437 | 0.7857 | -0.042 | 0.512 / 0.566 |
| s1 epoch 150 | 0.7519 | 0.7885 | -0.037 | 0.529 / 0.562 |
| s0 epoch 300 | 0.6523 | 0.7771 | **-0.125** | 0.340 / 0.559 |
| s1 epoch 300 | 0.6421 | 0.7879 | **-0.146** | 0.359 / 0.550 |
| replicate_tmax30 (30 ep) | 0.8056 | 0.7940 | +0.012 | 0.637 / 0.614 |

- Distance-only linking is flat from epoch 80 to 300 (0.77-0.79 both
  seeds): **the detector plateaus by ~epoch 80 and never gets worse.** It
  also never gets better than the 30-epoch detector (0.794 distance-only).
- The edge scorer is worth +0.01 to +0.015 at epoch 30-80 (matches the
  +0.012-0.018 measured yesterday), is already a net *loss* by epoch 150,
  and costs 0.13-0.15 by epoch 300. The entire collapse is the MLP
  overfitting its ~2-positives-per-step training signal.



## 4. Decisions

- Reject z-flip TTA, the 3-checkpoint ensemble, link-gate and
  link-threshold changes (section 1). Reject exact assignment / ILP
  (section 2).
- Selection by `val_score` is verified; use it for every Phase 1 run.
- Proceed to Phase 1: 6 runs (`--val-embryo 44b6`, `--val-embryo 6bba`,
  random split) x seeds {0, 1}, 30 epochs, lr 5e-4, `--save-every 10`,
  scored with the interactive batch runner rather than waiting on `short`.

- **Long training: rejected as-is.** 300 epochs buys nothing for the
  detector (plateau by epoch 80, no better than 30 epochs) and ruins the edge
  scorer. Stay at 30 epochs for Phase 1. The 0.90 sample solution's 400-epoch
  budget does not transfer to this edge-scorer design.
- **The edge scorer needs its own regularization/early-stopping story
  before any richer scorer is trained longer.** Concretely, for plan step 3
  (cross-attention scorer): log `val_edge_loss` and treat its rise as the
  stopping signal for the scorer; stop reading `edge_acc`; consider a
  separate (lower) lr or freezing the scorer once `val_edge_loss` turns, and
  revisit `EDGE_NEG_ALPHA = 0.05` / positive sampling (~2 positives per step
  is a very thin signal to train 300 epochs on). Any of these is a
  measured-not-assumed change: score with and without.
- The `_best`-by-`val_loss` checkpoints of the two long runs (0.7566, 0.7339)
  are both worse than a `val_score`-selected snapshot would have been
  (~epoch 80, 0.79). Third independent confirmation; nothing more to learn here.


## 5. Launched

Phase 1 (six runs: `--val-embryo 44b6`, `--val-embryo 6bba`, random split,
seeds 0/1; 30 epochs, lr 5e-4, `--select-by val_score --save-every 10`) was
submitted to `short` at 01:47 as jobs 2996674-2996685 (training + chained
`slurm_score_sweep.sbatch`), outputs `dist/models/detector_p1_<split>_s<seed>*`.
