# 2026-09-16 -- Score sweep of our own 400-epoch checkpoints

Both 4-GPU DDP runs of the pack recipe (`scripts/train.py --ddp`, 179 train volumes, 400 epochs,
~7.5 min/epoch) completed today: `pack_s0_ddp` (best val_score 0.8716 @ ep 345) and
`pack_s314159_ddp` (best 0.8803 @ ep 385). The training-time `val_score` runs the ILP stage without
TTA, blending or post-processing, so this sweep scores the full pipeline on the same 20 held-out
volumes (`dist/heldout_split.json`, 5 x 44b6 + 15 x 6bba). Launcher: `dist/sweeps/2026-09-16-ddp-sweep.sh`
(13 `slurm_score.sbatch` jobs, 6-18 min each). Table: `python scripts/tabulate_scores.py <tags>`.

All arms use our own DeepCenter (`dist/models/deepcenter/best.pt`, epoch 25, `--no-deepcenter-epoch-check`)
unless noted. "blend" = dual-seed with the first-named checkpoint as primary.

| tag | SCORE | 44b6 | 6bba | adj edge J | div J (tp/fp/fn) | det recall % | FP | FN |
|---|---|---|---|---|---|---|---|---|
| **shipped weights, dual seed** (`pack_full`, trained on all 199 -> leaky) | 0.9175 | 0.8314 | 0.9305 | 0.9094 | 0.081 (3/18/16) | 98.3 | 659 | 639 |
| shipped weights, single seed | 0.9075 | 0.8241 | 0.9198 | 0.9013 | 0.062 (2/13/17) | 98.1 | 713 | 701 |
| ddp_best_s314159 (ep 385) | **0.8976** | 0.7835 | 0.9127 | 0.8879 | 0.097 (3/12/16) | 96.6 | 700 | 908 |
| ddp_best_blend_swap (s0 primary, s314159 secondary) | 0.8960 | 0.7801 | 0.9121 | 0.8892 | 0.068 (3/25/16) | 97.0 | 721 | 848 |
| ddp_best_blend (s314159 primary, s0 secondary) | 0.8952 | **0.7932** | 0.9085 | 0.8893 | 0.059 (2/15/17) | 96.7 | 705 | 884 |
| ddp_best_s0 (ep 345) | 0.8854 | 0.7734 | 0.9000 | 0.8829 | 0.024 (1/22/18) | 96.6 | 734 | 913 |
| ddp_last_s314159 (ep 400) | 0.8921 | 0.7798 | 0.9073 | 0.8828 | 0.094 (3/13/16) | 95.8 | 711 | 1019 |
| ddp_last_blend | 0.8898 | 0.7592 | 0.9077 | 0.8829 | 0.069 (2/10/17) | 95.7 | 699 | 1023 |
| ddp_last_s0 | 0.8743 | 0.7515 | 0.8897 | 0.8708 | 0.034 (1/10/18) | 95.0 | 754 | 1149 |
| gpu1_best_blend (1-GPU chains, ep 155 + ep 145) | 0.8961 | 0.7936 | 0.9105 | 0.8840 | 0.121 (4/14/15) | 96.2 | 705 | 961 |
| cross_best_blend (ddp s314159 best + 1-GPU s0 best) | 0.8927 | 0.7880 | 0.9074 | 0.8827 | 0.100 (3/11/16) | 96.5 | 739 | 941 |
| ddp_best_blend, `--stage ilp` (no post-processing) | 0.8903 | 0.7953 | 0.9018 | 0.8903 | 0.000 (0/0/19) | 97.1 | 654 | 892 |
| ddp_best_s314159, `--stage ilp` | 0.8879 | 0.7800 | 0.9010 | 0.8879 | 0.000 (0/0/19) | 97.0 | 661 | 920 |
| ddp_best_blend, public DeepCenter (epoch 2) | 0.8951 | 0.7921 | 0.9085 | 0.8892 | 0.059 (2/15/17) | 96.7 | 706 | 884 |
| ddp_best_blend, no DeepCenter | 0.8947 | 0.7914 | 0.9082 | 0.8888 | 0.059 (2/15/17) | 96.7 | 711 | 883 |

## Findings

1. **Our clean held-out number is ~0.895-0.898**, 0.02 below the shipped weights' 0.9175. The shipped
   weights saw these 20 volumes in training, so the true gap is smaller than 0.02 and may be zero. The
   gap that does exist is almost entirely **missed detections**: det recall 96.7 % vs 98.3 %, FN 884 vs
   639, while FP is the same (705 vs 659). Wrong links are not the problem; missed cells are.
2. **The 44b6 gap is unchanged**: 0.78-0.79 vs 0.91 on 6bba, the same 12-13 point spread as with the
   shipped weights (0.83 / 0.93). Training our own weights did not move the embryo problem.
3. **Dual-seed blending adds nothing measurable** here: the blend (0.8952 / 0.8960) sits between the two
   single seeds (0.8976 / 0.8854) rather than above them. The notebook's blend gain (single 0.9075 ->
   dual 0.9175 with the shipped weights) does not reproduce with our seeds. The blend does help 44b6
   by +0.01 and reduces FN by 24-60, so it is a coin flip, not a loss; the shipped-weights gain came
   with a much larger FN reduction (701 -> 639).
4. **Seed 314159 beats seed 0 by 0.012** in every pairing (best, last, single). Same ordering the pack
   authors reported. With n = 2 this is anecdote; seed variance of ~0.01 is the noise floor for
   cross-checkpoint comparisons in this table.
5. **`_best` (val_score-selected) beats `_last` (ep 400) by 0.005-0.011**, and the ep-400 checkpoints
   have lower det recall (95.7-96.0 vs 96.6-97.0). The detector is drifting late; val_score selection
   is doing its job.
6. **Epochs 150 -> 385 bought nothing.** The 1-GPU chains' bests at ep ~150 blend to 0.8961, equal to
   the DDP ep-345/385 blend (0.8952). Consistent with the flat val_score curve from ep ~120.
   A second 400-epoch run for a new seed is not worth 2 GPU-days; 150-200 epochs is enough.
7. **Post-processing is +0.005 total**: -0.001 adjusted edge Jaccard, +0.006 from the division term
   (2/15/17 vs 0/0/19). Same wash as measured with the shipped weights on 09-14. The division term is
   19 GT divisions on 20 volumes; every number in that column is noise.
8. **DeepCenter has no measurable effect** (0.8947 none / 0.8951 public / 0.8952 ours). It only gates
   synthetic gap nodes spanning >= 8.5 um; there are too few of those to matter. Keep it for parity with
   the notebook, but do not spend time on it.

## Decision

Ship `pack_s314159_ddp_best.pt` as primary and `pack_s0_ddp_best.pt` as secondary with our DeepCenter
(the blend is within noise of the single seed and matches the notebook's structure, so the Kaggle
notebook runs unchanged). The leaderboard is the only cross-embryo signal; the local ~0.896 will not
predict it (see CLAUDE.md on cross-checkpoint anti-correlation).

## Next

- Detection recall is the lever (FN 884 vs 639 at equal FP). Candidates, each an experiment: lower
  `det_threshold` (0.965) or a (3,3,3) -> smaller pool kernel, scored on held-out with these weights;
  error analysis (`scripts/analyze_errors.py`) on `dist/preds_val_ddp_best_blend` to see whether the
  extra misses are crowded-neighbour absorptions as in the 0.842 pipeline.
- Jump-frame check with our weights: `--stage ilp` is +0.002 on 44b6 (0.7953 vs 0.7932) and -0.007 on
  6bba; per-volume diff on `44b6_d754aa59` / `6bba_3abfe10a` still to do.
