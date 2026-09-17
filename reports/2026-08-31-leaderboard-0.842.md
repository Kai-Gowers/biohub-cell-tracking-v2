# Leaderboard confirms the T_max=30 checkpoint: 0.842, new best

Submitted `detector_tmax30_epoch23.pt` (packaged via `package_for_kaggle.py`,
manifest sha256-verified against the checkpoint before upload -- see
`2026-08-31-tmax30-held-out-score.md`) to the Kaggle leaderboard.

**Result: 0.842**, vs. the 0.841 baseline's real submitted score. A real,
if small, improvement -- new best submitted checkpoint in this repo.

## Reading against the local held-out prediction

| | held-out `SCORE` | leaderboard |
|---|---|---|
| 0.841 baseline | 0.8031 | 0.841 |
| this checkpoint | 0.8074 | 0.842 |
| delta | +0.0043 | +0.001 |

Direction agreed (4th confirmation in [[local-scoring-validation]]'s
tracking, all 4/4 so far). The absolute gap between the predicted
(+0.0043) and actual (+0.001) delta is 0.0033 -- inside the ~0.004-0.006
agreement band the three prior (much larger-effect) comparisons
established, not outside it. This is just the smallest true effect
measured yet, so the same absolute noise floor looks proportionally larger.
**Takeaway: trust the sign of a `score_local.py --held-out` delta, treat
the magnitude as accurate to roughly +/-0.005 absolute, not as a precise
forecast** -- especially for small deltas like this one.

## Where this leaves things

- `dist/models/detector_best.pt` (epoch 23, T_max=30) is now the working
  best checkpoint, both locally and on the leaderboard.
- The underlying NaN-divergence numerics-margin condition
  (`conv_grad_norm_max` pinned at the clip ceiling) is still present and
  still occasionally triggers recoverable `GradScaler` skips -- T_max=30
  is the current best mitigation but is n=1 on the "no hard divergence"
  claim. Worth a repeat run before fully trusting it as solved.
- `select_by="val_loss"` inside `train()` picked the wrong checkpoint this
  run (see `2026-08-31-tmax30-held-out-score.md`) -- still an open
  methodological question, not addressed by this submission.
