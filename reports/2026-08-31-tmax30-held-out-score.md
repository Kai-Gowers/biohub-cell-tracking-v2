# T_max=30 run: held-out score beats the 0.841 baseline, and val_loss picked the wrong checkpoint

## Setup

Per `2026-08-31-training-survives-full-schedule.md`, the `epochs=30`
(shrunk cosine `T_max`) run completed without a hard divergence, early
stopping at epoch 23/30 with `detector_best.pt` selected at epoch 8
(`val_loss=0.01716`, the argmin). Open question from that report: is
`detector_best.pt` (epoch 8) actually the better checkpoint, or could
`detector.pt` (epoch 23, last epoch, `val_loss=0.02214`) be better despite
its higher val_loss -- val_loss here is noisy (bounced 0.017-0.029 with no
clear trend after epoch ~7) while train loss kept dropping smoothly through
epoch 23, so the two could plausibly be within each other's noise band.

Downloaded both `detector_best.zip` and `detector.zip` from the Kaggle
session (landed in `~/Downloads`, no run info in the filename -- verified
each one's `epoch`/`val_loss` via `torch.load` against the training log
before using it, per `checkpoint_management_gotchas`). Kaggle's "zip" is
actually just the raw `.pt` file (torch's own zip-based serialization) --
no unzip step needed, only a rename. Installed as
`dist/models/detector_best_tmax30_epoch8.pt` and
`dist/models/detector_tmax30_epoch23.pt`. Ran `scripts/predict.py` against
the fixed 20-volume held-out split (`dist/heldout_split.json`, same split
every checkpoint in this repo has used) for both, then
`scripts/score_local.py --held-out`.

(Also re-verified the 0.841 baseline's number against its own checkpoint
file directly, `dist/models/detector_best_0841.pt` -- confirms
`local_scoring_validation`'s recorded 0.8031, and confirms a *different*
file, `~/Downloads/detector_best_baseline.zip`, is NOT the 0.841 checkpoint
despite the name -- it's an older 6-epoch pre-edge-scorer run, val_loss
0.02005. Another instance of the exact ambiguous-Kaggle-filename gotcha.)

## Result

| checkpoint | val_loss | held-out `SCORE` | det. recall (7µm) | node ratio |
|---|---|---|---|---|
| 0.841 baseline (epoch 23) | 0.01879 | 0.8031 | 93.7% | 0.994x |
| `detector_best_tmax30_epoch8.pt` (epoch 8, val_loss-selected) | 0.01716 | 0.7972 | 93.5% | 0.998x |
| `detector_tmax30_epoch23.pt` (epoch 23, last) | 0.02214 | **0.8074** | 95.2% | 1.055x |

**Epoch 23 wins, despite the "worse" val_loss.** It beats epoch 8 by
+0.0102 and beats the 0.841 baseline itself by +0.0043 -- the best held-out
`SCORE` recorded in this repo to date. Per `local_scoring_validation`
(3/3 prior comparisons agreed with the leaderboard in both direction and
rough magnitude, ~0.004-0.006), this is a real signal worth a Kaggle
submission to confirm, not likely to be pure noise -- though n=1 for this
specific checkpoint pair.

**`val_loss`-based checkpoint selection produced the wrong pick this run.**
`train.py`'s `select_by="val_loss"` chose epoch 8 over epoch 23, and epoch
8 turned out to be the *worse* checkpoint by the metric that actually
matters. Reading: `val_loss` is pure detection BCE on ~16 sampled
frames/volume (noisy, and structurally blind to `edge_acc`/linking
quality, which climbed the entire run, 0.576 -> 0.780), while the held-out
`SCORE` integrates detection recall and edge-linking together across the
full volume. Epoch 23's higher recall (95.2% vs 93.5%) and higher node
ratio (1.055x vs 0.998x, i.e. slightly more nodes, but not egregiously
over-predicting) point at continued real improvement that a noisy 16
frames/volume val_loss estimate wasn't sensitive enough to detect.

**Promoted `detector_tmax30_epoch23.pt` to the canonical
`dist/models/detector_best.pt`** (the file every default path resolves).
The previous canonical file there was already a duplicate of the 0.841
baseline, independently backed up as `detector_best_0841.pt`, so nothing
was lost.

## Open question for next time

`select_by="val_loss"` inside `train()` may not be a reliable checkpoint
picker for this pipeline -- one clean counter-example now exists. Not
proposing a specific fix yet (options: checkpoint every epoch and score
the top few candidates against held-out `SCORE` directly, which is
expensive; track `edge_acc` alongside val_loss; or just always compare
`_best` vs. last-epoch with `score_local.py` before trusting either, which
is what happened here). Flagging so a future session doesn't blindly trust
`detector_best.pt` without this check when it matters.

## Next step

Submit `detector_best.pt` (now epoch 23's weights) to the Kaggle
leaderboard to confirm the +0.0043 held-out improvement over 0.841 holds
up -- this repo's local-vs-leaderboard tracking has been reliable so far,
but every real confirmation still matters given the sibling repo's history
of this exact metric failing badly.
