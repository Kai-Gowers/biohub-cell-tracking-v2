# First training run to complete without a hard NaN divergence

## Setup

Resumed-from-scratch run (not resumed from the epoch-14 checkpoint as
previously flagged as the fallback plan -- a clean run instead) on the
reverted-to-joint-clip `train.py` (split conv-only clipping tried and
reverted per `2026-08-30-nan-recurrence-after-conv-clip.md`) plus the
`is_bf16_supported()` -> `get_device_capability() >= 8` dtype-gate fix
(`2026-08-30-cudnn-benchmark-bf16-slowdown.md`), on Kaggle's usual T4.
`kaggle_train.ipynb` changed `epochs=90` -> `epochs=30` (shrinking the
cosine `T_max`) and `max_hours` 11.0 -> 9.5, on the theory that `T_max=90`
left the LR still near its initial value at every historical divergence
epoch (22-34), so a schedule matched to where training actually survives
might anneal the LR low enough, early enough, to avoid the numerics-margin
failure outright.

## Result: no hard divergence. Early stop at epoch 23/30 on patience instead.

```
Early stop at epoch 23/30: no val_loss improvement in 15 epochs (best 0.01716 at epoch 8).
```

This is the **first run in this investigation to reach a stopping
condition it chose, rather than being killed by a `TrainingDivergedError`**
-- every prior run (0.841 baseline, weight_decay test, parent-softmax,
edge-feature-neck, the split-clip test) hard-crashed somewhere in the
epoch 9-34 range, most clustering at 22-26. This run passed through that
exact window (epochs 22, 23 included) with only recoverable `GradScaler`
skips, no hard crash.

6 skip events total, epochs 1, 4, 12, 13 (implied by diag, no printed
sample list in this excerpt), 18, 20 -- same "every several epochs"
background rate as every prior run, not accelerating and not eliminated.
`conv_grad_norm_max` sits at the 0.99-1.0 clip ceiling from epoch 1 onward,
same steady-state signature as always. So the underlying numerics-margin
condition is still present and still occasionally triggering `inf`/`nan`
gradients -- what changed is that this run never escalated one of those
events into an unrecoverable divergence.

**Reading:** consistent with the `T_max=30` hypothesis -- annealing the LR
faster (rather than clipping tighter, rather than weight decay, rather
than bf16) may be what actually keeps the recoverable skips recoverable.
n=1, and the mechanism isn't proven (an early-stop-on-patience run
surviving further than a fixed-90-epoch run's crash point is exactly what
the hypothesis predicts, but it's also just consistent with this being a
lucky run) -- worth one more same-config run before trusting it, budget
permitting.

## Val loss: best checkpoint beats the 0.841 baseline's on a fair comparison

| run | config vs. baseline | best val_loss | epoch |
|---|---|---|---|
| 0.841 baseline | -- | 0.01879 | (crashed before this epoch's schedule completed) |
| parent-softmax | different edge loss | 0.01582 | 24 |
| edge-feature-neck | different edge model | 0.01888 | 6 |
| **this run** | **same architecture, T_max=30 instead of 90** | **0.01716** | **8** |

Per `metric.py`'s own guidance, only same-architecture/different-config
rows are trustworthy against each other. This run vs. the 0.841 baseline is
exactly that comparison (same model, same loss, same edge scorer -- only
the LR schedule and clip-fix bookkeeping differ) and shows a ~9% relative
val_loss improvement, reached by epoch 8 of a 23-epoch run. The
parent-softmax and edge-feature-neck rows are different-architecture
comparisons and already known (`edge_scorer_experiments` memory) to have
regressed on held-out score despite better/similar val_loss -- not a
counter-argument to this row, just a reminder not to over-read val_loss
across architecture changes.

Best checkpoint plateaued early (epoch 8) then got noisier/worse under
patience=15 for 15 more epochs before stopping -- val_loss bounced between
0.0184 and 0.0289 with no further improvement. Consistent with the
previously-flagged `val_frames_per_volume=16` sampling noise, not
necessarily a sign of overfitting (train loss kept dropping smoothly the
whole time, 0.0177 at epoch 4 down to 0.0127 at epoch 23).

## Next step

Score `detector_best.pt` (epoch 8) with `score_local.py --held-out` --
that's the number this repo actually trusts, not val_loss. If it holds up
the T_max=30 schedule change is worth keeping regardless of whether it
"fixed" the divergence outright, since (a) it's the best same-architecture
val_loss seen yet and (b) it got there without needing the full 90-epoch
budget.
