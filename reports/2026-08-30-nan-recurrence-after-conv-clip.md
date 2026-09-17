# Split conv gradient clipping: negative result

## Setup

Per reports/2026-08-30-conv-grad-clip.md: `conv_params` clipped separately
at `CONV_GRAD_CLIP_NORM=0.5` instead of the old joint
`clip_grad_norm_(params, 1.0)` across conv+attn+edge combined. Run with the
user's remaining ~10 GPU-hours (essentially one retrain), fp16 (dtype-gate
bug already fixed, see reports/2026-08-30-cudnn-benchmark-bf16-slowdown.md),
same lr=5e-4 / architecture as every prior comparable run.

## Result: hard divergence at epoch 15, earlier than the comparable baseline

```
non-finite loss (nan) at epoch 15 batch 300/895
diag at divergence: conv_grad_norm_max=0.500 (hit the new cap, as designed)
                    scaler_scale=16384  skipped_steps_this_epoch=0
                    |W|_conv=47.80  |W|_attn=30.31  |W|_edge=8.60
last good checkpoint: epoch 6 (val_loss=0.01877)
```

Speed was confirmed fixed first (~6.3 min/epoch, ~2.45 batch/s, matching
every historical fp16 run -- the dtype-gate fix worked as intended).

Four `GradScaler`-caught skips occurred before the hard divergence, at
epochs 1, 3, 7, and 12 -- notably *more* frequent than the historical
"every 6-8 epochs" description, not less.

Comparing to the only two directly comparable runs (same architecture,
same lr=5e-4, joint clip=1.0):

| run | clip | divergence epoch |
|---|---|---|
| 0.841 baseline | joint, 1.0 | 23 |
| weight_decay=1e-2 test | joint, 1.0 | 22 |
| **this run** | **split, conv=0.5** | **15** |

Divergence happened *earlier* than both. n=1, so this isn't proof the
change made things worse -- historical divergence epochs already ranged
22-34 across different architectures, a wide spread -- but there is **no
evidence this change helped**, and the one data point available leans
negative, not positive.

## Sample-overlap check (the other thing this run was designed to answer)

Checked all `(volume, t)` pairs across the 4 skip events + the hard
divergence (50 samples total) for a recurring specific bad sample:

- **Zero exact `(volume, t)` matches** anywhere.
- 3 pairs of events shared a volume name (different frame each time, e.g.
  `44b6_18ced818` at t=42 in epoch 1's skip vs. t=39 in epoch 15's
  divergence). Base-rate check: drawing 10 samples from ~179 training
  volumes across 5 events gives ~10 event-pairs, each with a ~44% chance of
  sharing at least one volume name by pure chance -- expected ~4.4 such
  pairs, observed 3. **Consistent with noise, not a recurring bad sample.**

This doesn't validate the growth hypothesis (the fix built on it didn't
help), but it does rule out "one specific corrupted training example" as a
simpler alternative explanation.

## Verdict and action taken

**Reverted** the split clip back to the original joint
`clip_grad_norm_(params, 1.0)` (commit pending). Given the extremely
limited remaining GPU budget, continuing to iterate on unproven clipping
variants was judged not worth the risk -- the safest use of the remaining
~7 hours is the best-evidenced configuration (joint clip, known to reach
epoch 22-34 previously), resumed from the epoch-14 checkpoint already on
disk (`/kaggle/working/detector.pt` -- checkpoints only save at completed
epoch boundaries, so epoch 15's crash did not touch it) rather than
restarting from epoch 0 and losing that compute.

## How to apply

Don't re-propose conv-specific or other per-group gradient clipping as a
fix for this divergence without new evidence -- tried once, no positive
signal, one data point suggesting it's not the answer (or at least that
0.5 isn't the right threshold, untested whether a different value would
behave differently, but not worth spending more of an already-scarce GPU
budget speculatively retuning it). The "cumulative growth erodes fp16
headroom" hypothesis is still unconfirmed either way -- this result doesn't
kill it (the fix could simply have been the wrong lever or magnitude for
it), but it doesn't support it either. The sample-log evidence (no
recurring bad sample across two separate divergence-chasing sessions now)
is the more solid conclusion to carry forward: this is very likely a
numerics-margin problem in some form, not a data-quality one, even though
the specific fix tried against it didn't pan out.
