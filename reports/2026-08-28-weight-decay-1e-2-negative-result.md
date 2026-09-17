# weight_decay 1e-4 -> 1e-2: negative result on the NaN-divergence chase

## Context

Per-component gradient-norm diagnostics (`46b3c4a`) on the prior epochs=90 run
(`lr=5e-4`, `weight_decay=1e-4`) showed the conv/InstanceNorm stack's weight
norm growing ~62% over its first 19 epochs with no sign of decelerating,
against ~9% for the attention stack. That inverted the earlier assumption
that attention was the fragile part under AMP, and motivated testing whether
stronger weight decay could curb the conv-side growth and, with it, the
periodic inf-gradient events GradScaler was silently catching and skipping.
`kaggle_train.ipynb` was changed to `weight_decay=1e-2` (`b6730e1`) and rerun
with everything else held fixed (`lr=5e-4`, same architecture/dropout as the
prior run).

## Result

The run still diverged to hard NaN, at epoch 22 batch 153/895 (`grad_norm`
inf/inf, non-finite loss, training stopped immediately per the
divergence-detection added in `702ab5b`). Last good checkpoint: epoch 19,
`val_loss=0.02353`.

Conv weight-norm growth over the same 19-epoch horizon used for the prior
measurement:

| run                         | \|W\|_conv @ epoch 1 | @ epoch 19 | growth  |
|------------------------------|----------------------:|-----------:|--------:|
| `wd=1e-4` (prior, per `46b3c4a` commit msg) | -                | -          | ~62%    |
| `wd=1e-2` (this run)          | 31.63                 | 50.56      | ~60%    |

A 100x increase in weight decay produced no measurable change in the growth
rate. The isolated `skipped_steps=1` events (GradScaler catching an inf
gradient and skipping that step) still recur every 6-8 epochs (epochs 1, 7,
15, 21 in this run) right up to the hard failure at epoch 22 -- same cadence
as before.

## Why this was underpowered by construction

AdamW's decay term multiplies each weight by `(1 - lr * weight_decay)` per
optimizer step, not per epoch. With `lr ≈ 5e-4` (roughly constant this early
in the cosine schedule) and `weight_decay = 1e-2`, that factor is
`1 - 5e-6` per step. Over the ~19,700 steps completed before divergence, the
decay-only pull (i.e. with gradients zeroed out) is:

```
(1 - 5e-6)^19700 ≈ exp(-5e-6 * 19700) ≈ 0.91
```

-- about a 9% shrink toward zero from decay alone, against 60%+ observed
growth from the gradient side. At the old `wd=1e-4` the equivalent pull was
`exp(-5e-8 * 19700) ≈ 0.999`, i.e. ~0.1%. Both are negligible next to the
gradient-driven growth; moving from "negligible" to "10x less negligible"
was never going to be visible against a 60% signal. Weight decay at any
value AdamW can sanely use at this learning rate is not the lever that
controls conv weight growth here.

## Conclusion

**Rejected**: raising `weight_decay` (tested at 1e-4 -> 1e-2, `lr=5e-4`
fixed) has no measurable effect on conv weight-norm growth or on the
NaN-divergence timing. Do not spend further tuning budget on weight decay
alone as a fix for this; it would need to be pushed to a regime (~0.1-1,
or paired with a much lower LR) that risks damaging the main task loss
before it could compete with the growth observed here, and that has not
been tried.

## Suggested next step

The `skipped_steps=1` events recur on a roughly fixed cadence (every 6-8
epochs) rather than escalating smoothly, which looks more consistent with an
occasional bad batch/sample than with a slowly-growing weight-magnitude
instability. Nothing currently logs which volume/frame produced the
non-finite loss. Before trying more optimizer-side knobs, add that logging
(source volume name + frame index for the batch that produced a non-finite
loss or a caught inf gradient) to check whether this is data-driven rather
than an optimization-dynamics problem.
