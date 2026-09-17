# Split gradient clipping: tighter cap for conv, isolated from attn/edge

## Context

Every fp16 training run to date has hard-diverged to NaN between epoch 9
and 34 of an intended 90 ([[nan-divergence-investigation]]). Per-component
gradient diagnostics show `conv_grad_norm_max` sitting at ~0.97-1.0 (the
clip ceiling) almost every epoch from early in training, while
`attn_grad_norm_max` stays comfortably low (0.03-0.27). Clipping was one
joint `clip_grad_norm_(params, 1.0)` across conv+attn+edge combined --
since conv's own raw gradient evidently accounts for nearly the entire
combined vector, the shared clip was really only ever constraining conv,
and attn/edge were being shrunk by the same factor as an incidental side
effect even though their own raw norms don't need it.

weight_decay was already tried and ruled out
(reports/2026-08-28-weight-decay-1e-2-negative-result.md) as a lever on the
same underlying concern (conv/InstanceNorm weight-norm growth) -- but for a
reason specific to *that* mechanism: AdamW's decay pull at this LR is too
weak (~9% cumulative shrink over a run) against ~60%+ observed growth from
gradient updates. Gradient clipping is mechanistically different -- it caps
the *size of each update step* directly, not a continuous post-hoc shrink
-- so that negative result doesn't carry over automatically.

## The change

`train.py`: replaced the single `clip_grad_norm_(params, 1.0)` with two
independent calls -- `clip_grad_norm_(conv_params, CONV_GRAD_CLIP_NORM=0.5)`
and `clip_grad_norm_(attn_params + edge_params, OTHER_GRAD_CLIP_NORM=1.0)`.
Conv's per-step update magnitude is now genuinely capped at half of before;
attn/edge are clipped independently at the old threshold and are no longer
dragged down by conv's gradient dominating a shared vector. The overall
`grad_norm` diagnostic is preserved (combined via sqrt of sum of squares of
the two pre-clip norms, mathematically identical to what the old joint call
would have reported, since conv/attn/edge are disjoint parameter sets).

Verified locally with a 1-epoch, tiny-sample CPU smoke run before spending
any Kaggle GPU time: no crash, and the diagnostic confirms the split
behaves as designed --
`conv_grad_norm_max=0.500000` (hit the new tighter cap exactly) vs.
`edgescorer_grad_norm_max=0.999988` (now visibly hitting *its own* ceiling
independently -- previously invisible, masked by conv dominating the joint
vector).

## What this is, and isn't

This targets one specific hypothesis: that the recurring NaN divergence is
driven by *cumulative* growth (conv/decoder weights and activations slowly
eroding fp16's dynamic range margin) rather than, say, a specific recurring
data sample triggering a numerically fragile computation (e.g.
`InstanceNorm3d` dividing by a near-zero variance on some input) regardless
of how much weights have grown. The two hypotheses are genuinely
distinguishable: the skipped-steps rate observed so far (every 6-8 epochs)
has been roughly *constant* across a run rather than accelerating, which is
at least as consistent with "a rare recurring trigger" as with "closing
margin." Not resolved before spending the next Kaggle run on this.

The sample-level divergence logging added in `89131af` (unconditionally
active -- any skipped step or hard divergence already prints the exact
`(volume, t)` pairs in that batch) has never actually produced data, because
the only fp16-eligible run since it was added accidentally got bf16 instead
(the `is_bf16_supported()` bug, see
reports/2026-08-30-cudnn-benchmark-bf16-slowdown.md). The next run will
finally produce this data regardless of whether the clipping change works,
and it should be checked either way:

- Diverges again, **same** volume/frame recurring across events -> points
  to a data/sample issue, not cumulative growth -- clipping was the wrong
  lever, don't tighten it further.
- Diverges again, **different/scattered** samples each time -> consistent
  with the growth hypothesis this change targets.
- Survives meaningfully further than epoch ~23-34 -> supports the fix.

## Result

Not yet run on Kaggle as of 2026-08-30. This is a single-shot test (the
user's remaining GPU budget is ~10 hours, essentially one retrain) -- record
the outcome here regardless of which of the three cases above it turns out
to be, not just if it "worked."
