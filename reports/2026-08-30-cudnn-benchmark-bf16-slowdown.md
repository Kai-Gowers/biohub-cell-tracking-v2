# bf16 run is ~5x slower per epoch than every prior fp16 run

## Context

`89131af` ("NaN divergence: prefer bf16 autocast, log the triggering sample")
switched `train.py`'s AMP autocast dtype from an implicit `float16` to
`bfloat16` when `torch.cuda.is_bf16_supported()`, to fix the recurring NaN
divergence (see `reports/2026-08-29-nan-divergence-bf16.md`). That part
worked: the resulting Kaggle run (Kaggle notebook version 12) completed 15
clean epochs with `n_skipped_steps=0` and `scaler_scale=1` the entire way --
further than any prior run (previous longest: epoch 34, parent-softmax
variant) -- and never diverged. It stopped only because it ran out of the
session's 11h time budget.

## The problem

That same run took **10h39m** wall clock for 15 epochs (`AMP: bf16`,
`199/199 volumes cached`, `0.39 batch/s` flat from epoch 1 through 15,
~38.9 min/epoch). Kaggle's notebook version history shows every prior run
(versions 7-11, all pre-bf16, i.e. running the old implicit `float16`
autocast) completed in **3h04m-4h31m total**, including the same ~54 min
cache-build step every session pays. Netting that out:

| version | dtype | total wall time | cache build | training time | epochs reached | est. min/epoch |
|---|---|---|---|---|---|---|
| 7-11 (pre-bf16) | fp16 | 3h04m-4h31m | ~54 min | ~130-190 min | up to ~34 | ~7-9 |
| 12 (bf16) | bf16 | 10h39m | ~54 min | ~585 min | 15 | ~39 |

That's a real ~5x per-epoch regression, not just improved visibility from
surviving longer -- an earlier version of this report incorrectly claimed the
slowdown was pre-existing and simply never observed because prior runs always
crashed first. The version-history timings rule that out: prior runs were
genuinely faster per epoch, not just shorter overall.

## Diagnosis

Diffing `89131af` against its parent, the *only* change touching the training
loop is the AMP dtype selection (`torch.autocast(..., dtype=autocast_dtype)`
replacing the old implicit-fp16 `torch.autocast(..., enabled=amp)`) plus
unrelated sample-logging plumbing -- `batch_size`, `edge_every`,
`frames_per_volume`, and the edge-scorer's per-step cost
(`edge_train.sample_edge_pair`, which already ran every step before this
change too) are all untouched. That isolates the dtype switch as the single
controlled variable behind the regression.

The model is a 3D U-Net (`UNet3D`, all `Conv3d`). cuDNN's tensor-core kernel
coverage for `Conv3d` has historically been much thinner for `bfloat16` than
for `float16` (fp16 3D-conv tensor-core kernels are older and more complete).
`torch.backends.cudnn.benchmark` was never set anywhere in this codebase, so
it defaulted to `False` -- cuDNN picks one algorithm per call via its built-in
heuristic rather than timing several candidates and caching the fastest. That
heuristic has years more tuning behind it for fp16 Conv3d than for bf16
Conv3d, so it's a plausible spot for bf16 to land on a slow, non-tensor-core
fallback algorithm while fp16 kept landing on a fast one -- and it would
produce exactly the flat, high-slowdown-factor, zero-variance profile
actually observed (0.39 batch/s dead flat across all 15 epochs, no matter the
loss dynamics).

This is a plausible mechanism, not a confirmed one -- there was no
opportunity to isolate it with a standalone timing test (no CUDA available
locally) before deciding how to proceed.

## Fix attempted: negative result

`train.py` was changed to set `torch.backends.cudnn.benchmark = True` when
`device.type == "cuda"` (commit `5fb620e`), on the theory that cuDNN's
default heuristic algorithm pick for bf16 `Conv3d` was landing on a slow
fallback that benchmark mode's try-several-and-cache-the-fastest approach
would route around.

**Verified on Kaggle (notebook version 13): no effect.** ~38.2 min/epoch,
statistically identical to version 12's ~38.9 min/epoch pre-fix. Pulling
exact `seconds`/epoch out of the checkpoint histories for a clean
same-step-count comparison confirms this is a real, reproducible ~6x
regression, not noise or GPU-assignment variance:

| checkpoint | dtype | seconds/epoch |
|---|---|---|
| 0.841 baseline | fp16 | ~390-400 |
| parent-softmax | fp16 | ~379-381 |
| edge-feature-neck | fp16 | ~414-418 |
| bf16 run (v12) | bf16 | ~2334 (~38.9 min) |
| bf16 + cudnn.benchmark (v13) | bf16 | ~2292 (~38.2 min) |

cudnn.benchmark doing nothing rules out algorithm *selection* as the
mechanism -- it implies there is no fast tensor-core bf16 `Conv3d` kernel
available for this shape on this cuDNN/GPU combination at all, so exhaustive
benchmarking still lands on the same slow fallback every time. The
`cudnn.benchmark = True` line is harmless (numerics unaffected,
`n_skipped_steps=0` etc. all still held) but did not solve the problem and
should not be relied on as the fix.

## Root cause found: `torch.cuda.is_bf16_supported()` is a false positive on Kaggle's T4s

`scripts/bench_amp_dtype.py` (a standalone, no-data timing script, isolating
the model's Conv3d/InstanceNorm3d stack from its attention layers) run on a
real Kaggle session confirmed the GPU is a **Tesla T4** (Turing, compute
capability 7.5 -- no bf16 tensor cores at all), yet
`torch.cuda.is_bf16_supported()` returns `True` there on this torch/CUDA
build (`torch 2.10.0+cu128`, `cudnn 91002`). `train.py` trusted that check
and picked bf16 on exactly the hardware its own comment said to avoid.

Isolated timings, `cudnn.benchmark` both on and off (confirms again it's
irrelevant -- identical either way):

| component | fp16 | bf16 | bf16/fp16 |
|---|---|---|---|
| full model (fwd+bwd) | 233.9-239.1ms | 1595.9-1616.0ms | ~6.8x |
| Conv3d/InstanceNorm3d stack alone | 36.2-36.8ms | 712.9-716.1ms | **~19.5-19.7x** |
| attention (`TemporalAttention3d`) alone | 57.3-58.2ms | 109.8-110.5ms | ~1.9x |

The Conv3d stack is almost the entire story -- ~20x slower in bf16, vs. only
~1.9x for attention. This also almost certainly explains the original bf16
training run's slowdown (Kaggle notebook version 12/13): very likely also a
T4, hitting the same false-positive check, not an unlucky one-off.

## Fix

`train.py` now checks `torch.cuda.get_device_capability(device)[0] >= 8`
(Ampere or newer) instead of `torch.cuda.is_bf16_supported()` to decide
between bf16 and fp16+GradScaler. This restores the originally-intended
behavior: fp16 on a T4 (fast, with the pre-existing divergence risk), real
tensor-core bf16 only on hardware that can actually run it fast.

**This does not solve the NaN divergence** -- it only fixes which dtype gets
chosen on which hardware. Since Kaggle's default/free GPU tier is
commonly a T4, the practical result is: training is very likely back to the
original fast-but-occasionally-diverging fp16 behavior (crashes observed
between epoch 9-34 historically, patience/checkpointing already handles
this and every such crashed run still produced a usable checkpoint,
including the real submitted 0.841 LB score). bf16 remains available and
correct for whenever a session lands on Ampere+ hardware.

## How to apply

Don't re-conclude "the joint edge-scorer path is what's slow" or "cuDNN
algorithm selection" -- both were investigated and ruled out earlier in
this thread. Don't trust `torch.cuda.is_bf16_supported()` as a proxy for
"bf16 will be fast" anywhere in this codebase -- it answers "can this
tensor type exist," not "does this GPU accelerate it," and Kaggle's T4s are
proof the two diverge. The remaining open question, now cleanly separated
from the dtype-selection bug: does fp16's divergence need a real fix (e.g.
tighter/separate gradient clipping for `conv_params`, since diagnostics
show `conv_grad_norm_max` pinned at the clip ceiling almost every epoch), or
is the existing patience/checkpoint behavior (get however far a run gets,
keep the best checkpoint) good enough to keep training on? Not decided as
of 2026-08-30.
