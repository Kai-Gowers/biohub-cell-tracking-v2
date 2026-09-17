# 2026-09-14 -- Can more GPUs / more memory speed up training? Plus a view on model size and augmentation

Prompted by the PI: "request 4 GPUs on one machine, max out the memory for as big a batch as
possible; is the model too big for the limited training data? add more augmentation." The
user then narrowed the ask: measure the speed question, answer the other two as an opinion
(no ablation runs). State of the recipe chains at the time of writing: seed 0 at epoch 114
(best `val_score` 0.8605 @ 95), seed 314159 at epoch 108 (best 0.851), both chained on `short`.

Artefacts: `dist/profile/profile_train_step.py` (the real `pack_train` step, cuda-synchronised
per phase, trained weights `pack_s0.pt` loaded so detection counts are realistic; `run_profile*.sh`
drivers; `results_*.jsonl`), `dist/profile/bench_ddp.sh` (end-to-end 1/2/4-GPU epoch timing),
logs `dist/profile/srun_*.out`. All on g019 (4 x L40S 48 GB, torch 2.6, TF32 conv on by default).

## 1. Where a training step goes (batch 8, one L40S)

| phase | s | share |
|---|---|---|
| data wait (6 workers, memory-mapped cache) | 0.002 | 0% |
| host-to-device | 0.001 | 0% |
| U-Net forward (with gradient checkpointing) | 0.08 | 12% |
| detection loss | 0.002 | 0% |
| detect-and-match + feature gather (the per-sample Python loops) | 0.013 | 2% |
| edge targets | 0.001 | 0% |
| node transformer + pair MLP forward | 0.06-0.07 | 10% |
| edge loss (per-sample loop) | 0.003 | 1% |
| backward | 0.49 | 74% |
| clip + AdamW step | 0.002 | 0% |
| **total** | **0.66-0.69** | |

0.66 s/step x 2099 steps = 1390 s, i.e. the ~1450 s/epoch the chains report (the rest is the
pack validation pass). The step is **GPU-bound**: the torch profiler shows 4.66 s of CUDA kernel
time in 5.4 s of wall time over 10 steps. The 2026-09-11 hypothesis ("CPU-side per-sample loops
dominate") is wrong for the trained regime; the loops are ~3%. Top kernels (fp32/TF32): 3D
convolution backward 28% of CUDA time, convolution forward 13%, the per-voxel temporal
attention's `bmm` 9%, cuDNN NCHW->NHWC transposes 7%, BatchNorm fwd+bwd 8%, elementwise 10%.
Peak memory at batch 8: 8.7 GiB of 48.

## 2. More memory / bigger batch: no

| config | GPUs | batch | s/step | samples/s | peak GiB |
|---|---|---|---|---|---|
| recipe (fp32/TF32, grad ckpt) | 1 | 8 | 0.66 | 12.2 | 8.7 |
| no gradient checkpointing | 1 | 8 | 0.61 | 13.2 | 11.3 |
| cudnn.benchmark | 1 | 8 | 0.64 | 12.6 | 9.1 |
| channels_last_3d | 1 | 8 | 0.64 | 12.5 | 8.7 |
| torch.compile (U-Net) | 1 | 8 | 0.64 | 12.5 | 8.7 |
| recipe | 1 | 16 | 1.49 | 10.8 | 17.3 |
| recipe | 1 | 32 | 3.60 | 8.9 | 34.5 |
| no ckpt | 1 | 32 | 3.74 | 8.6 | 44.9 |
| bf16 autocast | 1 | 8 | 0.48 | 16.6 | 6.6 |
| bf16 + no ckpt | 1 | 8 | 0.45 | 17.9 | 7.0 |
| bf16 + channels_last | 1 | 8 | 0.45 | 17.7 | 6.4 |
| **bf16 + torch.compile** | 1 | 8 | **0.39** | **20.7** | 6.5 |
| bf16 | 1 | 32 | 2.64 | 12.1 | 26.3 |
| DataParallel (U-Net only; the old `data_parallel=True` path) | 4 | 8 | 0.43 | 18.5 | 3.3 |
| DataParallel | 4 | 32 | 2.36 | 13.5 | 13.1 |

1. **Throughput per sample *falls* as the batch grows** (12.2 -> 10.8 -> 8.9 samples/s for
   8 -> 16 -> 32). Eight two-frame windows are already 16 volumes of 64^3 through a
   32/64/128-channel 3D U-Net; the GPU is saturated, so a bigger batch only lengthens the
   step. Worse, the node transformer pads every sample to the batch's maximum detection count
   and runs the pair MLP on the padded N x N grid, so its cost grows faster than linearly
   (0.06 -> 0.16 -> 0.40 s). Filling the 48 GB is possible (batch 32 without checkpointing
   uses 45 GB) and makes an epoch slower, not faster. Memory is not the constraint; FLOPs are.
   A bigger batch would also change the recipe (batch 8, lr 1e-4, no scheduler).
2. **Same-numerics tricks are worth 3-8%** (no grad checkpointing, cuDNN autotune,
   channels-last, torch.compile in fp32). Not worth touching the running chains for.
3. **bf16 autocast is worth 1.4x, 1.7x with torch.compile** (0.66 -> 0.39 s/step; an epoch
   ~24 -> ~14 min). This changes the numerics of the recipe (the pack trained in fp32), so it
   is an experiment, not a free speedup. Exposed as `scripts/train.py --amp bf16 --compile`,
   off by default; smoke-tested for 2 epochs (`dist/profile/srun_smoke_bf16.out`). The
   earlier repo's bf16 runs hit NaN divergence on a different model
   (reports/2026-08-29-nan-divergence-bf16.md); `TrainingDivergedError` still guards.

## 3. More GPUs: yes, with DDP -- not with DataParallel

`nn.DataParallel` over the U-Net gives 1.5x on 4 GPUs at batch 8 and 1.1x at batch 32 (the
same 1.6x measured on 2026-09-11). Structural reason: DP scatters the batch, runs the U-Net
on 4 replicas, then gathers the full (B, 2, 32, 64^3) feature map (0.5 GB per 8 samples) back
to GPU 0, where the node transformer, both losses and the backward through the gather run
alone; the other three GPUs idle for most of the step.

So `pack_train` now has a proper multi-process path, `scripts/train.py --ddp`
(`TrainConfig.ddp`): one process per visible GPU, each with `batch_size / n_gpu` samples,
gradients averaged with one flat all-reduce after `backward()`, **SyncBatchNorm** so BN
statistics are computed over the global batch of 8 exactly as on one GPU, `DistributedSampler`
with `drop_last`, rank 0 does the evaluation, checkpointing and the `max_hours` stop
decision (broadcast to the others). Per-sample losses are means, so averaged gradients equal
the single-GPU batch-8 gradient up to floating point; the optimizer trajectory is the recipe's,
not a new one. Checkpoints are identical in format (`SyncBatchNorm` keeps `BatchNorm3d`'s
keys) so the running chains can be *resumed* under `--ddp` and their checkpoints still load on
one GPU for inference.

Measured end-to-end (30-volume subset, 2381 windows, one epoch, `dist/profile/srun_bench_ddp.out`):

| config | GPUs | per-GPU batch | epoch train time | speedup |
|---|---|---|---|---|
| single GPU (`--single-gpu`) | 1 | 8 | 209 s | 1.0x |
| `--ddp` | 2 | 4 | 105 s | 2.0x |
| `--ddp` | 4 | 2 | **55 s** | **3.8x** |
| `--ddp --amp bf16 --compile` | 4 | 2 | 88 s (includes ~60 s of torch.compile warm-up; steady state was not separated in a 1-epoch run) | n/a |

Losses after the epoch were the same to within run-to-run variation (single: edge 0.00576 /
det 0.0332; DDP x4: 0.00628 / 0.0365; DDP x2: 0.00582 / 0.0331), the pack validation recall
0.82-0.83 in all three, and the DDP checkpoint loads through `load_pack_model` on one GPU.
Scaled to the full run: ~1450 s/epoch -> ~380 s/epoch, 400 epochs in ~1.8 days of running
time instead of ~6.7, once a 4-GPU allocation is held.

The scaling is close to linear because the step is compute-bound and the model is tiny
(one 8 MB gradient all-reduce per step); SyncBatchNorm's per-layer all-reduces and the
smaller per-GPU batch cost the remaining 5%. This is the largest speedup available *without
changing the recipe*. `--ddp --amp bf16 --compile` should compound (bf16+compile was 1.7x on
one GPU) but needs a multi-epoch measurement to separate the compile warm-up, and inherits
bf16's numerics caveat.

Practicalities on this cluster: `sbatch --gres=gpu:4 --cpus-per-task=32 scripts/slurm_train.sbatch
... --ddp` (the sbatch script needs no other change; `--num-workers 6` is per rank, 24 loader
processes in total). Today every L40S/A100 node on `short` was partially allocated by other
users and even 1-GPU jobs sat on `Priority`; a whole-node request will typically queue longer
than a 1-GPU one, so the wall-clock gain is (speedup) x (fraction of time actually running).
Pinning `--gres=gpu:l40s:4` or `gpu:a100:4` is not needed since v100 nodes are excluded already.

## 4. Is the model too big for the data? (opinion, with the numbers that inform it)

**Short answer: no, 2 M parameters is not "too big"; the risk is sparse *labels*, not
capacity, and the remedy for that is regularisation and early selection, not a smaller net.**

Sizes: 2.08 M parameters -- U-Net 1.50 M (encoder 0.86 M, decoder 0.55 M, temporal attention
0.08 M), node transformer 0.58 M (four cross-attention blocks 0.53 M, pair MLP 0.04 M). That is
small for a 3D U-Net (three stages, widths 32/64/128); typical biomedical 3D U-Nets are
5-30 M. The data: 179 training crops, 16 788 two-frame windows of 64^3 voxels (~4.4 G voxels
per epoch of *image*), but labels are sparse:

| | `44b6` | `6bba` | total |
|---|---|---|---|
| training crops | 71 | 128 | 199 |
| annotated nodes | 20 197 | 113 121 | 133 318 |
| annotated edges | 19 826 | 109 057 | 128 883 |
| divisions | 26 | 125 | 151 |
| annotated nodes per frame | | | mean 7.0, median 6, max 33 |

So the detector gets ~133 k positive voxels out of ~4.4 G, the edge head ~129 k positive
pairs, and every *unannotated* cell (hundreds per frame) is pushed towards "background" only
through the 0.01 negative weight. Two embryos only. The failure mode to expect is therefore not
classic over-parameterisation but the detector memorising the annotated frames of two embryos
and the edge head fitting their motion statistics. What we can see of that so far:

- Training losses track the pack's own history almost exactly (det 0.00396 vs their 0.00418 at
  epoch 110), so the port trains like the original.
- Held-out `val_score` is still rising but decelerating: +0.0106 per 25 epochs over epochs
  5-110, +0.0053 from epoch 50, +0.0030 from epoch 75 (residual std 0.005). Seed 314159 is
  flat at 0.845-0.851 from epoch 60.
- A direct train-vs-held-out comparison (same metric, same ILP-stage pipeline, 20 training
  volumes with the same 5/15 embryo mix vs the 20 held-out; `dist/profile/train_vs_heldout.py`;
  only two snapshots finished before the job was cancelled):

  | seed 0 epoch | train total | held-out total | train `44b6` | held-out `44b6` | train `6bba` | held-out `6bba` |
  |---|---|---|---|---|---|---|
  | 25 | 0.851 | 0.836 | 0.843 | 0.673 | 0.852 | 0.856 |
  | 100 | 0.880 | 0.859 | 0.860 | 0.747 | 0.881 | 0.872 |

  The gap is small on `6bba` (0.01) and large on `44b6` (0.11-0.17), and it is not growing
  much between epochs 25 and 100. That is a *data* gap -- the crowded embryo is under-represented
  (71 crops, 20 k nodes) and its held-out crops are different fields of view -- not a
  capacity gap. A smaller model would not close it; more `44b6`-like data or stronger
  invariances might.
- The pack's reported validation recall of 0.978 at epoch 381 was measured on 40 volumes they
  also trained on, so it is a training number; it is not evidence that 400 epochs generalise.
  Our `_best` selection on the genuinely held-out `val_score` is the right guard.

If one wanted a capacity test anyway, the cheap version is `--unet-layers 16,32,64` (0.96 M)
and `48,96,192` (3.94 M) at matched epochs; note the two recipe seeds differ by 0.013 at
epoch 100, so single runs need to differ by more than that to mean anything. Not run.

## 5. Augmentation (opinion)

What the recipe already does at *training* time: an additive brightness shift (+-0.1) and
independent z/y/x flips (2^3 = 8 axis-aligned symmetries). The "8 views" at *inference* are
the same 8 flips used as test-time augmentation in `pack_predict` (`det_tta`); TTA does not
regularise training. So training-side augmentation is minimal: 8 geometric views and one
scalar intensity offset. My view on what would help, in order:

1. **Intensity/contrast jitter** (per-window gain and gamma). Normalisation is per video by
   its own quantiles, so the residual brightness/contrast differences *between embryos* are
   exactly the shift the hidden test embryos present; the recipe's +-0.1 offset barely covers
   it. This is the one I would add first.
2. **In-plane 90-degree rotations.** The model grid is isotropic (1.625 um), so rotating the
   (y, x) plane is as legitimate as the flips and takes the symmetry group from 8 to 16 views
   for free (z stays put: the optical axis is not exchangeable).
3. **Additive noise** (small sigma) -- cheap robustness for the detector at its 0.965 threshold.
4. **Time reversal** (swap the frames, transpose the transition matrix; skip windows with a
   division since a reversed division is a merge). Doubles edge supervision, but cell motion is
   not time-symmetric and the transformer is directional, so it may hurt; test alone.

Not recommended: elastic deformations (they move the sparse point labels and the detector's
7 um match tolerance is tight), mixup/cutout (sparse labels make the target ill-defined), and
xy<->z axis permutations (PSF anisotropy).

All four are implemented as opt-in flags -- `scripts/train.py --aug rot90,intensity,noise,treverse`
(`TrainConfig.extra_augs`, off by default; coordinate/target consistency unit-tested in
`tests/test_extra_augs.py`) -- and were smoke-tested, but *no training run was launched with
them*, per the user's redirect. The expected gain is modest (a few hundredths) and shows up on
`44b6` first if it is real; the two-seed spread (0.013 at epoch 100) is the bar.

## 6. Decisions

- The two recipe chains stay as they are (fp32, batch 8, one GPU each).
- `--ddp` is the recipe-preserving way to use a 4-GPU node; `--amp bf16 --compile` is a
  further 1.7x as an experiment; both default off. `--aug` and `--unet-layers` exist for
  later ablations; four such 100-epoch arms were submitted and cancelled before starting.
- Bigger batches and DataParallel are closed as options for this model.

## 7. Addendum, 02:35 -- moving the two recipe runs to 4-GPU DDP

Submitted `ct-pack-s0-ddp` (3003485) and `ct-pack-s314159-ddp` (3003486) on `long`
(`--time=3-00:00:00 --gres=gpu:4 --cpus-per-task=32 --mem=160G`, `--ddp --max-hours 70`). A
5-day limit was refused: maintenance reservation `root_67` covers every node 2026-09-18
08:00-16:00. Each job forks at start from the chain's live checkpoint into its own file
(`CT_FORK_FROM=dist/models/pack_s0.pt` -> `--out dist/models/pack_s0_ddp.pt`, new feature in
`slurm_train.sbatch`, alongside `CT_CANCEL_JOBNAME=<name>` for an automatic hand-over). The
1-GPU chains were deliberately left running as the fallback: the DDP resume has only been
tested for one epoch on a subset, so the first `val_score` of each `_ddp` log must land on the
existing curve (~0.85-0.86) before the old chain is cancelled by hand
(`scancel -n ct-pack-s0`, `scancel -n ct-pack-s314159`). Cost of the overlap: one night of two
redundant single-GPU jobs.
