# Comparison: v2 baseline vs. `sample_solution` (real leaderboard 0.90)

Source reviewed: `../sample_solution` (a repackaged public artifact,
`biohub-tracking-support-pack-400ep-snapshot-v1`). Code read in full:
`repo/src/biohub_tracking/{io,img_proc,metrics,division_metrics}.py`,
`repo/src/biohub_tracking/models/{temporal_unet,simple_node_transformer}.py`,
`repo/scripts/{train_unet_transformer,predict_unet_transformer,augmentations,dataspec,evaluate}.py`,
`source_scripts/train_full_frame_center_detector.py` (header + config only),
`ARTIFACT_MANIFEST.json`, `weights/unet_transformer/split_0/config.json`.

## Architecture / process diff

| | v2 (this repo) | v1 (`biohub-cell-tracking`) | sample_solution |
|---|---|---|---|
| Detector | single-frame 3D U-Net | 3-frame window, temporal mix at bottleneck only (1x1x1 conv) | 2-frame window, full multi-head temporal self-attention at every encoder stage except full-res |
| Edge model | none (pure µm-distance bipartite assignment) | learned cross-attention head, trained on GT + nearest-peak decoys | learned cross-attention head (`SimpleNodeTransformer`), trained via `detect_and_match`: runs real peak detection + NMS on the live logits every step, matches to GT within 5um, trains on that exact detected/matched set |
| Selection | scipy exact assignment, degree <=1/<=1, no divisions | scipy exact assignment, degree <=1/<=2, ILP-costed | greedy probability-sorted, degree <=1/<=2, used by default; a global tracksdata+SCIP ILP (`--use-ilp`) exists but is OFF by default |
| Repair | none | 5 deterministic passes | unknown -- manifest states the shipped graph is "before notebook-level graph repair," and that notebook is not in this artifact |
| Training | 20 epochs, batch 4, no augmentation | 50 epochs, batch 2, no augmentation | 400 epochs, batch 16, multi-GPU, external cloud (RunPod scripts present), random brightness + random 3D flip augmentation every step |
| Inference | none | none | flip TTA: averages detection logits over identity + 3 flips (z excluded, correctly, since z is ~4x coarser) |
| Grid | isotropic 1.625 um, mean-pool downsample | same | same isotropic 1.625 um grid (`downsample=(1,4,4)` against the same anisotropic SCALE), but via strided decimation, not mean-pooling |

Grid resolution is NOT a difference -- `(1,4,4)` decimation against
`SCALE=(1.625, 0.40625, 0.40625)` lands on the same isotropic 1.625 um grid
either way. Their strided (non-mean-pooled) downsample is, if anything, a
theoretically worse choice by our own docs' reasoning, and they still scored
higher -- reinforcing that this specific choice is not decisive.

## Ranked reasons for the 0.90 vs. our score

1. **Training budget.** 400 epochs, batch 16, multi-GPU cloud compute vs.
   our 20-50 epochs on a laptop/Kaggle session. For a model with a learned
   edge head, this plausibly explains most of the gap on its own -- an
   undertrained edge transformer is not meaningfully better than no edge
   model, which is consistent with v1's own history of edge-model tweaks
   not moving the score much.
2. **Edge model trained on its own detections, not GT+decoys.** The single
   most sophisticated design choice here. It directly targets what v1
   already measured (rank-1 edge accuracy 0.98 -> 0.58 as local crowding
   increases): the transformer never trains against a synthetic proxy
   candidate set, only the real one.
3. **Free accuracy from augmentation + TTA.** Neither v1 nor v2 do either.
   Flip TTA alone is close to a free 4-way ensemble at inference.
4. **Richer temporal modeling** (attention at every encoder scale vs. one
   1x1 conv at the bottleneck vs. v2's none at all).
5. **NOT the selection algorithm.** Even the 0.90 run uses plain greedy
   per-pair thresholding by default; the "proper" global ILP solver ships
   but is disabled. This matches v1's own refuted finding that selection is
   not where the missing points are -- do not prioritize re-adding ILP.
6. **Unknown last mile.** A "notebook-level graph repair" step and an
   auxiliary rescue detector are referenced in the manifest but not present
   in this artifact (`full_frame_center_included: false`). Some slice of
   0.90 is not auditable from what we have.

## Implication for v2

Given the "start narrow, add on evidence" rule, the highest-leverage next
step is **not** reintroducing ILP or repair (item 5 says that's not where
the points are). It's training the current single-frame detector much
longer, with augmentation, before deciding whether a learned edge model is
worth adding back at all. That isolates whether the *detector* is the
bottleneck first.
