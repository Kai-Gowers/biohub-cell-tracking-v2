# 2026-09-14 — The data, and a first look at the 0.842 pipeline

Overview for the PI. Deliberately short: what the data is, how much of it is annotated, a few
plain numbers per pipeline stage, then why links fail and where. Deeper breakdowns (by crowding, depth, score margin, tracks,
qualitative crops) are in `2026-09-14-error-analysis-0842-detailed-appendix.md` and can be pulled
in as needed. Figures live in `reports/figures/2026-09-14-error-analysis-0842/`.

## 1. What we train and test on

**Videos.** 199 training videos ("volumes"), every one 100 frames of 64 x 256 x 256 voxels, uint16.
The voxel is 1.625 µm in z and 0.406 µm in x/y, so each frame is a 104 µm cube of a developing
embryo (light-sheet, nuclear label). Frame-to-frame a cell moves 1.8 µm (median).

**Embryos.** The 199 videos are crops from just two embryos: `44b6` (71 videos) and `6bba` (128).
Kaggle's hidden test set is from *different* embryos. The visible `test/` folder holds 4 videos
that are also in `train/` (placeholders for notebook debugging), so the leaderboard is the only
cross-embryo signal we have.

**Our split.** We train on 179 videos and hold out 20 (5 x `44b6`, 15 x `6bba`,
`dist/heldout_split.json`). Every number below is on those 20 held-out videos, which the checkpoint
never saw.

## 2. How much is annotated

| | all 199 | 44b6 | 6bba | held-out 20 |
|---|---|---|---|---|
| annotated cells (nodes) | 133,318 | 20,197 | 113,121 | 14,177 |
| annotated links (edges, t -> t+1) | 128,883 | 19,826 | 109,057 | 13,692 |
| annotated divisions | 151 | 26 | 125 | 19 |
| annotated tracks | 4,726 | 421 | 4,305 | 518 |
| median track length (frames) | 26 | 50 | 22 | 23 |
| annotated cells per frame | 6.9 | 3.0 | 9.0 | 7.2 |
| estimated true cells per frame | 237 | 369 | 165 | 202 |
| **fraction of cells annotated** | **2.8%** | **0.8%** | **5.4%** | 3.5% |

The annotation is sparse: on average 7 of ~240 cells in a frame are labelled, and those 7 are
followed through the video as tracks (median 26 frames). `44b6` is the denser embryo (369 cells per
frame vs 165) and the more thinly annotated one (3 cells per frame, 0.8%).
`data01_annotated_vs_estimated_cells_per_frame.png` shows this per video.

**What the metric sees.** The score is an edge Jaccard (TP / (TP + FP + FN)) over annotated links,
with predicted cells matched to annotated ones within 7 µm, plus 0.1 x a division Jaccard, minus a
small penalty when the number of predicted cells is far from the estimated true count. A predicted
link that touches an *unannotated* cell is ignored, not penalised. On the held-out set the pipeline
predicts 363,745 links and only 13,089 of them (3.6%) are scored at all. Over-linking of
unannotated cells is invisible locally.

## 3. The 0.842 pipeline, by stage

`main` branch: 2-frame 3D U-Net detects cell centres on a 4x-decimated grid; a small MLP scores
every candidate pair of detections within 15 µm between consecutive frames; a greedy pass keeps the
highest-scoring non-conflicting links. No divisions are predicted. Checkpoint
`detector_tmax30_epoch23.pt`, leaderboard 0.842 (2026-08-31).

| stage | metric | all 20 | 44b6 (5) | 6bba (15) |
|---|---|---|---|---|
| **Detector** | annotated cells found (within 7 µm) | 95.2% | 93.0% | 95.4% |
| | position error of found cells, median | 1.8 µm | 2.1 µm | 1.8 µm |
| | predicted cells / estimated true cells | 1.06 | 1.01 | 1.08 |
| **Linker** | correct links, given both cells were found | 94.6% | 86.0% | 95.5% |
| | links the greedy pass output | 363,745 | 137,817 | 225,928 |
| | of which touch an annotated cell and are scored: correct (TP) / wrong (FP) | 12,036 / 1,053 | 1,069 / 306 | 10,967 / 747 |
| | annotated links not reproduced (FN) | 1,656 | 350 | 1,306 |
| **Divisions** | annotated / predicted | 19 / 0 | 4 / 0 | 15 / 0 |
| **Score** | edge Jaccard | 0.816 | 0.620 | 0.842 |
| | **competition score (held-out)** | **0.807** | **0.615** | **0.833** |

A predicted link that touches no annotated cell (96% of them) is ignored by the metric. An FN is
an annotated link that no predicted link reproduced, because an endpoint was missed by the detector
or because the linker connected the detected cells elsewhere.

`fig01_per_volume_score.png` shows the same per video: 0.54-0.84 on `44b6`, 0.66-0.95 on `6bba`.

## 4. Reading it

- The detector finds 95 of every 100 annotated cells in both embryos and places them within about
  one grid cell (1.6 µm) of the annotation. The 5% it misses cost roughly as much as everything the
  linker gets wrong (57% vs 42% of missed links trace back to a missed cell).
- Given two found cells, the linker connects them correctly 95.5% of the time on `6bba` but only
  86.0% on `44b6`. That 10-point gap in linking, in the embryo with twice the cell density, is
  what turns 0.83 into 0.62.
- Divisions contribute nothing yet: 19 in the held-out set, 0 predicted, so the 0.1 division term
  is exactly 0.

## 5. Why links fail

Every annotated link that was not reproduced (FN, 1,656) is given exactly one cause, by checking
whether its two cells were detected and, if so, what the linker did with them
(`fig02_error_decomposition.png`).

| missed annotated links (FN) | all 20 | 44b6 | 6bba |
|---|---|---|---|
| a cell at one or both ends was never detected | 948 (57%) | 174 (50%) | 774 (59%) |
| -- both ends missed | 344 | 23 | 321 |
| -- one end missed | 604 | 151 | 453 |
| both cells detected, but linked to the wrong cell(s) | 593 (36%) | 168 (48%) | 425 (33%) |
| both cells detected, left unlinked | 97 (6%) | 5 (1%) | 92 (7%) |
| second child of a division (no divisions predicted) | 18 (1%) | 3 (1%) | 15 (1%) |

The same for the wrong predicted links that were counted (FP, 1,053):

| counted wrong links (FP) | all 20 | 44b6 | 6bba |
|---|---|---|---|
| true partner was detected, the linker chose another cell | 767 (73%) | 227 (74%) | 540 (72%) |
| true partner was never detected, so nothing right was available | 282 (27%) | 79 (26%) | 203 (27%) |
| both ends matched annotated cells, wrong pair | 4 | 0 | 4 |

Reading it:

- **On 6bba the detector is the larger problem**: 59% of missed links trace back to a missed cell,
  and the detector misses 4.6% of annotated cells there.
- **On 44b6 it is half and half**: the detector misses about as many cells (7%), but with cells
  twice as dense, the linker's wrong choices are as costly as the detector's misses (48% of FN),
  and 74% of its counted wrong links are pure linker choices with the right cell available.
- The linker almost never leaves two detected cells unlinked (6%). Its failure mode is choosing the
  wrong cell, not being too cautious.
- Fixing every linker error while keeping the detector as is would lift the edge Jaccard from
  0.816 to about 0.91; fixing every detector miss with the linker as is, to about 0.90. The two
  halves are worth about the same.

## 6. Linker: where the wrong links happen

Take every annotated link whose two cells were both found (12,710 links) and ask how close the
*nearest other detected cell* is to the source cell. That distance is "crowding".

| nearest other detected cell | < 7 µm | 7-8 | 8-10 | 10-14 | >= 14 |
|---|---|---|---|---|---|
| correct links, all | 72% | 86% | 92% | 96% | 99% |
| correct links, 44b6 | 75% | 84% | 89% | 92% | 100% |
| correct links, 6bba | 71% | 87% | 93% | 96% | 99% |
| share of 44b6 links in this bin | 18% | 22% | 41% | 17% | 1% |
| share of 6bba links in this bin | 3% | 5% | 21% | 32% | 39% |

Two things follow (`fig03_recall_vs_crowding.png`, left panel):

- **Crowding, not embryo, decides the link.** At the same crowding the two embryos link equally
  well; the curves lie on top of each other. Below 7 µm about one link in four is wrong; above
  14 µm one in a hundred.
- **44b6 is worse because it is denser.** 40% of its links have another cell within 8 µm, against
  8% for 6bba, and 71% of 6bba's links sit in the two easiest bins. Applying the pooled curve (top row) to
  each embryo's own distribution predicts 88% vs 95%, close to the observed 86% vs 95.5%.

**What a wrong link looks like.** The linker's edge scorer gives every candidate pair a score, and
greedy keeps the best. Of the 593 "both cells detected, linked to the wrong cell(s)" cases in
section 5, 386 are ones where the source cell itself was given a wrong outgoing link (in the other
207 the source got no link and its true child was claimed by another source, so there is no chosen
link to compare against). For those 386 (382 with scorer output in the dump) we can ask where the
true partner ranked: it was the scorer's 2nd choice in 66%, 3rd or lower in 28%, and in 6% it was
the scorer's *first* choice but that cell had already been claimed by a competing source earlier in
the greedy pass. A third of the wrong links are confident (the chosen pair outscored the true one by more
than 0.5), so a higher score threshold would not remove them. Exact (Hungarian) assignment on the
same scores was tested and is worse than greedy (-45 TP, +190 FP): the scores are the limit, not
the selection rule.

## 7. Detector: where the missed cells are

The detector missed 685 of the 14,177 annotated cells (101 on 44b6, 584 on 6bba). Its recall
barely depends on *where* a cell is; it depends on how bright the cell is
(`fig04_detection_depth_time_intensity_localisation.png`).

| cell brightness (3x5x5 voxel mean / frame median) | < 2 | 2-3 | 3-4 | 4-6 | 6-8 | >= 8 |
|---|---|---|---|---|---|---|
| cells found, all | 89.0% | 92.7% | 94.9% | 96.5% | 96.3% | 98.0% |
| number of annotated cells | 1,973 | 1,891 | 1,396 | 2,735 | 2,174 | 4,008 |

- **Brightness**: the dimmest fifth of cells is found 89% of the time, the brightest 98%. Missed
  cells are half as bright as found ones (median 2.9x vs 5.4x the frame median).
- **Not depth, time or crowding**: recall stays within 93-97% across z (< 20 µm to > 80 µm deep),
  across the video (first fifth to last fifth), and across local density (from < 2 to > 11 other
  detections within 15 µm). On 44b6 the deepest cells dip to 89-90% (n ~ 220 each), the one spatial
  effect worth noting.
- **Position error** of found cells: median 1.8 µm on 6bba, 2.1 µm on 44b6, i.e. about one cell of
  the 1.625 µm detection grid; the 90th percentile is 4.0 and 5.4 µm, against a 7 µm matching radius.

What sits near a missed cell:

| the 685 missed cells | all | 44b6 | 6bba |
|---|---|---|---|
| a detection within 7 µm, but already matched to a neighbouring annotated cell (two cells, one detection) | 91 (13%) | 27 (27%) | 64 (11%) |
| nearest detection 7-10 µm away | 431 (63%) | 70 (69%) | 361 (62%) |
| no detection within 10 µm | 163 (24%) | 4 (4%) | 159 (27%) |

Three kinds of miss show up in the image crops (`fig10_qualitative_detector.png`): very dim,
shallow nuclei with nothing detected nearby (the 6bba "no detection within 10 µm" group); two
nuclei 3-6 µm apart merged into one detection on the 4x-decimated grid (the dominant 44b6 miss);
and large, very bright nuclei that look pre-mitotic, where the detector puts its peak on the
neighbour instead. The last kind also blocks any future division prediction.

## 8. What can be added next (already computed, in the appendix)

Score-margin distributions for the linker; track-level fragmentation (12% of 44b6 tracks fully
recovered vs 52% on 6bba); and the image crops of typical linker errors.

## 9. Interactive 3D error browser

Every counted error above (1,656 FN + 1,053 FP = 2,709 cases) can be stepped through in 3D:
a ray-marched rendering of the raw intensity around the cell, frames t (orange) and t+1 (cyan)
overlaid, with GT cells (green), detections (red), the GT link (solid green), the predicted link(s)
(dashed red) and the source's candidate scores. Code on `main` (worktree
`/projects/twist2d/gowers/biohub-cell-tracking-v2-main`): `scripts/error_browser.py` (stdlib HTTP
server: index + scenes + crops cut from the zarr on demand), `viewer/index.html`, `viewer/app.js`,
`viewer/render3d.js` (plain WebGL2, no dependencies).

```
cd /projects/twist2d/gowers/biohub-cell-tracking-v2-main
PYTHONPATH=src /projects/twist2d/gowers/miniconda3/envs/cell-tracking/bin/python scripts/error_browser.py \
    --geff-dir dist/preds_val_0842_dump --candidates dist/preds_val_0842_dump/candidates --port 8765
# ~20 s to build the index, then open http://localhost:8765/  (Cursor forwards the port; if not,
# Ports panel -> Forward a Port -> 8765).  One case: http://localhost:8765/#i=3
```

Reference case `#i=3` is the wrong link from section 6 / fig09 row 1 (`44b6_1574802b`, t = 21):
true partner scored 0.96 but taken by a competing source, chosen 0.63 alternative 9 µm away in z.
Keys: arrows step, 1/2/3 frame t / t+1 / both, z/x/y/r camera presets (x = side view with z up,
which exposes z offsets), m = 7 µm matching sphere, n = candidate edges, c = copy link. Filters by
cause, embryo, video; sort by score margin, |dz|, displacement, crowding. Any other prediction
(e.g. a pack checkpoint's `--stage ilp` output with a candidate dump) can be browsed by pointing
`--geff-dir` at it.

## 10. Is there global motion worth compensating? (PI question, 2026-09-14)

Measured two ways (`scripts/motion_analysis.py` on `main`, `motion01_global_shift.png`,
`motion_stats.json`): the rigid whole-frame shift between consecutive frames by sub-voxel phase
correlation of the raw images (20 held-out videos, 1,980 frame pairs), and the median displacement
vector of the annotated links in each frame pair (all 199 videos, 14,559 pairs with >= 3 links).

| per frame pair | median | p90 | p99 | > 3 µm | > 5 µm |
|---|---|---|---|---|---|
| rigid image shift | 0.29 µm | 0.77 µm | 5.4 µm | 4.1% of pairs | 1.2% |
| median annotated-link vector | 1.46 µm | 3.6 µm | 7.8 µm | 15% | 4.3% |
| per-cell displacement, for reference | 1.82 µm | 4.1 µm | 7.8 µm | | |

- **Rigid global shift is negligible** (0.3 µm median, one fifth of a voxel in z). A whole-frame
  shift compensation would remove nothing in 95% of frames.
- **A few real acquisition jumps exist**: 4% of frame pairs shift > 3 µm and 1.2% > 5 µm; in one
  video (`6bba_f20478e9`, t = 65) the whole frame is displaced ~46 µm in z for one frame and the
  annotations follow it. Across all 199 videos 10 frame pairs (0.07%, 58 links) have a median
  annotated shift > 15 µm, i.e. beyond the linker's gate. In frame pairs with a shift > 5 µm the
  held-out missed-link rate is 21-25% vs 11% elsewhere, but they are so few that the total cost is
  well under 1% of links.
- **The annotated cells move together far more than the frame does**: the median link vector of a
  frame pair (1.46 µm) is five times the rigid shift, and its direction is consistent over time
  within a video (|mean| / mean|.| = 0.69). That is collective tissue flow in the crop, not camera
  or stage motion. Removing each pair's median vector from every link reduces per-cell motion from
  1.82 µm to 1.15 µm (residual), so a *local* flow prior could take about a third of the apparent
  motion out. It would not, by itself, separate neighbours 5-8 µm apart, which is where the linker
  fails; a scorer that sees the neighbouring candidates jointly (the node transformer in the 0.942
  pipeline) learns this flow implicitly.

Conclusion: global shift compensation is not applicable here; a local flow prior is a modest,
second-order lever, and is already covered by the transformer-based linker being trained.

**Where the jumps are** (`scripts/motion_jumps.py` -> `motion_jumps.csv`, one row per frame pair
with image shift >= 3 µm: 1,030 pairs in 106 videos; 337 >= 5 µm; 48 >= 10 µm). Three kinds:

1. **Single displaced frames** (10 videos, all `6bba`): one frame whose whole stack sits ~12 slices
   higher in z and ~15% dimmer, then the next frame is back. The image shift reads 45-52 µm into and
   out of that frame; the annotators left these frames empty (0 GT nodes, except 4 in
   `6bba_f20478e9` t=65). Frames: `6bba_f17befbc` 36, `6bba_b693381b` 67, `6bba_062c8d37` 40,
   `6bba_3fda6b25` 1, `6bba_4f99ce20` 62, `6bba_f1fde7e0` 87, `6bba_bb9f20c3` 3, `6bba_f20478e9` 65,
   `6bba_7f87b3d8` 85, `6bba_ebdf3b34` 10. Acquisition glitches; a model cannot link across them
   and, since they are unannotated, it is not penalised for failing to.
2. **Real in-plane jumps of 10-18 µm** confirmed by the annotations (image and GT-median shift
   agree): e.g. `44b6_e29f0176` 37->38 and 40->41, `44b6_d754aa59` 55->56 (held-out),
   `6bba_e5e44988` 53->54, `44b6_cf8fed6b` 60->61, `44b6_668e0cc7` 23->24. Mostly y/x, i.e. the
   embryo or stage moved between frames.
3. **Videos with sustained motion**: a dozen videos have 25-55% of their frame pairs shifting
   >= 3 µm (`6bba_0c7fa718` 54%, `6bba_767a1e17` 53%, `6bba_cdcfe533` 42%, `6bba_3db54e20` 39%,
   `44b6_d754aa59` 31%, ...). This is where a per-frame flow prior would actually change the
   linker's inputs. In the held-out set that is `44b6_d754aa59` (12 pairs >= 5 µm) and
   `6bba_3abfe10a` (4), both among the lowest-scoring videos of their embryo.

**Follow-up (same day):** on the jump frames the 0.942 pipeline's node transformer links correctly
(missed-link rate 10% vs 30% for `main` in pairs shifting >= 5 µm) but its notebook post-processing
undoes it (32%): the motion relink caps hops at 5.5 / 10 µm and drops edges over 14 µm. On our
held-out set the post-processing is a wash overall (edge Jaccard -0.009, division term +0.008) and
costs 0.078 on the one video with sustained motion. Details and proposed experiments:
`2026-09-14-jump-frames-postprocessing.md`.
