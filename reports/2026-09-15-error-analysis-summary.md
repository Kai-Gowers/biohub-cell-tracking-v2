# Error analysis of the cell-tracking pipeline: summary

*Kai Gowers, 2026-09-15. Condensed from `2026-09-14-error-analysis-0842.md`, its appendix and
`2026-09-14-jump-frames-postprocessing.md`. All numbers are on 20 held-out videos the model never saw.*

## The data

Each of the 199 training videos is 100 frames of a 104 µm cube of a developing embryo (light-sheet, nuclear label; voxel 1.625 µm in z, 0.406 µm in x/y). A cell moves 1.8 µm per frame (median). The videos are crops of only two embryos, `44b6` (71 videos, ~370 cells per frame) and `6bba` (128 videos, ~165 cells per frame). The hidden test set is from different embryos.

Annotation is sparse. On average 7 of about 240 cells per frame are labelled and followed as tracks (median 26 frames): 2.8% of cells overall, 0.8% on `44b6`, 5.4% on `6bba`. The competition metric only scores predicted links that touch an annotated cell, which is 3.6% of the links we output. Over-linking of unannotated cells is invisible locally, so the leaderboard is the only true cross-embryo signal.

## Where the 0.842 pipeline loses points

Held-out score 0.807 overall, 0.615 on `44b6`, 0.833 on `6bba`. Per stage:

| | all | 44b6 | 6bba |
|---|---|---|---|
| annotated cells detected (within 7 µm) | 95.2% | 93.0% | 95.4% |
| correct link, given both cells detected | 94.6% | 86.0% | 95.5% |
| divisions predicted (19 annotated) | 0 | 0 | 0 |

Of every annotated link we miss, 57% trace back to a cell the detector never found and 36% to the linker choosing the wrong partner while both cells were detected. Fixing all linker errors would raise the edge Jaccard from 0.82 to about 0.91; fixing all detector misses, to about 0.90. The two halves are worth the same.

## Findings

1. **Crowding, not embryo, decides whether a link is right.** At the same distance to the nearest other cell, both embryos link equally well: about 72% correct when a neighbour is within 7 µm, 99% beyond 14 µm. `44b6` scores 0.2 lower only because 40% of its links have a neighbour within 8 µm, against 8% on `6bba`. Applying the pooled curve to each embryo's density predicts 88% vs 95% linking; observed 86% vs 95.5%.

2. **The link scores are the limit, not how we pick among them.** When the linker is wrong, the true partner was its second choice 66% of the time, and a third of wrong links beat the true one by a wide margin. Exact (Hungarian) assignment on the same scores was worse than greedy; a stricter threshold would not remove the confident mistakes. The pair scorer sees only the two cells and their offset, never the neighbours.

3. **Detector misses are about brightness, not position.** Recall is 89% for the dimmest fifth of cells and 98% for the brightest, and flat across depth, time and local density. On `44b6` the typical miss is two nuclei 3 to 6 µm apart merged into one detection on the 4x-decimated grid; on `6bba` it is a dim, shallow nucleus with nothing detected nearby.

4. **Global motion compensation does not apply.** The rigid frame-to-frame image shift is 0.3 µm (median), a fifth of a voxel. Annotated cells move together five times more than the frame does; that is tissue flow, not stage motion. Subtracting each frame pair's median displacement would remove a third of apparent cell motion but would not separate neighbours 5 to 8 µm apart, where the errors are.

5. **Whole-frame jumps are the one systematic failure with a concrete fix.** In 4% of frame pairs the field shifts more than 3 µm. The pair scorer cannot tell which way the field moved and misses 30% of links in pairs shifted over 5 µm. The 0.942 public pipeline's node transformer, which attends over all detections of both frames, cuts this to 10%. Its notebook post-processing then raises it back to 32%, because the motion relink caps hops at 5.5 or 10 µm and drops links over 14 µm. On the held-out set the post-processing is a wash (edge Jaccard -0.009, division term +0.008), but it costs 0.078 on the one video with sustained motion.

## What this means for the current work

- Switching to the 0.942 pipeline (in progress on branch `replicate-0942`, two seeds at epoch ~260 of 400) addresses the linker half directly: its transformer sees neighbouring candidates, which is exactly what the crowding and jump-frame results call for.
- The detector half needs better recall on dim cells and on touching nuclei; that is a data and resolution question, not model capacity.
- The first experiment once our own checkpoints finish: compare the ILP stage against the full pipeline on the jump videos, and if the pattern holds, make the post-processing's motion relink motion-aware (subtract the frame pair's median displacement, or skip the relink above about 4 µm) while keeping the division step.
- Every counted error can be inspected in 3D in the error browser (overview report, section 9).
