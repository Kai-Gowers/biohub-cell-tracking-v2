# Deep Research Critique: Biohub Cell Tracking Pipeline

## Scope and assumptions

This report critiques the pipeline described in `pipeline-diagram(1).html` for the Kaggle **Biohub – Cell Tracking During Development** competition. I assume GPU memory, training time, and inference time are not design constraints, as requested. I still distinguish choices that appear to have been inherited from compute/runtime constraints from choices that were motivated by the metric or prior experiments.

The key competition facts are: the data are 3D+time fluorescence movies with anisotropic raw spacing `(z,y,x) = (1.625, 0.40625, 0.40625) µm`; the training annotations are sparse; the hidden test embryos are disjoint from training embryos; nodes match within 7 µm; and the official score is **adjusted edge Jaccard + 0.1 × division Jaccard**. The edge Jaccard is also scaled by a total-node overprediction penalty.

## Executive assessment

The current pipeline is technically coherent and has several good engineering choices, but it is still structurally close to a compact baseline. Its largest limitation is **not detector capacity**. The pipeline’s own error analysis says roughly **57% of counted edge errors occur when the correct cells were already detected**, versus 42% from missed detections. That points directly at association as the largest source of headroom.

The current association model reduces every candidate pair to two 16-dimensional endpoint features plus displacement/distance, scores that pair independently with a small MLP, then makes irreversible greedy decisions. This means the model does not know what *other* candidate cells are competing for the same source or destination, has no explicit track history, no velocity/acceleration state, no multi-frame context, and no ability to revise a locally attractive edge when it makes the rest of the movie implausible. Modern cell-tracking work such as **Trackastra** explicitly targets this weakness by contextualizing all detections across a temporal window, while the official Biohub baseline already uses cross-attention between all nodes in adjacent frames instead of independent pair scoring.

The second major weakness is **hard, early commitment**. The detector is thresholded at 0.985, hard NMS chooses one peak, and everything below threshold disappears before tracking. Yet temporal continuity can be exactly the evidence needed to rescue a weak or ambiguous detection. Ultrack’s central insight is to retain multiple hypotheses and let temporal consistency resolve ambiguity later. A point-tracking version of that idea fits this competition extremely well: internally over-generate candidates, give them learned node/edge costs, and let a whole-movie optimizer decide which candidates to output. Because only selected nodes are submitted, the internal proposal pool does not itself incur the node-count penalty.

The third major weakness is **sparse-label handling**. The detector puts a 1 on each annotated center voxel and treats every other voxel as a negative with 0.01 weight. Those “negatives” include many real, unlabeled nuclei. In a 64³ grid there are 262,144 voxels; even after a 100× negative downweight, the aggregate negative weight remains enormous relative to a handful of labeled centers per frame. This explains the extreme sigmoid saturation and the need for a 0.985 threshold and plateau tie-break. Sparse-microscopy methods such as ELEPHANT instead maintain an explicit *unlabeled* state and ignore unknown voxels; positive-unlabeled learning literature gives a more formal alternative. I would replace the one-hot BCE target with masked sparse supervision plus a smooth center heatmap and an offset/flow head.

Finally, there are two evaluation issues that should be audited **before changing the model**:

1. The HTML says 20 of 199 *videos* are held out with a fixed seed, but the official competition says multiple crops share an embryo and the hidden test is embryo-disjoint. If those 20 videos are not grouped by embryo ID, the validation split is not truly honest for this hidden-test regime.
2. The HTML describes plain edge Jaccard plus the division term but does not describe the official **adjusted node-count penalty**. If `score_local.py` really omits that adjustment, model selection is optimizing the wrong objective. If the code does include it and the HTML is merely incomplete, update the documentation and verify it against the official package.

These two audits have higher priority than training another checkpoint.

---

## What is already strong

### Physical units and coordinate handling

Centralizing raw-voxel, isotropic-grid, and micron conversions is exactly right. The 15 µm candidate-link radius is also data-driven rather than arbitrary. I would keep the physical-unit representation for every motion, gating, and scoring feature.

### Shared train/inference preprocessing

Using the same frame preparation function for training and prediction avoids a very common calibration bug. Keep this invariant even if the preprocessing itself is changed.

### Live-detection edge training

Training the link model on detections produced by the detector, rather than only pristine ground-truth coordinates, is directionally correct because the linker then sees localization error and false/missed proposals. I would preserve this, but add a staged curriculum so the linker is not starved of training signal early.

### Logit-space TTA and subvoxel refinement

Averaging logits across spatial transforms is reasonable. Subvoxel refinement is clean and harmless, although because the metric tolerance is 7 µm it is unlikely to be where the largest score gain lives.

### Empirical ablation discipline

The pipeline records measured changes, seed variance, and failed ideas. That discipline is valuable. The main change I would make is the **validation unit and selection metric**, not the fact that you ablate.

---

## Stage-by-stage critique

### Stage 0 / validation design: the first thing I would fix

The official data description says the first token of each sample name identifies the embryo, multiple crops can come from one embryo, and hidden train/test embryos are disjoint. Therefore a random video-level holdout can leak embryo-specific acquisition appearance, density, morphology, and motion statistics into both train and validation.

The current HTML only says “20 of 199 videos” are held out with a fixed seed. If this is not embryo-grouped, rerun all headline numbers with embryo-disjoint validation. With the visible training data apparently dominated by very few embryo identities, a leave-one-embryo-out style evaluation may be harsh, but it is much closer to the actual generalization problem.

The four visible `test/` clips should not be treated as a test set at all. Kaggle explicitly states they are copies from train used for notebook debugging; submission reruns swap in a hidden embryo-disjoint test set of roughly training-set scale.

The local metric also needs exact parity with the official evaluator. The official metric penalizes total predicted-node overcount:

`adjusted_jaccard = max(0, edge_jaccard * (1 - 0.1 * (T_pred - T_true) / T_true))`.

The current HTML explains ignored sparse-label edges and plain edge Jaccard, but not this adjustment. That omission matters because a detector can look better locally while becoming too dense for the hidden test. Public competition experiments have already shown that exact failure mode: a participant increased local edge Jaccard with higher-recall detection but saw leaderboard score fall sharply because the denser predictions incurred hidden-test false links and count penalties.

**Recommendation:** checkpoint selection should use the official tracking metric on full held-out movies, not detector validation loss. Run multiple seeds per ablation and report confidence intervals because the current pipeline itself observed up to ~0.05 spread between nominally identical runs.

### Stage 1 / preprocessing: useful simplification, but too aggressive for an unconstrained design

Per-frame percentile normalization solves gross microscope gain variation, but doing it independently on every frame can erase or distort temporal intensity continuity. Since endpoint appearance is used for identity, a nucleus should not look artificially brighter or dimmer simply because the frame’s global histogram changed.

I would test per-video robust normalization or temporally smoothed percentile statistics, then augment explicitly for gain, bias, gamma, photobleaching, depth attenuation, shot noise, blur, and PSF variation. This gives the model invariance without injecting frame-to-frame normalization jitter.

The 4×4 mean pool in y/x is more nuanced. It makes the grid physically isotropic at 1.625 µm and massively reduces compute, but it also discards 15 of every 16 in-plane samples. Under a 7 µm matching tolerance this may not hurt centroid localization much, which is why I would **not** call it the first bottleneck. The concern is crowded-cell separation and discriminative appearance for linking.

With unlimited compute, compare three controlled variants:

- current `(1,4,4)` pooling,
- `(1,2,2)` pooling with anisotropic early convolutions,
- native resolution with anisotropic kernels/downsampling until physical receptive fields become roughly isotropic.

Modern biomedical pipelines such as nnU-Net explicitly treat strongly anisotropic axes separately rather than requiring an immediate box-average to cubic voxels.

The uint8 memory-mapped cache is explicitly a speed/storage trick. Under the stated assumptions, preserve normalized data in fp16/bf16 or recompute it; quantizing to 8 bits is unnecessary. The custom chunk decoder is operationally useful for Kaggle’s internet-off environment, not a modeling choice.

### Stage 2 / detector: two frames and co-located temporal attention are too weak

The detector encodes `t-1` and `t`, then applies attention across the two temporal tokens at the **same spatial index**. Convolutional receptive fields mean this is not literally blind to nearby motion, but the attention mechanism itself cannot say “the feature at `(z,y,x)` now corresponds to `(z+1,y-3,x+2)` previously.” It only mixes co-located features.

This is a poor match to tracking, where the core latent variable is displacement.

With no compute limit, I would use a centered temporal window, for example 5 frames for detection (`t-2 … t+2`), and one of:

- local 3D cross-attention over a spatial neighborhood in adjacent frames,
- deformable attention that predicts where to sample the neighboring frame,
- a learned 3D flow/correlation volume,
- or a ConvGRU/video-transformer hierarchy with explicit spatial motion.

ELEPHANT is useful evidence here: it learns a voxel-wise 3D displacement field from sparse links and combines it with nearest-neighbor association. Ultrack likewise reports gains from nonlinear registration in deforming tissues. The design principle is that motion should be an explicit prediction, not only an implicit feature.

The current U-Net width `16/32/64`, two-frame window, and training time of roughly half an hour on one L40S are all signs of an experiment-throughput-oriented model. The official baseline itself defaults to `32/64/128` and explicitly uses gradient checkpointing and skips full-resolution temporal attention to reduce memory/runtime. With unlimited resources I would remove those restrictions — but only after fixing the supervision and association model. A 4× larger U-Net feeding the same independent pair MLP is unlikely to be the best use of compute.

### Stage 2 loss: treat unknown as unknown

This is one of the changes I expect to matter most.

Current target:

- labeled center voxel = positive,
- everything else = negative with weight 0.01.

But sparse annotations mean “not labeled” is not equivalent to “background.” ELEPHANT directly models center, periphery, background, and unlabeled voxels and excludes unlabeled voxels from supervised loss. Positive-unlabeled cell-detection research formalizes the same issue.

I would use a 3-part mask:

- **positive:** a Gaussian/ellipsoidal heatmap around an annotated nucleus,
- **known background:** only voxels that can be labeled confidently, for example low-intensity regions and/or regions sufficiently far from any plausible nucleus,
- **unknown:** everything else, excluded from the supervised classification term.

Then add a 3D offset head that predicts the vector from a positive neighborhood voxel to the subvoxel center. Spotiflow’s multiscale heatmap + flow-regression formulation is a strong example of why a smooth localization target is preferable to a saturated one-voxel classifier.

This removes the pathological incentive that produces 1.0 plateaus, makes confidence more calibratable, and gives you a natural subvoxel output.

A second option is a positive-unlabeled risk estimator. I would test both masked supervision and PU loss; the former is easier to debug.

### Stage 3 / peak extraction: hard thresholding is too early

`TAU=0.985` and hard 5×5×5 NMS make irreversible decisions before temporal context can help. The threshold is also partly compensating for the unusual sparse BCE setup rather than representing calibrated probability.

For an unconstrained tracker, I would split **proposals** from **submitted nodes**.

Generate a high-recall internal candidate set at each frame, retaining multiple plausible peaks and a detector confidence. Do not submit them all. Pass them to the association model and global optimizer. A weak candidate that forms a highly coherent track can be selected; a strong isolated candidate can be rejected. This is the point-detection analogue of Ultrack’s multiple-hypothesis philosophy.

The hard cap of 1,500 detections per frame should be removed unless EDA demonstrates it can never bind. A fixed safety cap is a classic resource/defensive-engineering choice, not a scientific assumption.

The stated reason for excluding z-flip TTA is also incorrect as written. **Anisotropic sampling does not by itself make reflection along z invalid.** A z reflection preserves the z spacing; it is not an axis permutation. There may still be valid reasons not to use z-flips — depth-dependent optics, illumination, or biological orientation — but that must be established empirically. Test it rather than rejecting it because z is coarser.

### Stage 4 / edge scorer: this is the clearest architectural bottleneck

The current edge scorer sees:

- a 16-D feature at source,
- a 16-D feature at destination,
- `dz, dy, dx`,
- distance.

It then scores each pair independently.

This ignores the information that actually disambiguates tracking in crowded scenes: **competition and context**. If source A has two plausible daughters B and C, whether A→B is correct depends partly on who can plausibly link to C, what A was doing over the previous frames, local tissue motion, neighboring cells, and whether B participates in a coherent future track.

The official Biohub baseline already uses bidirectional cross-attention between *all nodes* in frame `t` and *all nodes* in `t+1`. Trackastra goes further and uses the full spatiotemporal context of detections over a temporal window while supporting divisions.

I would replace the edge MLP with a temporal detection-token transformer or graph neural network. Each candidate token should contain:

- a richer 64–256D image embedding from a local 3D patch / detector feature pyramid,
- physical `(z,y,x)` coordinates and time,
- detector confidence,
- local density / nearest-neighbor geometry,
- estimated radius/scale/intensity if available,
- learned motion/flow features.

Use 5–9 frames in the association window. Alternate within-frame self-attention with cross-frame/local temporal attention, or use a sparse spatiotemporal graph. Predict pair logits jointly, plus birth/death and division logits.

Train with hard negatives chosen from realistic crowded neighborhoods and with listwise/matching losses, not only independent weighted BCE. A row/column softmax or Sinkhorn-style auxiliary objective can explicitly teach candidate competition.

### Stage 5 / greedy matching: upgrade it, but only after upgrading the scores

Greedy sorting can make an irreversible local error: a slightly over-scored edge can consume a source or destination and block a better global configuration. So yes, under unlimited resources I would replace it.

However, evidence from this very competition suggests that **global optimization by itself is not the main lever**. A public competitor replaced a motion-aware rule linker with a whole-graph ILP using plain distance costs and tied/lost. This is exactly what one should expect: an optimizer cannot recover information that is absent from its costs.

Therefore the order should be:

1. learn a much stronger contextual edge/node score,
2. then run whole-movie global selection.

A min-cost-flow or ILP formulation can include node selection, ordinary edge selection, birth/death, division, motion smoothness, and count priors. It should allow one outgoing edge normally and two only under a division variable. The optimizer can also penalize implausible acceleration and favor track continuity.

The important conceptual change is that **detection and association hypotheses remain revisable until the final global selection**.

### Missed detections: use tracking-guided redetection, not just gap edges

The pipeline attributes ~42% of counted edge errors to missed detections. If a cell is confidently tracked at `t-1` and `t+1` but weak at `t`, that is valuable evidence that the detector should revisit a local region at `t`.

Public competition experiments have reported sizable gains from motion-aware association plus one-frame gap handling. But because the official graph expects consecutive-frame lineage, I would not simply output `t → t+2`. Instead:

1. predict the missing location from the track state / flow,
2. query a lower-threshold detector or local refinement network around that location in frame `t`,
3. insert the node only if image evidence plus temporal evidence is strong,
4. output the normal consecutive edges.

This can recover edge recall without flooding the entire frame with low-threshold detections.

### Divisions: currently rational to postpone, irrational to permanently omit

Ignoring divisions was reasonable while the edge tracker was still weak: divisions are rare and weighted only 0.1, and naive rule-based fork attachment often creates many false positives. Public competition reports confirm that post-hoc geometry can hurt.

But in the unlimited-resource design, divisions should be part of the association model and global constraints. Trackastra specifically models dividing objects, and Ultrack’s ILP includes division states.

There is also a competition-specific opportunity: publicly shared synthetic 3D microscopy data with a very large number of labeled division events is available under a permissive license. Since Kaggle allows freely and publicly available external data and pretrained models, this could be used to pretrain a division representation, followed by careful in-domain fine-tuning. Domain shift must be validated.

Do this **after** edge association is strong. A perfect division Jaccard is worth only +0.1, whereas edge mistakes dominate the base score and false division edges can also damage edge Jaccard.

### Stage 7 / training: current setup is optimized for fast iterations, not maximum performance

The current schedule — 20 random two-frame windows per video per epoch, 30 epochs, batch 4 with accumulation, a compact U-Net, and ~30 minutes per L40S run — is a resource-conscious experiment loop.

With unlimited resources, I would change the *training regime* more than simply increasing epochs.

First pretrain the detector with sparse-aware center/offset losses. Then train the association model on a mixture of ground-truth-centered candidates, detector-jittered candidates, and live predictions. Finally joint-finetune the whole system.

This fixes a current bootstrapping problem: the edge loss is near zero early because it only exists after the detector generates peaks above a very high hard threshold. It also addresses the fact that peak extraction is non-differentiable: “joint training” does not mean the edge loss can move a chosen peak coordinate in a smooth end-to-end way.

The reported gradient spikes of 30–99 before clipping and the fact that clipping at 1.0 is “load-bearing” are warning signs. I would inspect per-loss gradient norms, use separate learning rates/optimizers or explicit loss balancing, add warmup, and consider EMA/SWA. The goal is not to remove clipping at all costs; it is to understand why the optimization is so unstable.

The current validation loss only measures detection and is a poor checkpoint selector for a tracking competition. With unlimited time, run full tracking validation on every checkpoint and select by the exact competition metric.

Finally, the observed run-to-run score spread near 0.05 means single-seed ablations are not trustworthy. Use at least 3–5 seeds for important architecture decisions and ensemble the best cross-validation models at inference.

---

## A stronger unconstrained architecture

A research-grade redesign could look like this.

### 1. Robust preprocessing

Use per-video or temporally smoothed robust intensity normalization. Retain native or 2×-downsampled y/x resolution. Use anisotropic convolution/downsampling early rather than forcing immediate cubic voxels. Train with microscopy-specific augmentation: gain/bias/gamma, shot noise, blur, depth attenuation, photobleaching, small affine transforms, elastic deformation, and frame-to-frame drift.

### 2. Multi-frame sparse-aware detector

Use a 5-frame centered window and a shared 3D encoder. Add local/deformable temporal attention or a 3D motion/correlation head. Predict:

- center heatmap,
- subvoxel offset,
- optional nucleus scale/radius,
- feature embedding,
- motion/flow,
- uncertainty.

Train annotated centers with smooth heatmaps and offsets; mask unknown voxels rather than treating them as negatives.

### 3. High-recall internal proposal graph

Decode a generous candidate set, retaining lower-confidence alternatives. Each candidate becomes a node with detector confidence, embedding, coordinates, uncertainty, local morphology, and motion features. The proposal set can be much denser than the final submission because only selected nodes will be output.

### 4. Multi-frame contextual association model

Run a sparse transformer/GNN over 5–9 frames. Let each candidate compare against all plausible competing nodes, not one pair at a time. Predict edge scores jointly, ordinary-vs-division state, birth/death confidence, and perhaps node validity.

ASCENT suggests that self-supervised contrastive embeddings can also be effective in dense 3D fluorescence data. Pretraining the token representation on unlabeled movies is particularly attractive here because sparse annotation, not raw imagery, is the real data bottleneck.

### 5. Whole-movie inference

Solve a global min-cost flow / ILP over all candidates in the movie with:

- node costs,
- edge costs,
- birth/death costs,
- division variables,
- motion/acceleration consistency,
- optional learned count prior,
- biological degree constraints.

Use internal t→t+2 hypotheses only to trigger redetection; submit normal consecutive-time nodes and edges.

### 6. Tracking-guided redetection

For gaps or uncertain regions, use the predicted trajectory and image data to search for a missing nucleus locally. This is substantially safer than lowering the global detection threshold.

### 7. Ensemble

Train embryo-disjoint folds and multiple seeds. Ensemble heatmaps/embeddings/edge logits, cluster candidate points, then perform one final global solve. Given the current model’s high seed variance, ensembling should be more valuable here than in a perfectly stable setup.

---

## What I would *not* prioritize first

I would **not** start by merely doubling U-Net channels. Detection recall is already reported near 95% within a generous 7 µm tolerance, and most counted edge errors are association errors.

I would **not** simply turn on a global ILP with the existing pair MLP and expect a dramatic gain. Improve the edge signal first.

I would **not** spend much time refining subvoxel localization before fixing identity. The metric’s 7 µm match radius makes identity more valuable than another fraction of a voxel of centroid accuracy.

I would **not** add heuristic divisions as a bolt-on. Division needs contextual learned evidence and a topology-aware selector.

---

## Resource-constraint audit

| Current choice | Resource-driven? | Assessment with unlimited compute |
|---|---|---|
| uint8 mmap cache | **Definitely** | Remove quantization or use fp16/bf16; cache only for convenience. |
| custom no-internet Zarr decoding | **Operational constraint** | Keep only if Kaggle packaging still matters; not a modeling choice. |
| plain attention backend because fused kernels crash | **Definitely** | Fix software/hardware path and use efficient kernels; do not let this constrain architecture. |
| 16/32/64 U-Net | **Probably partly** | Increase only after fixing supervision/context; official baseline itself is wider. |
| two-frame temporal window | **Probably partly** | Move to centered 5-frame detector and 5–9-frame association. |
| no full-resolution temporal attention | **Likely compute-driven** | Re-test with local/deformable attention; official baseline documents large memory/runtime savings from skipping it. |
| 30 epochs, 20 windows/video/epoch, ~30 min/run | **Definitely experiment-throughput oriented** | Train to convergence and select by whole-movie tracking metric. |
| batch 4 + grad accumulation | **Memory-driven** | Increase effective batch if useful, but optimization stability matters more than raw batch size. |
| 1,500 detections/frame cap | **Likely defensive/runtime** | Remove unless proven never to bind or biologically justified. |
| 4-way x/y TTA only | **Partly runtime / partly assumption** | Broaden TTA/ensemble; empirically test z reflection. |
| 4×4 y/x mean-pool | **Mixed** | Isotropy is a valid motive, but the degree of downsampling also saves 16× spatial compute; ablate 2× and native anisotropic models. |
| tiny independent edge MLP | **Strongly simplicity/throughput oriented** | Replace with contextual temporal transformer/GNN. |
| greedy linking | **Not primarily resource-driven** | It was a measured simplification; upgrade only together with stronger scores. |
| no divisions | **Metric/data driven** | Reasonable early; add jointly later. |
| no ILP/repair | **Prior-experiment driven** | Do not reverse blindly; revisit with better node/edge scores. |

---

## Priority experiment sequence

1. **Audit validation and metric.** Make the split embryo-disjoint; use the official adjusted score; verify local scorer byte-for-byte/count-for-count against the RoyerLab package. Re-score all saved checkpoints before retraining anything.
2. **Association-only A/B on frozen detections.** Current MLP vs official-style all-node cross-attention vs a 5-frame Trackastra-like transformer. Measure edge recall *conditioned on both endpoints being detected*, true-partner rank, and performance by crowding level.
3. **Proposal graph instead of hard final peaks.** Lower the internal proposal threshold, retain alternatives, and train node/edge validity jointly. Keep final output count conservative through global selection.
4. **Sparse-aware detector loss.** Compare current weak-negative BCE against masked unknown supervision and PU learning; change one-voxel targets to Gaussian heatmap + offset regression.
5. **Explicit motion.** Add flow/deformable temporal correspondence and tracking-guided redetection. Measure recovery of the current 42% detection-related edge failures.
6. **Whole-movie global optimization.** Once contextual edge scores are strong, compare greedy, Hungarian, min-cost-flow, and ILP with identical proposals/scores.
7. **Resolution ablation.** Only now compare 4× pool vs 2× pool vs native anisotropic input; inspect crowded-region performance rather than only overall recall.
8. **Joint division modeling.** Pretrain on public synthetic divisions if useful, then fine-tune in-domain and let the global tracker decide one-vs-two children.
9. **Scale and ensemble.** Wider/deeper encoders, longer training, self-supervised pretraining, multi-seed and multi-fold ensembles.

If I had to bet on a single modeling change after the validation audit, I would bet on **replacing the independent edge MLP with a contextual multi-frame transformer/GNN**. If I could choose two changes, I would pair that with **high-recall internal proposals + global hypothesis selection**, so temporal evidence can rescue weak detections instead of being forced to accept the detector’s thresholded decisions.

---

## Suggested evaluation dashboard

For each validation embryo/density stratum, report:

| Metric | Why it matters |
|---|---|
| node recall within 7 µm | isolates detection coverage |
| submitted node count / estimated true count | exposes adjusted-Jaccard risk |
| edge recall given both endpoints detected | isolates association model quality |
| end-to-end edge Jaccard | main objective |
| true partner rank among candidates | tells whether scorer or optimizer is failing |
| greedy-vs-global assignment regret | quantifies value of the optimizer |
| gap/redetection recovery rate | isolates missed-node recovery |
| performance vs local cell density | crowded frames are the hardest regime |
| division precision/recall/Jaccard | secondary 0.1 objective |
| per-seed mean ± std | prevents noise-driven model selection |

This decomposition is more informative than detector recall plus one aggregate tracking score.

---

## Bottom line

The current pipeline is a good compact baseline, but the design is still **too local and too irreversible** for an unconstrained attempt at the competition ceiling. The three most important redesign principles are:

**(1) validate on the right distribution and exact metric; (2) treat sparse labels as unknown rather than weak negatives; (3) let detections reason jointly through time before making final node/edge decisions.**

The current 3D U-Net is not the part I would discard first. I would keep its image encoder idea, make its temporal reasoning motion-aware, change its detector target/loss, and replace the small pair MLP + greedy linker with a multi-frame contextual association model followed by global hypothesis selection.

## Primary and high-value sources

1. Kaggle competition data and evaluation pages, Biohub – Cell Tracking During Development:
   - https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data
   - https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
2. RoyerLab official baseline and metric:
   - https://github.com/royerlab/kaggle-cell-tracking-competition
   - https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
3. Gallusser & Weigert, **Trackastra: Transformer-based cell tracking for live-cell microscopy**, ECCV 2024:
   - https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/9819_ECCV_2024_paper.php
4. Bragantini et al., **Ultrack: pushing the limits of cell tracking across biological scales**, Nature Methods 2025:
   - https://www.nature.com/articles/s41592-025-02778-0
5. Sugawara et al., **Tracking cell lineages in 3D by incremental deep learning (ELEPHANT)**, eLife 2022:
   - https://elifesciences.org/articles/69380
6. Dominguez Mantes et al., **Spotiflow**, Nature Methods 2025:
   - https://www.nature.com/articles/s41592-025-02662-x
7. Zhao et al., **Positive-unlabeled learning for cell detection with incomplete annotations**, MELBA:
   - https://www.melba-journal.org/papers/2022%3A027.html
8. Han & Lu, **ASCENT: Annotation-free Self-supervised Contrastive Embeddings for 3D Neuron Tracking in Fluorescence Microscopy**, ICCV 2025:
   - https://openaccess.thecvf.com/content/ICCV2025/html/Han_ASCENT_Annotation-free_Self-supervised_Contrastive_Embeddings_for_3D_Neuron_Tracking_in_ICCV_2025_paper.html
9. Public Biohub competition experiment log showing motion-aware/gap-linking gains and a dense-detection leaderboard regression:
   - https://github.com/JunhaoLiXD/Biohub_Cell_Tracking/blob/main/docs/experiments.md

Community competition experiments are treated as anecdotal evidence, not peer-reviewed authority. The architectural recommendations above are primarily grounded in the official metric/baseline and peer-reviewed cell-tracking literature.
