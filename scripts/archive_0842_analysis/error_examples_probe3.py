#!/usr/bin/env python3
"""For every frame that contains a missed GT cell: re-run the detector (TTA, CPU) and extract peaks with
(a) the pipeline's rule (5x5x5 NMS on the sigmoid, tie-break by index), (b) the same NMS on the LOGITS
(no saturation ties), (c) 3x3x3 NMS on the sigmoid, (d) 3x3x3 NMS on the logits.
Variant "rescue": 5x5x5 peaks plus any 3x3x3 peak that lies >= R um from every 5x5x5 peak."""
import pickle, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(8)
from cell_tracking.detect import load_model, predict_logits_tta, extract_detections
from cell_tracking.peaks import local_maxima
from cell_tracking.cache import VolumeFrames, cache_path
from cell_tracking.config import get_cache_dir, WINDOW_SIZE, TAU, grid_to_voxels, SCALE

d = pickle.load(open("dist/error_examples/tables.pkl", "rb")); misses = d["misses"]
model, _, dev = load_model("dist/models/detector_tmax30_epoch23.pt", device=torch.device("cpu"))
by_frame = defaultdict(list)
for i, m in enumerate(misses):
    by_frame[(m["name"], int(m["t"]))].append(i)
train = Path("data/biohub-cell-tracking-during-development/train")
logit_tau = float(np.log(TAU / (1 - TAU)))
out, frames_out, frames = {}, {}, {}
t0 = time.time()
for k, (key, idxs) in enumerate(sorted(by_frame.items())):
    name, t = key
    if name not in frames:
        frames[name] = VolumeFrames(train / f"{name}.zarr", cache_path(get_cache_dir(), name))
    w = frames[name].window(t - (WINDOW_SIZE - 1), WINDOW_SIZE)
    logits, _ = predict_logits_tta(model, w, dev, tta=True)
    prob = torch.sigmoid(logits)
    variants = {}
    det = extract_detections(prob, t, logits=logits)                       # (a) pipeline
    variants["pipeline"] = det.zyx
    idx1, _ = local_maxima(prob, threshold=TAU, radius=1)
    p1 = grid_to_voxels(idx1.numpy().astype(np.float64)); p2 = variants["pipeline"]
    from scipy.spatial import cKDTree
    dmin = cKDTree(p2 * SCALE).query(p1 * SCALE, k=1)[0] if len(p2) and len(p1) else np.full(len(p1), np.inf)
    for R in (4.0, 5.0, 5.5, 6.5):
        extra = p1[dmin >= R]
        variants[f"rescue_{R}"] = np.concatenate([p2, extra], axis=0) if len(extra) else p2
    frames_out[key] = {lab: len(v) for lab, v in variants.items()}
    for i in idxs:
        m = misses[i]; g = np.array([m["z"], m["y"], m["x"]], float) * SCALE
        rec = {}
        for lab, zyx in variants.items():
            rec[lab] = float(np.linalg.norm(zyx * SCALE - g, axis=1).min()) if len(zyx) else np.inf
        out[i] = rec
    if k % 50 == 0:
        print(f"{k}/{len(by_frame)} frames, {time.time()-t0:.0f}s", flush=True)
pickle.dump(dict(misses=out, frames=frames_out), open("dist/error_examples/probe3.pkl", "wb"))
print("done in", round(time.time() - t0), "s")
