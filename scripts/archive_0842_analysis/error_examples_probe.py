#!/usr/bin/env python3
"""Probe the 0.842 detector's probability field at every missed GT node (CPU, ~1 s/frame with TTA).
Writes dist/error_examples/probe.pkl: per miss, prob at the GT grid cell, max within +-1 / +-2 cells,
offset of that max, and whether that max is itself a 3x3x3 local maximum (i.e. a peak the NMS would keep)."""
import pickle, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
torch.set_num_threads(8)
from cell_tracking.detect import load_model, predict_logits_tta
from cell_tracking.cache import VolumeFrames, cache_path
from cell_tracking.config import get_cache_dir, WINDOW_SIZE, voxels_to_grid, TAU

d = pickle.load(open("dist/error_examples/tables.pkl", "rb"))
misses = d["misses"]
model, _, dev = load_model("dist/models/detector_tmax30_epoch23.pt", device=torch.device("cpu"))
by_frame = defaultdict(list)
for i, m in enumerate(misses):
    by_frame[(m["name"], int(m["t"]))].append(i)
train = Path("data/biohub-cell-tracking-during-development/train")
out = {}
t0 = time.time(); frames = {}
for k, (key, idxs) in enumerate(sorted(by_frame.items())):
    name, t = key
    if name not in frames:
        frames[name] = VolumeFrames(train / f"{name}.zarr", cache_path(get_cache_dir(), name))
    w = frames[name].window(t - (WINDOW_SIZE - 1), WINDOW_SIZE)
    logits, _ = predict_logits_tta(model, w, dev, tta=True)
    prob = torch.sigmoid(logits)
    pooled = F.max_pool3d(prob[None, None], 5, stride=1, padding=2)[0, 0]
    is_peak = (prob == pooled) & (prob > TAU)
    P = prob.numpy(); PK = is_peak.numpy(); shp = np.array(P.shape)
    for i in idxs:
        m = misses[i]
        g = np.round(voxels_to_grid(np.array([m["z"], m["y"], m["x"]], float))).astype(int)
        g = np.clip(g, 0, shp - 1)
        rec = dict(p_at=float(P[tuple(g)]))
        for r in (1, 2, 3):
            lo = np.maximum(g - r, 0); hi = np.minimum(g + r + 1, shp)
            sub = P[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
            j = np.unravel_index(int(sub.argmax()), sub.shape); pos = lo + np.array(j)
            rec[f"p_max{r}"] = float(sub.max()); rec[f"peak_in{r}"] = bool(PK[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].any())
            if r == 2:
                rec["argmax_off_grid"] = (pos - g).tolist()
        # where is the nearest kept peak, in grid cells?
        pk = np.argwhere(PK)
        if len(pk):
            dd = np.linalg.norm((pk - g) * 1.625, axis=1); rec["nearest_peak_um_grid"] = float(dd.min())
        out[i] = rec
    if k % 50 == 0:
        print(f"{k}/{len(by_frame)} frames, {time.time()-t0:.0f}s", flush=True)
pickle.dump(out, open("dist/error_examples/probe.pkl", "wb"))
print("done", len(out), "misses probed in", round(time.time() - t0), "s")
