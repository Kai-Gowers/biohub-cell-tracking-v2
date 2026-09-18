#!/usr/bin/env python3
"""napari 3D viewer for the tracking errors of one held-out volume.

    pip install "napari[all]"            # once, on a machine with a display
    python scripts/error_viewer_napari.py dist/error_viewer3d/6bba_3abfe10a.npz

Input: the bundle written by scripts/error_viewer3d_build.py. Layers (top to bottom in the layer list):

  selected track          the annotated track you picked in the dock widget (cyan), with its full history
  annotated links         green arrows from each annotated cell to its child in the next frame
  predicted links: false  red arrows (the metric's FP: a link that contradicts an annotation)
  predicted links: correct green arrows (TP)
  annotated cells         points coloured by outcome: green linked ok, amber linked wrong,
                          magenta dropped by the ILP (a peak > 0.965 existed), red only a weak peak, grey no peak
  predicted tracks        the pipeline's own tracks with a fading tail (napari Tracks layer), hidden by default
  predicted cells         all predicted nodes, small grey points, hidden by default
  ILP-dropped candidates  magenta squares, hidden by default
  raw                     the video, attenuated maximum-intensity projection

Use the time slider (or its play button) to move through frames; the viewer opens in 3D. Everything is in
physical units (z 1.625 um, xy 0.406 um) so the volume is not squashed. In the dock widget pick an
annotated track (sorted by error count) to jump to its first error frame, centre the camera on it and
show only that track's cells. Ctrl+Y toggles 2D/3D; the tail length slider on the Tracks layer controls how
much history is drawn.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

STATUS_COLORS = {
    "link_ok": "#56C271", "link_wrong": "#F2A93B", "dropped": "#D86CD1", "weak": "#EF6363", "none": "#8C96A1",
}
STATUS_LABEL = {
    "link_ok": "detected, linked correctly", "link_wrong": "detected, linked to the wrong cell",
    "dropped": "not in output: peak existed, ILP dropped it", "weak": "not in output: only a weak peak", "none": "not in output: no peak",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--tail", type=int, default=12, help="tail length (frames) for the Tracks layers")
    ap.add_argument("--start-2d", action="store_true", help="open in 2D instead of 3D")
    a = ap.parse_args()
    try:
        import napari
    except ImportError:
        print("napari is not installed: pip install 'napari[all]'", file=sys.stderr)
        return 1

    b = np.load(a.bundle, allow_pickle=False)
    meta = json.loads(str(b["meta"]))
    T, Z, Y, X = meta["shape"]
    sz, sy, sx = meta["scale_um"]
    ds = meta.get("xy_downsample", 1)
    status = meta["status"]
    scale4 = (1.0, sz, sy, sx)                 # world units for [t, z, y, x] data in raw voxels
    img_scale = (1.0, sz, sy * ds, sx * ds)   # frames may be xy-downsampled; points are not

    viewer = napari.Viewer(title=f"Lost tracks -- {meta['name']}", ndisplay=2 if a.start_2d else 3)
    viewer.add_image(b["frames"], name="raw", scale=img_scale, rendering="attenuated_mip", attenuation=0.05,
                     colormap="gray", contrast_limits=(0, 255), blending="additive")

    # --- predictions (hidden by default) ------------------------------------------------------------
    dropped = b["dropped"]
    if len(dropped):
        viewer.add_points(dropped, name="ILP-dropped candidates", scale=scale4, size=8, symbol="square",
                          face_color="transparent", border_color="#D86CD1", border_width=0.15, visible=False,
                          features={"p": b["dropped_p"]})
    pred = b["pred"]
    pstat = np.array(["none", "correct link", "false link"])[b["pred_status"]]
    viewer.add_points(pred, name="predicted cells", scale=scale4, size=5, face_color="#A8B2BF", opacity=0.6, visible=False,
                      features={"link": pstat, "track": b["pred_track"]})
    ptr = b["pred_track"]
    keep = ptr >= 0
    if keep.any():
        tdata = np.column_stack([ptr[keep], pred[keep]])  # [track_id, t, z, y, x]
        order = np.lexsort((tdata[:, 1], tdata[:, 0]))
        viewer.add_tracks(tdata[order], name="predicted tracks", scale=scale4, tail_length=a.tail, head_length=0,
                          tail_width=2, features={"link": b["pred_status"][keep][order].astype(float)}, color_by="link",
                          colormap="viridis", visible=False)

    # --- annotated cells and links ------------------------------------------------------------------
    gt = b["gt"]
    gstat = np.array(status)[b["gt_status"]]
    cells = viewer.add_points(
        gt, name="annotated cells", scale=scale4, size=9, features={"status": gstat, "track": b["gt_track"], "id": b["gt_id"]},
        face_color="status", face_color_cycle=STATUS_COLORS, border_color="black", border_width=0.05, opacity=0.95,
    )
    cells.face_color_mode = "cycle"
    pl, pc = b["pred_links"], b["pred_link_cls"]
    if (pc == 1).any():
        viewer.add_vectors(pl[pc == 1], name="predicted links: correct", scale=scale4, edge_color="#56C271", edge_width=1.2,
                           vector_style="arrow", length=1.0)
    if (pc == 2).any():
        viewer.add_vectors(pl[pc == 2], name="predicted links: false", scale=scale4, edge_color="#FF5C5C", edge_width=2.0,
                           vector_style="arrow", length=1.0)
    gl = b["gt_links"]
    if len(gl):
        viewer.add_vectors(gl, name="annotated links", scale=scale4, edge_color="#56C271", edge_width=0.8, opacity=0.7,
                           vector_style="line", length=1.0)
    gtr = b["gt_track"]
    gtracks = np.column_stack([gtr, gt])
    order = np.lexsort((gtracks[:, 1], gtracks[:, 0]))
    viewer.add_tracks(gtracks[order], name="annotated tracks", scale=scale4, tail_length=a.tail, head_length=0, tail_width=2,
                      features={"status": b["gt_status"][order].astype(float)}, color_by="status", colormap="turbo", visible=False)
    sel = viewer.add_points(np.zeros((0, 4)), name="selected track", scale=scale4, size=13, face_color="transparent",
                            border_color="#4FD1E0", border_width=0.2)

    # --- dock widget: pick a track ------------------------------------------------------------------
    tracks = b["tracks"]
    tracks = tracks[np.argsort(-tracks["errors"], kind="stable")]
    labels = [f"#{tr['tid']}  errors {tr['errors']}/{tr['n']}  frames {tr['start']}-{tr['end']}" for tr in tracks if tr["n"] >= 2]
    tid_by_label = {lab: int(tr["tid"]) for lab, tr in zip(labels, [tr for tr in tracks if tr["n"] >= 2])}
    first_err = {int(tr["tid"]): int(tr["first_error"]) for tr in tracks}
    try:
        from magicgui import magicgui

        @magicgui(auto_call=True, track={"choices": ["(all annotated cells)"] + labels, "label": "track"},
                  only_this_track={"label": "hide other annotated cells"})
        def follow(track: str = "(all annotated cells)", only_this_track: bool = True):
            if track == "(all annotated cells)":
                sel.data = np.zeros((0, 4)); cells.shown = np.ones(len(gt), dtype=bool)
                return
            tid = tid_by_label[track]
            m = gtr == tid
            sel.data = gt[m]
            cells.shown = m if only_this_track else np.ones(len(gt), dtype=bool)
            t0 = first_err.get(tid, -1)
            if t0 < 0:
                t0 = int(gt[m][:, 0].min())
            viewer.dims.set_point(0, t0)
            row = gt[m][np.argmin(np.abs(gt[m][:, 0] - t0))]
            viewer.camera.center = (row[1] * sz, row[2] * sy, row[3] * sx)
            viewer.camera.zoom = max(viewer.camera.zoom, 4.0)
            viewer.status = f"track #{tid}: {STATUS_LABEL[gstat[m][np.argmin(np.abs(gt[m][:, 0] - t0))]]} at t={t0}"

        viewer.window.add_dock_widget(follow, area="right", name="follow a track")
    except Exception as e:  # magicgui missing or API drift: the layers still work
        print(f"(dock widget unavailable: {e})", file=sys.stderr)

    # hover text for annotated cells
    @cells.mouse_move_callbacks.append
    def _hover(layer, event):
        i = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
        if i is not None:
            t, z, y, x = gt[i]
            viewer.status = f"annotated cell {int(b['gt_id'][i])} track #{int(gtr[i])} t={int(t)} z={int(z)} y={int(y)} x={int(x)}: {STATUS_LABEL[gstat[i]]}"

    viewer.dims.set_point(0, int(first_err.get(int(tracks[0]['tid']), 0)) if len(tracks) and tracks[0]["first_error"] >= 0 else 0)
    s = meta["summary"]
    print(f"{meta['name']}: {s['gt_nodes']} annotated cells, TP {s['tp']} FP {s['fp']} FN {s['fn']}; "
          f"status counts {s['status_counts']}; ILP-dropped candidates {s['dropped_candidates']}")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
