#!/usr/bin/env python3
"""napari 3D viewer for the tracking errors of one held-out volume: spheres, not voxels.

    pip install "napari[all]"            # once, on a machine with a display
    python scripts/error_viewer_napari.py dist/error_viewer3d/6bba_3abfe10a.npz

Input: the bundle written by scripts/error_viewer3d_build.py. Everything is in micrometres.

Main window (3D)
  embryo shell            translucent isosurface of the video, just enough to see the tissue outline (with a
                          bounding box); the dense MIP volume is a separate layer, off by default
  annotated cells         spheres coloured by outcome: green linked ok, amber linked wrong, magenta dropped by
                          the ILP (a peak > 0.965 existed), red only a weak peak, grey no peak
  predicted cells         smaller spheres: grey = no counted link leaves it, green = correct link, red = false link
  match offsets           thin cyan line from each annotated cell to the prediction it was matched to (7 um)
  annotated links / predicted links: correct / predicted links: false   arrows to the next frame
  annotated tracks / predicted tracks   fading tails (Tracks layers)
  selected track          cyan halo on the track picked in the dock widget
  ILP-dropped candidates  magenta squares, off by default

Dock widgets (right)
  follow a track          pick an annotated track (sorted by error count): jumps to its first error frame,
                          centres the camera, and optionally hides every other annotated cell
  status strip            one cell per frame for the followed track (same colours); click a cell to jump there

Second window (2D)       xy maximum projection with the same cells, kept in sync with the main window's time
                          and selection -- the "when does it lose the cell" view

Time slider / play button move through frames in both windows. Ctrl+Y toggles 2D/3D in the main window.
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np

STATUS_COLORS = {"link_ok": "#56C271", "link_wrong": "#F2A93B", "dropped": "#D86CD1", "weak": "#EF6363", "none": "#8C96A1"}
STATUS_LABEL = {
    "link_ok": "detected, linked correctly", "link_wrong": "detected, linked to the wrong cell",
    "dropped": "not in output: peak existed, ILP dropped it", "weak": "not in output: only a weak peak",
    "none": "not in output: no peak",
}
PRED_COLORS = {"no counted link": "#9AA4B2", "correct link": "#56C271", "false link": "#FF5C5C"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--tail", type=int, default=12, help="tail length (frames) of the Tracks layers")
    ap.add_argument("--no-2d", action="store_true", help="do not open the synchronised 2D window")
    ap.add_argument("--shell-threshold", type=float, default=0.55, help="isosurface level of the embryo shell, 0-1")
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
    um = np.array([1.0, sz, sy, sx], dtype=np.float32)   # [t, z, y, x] raw voxels -> [t, um, um, um]

    gt = b["gt"] * um
    gstat = np.array(status)[b["gt_status"]]
    gtr = b["gt_track"]
    gid = b["gt_id"]
    pred = b["pred"] * um
    pstat = np.array(list(PRED_COLORS))[b["pred_status"]]
    ptr = b["pred_track"]
    gmatch = b["gt_match"] if "gt_match" in b.files else np.full(len(gt), -1)

    viewer = napari.Viewer(title=f"Lost tracks (3D) -- {meta['name']}", ndisplay=3)
    frames = b["frames"]
    img_scale = (1.0, sz, sy * ds, sx * ds)
    shell = viewer.add_image(frames, name="embryo shell", scale=img_scale, rendering="iso",
                             iso_threshold=float(a.shell_threshold), colormap="gray", contrast_limits=(0, 255),
                             opacity=0.18, blending="translucent")
    try:
        shell.bounding_box.visible = True
        shell.bounding_box.line_color = "#6B7280"
    except Exception:
        pass
    viewer.add_image(frames, name="raw volume (MIP)", scale=img_scale, rendering="attenuated_mip", attenuation=0.05,
                     colormap="gray", contrast_limits=(0, 255), blending="additive", visible=False)

    # --- predictions -------------------------------------------------------------------------------
    dropped = b["dropped"] * um
    if len(dropped):
        viewer.add_points(dropped, name="ILP-dropped candidates", size=5.0, symbol="square", face_color="transparent",
                          border_color="#D86CD1", border_width=0.15, visible=False, features={"p": b["dropped_p"]})
    keep = ptr >= 0
    if keep.any():
        tdata = np.column_stack([ptr[keep], pred[keep]])
        order = np.lexsort((tdata[:, 1], tdata[:, 0]))
        viewer.add_tracks(tdata[order], name="predicted tracks", tail_length=a.tail, head_length=0, tail_width=2,
                          features={"link": b["pred_status"][keep][order].astype(float)}, color_by="link", colormap="viridis",
                          opacity=0.8)
    pcells = viewer.add_points(pred, name="predicted cells", size=4.0, shading="spherical", features={"link": pstat, "track": ptr},
                               face_color="link", face_color_cycle=PRED_COLORS, border_width=0.0, opacity=0.9)
    pcells.face_color_mode = "cycle"

    # --- annotated cells, links, alignment ------------------------------------------------------------
    gl = b["gt_links"] * um
    if len(gl):
        viewer.add_vectors(gl, name="annotated links", edge_color="#56C271", edge_width=0.6, opacity=0.8, vector_style="line", length=1.0)
    pl, pc = b["pred_links"] * um, b["pred_link_cls"]
    if (pc == 1).any():
        viewer.add_vectors(pl[pc == 1], name="predicted links: correct", edge_color="#56C271", edge_width=0.9, vector_style="arrow", length=1.0)
    if (pc == 2).any():
        viewer.add_vectors(pl[pc == 2], name="predicted links: false", edge_color="#FF5C5C", edge_width=1.4, vector_style="arrow", length=1.0)
    m = gmatch >= 0
    if m.any():
        off = np.stack([gt[m], pred[gmatch[m]] - gt[m]], axis=1)
        viewer.add_vectors(off, name="match offsets (annotated -> matched prediction)", edge_color="#4FD1E0", edge_width=0.5,
                           vector_style="line", length=1.0, opacity=0.9)
    gtracks = np.column_stack([gtr, gt])
    order = np.lexsort((gtracks[:, 1], gtracks[:, 0]))
    viewer.add_tracks(gtracks[order], name="annotated tracks", tail_length=a.tail, head_length=0, tail_width=3,
                      features={"status": b["gt_status"][order].astype(float)}, color_by="status", colormap="turbo")
    cells = viewer.add_points(gt, name="annotated cells", size=6.5, shading="spherical",
                              features={"status": gstat, "track": gtr, "id": gid},
                              face_color="status", face_color_cycle=STATUS_COLORS, border_color="black", border_width=0.08)
    cells.face_color_mode = "cycle"
    sel = viewer.add_points(np.zeros((0, 4)), name="selected track", size=10.0, face_color="transparent",
                            border_color="#4FD1E0", border_width=0.25)
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    # --- 2D companion window --------------------------------------------------------------------------
    v2 = None
    if not a.no_2d:
        try:
            proj = frames.max(axis=1)  # (T, Y, X)
            v2 = napari.Viewer(title=f"Lost tracks (2D xy projection) -- {meta['name']}", ndisplay=2)
            v2.add_image(proj, name="xy max projection", scale=(1.0, sy * ds, sx * ds), colormap="gray", contrast_limits=(0, 255))
            gt2 = gt[:, [0, 2, 3]]; pred2 = pred[:, [0, 2, 3]]
            p2 = v2.add_points(pred2, name="predicted cells", size=3.0, features={"link": pstat}, face_color="link",
                               face_color_cycle=PRED_COLORS, border_width=0.0, opacity=0.85)
            p2.face_color_mode = "cycle"
            if (pc == 2).any():
                v2.add_vectors(pl[pc == 2][:, :, [0, 2, 3]], name="predicted links: false", edge_color="#FF5C5C", edge_width=1.2, vector_style="arrow", length=1.0)
            if (pc == 1).any():
                v2.add_vectors(pl[pc == 1][:, :, [0, 2, 3]], name="predicted links: correct", edge_color="#56C271", edge_width=0.7, vector_style="arrow", length=1.0)
            if len(gl):
                v2.add_vectors(gl[:, :, [0, 2, 3]], name="annotated links", edge_color="#56C271", edge_width=0.5, opacity=0.8, vector_style="line", length=1.0)
            v2.add_tracks(np.column_stack([gtr, gt2])[order], name="annotated tracks", tail_length=a.tail, head_length=0, tail_width=2,
                          features={"status": b["gt_status"][order].astype(float)}, color_by="status", colormap="turbo")
            c2 = v2.add_points(gt2, name="annotated cells", size=5.0, features={"status": gstat, "track": gtr}, face_color="status",
                               face_color_cycle=STATUS_COLORS, border_color="black", border_width=0.08)
            c2.face_color_mode = "cycle"
            sel2 = v2.add_points(np.zeros((0, 3)), name="selected track", size=8.0, face_color="transparent", border_color="#4FD1E0", border_width=0.25)
            v2.scale_bar.visible = True; v2.scale_bar.unit = "um"

            _syncing = {"on": False}

            def _sync(src, dst):
                def cb(event=None):
                    if _syncing["on"]:
                        return
                    _syncing["on"] = True
                    try:
                        dst.dims.set_point(0, src.dims.point[0])
                    finally:
                        _syncing["on"] = False
                return cb
            viewer.dims.events.current_step.connect(_sync(viewer, v2))
            v2.dims.events.current_step.connect(_sync(v2, viewer))
        except Exception as e:
            print(f"(2D window unavailable: {e})", file=sys.stderr)
            v2 = None

    # --- dock widgets: follow a track + status strip --------------------------------------------------
    tracks = b["tracks"]
    tracks = tracks[np.argsort(-tracks["errors"], kind="stable")]
    tracks = tracks[tracks["n"] >= 2]
    labels = [f"#{tr['tid']}  errors {tr['errors']}/{tr['n']}  frames {tr['start']}-{tr['end']}" for tr in tracks]
    tid_by_label = dict(zip(labels, (int(tr["tid"]) for tr in tracks)))
    first_err = {int(tr["tid"]): int(tr["first_error"]) for tr in tracks}
    state = {"tid": None}

    strip = None
    try:
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QColor, QPainter, QPen
        from qtpy.QtWidgets import QWidget

        class Strip(QWidget):
            """One rectangle per frame, coloured by the followed track's status; click to jump."""
            def __init__(self):
                super().__init__(); self.setMinimumHeight(46); self.status = [None] * T; self.title = "no track selected"
            def set_track(self, tid):
                mm = gtr == tid
                st = [None] * T
                for row, s_ in zip(gt[mm], gstat[mm]):
                    st[int(row[0])] = s_
                self.status = st; self.title = f"track #{tid}"; self.update()
            def paintEvent(self, ev):
                p = QPainter(self); w = self.width(); h = self.height(); top = 16
                p.setPen(QColor("#98A1AC")); p.drawText(4, 12, f"{self.title}   (t = {int(viewer.dims.point[0])})")
                cw = max(1.0, (w - 8) / T)
                for i, s_ in enumerate(self.status):
                    x0 = 4 + i * cw
                    p.fillRect(int(x0), top, int(np.ceil(cw)), h - top - 2, QColor(STATUS_COLORS[s_]) if s_ else QColor("#2A3139"))
                cur = int(viewer.dims.point[0])
                pen = QPen(QColor("#4FD1E0")); pen.setWidth(2); p.setPen(pen)
                p.drawRect(int(4 + cur * cw), top - 2, int(np.ceil(cw)), h - top + 1)
                p.end()
            def mousePressEvent(self, ev):
                cw = max(1.0, (self.width() - 8) / T)
                i = int((ev.position().x() if hasattr(ev, "position") else ev.x()) - 4) // int(np.ceil(cw))
                if 0 <= i < T:
                    viewer.dims.set_point(0, i)
        strip = Strip()
        viewer.window.add_dock_widget(strip, area="right", name="status strip (click a frame)")
        viewer.dims.events.current_step.connect(lambda e=None: strip.update())
    except Exception as e:
        print(f"(status strip unavailable: {e})", file=sys.stderr)

    def select(tid: int | None, hide_others: bool = True):
        state["tid"] = tid
        if tid is None:
            sel.data = np.zeros((0, 4)); cells.shown = np.ones(len(gt), dtype=bool)
            if v2 is not None:
                sel2.data = np.zeros((0, 3)); c2.shown = np.ones(len(gt), dtype=bool)
            return
        mm = gtr == tid
        sel.data = gt[mm]; cells.shown = mm if hide_others else np.ones(len(gt), dtype=bool)
        if v2 is not None:
            sel2.data = gt[mm][:, [0, 2, 3]]; c2.shown = mm if hide_others else np.ones(len(gt), dtype=bool)
        t0 = first_err.get(tid, -1)
        if t0 < 0:
            t0 = int(gt[mm][:, 0].min())
        viewer.dims.set_point(0, t0)
        row = gt[mm][np.argmin(np.abs(gt[mm][:, 0] - t0))]
        viewer.camera.center = (row[1], row[2], row[3]); viewer.camera.zoom = max(viewer.camera.zoom, 3.0)
        if v2 is not None:
            v2.camera.center = (row[2], row[3]); v2.camera.zoom = max(v2.camera.zoom, 3.0)
        if strip is not None:
            strip.set_track(tid)
        s_ = gstat[mm][np.argmin(np.abs(gt[mm][:, 0] - t0))]
        viewer.status = f"track #{tid} at t={t0}: {STATUS_LABEL[s_]}"

    try:
        from magicgui import magicgui

        @magicgui(auto_call=True, track={"choices": ["(all annotated cells)"] + labels, "label": "track"},
                  hide_others={"label": "hide other annotated cells"})
        def follow(track: str = "(all annotated cells)", hide_others: bool = True):
            select(None if track == "(all annotated cells)" else tid_by_label[track], hide_others)
        viewer.window.add_dock_widget(follow, area="right", name="follow a track")
    except Exception as e:
        print(f"(follow widget unavailable: {e})", file=sys.stderr)

    @cells.mouse_drag_callbacks.append
    def _click(layer, event):
        i = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
        if i is not None:
            select(int(gtr[i]), True)

    @cells.mouse_move_callbacks.append
    def _hover(layer, event):
        i = layer.get_value(event.position, view_direction=event.view_direction, dims_displayed=event.dims_displayed, world=True)
        if i is not None:
            t, z, y, x = gt[i]
            viewer.status = (f"annotated cell {int(gid[i])} track #{int(gtr[i])} t={int(t)} z={z:.0f} y={y:.0f} x={x:.0f} um: "
                             f"{STATUS_LABEL[gstat[i]]}" + (f" | matched prediction {abs(pred[gmatch[i]][1:] - gt[i][1:]).max():.1f} um off" if gmatch[i] >= 0 else ""))

    start = int(tracks[0]["first_error"]) if len(tracks) and tracks[0]["first_error"] >= 0 else 0
    viewer.dims.set_point(0, start)
    s = meta["summary"]
    print(f"{meta['name']}: {s['gt_nodes']} annotated cells, TP {s['tp']} FP {s['fp']} FN {s['fn']}; "
          f"status counts {s['status_counts']}; ILP-dropped candidates {s['dropped_candidates']}")
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
