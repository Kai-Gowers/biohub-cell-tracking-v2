#!/usr/bin/env python3
"""Interactive 3D error browser: step through every counted tracking error of a prediction in 3D.

Server side (this file, stdlib only): builds the error list once from `error_report.analyze()`
(cached as JSON), keeps the GT / predicted graphs in memory, and cuts two-frame 3D crops from the
zarr on request. Browser side: `viewer/index.html` + `viewer/render3d.js` (plain WebGL2 volume
ray-marching with the annotated nodes, detections and links drawn as 3D markers).

    PYTHONPATH=src python scripts/error_browser.py \
        --geff-dir dist/preds_val_0842_dump --candidates dist/preds_val_0842_dump/candidates \
        --cache /projects/twist2d/gowers/biohub-cell-tracking-v2-data/error_browser/index_0842.json \
        --port 8765
    # then open http://localhost:8765/  (Cursor / VS Code remote-SSH forwards the port automatically;
    # if not, Ports panel -> Forward a Port -> 8765).  A single error is addressable as
    # http://localhost:8765/#i=<index>

Endpoints
    GET /               the viewer
    GET /index.json     meta + one record per error (kind fn|fp|ok, cause, video, t, scores, ranks...)
    GET /scene/<i>      nodes / links / candidates around error i, absolute voxel coordinates
    GET /crop/<i>?hz=7&hyx=24[&frames=t,t1]   uint8 volume crop, layout (Z, Y, X, C) C-order,
                        C = number of frames (t in channel 0, t+1 in channel 1); headers X-Dims,
                        X-Origin, X-Frames.  Per-video normalisation
                        u8 = 255 * sqrt(clip((v - q0.1) / (q0.999 - q0.1), 0, 1)).

Definitions: an error is one counted metric error of `error_report.analyze()`: every annotated link
not reproduced (FN; cause = `outcome`), every counted wrong predicted link (FP; cause = `cat`), and
optionally `--n-correct` random correct links for contrast (kind `ok`). `dz_um` is the z component
of the *chosen* hop for wrong links (z(chosen) - z(source detection)), otherwise of the error's own
link.
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import random
import sys
import threading
import time
from collections import OrderedDict, defaultdict
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_report import analyze  # noqa: E402

from cell_tracking.config import SCALE  # noqa: E402
from cell_tracking.io_geff import embryo_of  # noqa: E402
from cell_tracking.io_zarr import read_array_meta, read_volume  # noqa: E402

FN_CAUSES = ["undetected_both", "undetected_parent", "undetected_child", "wrong_link_both",
             "wrong_link_parent", "wrong_link_child", "unlinked", "division_second_child"]
FP_CAUSES = ["partner_detected", "partner_undetected", "both_matched_wrong_pair"]
LABELS = {
    "undetected_both": "missed link: both cells undetected",
    "undetected_parent": "missed link: parent cell undetected",
    "undetected_child": "missed link: child cell undetected",
    "wrong_link_both": "missed link: both detected, both linked elsewhere",
    "wrong_link_parent": "missed link: parent linked to a wrong child",
    "wrong_link_child": "missed link: child linked to a wrong parent",
    "unlinked": "missed link: both detected, neither linked",
    "division_second_child": "missed link: 2nd child of a division",
    "partner_detected": "wrong link: true partner detected, another chosen",
    "partner_undetected": "wrong link: true partner undetected",
    "both_matched_wrong_pair": "wrong link: both ends annotated, wrong pair",
    "tp": "correct link (for contrast)",
}


def _clean(o):
    """NaN/None/numpy -> JSON-safe."""
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (o is None or math.isnan(float(o)) or math.isinf(float(o))) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    return o


def _quantiles(zarr_path: Path) -> tuple[float, float]:
    meta = json.load(open(zarr_path / "zarr.json"))
    q = meta.get("attributes", {}).get("image_statistics", {}).get("quantiles", {})
    return float(q.get("0.1", 0.0)), float(q.get("0.999", 1.0))


class ErrorStore:
    """Per-video graphs + the global error list."""

    def __init__(self, names, train_dir: Path, geff_dir: Path, cand_dir: Path | None, tag: str,
                 n_correct: int = 0, seed: int = 0):
        self.train_dir, self.geff_dir, self.tag = train_dir, geff_dir, tag
        self.vol: dict[str, dict] = {}
        self.errors: list[dict] = []
        rng = random.Random(seed)
        t0 = time.time()
        for name in names:
            res = analyze(name, train_dir, geff_dir, cand_dir)
            shape, dtype = read_array_meta(train_dir / f"{name}.zarr")
            lo, hi = _quantiles(train_dir / f"{name}.zarr")
            gt, pred = res["gt"], res["pred"]
            self.vol[name] = dict(
                res=res, shape=shape, dtype=dtype, lo=lo, hi=hi, emb=embryo_of(name),
                gt_c=gt.coords_of(), pred_c=pred.coords_of(), gt_t=gt.t_of(), pred_t=pred.t_of(),
                gt_by_t=gt.nodes_by_t(), pred_by_t=pred.nodes_by_t(),
                gt_out=self._adj(gt.edges), pred_out=self._adj(pred.edges),
                gt_in=self._adj(gt.edges, rev=True), pred_in=self._adj(pred.edges, rev=True),
                gt_to_pred={g: p for p, g in res["pred_to_gt"].items()},
            )
            errs = []
            for e in res["edges"]:
                if e["outcome"] == "tp":
                    continue
                errs.append(self._fn_record(name, e))
            for f in res["fps"]:
                errs.append(self._fp_record(name, f))
            if n_correct:
                tps = [e for e in res["edges"] if e["outcome"] == "tp"]
                k = max(1, round(n_correct * len(tps) / 13000))  # roughly proportional to the video's share
                for e in rng.sample(tps, min(k, len(tps))):
                    errs.append(self._fn_record(name, e, kind="ok"))
            errs.sort(key=lambda r: (r["t"], r["kind"], r["cat"], r["src_id"]))
            self.errors += errs
            print(f"  {name}: {len(errs)} cases  ({time.time() - t0:.0f} s)", flush=True)
        for i, r in enumerate(self.errors):
            r["i"] = i

    @staticmethod
    def _adj(edges, rev=False):
        d = defaultdict(list)
        for u, v in edges:
            u, v = int(u), int(v)
            d[v if rev else u].append(u if rev else v)
        return d

    def _fn_record(self, name, e, kind="fn"):
        V = self.vol[name]
        pu, pv, ch = e["pu"], e["pv"], e["chosen"]
        if ch is not None and pu is not None:
            dz = (V["pred_c"][ch][0] - V["pred_c"][pu][0]) * SCALE[0]
        else:
            dz = (V["gt_c"][e["v"]][0] - V["gt_c"][e["u"]][0]) * SCALE[0]
        return dict(kind=kind, cat=e["outcome"], video=name, emb=V["emb"], t=int(e["t"]),
                    src_id=int(e["u"]), u=int(e["u"]), v=int(e["v"]), pu=pu, pv=pv, chosen=ch,
                    disp_um=e["disp_um"], chosen_disp_um=e["chosen_disp_um"], dz_um=dz,
                    s_true=e["s_true"], s_chosen=e["s_chosen"], s_best=e["s_best"], rank=e["rank"],
                    n_cand=e["n_cand"], crowd_um=e["crowd_um"], z_um=e["z_um"], is_div=bool(e["is_div"]))

    def _fp_record(self, name, f):
        V = self.vol[name]
        s, d = int(f["s"]), int(f["d"])
        return dict(kind="fp", cat=f["cat"], video=name, emb=V["emb"], t=int(f["t"]), src_id=s, s=s, d=d,
                    disp_um=f["disp_um"], dz_um=(V["pred_c"][d][0] - V["pred_c"][s][0]) * SCALE[0],
                    s_chosen=f["score"], z_um=V["pred_c"][s][0] * SCALE[0])

    # ---------------------------------------------------------------- index
    def index_json(self) -> dict:
        counts = defaultdict(int)
        for r in self.errors:
            counts[f"{r['kind']}_{r['cat']}"] += 1
        videos = {n: dict(emb=V["emb"], shape=list(V["shape"]), lo=V["lo"], hi=V["hi"],
                          n=sum(1 for r in self.errors if r["video"] == n)) for n, V in self.vol.items()}
        keep = ["i", "kind", "cat", "video", "emb", "t", "disp_um", "chosen_disp_um", "dz_um", "s_true",
                "s_chosen", "s_best", "rank", "n_cand", "crowd_um", "z_um", "is_div"]
        return _clean(dict(
            meta=dict(tag=self.tag, geff_dir=str(self.geff_dir), created=time.strftime("%Y-%m-%d %H:%M"),
                      scale_um=list(SCALE), gate_um=15.0, match_um=7.0, videos=videos,
                      categories=dict(fn=FN_CAUSES, fp=FP_CAUSES, ok=["tp"]), labels=LABELS, counts=dict(counts),
                      n=len(self.errors)),
            errors=[{k: r.get(k) for k in keep} for r in self.errors],
        ))

    # ---------------------------------------------------------------- scene
    def centre(self, r) -> tuple[int, int, int]:
        V = self.vol[r["video"]]
        if r["kind"] == "fp":
            a, b = V["pred_c"][r["s"]], V["pred_c"][r["d"]]
        else:
            a, b = V["gt_c"][r["u"]], V["gt_c"][r["v"]]
        return tuple(int(round((a[k] + b[k]) / 2)) for k in range(3))

    def scene(self, i: int, hz: int, hyx: int) -> dict:
        r = self.errors[i]
        V = self.vol[r["video"]]
        cz, cy, cx = self.centre(r)
        origin = (cz - hz, cy - hyx, cx - hyx)
        dims = (2 * hz + 1, 2 * hyx + 1, 2 * hyx + 1)
        t = r["t"]
        pad_vox = 4.0 / SCALE  # 4 µm margin in voxels
        lo = np.array(origin) - pad_vox
        hi = np.array(origin) + np.array(dims) + pad_vox

        def inside(p):
            return bool(np.all(np.array(p) >= lo) and np.all(np.array(p) < hi))

        roles_gt, roles_pred = {}, {}
        if r["kind"] == "fp":
            roles_pred[r["s"]] = "s"; roles_pred[r["d"]] = "d"
            focus = ["pred", r["s"]]
            src_det = r["s"]
        else:
            roles_gt[r["u"]] = "u"; roles_gt[r["v"]] = "v"
            for k in ("pu", "pv", "chosen"):
                if r.get(k) is not None:
                    roles_pred.setdefault(r[k], k)
            focus = ["gt", r["u"]]
            src_det = r.get("pu")

        gt_nodes, pred_nodes = [], []
        gt_ids, pred_ids = set(), set()
        for tt in (t, t + 1):
            for nid, _ in V["gt_by_t"].get(tt, []):
                p = V["gt_c"][nid]
                if nid in roles_gt or inside(p):
                    gt_ids.add(nid)
                    gt_nodes.append(dict(id=nid, t=tt, p=list(p), role=roles_gt.get(nid),
                                         matched=V["gt_to_pred"].get(nid)))
            for nid, _ in V["pred_by_t"].get(tt, []):
                p = V["pred_c"][nid]
                if nid in roles_pred or inside(p):
                    pred_ids.add(nid)
                    pred_nodes.append(dict(id=nid, t=tt, p=list(p), role=roles_pred.get(nid),
                                           matched=V["res"]["pred_to_gt"].get(nid)))
        # role nodes may sit at other frames only through the graph, all are at t / t+1 by construction
        gt_links = [dict(a=a, b=b, role="true" if (a == r.get("u") and b == r.get("v")) else "other")
                    for a in gt_ids for b in V["gt_out"].get(a, []) if b in gt_ids]
        pred_links = []
        for a in pred_ids:
            for b in V["pred_out"].get(a, []):
                if b in pred_ids:
                    if r["kind"] == "fp":
                        role = "chosen" if (a == r["s"] and b == r["d"]) else "other"
                    else:
                        role = "chosen" if (a == r.get("pu") and b == r.get("chosen")) else "other"
                    pred_links.append(dict(a=a, b=b, role=role))

        cands = []
        ci = V["res"]["cand_index"].get(t) if V["res"].get("cand_index") else None
        cand = V["res"].get("cand")
        if ci is not None and src_det is not None and src_det in ci["src_local"]:
            ls = ci["src_local"][src_det]
            rows = np.nonzero(ci["pairs"][:, 0] == ls)[0]
            sel = cand.get(f"{t}:selected") if cand is not None else None
            sel_set = {(int(a), int(b)) for a, b in sel} if sel is not None else set()
            dst_ids = cand[f"{t + 1}:node_ids"]
            src_ids = cand[f"{t}:node_ids"]
            taken_by = {int(b): int(a) for a, b in sel_set}
            for row in rows:
                ld = int(ci["pairs"][row, 1])
                b = int(dst_ids[ld])
                cands.append(dict(b=b, p=list(V["pred_c"][b]), score=float(ci["scores"][row]),
                                  selected=(ls, ld) in sel_set,
                                  taken_by=(int(src_ids[taken_by[ld]]) if ld in taken_by and taken_by[ld] != ls else None)))
                if b not in pred_ids:  # make sure every candidate is drawable
                    pred_ids.add(b)
                    pred_nodes.append(dict(id=b, t=t + 1, p=list(V["pred_c"][b]), role=None,
                                           matched=V["res"]["pred_to_gt"].get(b)))
            cands.sort(key=lambda c: -c["score"])
        return _clean(dict(i=i, origin=list(origin), dims=list(dims), t=t, focus=focus, centre=[cz, cy, cx],
                           gt_nodes=gt_nodes, pred_nodes=pred_nodes, gt_links=gt_links,
                           pred_links=pred_links, cands=cands, record=r))

    # ---------------------------------------------------------------- crops
    _frame_cache: OrderedDict = OrderedDict()
    _lock = threading.Lock()

    def frame(self, name: str, t: int) -> np.ndarray | None:
        V = self.vol[name]
        if not (0 <= t < V["shape"][0]):
            return None
        key = (name, t)
        with self._lock:
            if key in self._frame_cache:
                self._frame_cache.move_to_end(key)
                return self._frame_cache[key]
        vol = read_volume(self.train_dir / f"{name}.zarr", t, V["shape"], V["dtype"])
        with self._lock:
            self._frame_cache[key] = vol
            while len(self._frame_cache) > 24:
                self._frame_cache.popitem(last=False)
        return vol

    def crop(self, i: int, hz: int, hyx: int, frames: list[str]) -> tuple[bytes, dict]:
        r = self.errors[i]
        V = self.vol[r["video"]]
        cz, cy, cx = self.centre(r)
        origin = (cz - hz, cy - hyx, cx - hyx)
        dims = (2 * hz + 1, 2 * hyx + 1, 2 * hyx + 1)
        offsets = {"tm1": -1, "t": 0, "t1": 1, "t2": 2}
        out = np.zeros(dims + (len(frames),), dtype=np.uint8)
        lo, hi = V["lo"], V["hi"]
        for c, f in enumerate(frames):
            vol = self.frame(r["video"], r["t"] + offsets[f])
            if vol is None:
                continue
            z0, y0, x0 = origin
            zs, ys, xs = (slice(max(z0, 0), min(z0 + dims[0], vol.shape[0])),
                          slice(max(y0, 0), min(y0 + dims[1], vol.shape[1])),
                          slice(max(x0, 0), min(x0 + dims[2], vol.shape[2])))
            sub = vol[zs, ys, xs].astype(np.float32)
            enc = np.round(255.0 * np.sqrt(np.clip((sub - lo) / max(hi - lo, 1.0), 0.0, 1.0))).astype(np.uint8)
            out[zs.start - z0:zs.stop - z0, ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0, c] = enc
        return out.tobytes(), dict(dims=dims, origin=origin, frames=frames)


# -------------------------------------------------------------------- HTTP
class Handler(SimpleHTTPRequestHandler):
    store: ErrorStore = None  # set via partial
    index_cache: bytes = b""

    def log_message(self, fmt, *args):  # quieter log: only non-static requests
        if not (self.path.startswith("/crop") or self.path.startswith("/scene")):
            super().log_message(fmt, *args)

    def _send(self, body: bytes, ctype: str, extra: dict | None = None, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path == "/index.json":
                return self._send(self.index_cache, "application/json")
            if u.path.startswith("/scene/"):
                i = int(u.path.split("/")[2])
                hz, hyx = int(q.get("hz", [7])[0]), int(q.get("hyx", [24])[0])
                return self._send(json.dumps(self.store.scene(i, hz, hyx)).encode(), "application/json")
            if u.path.startswith("/crop/"):
                i = int(u.path.split("/")[2])
                hz, hyx = int(q.get("hz", [7])[0]), int(q.get("hyx", [24])[0])
                frames = q.get("frames", ["t,t1"])[0].split(",")
                body, meta = self.store.crop(i, hz, hyx, frames)
                return self._send(body, "application/octet-stream", {
                    "X-Dims": ",".join(map(str, meta["dims"])), "X-Origin": ",".join(map(str, meta["origin"])),
                    "X-Frames": ",".join(meta["frames"]), "Access-Control-Expose-Headers": "X-Dims,X-Origin,X-Frames"}, cache=True)
        except (IndexError, ValueError, KeyError) as exc:
            self.send_error(400, f"bad request: {exc!r}")
            return
        if u.path == "/":
            self.path = "/index.html"
        return super().do_GET()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geff-dir", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--split", type=Path, default=Path("dist/heldout_split.json"))
    ap.add_argument("--train-dir", type=Path, default=Path("data/biohub-cell-tracking-during-development/train"))
    ap.add_argument("--videos", nargs="*", default=None, help="subset of video names (default: the split)")
    ap.add_argument("--tag", default="0.842 checkpoint (detector_tmax30_epoch23.pt)")
    ap.add_argument("--n-correct", type=int, default=0, help="also include ~N random correct links")
    ap.add_argument("--cache", type=Path, default=None, help="index.json cache (written if missing)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--check-stats", type=Path, default=None, help="stats.json from error_report.py to compare counts")
    ap.add_argument("--no-serve", action="store_true", help="build the index, print counts, exit")
    args = ap.parse_args()

    names = args.videos or json.load(open(args.split))["datasets"]
    print(f"building error index for {len(names)} videos ...", flush=True)
    store = ErrorStore(names, args.train_dir, args.geff_dir, args.candidates, args.tag, n_correct=args.n_correct)
    index = store.index_json()
    if args.cache:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(index))
        print(f"index cached to {args.cache}")
    counts = index["meta"]["counts"]
    print(f"\n{len(store.errors)} cases")
    for k in FN_CAUSES:
        print(f"  fn {k:24s} {counts.get('fn_' + k, 0):5d}")
    for k in FP_CAUSES:
        print(f"  fp {k:24s} {counts.get('fp_' + k, 0):5d}")
    if args.n_correct:
        print(f"  ok tp                       {counts.get('ok_tp', 0):5d}")
    if args.check_stats:
        agg = json.load(open(args.check_stats))["aggregate"]["all"]
        ok = all(counts.get("fn_" + k, 0) == agg["fn_" + k] for k in FN_CAUSES) and \
            all(counts.get("fp_" + k, 0) == agg["fp_" + k] for k in FP_CAUSES)
        print("counts match stats.json:", ok)
    ref = [r["i"] for r in store.errors if r["video"] == "44b6_1574802b" and r["t"] == 21 and r["cat"].startswith("wrong_link")]
    if ref:
        print(f"reference example (44b6_1574802b t=21 wrong link): i={ref}")
    if args.no_serve:
        return
    Handler.store = store
    Handler.index_cache = json.dumps(index).encode()
    viewer_dir = Path(__file__).resolve().parent.parent / "viewer"
    handler = partial(Handler, directory=str(viewer_dir))
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"\nserving {viewer_dir} on http://localhost:{args.port}/   (Ctrl-C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
