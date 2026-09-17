#!/usr/bin/env python3
"""Build the per-miss / per-wrong-link tables used by error_examples.py (cached as a pickle)."""
import json, pickle, sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, str(Path(__file__).resolve().parent))
from error_report import analyze, add_intensity
from cell_tracking.config import SCALE

train_dir = Path("data/biohub-cell-tracking-during-development/train")
geff_dir = Path("dist/preds_val_0842_dump"); cand_dir = geff_dir / "candidates"
names = json.load(open("dist/heldout_split.json"))["datasets"]
out = Path("dist/error_examples/tables.pkl")

misses, wrongs, all_nodes, all_edges = [], [], [], []
for n in names:
    res = analyze(n, train_dir, geff_dir, cand_dir)
    add_intensity(res["nodes"], n, train_dir)
    gt, pred = res["gt"], res["pred"]
    gt_um, pred_um, p2g = res["gt_um"], res["pred_um"], res["pred_to_gt"]
    g2p = {g: p for p, g in p2g.items()}
    gt_t, pred_t = gt.t_of(), pred.t_of()
    pred_by_t, gt_by_t = pred.nodes_by_t(), gt.nodes_by_t()
    trees = {t: ([i for i, _ in ns], cKDTree(np.array([pred_um[i] for i, _ in ns]))) for t, ns in pred_by_t.items()}
    gtrees = {t: ([i for i, _ in ns], cKDTree(np.array([gt_um[i] for i, _ in ns]))) for t, ns in gt_by_t.items()}
    snr = {x["gid"]: x["snr"] for x in res["nodes"]}
    # detection logit per pred node (from the candidate dump)
    logit = {}
    if res["cand"] is not None:
        c = res["cand"]
        for t in range(int(c["n_t"])):
            if f"{t}:node_ids" in c:
                for i, l in zip(c[f"{t}:node_ids"], c[f"{t}:logit"]):
                    logit[int(i)] = float(l)
    gt_parent = {int(b): int(a) for a, b in gt.edges}
    gt_child = {}
    for a, b in gt.edges:
        gt_child.setdefault(int(a), []).append(int(b))
    pred_in_all = {int(b): int(a) for a, b in pred.edges}
    pred_out_all = {int(a): int(b) for a, b in pred.edges}
    detected = {x["gid"]: x["detected"] for x in res["nodes"]}
    def run_len(g):
        # consecutive missed GT nodes along the track through g (parent chain + single-child chain)
        n = 1; h = g
        while h in gt_parent and not detected.get(gt_parent[h], True):
            h = gt_parent[h]; n += 1
        h = g
        while len(gt_child.get(h, [])) == 1 and not detected.get(gt_child[h][0], True):
            h = gt_child[h][0]; n += 1
        return n
    def frames_to_division(g, maxf=10):
        h = g
        for k in range(maxf + 1):
            if len(gt_child.get(h, [])) >= 2:
                return k
            if len(gt_child.get(h, [])) != 1:
                return None
            h = gt_child[h][0]
        return None
    def frames_since_division(g, maxf=10):
        h = g
        for k in range(maxf + 1):
            if h not in gt_parent:
                return None
            h = gt_parent[h]
            if len(gt_child.get(h, [])) >= 2:
                return k + 1
        return None
    for x in res["nodes"]:
        all_nodes.append(dict(name=n, emb=x["emb"], gid=x["gid"], t=x["t"], snr=x["snr"], detected=x["detected"], z_um=x["z_um"]))
        if x["detected"]:
            continue
        g, t, um = x["gid"], x["t"], gt_um[x["gid"]]
        ids, tree = trees[t]
        d, j = tree.query(um, k=min(3, len(ids)))
        d, j = np.atleast_1d(d), np.atleast_1d(j)
        near = [(float(dd), ids[jj]) for dd, jj in zip(d, j)]
        d1, p1 = near[0]
        claimer = p2g.get(p1)
        off = pred_um[p1] - um
        # nearest other GT node in the same frame
        gids, gtree = gtrees[t]
        dg, jg = gtree.query(um, k=min(2, len(gids)))
        dg, jg = np.atleast_1d(dg), np.atleast_1d(jg)
        other = [(float(a), gids[b]) for a, b in zip(dg, jg) if gids[b] != g]
        # does the miss belong to a track that was detected in adjacent frames?
        misses.append(dict(name=n, emb=x["emb"], gid=g, t=t, z=x["z"], y=x["y"], x=x["x"], z_um=x["z_um"], snr=x["snr"],
                           intensity_rel=x["intensity_rel"], near_um=d1, near_pid=p1, near_dxy_um=float(np.hypot(off[1], off[2])),
                           near_dz_um=float(off[0]), near_claimed_by=claimer, near_claimed_um=(float(np.linalg.norm(pred_um[p1] - gt_um[claimer])) if claimer is not None else np.nan),
                           near_logit=logit.get(p1, np.nan), second_um=near[1][0] if len(near) > 1 else np.nan,
                           second_claimed=(p2g.get(near[1][1]) is not None) if len(near) > 1 else None,
                           other_gt_um=other[0][0] if other else np.nan, other_gt_gid=other[0][1] if other else None,
                           other_gt_detected=(g2p.get(other[0][1]) is not None) if other else None,
                           n_pred_10=len(tree.query_ball_point(um, 10.0)), t_frac=x["t_frac"],
                           parent_detected=detected.get(gt_parent[g]) if g in gt_parent else None,
                           child_detected=(detected.get(gt_child[g][0]) if len(gt_child.get(g, [])) == 1 else None),
                           run_len=run_len(g), has_parent=g in gt_parent, n_children=len(gt_child.get(g, [])),
                           to_div=frames_to_division(g), since_div=frames_since_division(g)))
    for e in res["edges"]:
        er = {k: e[k] for k in ("name", "emb", "u", "v", "t", "disp_um", "outcome", "both_detected", "crowd_um", "rank", "s_true", "s_best", "s_chosen", "n_cand", "is_div")}
        gd = gt_um[e["v"]] - gt_um[e["u"]]; er["gt_dz"] = float(gd[0]); er["gt_dxy"] = float(np.hypot(gd[1], gd[2]))
        if e["both_detected"]:
            dd = pred_um[e["pv"]] - pred_um[e["pu"]]; er["det_dz"] = float(dd[0]); er["det_dxy"] = float(np.hypot(dd[1], dd[2]))
            er["u_err_z"] = float(pred_um[e["pu"]][0] - gt_um[e["u"]][0]); er["v_err_z"] = float(pred_um[e["pv"]][0] - gt_um[e["v"]][0])
            er["u_err_um"] = float(np.linalg.norm(pred_um[e["pu"]] - gt_um[e["u"]])); er["v_err_um"] = float(np.linalg.norm(pred_um[e["pv"]] - gt_um[e["v"]]))
        all_edges.append(er)
        if e["outcome"] not in ("wrong_link_both", "wrong_link_parent", "wrong_link_child", "unlinked", "division_second_child"):
            continue
        rec = dict(e)
        pu, pv = e["pu"], e["pv"]
        ch = e["chosen"]
        rec["chosen_gt"] = p2g.get(ch) if ch is not None else None       # the chosen target is itself an annotated cell?
        rec["chosen_dz_um"] = float(pred_um[ch][0] - pred_um[pu][0]) if ch is not None else np.nan
        rec["true_dz_um"] = float(pred_um[pv][0] - pred_um[pu][0])
        rec["true_dxy_um"] = float(np.hypot(*(pred_um[pv] - pred_um[pu])[1:]))
        rec["chosen_dxy_um"] = float(np.hypot(*(pred_um[ch] - pred_um[pu])[1:])) if ch is not None else np.nan
        # who took the true target (pred_in)?
        pred_in = {int(b): int(a) for a, b in pred.edges}
        taker = pred_in.get(pv)
        rec["true_target_taken_by"] = taker
        rec["taker_gt"] = p2g.get(taker) if taker is not None else None
        rec["snr_u"] = snr.get(e["u"], np.nan); rec["snr_v"] = snr.get(e["v"], np.nan)
        rec["taker_dist_um"] = float(np.linalg.norm(pred_um[taker] - pred_um[pu])) if taker is not None else np.nan
        rec["taker_disp_um"] = float(np.linalg.norm(pred_um[pv] - pred_um[taker])) if taker is not None else np.nan
        # taker's score for the true target, and the source's best candidate score
        ci = res["cand_index"].get(e["t"])
        def score_of(a, b):
            if ci is None or a not in ci["src_local"] or b not in ci["dst_local"]:
                return np.nan
            rows = np.nonzero((ci["pairs"][:, 0] == ci["src_local"][a]) & (ci["pairs"][:, 1] == ci["dst_local"][b]))[0]
            return float(ci["scores"][rows[0]]) if len(rows) else np.nan
        rec["taker_score"] = score_of(taker, pv) if taker is not None else np.nan
        rec["taker_dz_um"] = float(pred_um[taker][0] - pred_um[pu][0]) if taker is not None else np.nan
        # is the taker itself a GT-annotated cell in frame t? (taker_gt) and is the taker's own GT child detected?
        tg = p2g.get(taker) if taker is not None else None
        rec["taker_gt_child_detected"] = (any(g2p.get(c) is not None for c in gt_child.get(tg, [])) if tg is not None else None)
        # duplicate-detection check: does the source have another detection within 4 um in frame t?
        ids_t, tree_t = trees[e["t"]]
        rec["src_dup_um"] = float(tree_t.query(pred_um[pu], k=2)[0][1]) if len(ids_t) > 1 else np.nan
        rec["src_logit"] = logit.get(pu, np.nan); rec["taker_logit"] = logit.get(taker, np.nan) if taker is not None else np.nan
        rec["chosen_logit"] = logit.get(ch, np.nan) if ch is not None else np.nan; rec["true_logit"] = logit.get(pv, np.nan)
        rec["chosen_true_um"] = float(np.linalg.norm(pred_um[ch] - pred_um[pv])) if ch is not None else np.nan
        ids1, tree1 = trees[e["t"] + 1]
        rec["true_dup_um"] = float(tree1.query(pred_um[pv], k=2)[0][1]) if len(ids1) > 1 else np.nan   # nearest other t+1 detection to the true target
        rec["to_div"] = frames_to_division(e["u"]); rec["since_div"] = frames_since_division(e["u"])
        # the chosen target: does it have its own strong competitor (another source scoring it >0.5)?
        if ch is not None and ci is not None and ch in ci["dst_local"]:
            rows = np.nonzero(ci["pairs"][:, 1] == ci["dst_local"][ch])[0]
            sc = ci["scores"][rows]; rec["chosen_n_strong_sources"] = int((sc > 0.5).sum())
        else:
            rec["chosen_n_strong_sources"] = None
        for k in ("gt", "pred"):
            rec.pop(k, None)
        wrongs.append(rec)
    print(n, "misses", sum(m["name"] == n for m in misses), "linker FN", sum(w["name"] == n for w in wrongs), flush=True)
pickle.dump(dict(misses=misses, wrongs=wrongs, nodes=all_nodes, edges=all_edges), open(out, "wb"))
print("wrote", out)
