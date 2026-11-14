"""44b6-only learned ranking / constrained selection for FNN detections.

Ranks pairwise candidates already inside the locked 12 µm / sqrt(dt) /
max_gap=2 envelope, then selects with max 1 incoming and max 2 outgoing
edges so a parent may keep two children. Trains the locked transformer
pair head on 44b6 only with hard negatives. Appearance Zarrs are optional;
this helper uses deterministic coordinate embeddings so the two exported
tuning volumes can be scored without a 69-volume train.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Does not retune the 12 µm gate, ELF, FNN, or TCF. Not a competition
submission. A local official score is not a leaderboard number. Do not
use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import DEFAULT_SCALE_ZYX
from .evaluate import match_nodes
from .graphs import TrackGraph
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LEARNED_ORGDET_FNN_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_HASH,
)
from .learned_orgdet_fnn_oracle import (
    GAP_SCALE,
    LINKER_GATE_UM,
    LOCKED_RANK_HASH,
    MAX_GAP_FRAMES,
    RANK_LABEL,
    TUNE_KEEP_THRESHOLD,
    TUNE_MARGIN,
    invert_mapping,
    pair_in_candidate,
    rank_config_payload,
    refuse_oracle_holdout,
)
from .learned_orgdet_fnn_rank_canary import public_lb_not_a_tune_target
from .learned_prune import drop_degree0
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, HoldoutLeakError, specimen_of
from .link import physical_coords

SCALE_ZYX = DEFAULT_SCALE_ZYX
FEAT_DIM = 64
MAX_IN = 1
MAX_OUT = 2
MIN_LOGIT = float("-inf")
SEED = 0
N_EPOCHS = 1
LR = 1e-3
KERNEL_SLUG = "mahdimzafari/biohub-learned-unet-orgdet-fnn-rank-v1"
PUBLIC_LB_NOT_TUNE_TARGETS = ("0.553", "0.602", "0.612")

assert LOCKED_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert abs(TUNE_KEEP_THRESHOLD - 0.7266526336456757) < 1e-12


def refuse_holdout_design(stems: Iterable[str]) -> None:
    leaked = [s for s in stems if specimen_of(str(s)) in HOLDOUT_SPECIMENS or str(s) in HOLDOUT_STEMS]
    if leaked:
        raise HoldoutLeakError(
            "learned_unet_orgdet_fnn_rank_v1 must not inspect 6bba; got " + ", ".join(leaked)
        )


def refuse_rank_holdout(paths: Iterable[str] | None = None, stems: Iterable[str] | None = None) -> None:
    if stems is not None:
        refuse_holdout_design(stems)
        refuse_oracle_holdout(stems=stems)
    if paths is not None:
        refuse_oracle_holdout(paths=paths)


def node_coord_features(
    z: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    t: np.ndarray,
    feat_dim: int = FEAT_DIM,
) -> np.ndarray:
    """Deterministic coordinate embedding. No image / Zarr / 199-video weights."""
    raw = np.stack(
        [
            np.asarray(z, dtype=np.float64) * SCALE_ZYX[0] / 64.0,
            np.asarray(y, dtype=np.float64) * SCALE_ZYX[1] / 64.0,
            np.asarray(x, dtype=np.float64) * SCALE_ZYX[2] / 64.0,
            np.asarray(t, dtype=np.float64) / 100.0,
        ],
        axis=-1,
    ).astype(np.float32)
    out = np.zeros((raw.shape[0], int(feat_dim)), dtype=np.float32)
    n = min(4, int(feat_dim))
    out[:, :n] = raw[:, :n]
    return out


def enumerate_candidates(graph: TrackGraph) -> list[dict[str, Any]]:
    """All in-envelope pairs on locked FNN detections. 6bba refused."""
    refuse_holdout_design([graph.dataset])
    if graph.n_nodes == 0:
        return []
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    nids = [int(x) for x in graph.node_id]
    by_t: dict[int, list[int]] = defaultdict(list)
    for i, t in enumerate(graph.t):
        by_t[int(t)].append(int(i))
    rows: list[dict[str, Any]] = []
    for t, src_idxs in by_t.items():
        src_arr = np.asarray(src_idxs, dtype=np.int64)
        src_xyz = xyz[src_arr]
        for dt in (1, MAX_GAP_FRAMES) if MAX_GAP_FRAMES != 1 else (1,):
            if dt not in (1, 2):
                raise RuntimeError("ranker would invent a gap dt; refusing")
            tgt_idxs = by_t.get(int(t) + int(dt))
            if not tgt_idxs:
                continue
            tgt_arr = np.asarray(tgt_idxs, dtype=np.int64)
            tgt_xyz = xyz[tgt_arr]
            dists = np.linalg.norm(src_xyz[:, None, :] - tgt_xyz[None, :, :], axis=2)
            limit = float(LINKER_GATE_UM) * float(np.sqrt(dt))
            ii, jj = np.where(dists <= limit + 1e-12)
            for a, b in zip(ii.tolist(), jj.tolist()):
                si = int(src_arr[a])
                ti = int(tgt_arr[b])
                dist = float(dists[a, b])
                if not pair_in_candidate(int(dt), dist):
                    continue
                rows.append(
                    {
                        "src": nids[si],
                        "tgt": nids[ti],
                        "src_idx": si,
                        "tgt_idx": ti,
                        "dt": int(dt),
                        "dist_um": dist,
                    }
                )
    return rows


def label_candidates(
    pred: TrackGraph,
    gt: TrackGraph,
    candidates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Label in-envelope pairs from sparse GT. GT is unused at inference."""
    refuse_holdout_design([pred.dataset, gt.dataset])
    mapping = match_nodes(pred, gt, scale_zyx=SCALE_ZYX)
    gt_to_pred = invert_mapping(mapping)
    true_pred: set[tuple[int, int]] = set()
    if gt.n_edges:
        for src, dst in np.asarray(gt.edges).reshape(-1, 2):
            gs, gd = int(src), int(dst)
            if gs in gt_to_pred and gd in gt_to_pred:
                true_pred.add((gt_to_pred[gs], gt_to_pred[gd]))
    labeled: list[dict[str, Any]] = []
    n_true = 0
    n_hard = 0
    for row in candidates:
        item = dict(row)
        label = 1 if (int(item["src"]), int(item["tgt"])) in true_pred else 0
        item["label"] = label
        labeled.append(item)
        if label:
            n_true += 1
        else:
            n_hard += 1
    return labeled


def ranking_index(labeled: Sequence[Mapping[str, Any]]) -> list[tuple[int, int, int]]:
    """(src_local, true_local, hard_local) triples per dt-grouped batch are built later."""
    by_src_dt: dict[tuple[int, int], dict[str, list[int]]] = defaultdict(lambda: {"true": [], "hard": []})
    for i, row in enumerate(labeled):
        key = (int(row["src"]), int(row["dt"]))
        by_src_dt[key]["true" if int(row["label"]) == 1 else "hard"].append(i)
    triples: list[tuple[int, int, int]] = []
    for bag in by_src_dt.values():
        for ti in bag["true"]:
            for hi in bag["hard"]:
                triples.append((int(labeled[ti]["src_idx"]), int(labeled[ti]["tgt_idx"]), int(labeled[hi]["tgt_idx"])))
    return triples


def constrained_select(
    candidates: Sequence[Mapping[str, Any]],
    scores: Sequence[float],
    *,
    max_in: int = MAX_IN,
    max_out: int = MAX_OUT,
    min_logit: float = MIN_LOGIT,
) -> list[int]:
    """Greedy constrained selection. Permits two children; one parent."""
    if max_in != 1 or max_out != 2:
        raise RuntimeError("ranker would retune graph constraints; refusing")
    if len(candidates) != len(scores):
        raise ValueError("candidate/score length mismatch")
    order = sorted(
        range(len(candidates)),
        key=lambda i: (-float(scores[i]), int(candidates[i]["src"]), int(candidates[i]["tgt"]), int(candidates[i]["dt"])),
    )
    kept: list[int] = []
    n_in: dict[int, int] = {}
    n_out: dict[int, int] = {}
    seen: set[tuple[int, int]] = set()
    for i in order:
        if float(scores[i]) < float(min_logit):
            continue
        src = int(candidates[i]["src"])
        tgt = int(candidates[i]["tgt"])
        if (src, tgt) in seen:
            continue
        if n_out.get(src, 0) >= max_out:
            continue
        if n_in.get(tgt, 0) >= max_in:
            continue
        kept.append(i)
        seen.add((src, tgt))
        n_out[src] = n_out.get(src, 0) + 1
        n_in[tgt] = n_in.get(tgt, 0) + 1
    return kept


def graph_from_selected(
    graph: TrackGraph,
    candidates: Sequence[Mapping[str, Any]],
    kept: Sequence[int],
) -> TrackGraph:
    refuse_holdout_design([graph.dataset])
    edges = [(int(candidates[i]["src"]), int(candidates[i]["tgt"])) for i in kept]
    leftover = set(int(x) for x in graph.node_id)
    if any(s not in leftover or d not in leftover for s, d in edges):
        raise RuntimeError("ranker selected a dangling edge; refusing")
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=graph.node_id,
        t=graph.t,
        z=graph.z,
        y=graph.y,
        x=graph.x,
        edges=np.asarray(edges, dtype=np.int64).reshape(-1, 2) if edges else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    pruned = drop_degree0(out)
    pruned.metadata["overlay"] = RANK_LABEL
    pruned.metadata["config_hash"] = LOCKED_RANK_HASH
    pruned.metadata["parent_fnn_hash"] = LOCKED_ORGDET_FNN_HASH
    pruned.metadata["n_candidates"] = int(len(candidates))
    pruned.metadata["n_selected_edges"] = int(len(edges))
    pruned.metadata["n_dropped_isolates"] = int(pruned.metadata.get("n_dropped_isolates") or 0)
    pruned.metadata["uses_transformer"] = True
    pruned.metadata["uses_ilp"] = False
    pruned.metadata["is_12um_assignment"] = False
    pruned.metadata["retunes_nn_tie_eps"] = False
    return pruned


def distance_fallback_scores(candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    """Deterministic non-learned scores for unit tests only. Not official-scored."""
    return [-float(row["dist_um"]) for row in candidates]


def apply_rank_overlay_with_scores(
    graph: TrackGraph,
    scores: Sequence[float],
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> TrackGraph:
    cands = list(candidates if candidates is not None else enumerate_candidates(graph))
    if len(cands) != len(scores):
        raise ValueError("apply_rank_overlay_with_scores score mismatch")
    kept = constrained_select(cands, scores)
    return graph_from_selected(graph, cands, kept)


def build_pair_transformer(feat_dim: int = FEAT_DIM):
    """Locked SimpleNodeTransformer pair head. Imported from the train notebook."""
    import torch
    from torch import nn

    class CrossAttentionBlock(nn.Module):
        def __init__(self, hidden_dim: int) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )

        def forward(self, q, kv, kv_mask=None):
            key_padding_mask = ~kv_mask if kv_mask is not None else None
            attn_out, _ = self.cross_attn(self.norm1(q), self.norm1(kv), self.norm1(kv), key_padding_mask=key_padding_mask)
            q = q + attn_out
            return q + self.mlp(self.norm2(q))

    class SimpleNodeTransformer(nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            hidden_dim = 128
            self.proj = nn.Linear(dim, hidden_dim)
            self.norm_in = nn.LayerNorm(hidden_dim)
            self.blocks = nn.ModuleList([CrossAttentionBlock(hidden_dim) for _ in range(4)])
            self.norm_out = nn.LayerNorm(hidden_dim)
            self.pair_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2 + 3, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, feat_t, feat_t1, coords_t, coords_t1, mask_t=None, mask_t1=None):
            q = self.norm_in(self.proj(feat_t))
            k = self.norm_in(self.proj(feat_t1))
            for block in self.blocks:
                q = block(q, k, kv_mask=mask_t1)
                k = block(k, q, kv_mask=mask_t)
            q = self.norm_out(q)
            k = self.norm_out(k)
            n1 = k.shape[1]
            qe = q.unsqueeze(2).expand(-1, -1, n1, -1)
            ke = k.unsqueeze(1).expand(-1, q.shape[1], -1, -1)
            rel = (coords_t.unsqueeze(2) - coords_t1.unsqueeze(1)) / 100.0
            return self.pair_mlp(torch.cat([qe, ke, rel], dim=-1)).squeeze(-1)

    return SimpleNodeTransformer(int(feat_dim))


def _frame_batches(graph: TrackGraph, labeled: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    refuse_holdout_design([graph.dataset])
    feats = node_coord_features(graph.z, graph.y, graph.x, graph.t)
    coords = graph.coords_zyx().astype(np.float32)
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in labeled:
        src_t = int(graph.t[int(row["src_idx"])])
        by_key[(src_t, int(row["dt"]))].append(row)
    batches = []
    for (t, dt), rows in by_key.items():
        src_idxs = sorted({int(r["src_idx"]) for r in rows})
        tgt_idxs = sorted({int(r["tgt_idx"]) for r in rows})
        src_map = {old: i for i, old in enumerate(src_idxs)}
        tgt_map = {old: i for i, old in enumerate(tgt_idxs)}
        pair_index = [[src_map[int(r["src_idx"])], tgt_map[int(r["tgt_idx"])]] for r in rows]
        labels = [float(r["label"]) for r in rows]
        rank_rows = []
        by_src: dict[int, dict[str, list[int]]] = defaultdict(lambda: {"true": [], "hard": []})
        for r in rows:
            bag = by_src[int(r["src_idx"])]
            bag["true" if int(r["label"]) == 1 else "hard"].append(int(r["tgt_idx"]))
        for src_idx, bag in by_src.items():
            for ti in bag["true"]:
                for hi in bag["hard"]:
                    rank_rows.append([src_map[src_idx], tgt_map[ti], tgt_map[hi]])
        batches.append(
            {
                "dataset": graph.dataset,
                "t": t,
                "dt": dt,
                "src_idx": src_idxs,
                "tgt_idx": tgt_idxs,
                "feat_t": feats[src_idxs],
                "feat_t1": feats[tgt_idxs],
                "coords_t": coords[src_idxs],
                "coords_t1": coords[tgt_idxs],
                "pair_index": pair_index,
                "labels": labels,
                "ranking_index": rank_rows,
                "rows": rows,
            }
        )
    return batches


def train_ranker(
    graphs: Sequence[TrackGraph],
    gts: Sequence[TrackGraph],
    *,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    seed: int = SEED,
) -> dict[str, Any]:
    """Train the pair head on 44b6 hard negatives. Fail closed on collapse."""
    import torch
    import torch.nn.functional as F

    if len(graphs) != len(gts):
        raise ValueError("graph/gt count mismatch")
    refuse_holdout_design([g.dataset for g in graphs] + [g.dataset for g in gts])
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = build_pair_transformer(FEAT_DIM)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr))
    batches: list[dict[str, Any]] = []
    n_true = 0
    n_hard = 0
    for pred, gt in zip(graphs, gts):
        labeled = label_candidates(pred, gt, enumerate_candidates(pred))
        n_true += sum(1 for r in labeled if r["label"] == 1)
        n_hard += sum(1 for r in labeled if r["label"] == 0)
        batches.extend(_frame_batches(pred, labeled))
    if n_true < 1 or n_hard < 1:
        raise RuntimeError("ranker train found no true/hard envelope pairs; refusing")

    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(int(n_epochs)):
        for batch in batches:
            feat_t = torch.tensor(batch["feat_t"], dtype=torch.float32).unsqueeze(0)
            feat_t1 = torch.tensor(batch["feat_t1"], dtype=torch.float32).unsqueeze(0)
            coords_t = torch.tensor(batch["coords_t"], dtype=torch.float32).unsqueeze(0)
            coords_t1 = torch.tensor(batch["coords_t1"], dtype=torch.float32).unsqueeze(0)
            pair_index = torch.tensor(batch["pair_index"], dtype=torch.long)
            labels = torch.tensor(batch["labels"], dtype=torch.float32)
            logits = model(feat_t, feat_t1, coords_t, coords_t1)[0]
            cand = logits[pair_index[:, 0], pair_index[:, 1]]
            loss = F.binary_cross_entropy_with_logits(cand, labels)
            if batch["ranking_index"]:
                rank = cand.new_zeros(())
                for si, ti_true, ti_hard in batch["ranking_index"]:
                    rank = rank + F.softplus(logits[int(si), int(ti_hard)] - logits[int(si), int(ti_true)])
                loss = loss + rank / float(len(batch["ranking_index"]))
            opt.zero_grad()
            loss.backward()
            grad = model.pair_mlp[5].weight.grad
            gnorm = float(grad.norm()) if grad is not None else 0.0
            opt.step()
            history.append({"epoch": float(epoch), "link_loss": float(loss.detach()), "grad_l2": gnorm})

    finite = all(
        x == x and abs(x) != float("inf")
        for row in history
        for x in (row["link_loss"], row["grad_l2"])
    )
    nonzero = any(row["grad_l2"] > 0.0 for row in history[:5])
    if not finite or not nonzero:
        raise RuntimeError("ranker link-head failed finite/nonzero gates; refusing")
    clone = build_pair_transformer(FEAT_DIM)
    clone.load_state_dict(model.state_dict(), strict=True)
    reload_max = 0.0
    model.eval()
    clone.eval()
    with torch.no_grad():
        probe = batches[0]
        a = model(
            torch.tensor(probe["feat_t"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(probe["feat_t1"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(probe["coords_t"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(probe["coords_t1"], dtype=torch.float32).unsqueeze(0),
        )
        b = clone(
            torch.tensor(probe["feat_t"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(probe["feat_t1"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(probe["coords_t"], dtype=torch.float32).unsqueeze(0),
            torch.tensor(probe["coords_t1"], dtype=torch.float32).unsqueeze(0),
        )
        reload_max = float((a - b).abs().max())
    if reload_max >= 1e-6:
        raise RuntimeError("ranker checkpoint reload drifted; refusing")
    return {
        "model": model,
        "n_true": n_true,
        "n_hard": n_hard,
        "n_batches": len(batches),
        "history": history,
        "finite_losses": finite,
        "nonzero_edge_grad": nonzero,
        "reload_max_abs": reload_max,
        "opened_holdout": False,
        "used_199_video_weights": False,
        "used_zarr_appearance": False,
    }


def score_candidates_with_model(model, graph: TrackGraph, candidates: Sequence[Mapping[str, Any]]) -> list[float]:
    import torch

    refuse_holdout_design([graph.dataset])
    if not candidates:
        return []
    feats = node_coord_features(graph.z, graph.y, graph.x, graph.t)
    coords = graph.coords_zyx().astype(np.float32)
    scores = [0.0] * len(candidates)
    by_key: dict[tuple[int, int], list[int]] = defaultdict(list)
    for i, row in enumerate(candidates):
        src_t = int(graph.t[int(row["src_idx"])])
        by_key[(src_t, int(row["dt"]))].append(i)
    model.eval()
    with torch.no_grad():
        for (t, dt), idxs in by_key.items():
            src_idxs = sorted({int(candidates[i]["src_idx"]) for i in idxs})
            tgt_idxs = sorted({int(candidates[i]["tgt_idx"]) for i in idxs})
            src_map = {old: j for j, old in enumerate(src_idxs)}
            tgt_map = {old: j for j, old in enumerate(tgt_idxs)}
            logits = model(
                torch.tensor(feats[src_idxs], dtype=torch.float32).unsqueeze(0),
                torch.tensor(feats[tgt_idxs], dtype=torch.float32).unsqueeze(0),
                torch.tensor(coords[src_idxs], dtype=torch.float32).unsqueeze(0),
                torch.tensor(coords[tgt_idxs], dtype=torch.float32).unsqueeze(0),
            )[0]
            for i in idxs:
                si = src_map[int(candidates[i]["src_idx"])]
                ti = tgt_map[int(candidates[i]["tgt_idx"])]
                scores[i] = float(logits[si, ti])
    return scores


def apply_trained_rank_overlay(graph: TrackGraph, model) -> TrackGraph:
    candidates = enumerate_candidates(graph)
    scores = score_candidates_with_model(model, graph, candidates)
    return apply_rank_overlay_with_scores(graph, scores, candidates)


def overlay_summary(before: TrackGraph, after: TrackGraph, n_candidates: int) -> dict[str, Any]:
    n_div = 0
    if after.n_edges:
        counts: dict[int, int] = defaultdict(int)
        for src, _dst in np.asarray(after.edges).reshape(-1, 2):
            counts[int(src)] += 1
        n_div = sum(1 for v in counts.values() if v >= 2)
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_candidates": int(n_candidates),
        "n_predicted_divisions": int(n_div),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "overlay": RANK_LABEL,
        "config_hash": LOCKED_RANK_HASH,
    }


def tuning_decision(learned_combined: float) -> str:
    public_lb_not_a_tune_target(0.6557507105877639)
    if float(learned_combined) <= float(LEARNED_ORGDET_FNN_LOCKED_TUNE) + 1e-15:
        return "reject_learned_unet_orgdet_fnn_rank_v1"
    if float(learned_combined) + 1e-15 < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_fnn_rank_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_rank_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(LEARNED_ORGDET_FNN_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_rank_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_rank_v1"


def runtime_estimate(
    n_volumes: int = 2,
    train_sec: float = 180.0,
    infer_sec_per_volume: float = 20.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1800.0,
) -> dict[str, Any]:
    sec = float(train_sec) + float(n_volumes) * float(infer_sec_per_volume)
    return {
        "n_volumes": int(n_volumes),
        "train_sec": float(train_sec),
        "infer_sec_per_volume": float(infer_sec_per_volume),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "prefers_tuning_zarrs": True,
        "uses_69_volume_train": False,
        "source": (
            "CPU/GPU pair-head ranker on locked FNN detections inside the "
            "12 µm / sqrt(dt) envelope; coordinate embeddings; no 69-volume train"
        ),
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }


def helper_config_payload() -> dict[str, Any]:
    """Packaging metadata. Does not change the locked rank hash."""
    return {
        "label": RANK_LABEL,
        "parent_hash": LOCKED_ORGDET_FNN_HASH,
        "config_hash": LOCKED_RANK_HASH,
        "kernel_slug": KERNEL_SLUG,
        "feat_dim": FEAT_DIM,
        "max_in": MAX_IN,
        "max_out": MAX_OUT,
        "min_logit": None,
        "n_epochs": N_EPOCHS,
        "lr": LR,
        "seed": SEED,
        "uses_coord_embeddings": True,
        "uses_zarr_appearance": False,
        "uses_69_volume_train": False,
        "writes_submission_csv": False,
        "opens_6bba": False,
        "kernel_push_default": False,
    }
