"""Drop abrupt dt=1 speed changes after locked ACF (acceleration, not angle).

Official 44b6 ACF residue is still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. ELF only
drops dt=1 edges longer than 8.0 µm. Gap-edge appearance, gap-edge
length, and dt=1 turning-angle are closed. Remaining conflict FPs
are almost all dt=1.

This overlay keeps locked ACF, then drops a dt=1 continuation edge
when the source has a unique dt=1 predecessor and the absolute
speed-change of that two-hop exceeds the locked gate, and drops new
degree-0 leftovers. It does not reassign, does not shrink the 12 µm
linker/gap gates, and is not an ELF / FNN / ACF / BNN / IPC / TCF /
GACF / GELF / TAC / rank / division retune. Do not retune
``max_dt1_edge_um`` or ``max_rel_appearance``.

The speed-change gate is locked from 44b6 ACF official-matched TP
dt=1 unique-predecessor |Δv| values only. Leftover unmatched edges
are not negatives. If that diagnostic does not separate conflict FPs
above a 0.1 µm/frame round-up of TP max, the overlay is rejected
without a threshold sweep and without an official 44b6 overlay score.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number. Do not overwrite locked slugs. Do not retune locked
parent hashes. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .evaluate import match_nodes
from .graphs import TrackGraph
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH, MAX_DT1_EDGE_UM
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_ORGDET_FNN_HASH,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
from .learned_orgdet_fnn_acf import (
    LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_ACF_HASH,
    apply_orgdet_fnn_acf_overlay,
    resolve_sequence_zarr,
)
from .learned_orgdet_fnn_acf_bnn import LOCKED_ORGDET_FNN_ACF_BNN_HASH
from .learned_orgdet_fnn_acf_gacf import LOCKED_ORGDET_FNN_ACF_GACF_HASH
from .learned_orgdet_fnn_acf_gelf import (
    LOCKED_ORGDET_FNN_ACF_GELF_HASH,
    hop_class,
    round_up_tenth_um,
)
from .learned_orgdet_fnn_acf_ipc import LOCKED_ORGDET_FNN_ACF_IPC_HASH
from .learned_orgdet_fnn_acf_tac import LOCKED_ORGDET_FNN_ACF_TAC_HASH
from .learned_orgdet_fnn_div import LOCKED_ORGDET_FNN_DIV_HASH
from .learned_orgdet_fnn_rank import LOCKED_RANK_HASH as LOCKED_ORGDET_FNN_RANK_HASH
from .learned_orgdet_fnn_reas import LOCKED_ORGDET_FNN_REAS_HASH
from .learned_orgdet_gap import LOCKED_ORGDET_GAP_HASH
from .learned_orgdet_gap_prune import LOCKED_ORGDET_GAP_PRUNE_HASH
from .learned_orgop import (
    LOCKED_ORGOP_HASH as REJECTED_ORGOP_HASH,
    ORGANIZER_DET_THRESHOLD,
    ORGANIZER_POOL_KERNEL_UM,
)
from .learned_prune import drop_degree0
from .learned_split import HOLDOUT_SPECIMENS, TUNING_SPECIMEN, specimen_of
from .link import physical_coords

SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_FNN_ACF_LOCKED_TUNE = LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE
ORGDET_FNN_ACF_LOCKED_HOLDOUT = LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN
# Locked association_v2 / learned gap-close span. Do not retune.
MAX_GAP_DT = 3
# Relative-change denominator floor in µm/frame. Not a gate.
SPEED_EPS_UM = 1e-9
# Locked from 44b6 ACF TP dt=1 unique-predecessor |Δv| only. None means
# the diagnostic has not supported a gate. Do not invent one.
MAX_SPEED_CHANGE_UM: float | None = None

LOCKED_GATES = {
    "drop_abrupt_dt1_speed_change": True,
    "drop_degree0_after": True,
    "max_speed_change_um": MAX_SPEED_CHANGE_UM,
    "max_gap_dt": MAX_GAP_DT,
    "assignment": "unchanged_locked_orgdet_fnn_acf",
}


def orgdet_fnn_acf_scf_config_payload() -> dict[str, Any]:
    """dt=1 speed-change filter after locked ACF. Lock the µm/frame gate before 44b6."""
    return {
        "label": "learned_unet_orgdet_fnn_acf_scf_v1",
        "train_label": "learned_unet_v1",
        "infer_label": "learned_unet_infer_v1",
        "orgdet_label": "learned_unet_orgdet_v1",
        "orgdet_gap_label": "learned_unet_orgdet_gap_v1",
        "orgdet_gap_prune_label": "learned_unet_orgdet_gap_prune_v1",
        "orgdet_elf_label": "learned_unet_orgdet_elf_v1",
        "orgdet_fnn_label": "learned_unet_orgdet_fnn_v1",
        "orgdet_fnn_acf_label": "learned_unet_orgdet_fnn_acf_v1",
        "rejected_fnn_rank_label": "learned_unet_orgdet_fnn_rank_v1",
        "rejected_fnn_div_label": "learned_unet_orgdet_fnn_div_v1",
        "rejected_fnn_reas_label": "learned_unet_orgdet_fnn_reas_v1",
        "rejected_fnn_acf_bnn_label": "learned_unet_orgdet_fnn_acf_bnn_v1",
        "rejected_fnn_acf_ipc_label": "learned_unet_orgdet_fnn_acf_ipc_v1",
        "rejected_fnn_acf_gacf_label": "learned_unet_orgdet_fnn_acf_gacf_v1",
        "rejected_fnn_acf_gelf_label": "learned_unet_orgdet_fnn_acf_gelf_v1",
        "rejected_fnn_acf_tac_label": "learned_unet_orgdet_fnn_acf_tac_v1",
        "train_config_hash": TRAIN_CONFIG_HASH,
        "infer_config_hash": LOCKED_INFER_HASH,
        "orgdet_config_hash": LOCKED_ORGDET_HASH,
        "orgdet_gap_config_hash": LOCKED_ORGDET_GAP_HASH,
        "orgdet_gap_prune_config_hash": LOCKED_ORGDET_GAP_PRUNE_HASH,
        "orgdet_elf_config_hash": LOCKED_ORGDET_ELF_HASH,
        "orgdet_fnn_config_hash": LOCKED_ORGDET_FNN_HASH,
        "orgdet_fnn_acf_config_hash": LOCKED_ORGDET_FNN_ACF_HASH,
        "rejected_fnn_rank_hash": LOCKED_ORGDET_FNN_RANK_HASH,
        "rejected_fnn_div_hash": LOCKED_ORGDET_FNN_DIV_HASH,
        "rejected_fnn_reas_hash": LOCKED_ORGDET_FNN_REAS_HASH,
        "rejected_fnn_acf_bnn_hash": LOCKED_ORGDET_FNN_ACF_BNN_HASH,
        "rejected_fnn_acf_ipc_hash": LOCKED_ORGDET_FNN_ACF_IPC_HASH,
        "rejected_fnn_acf_gacf_hash": LOCKED_ORGDET_FNN_ACF_GACF_HASH,
        "rejected_fnn_acf_gelf_hash": LOCKED_ORGDET_FNN_ACF_GELF_HASH,
        "rejected_fnn_acf_tac_hash": LOCKED_ORGDET_FNN_ACF_TAC_HASH,
        "source_graphs": "locked_learned_unet_orgdet_fnn_acf_graphs",
        "source_zarrs": "locked_tuning_zarr_export_two_44b6",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "retunes_max_rel_appearance": False,
        "uses_intensity": False,
        "uses_transformer": False,
        "uses_velocity": True,
        "uses_ilp": False,
        "linker": "orgdet_fnn_acf_then_drop_abrupt_dt1_speed_change",
        "drop_abrupt_dt1_speed_change": LOCKED_GATES["drop_abrupt_dt1_speed_change"],
        "drop_degree0_after": LOCKED_GATES["drop_degree0_after"],
        "max_speed_change_um": LOCKED_GATES["max_speed_change_um"],
        "max_gap_dt": LOCKED_GATES["max_gap_dt"],
        "assignment": LOCKED_GATES["assignment"],
        "tune_keep_threshold": TUNE_KEEP_THRESHOLD,
        "tune_margin": TUNE_MARGIN,
        "det_threshold": ORGANIZER_DET_THRESHOLD,
        "det_tta": True,
        "pool_kernel_um": ORGANIZER_POOL_KERNEL_UM,
        "scale_zyx": list(SCALE_ZYX),
        "writes_submission_csv": False,
    }


def orgdet_fnn_acf_scf_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_acf_scf_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_ACF_SCF_HASH = orgdet_fnn_acf_scf_config_hash(orgdet_fnn_acf_scf_config_payload())
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH == "a259d184c9f99046"
assert LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_FNN_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
assert LOCKED_ORGDET_FNN_ACF_BNN_HASH == "bdb38cf13e226ced"
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH == "7302409595bee523"
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH == "5ce49a6d175c8032"
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH == "59812f90385ed4d2"
assert LOCKED_ORGDET_FNN_ACF_TAC_HASH == "4dfc5514c65496e5"
assert TUNE_KEEP_THRESHOLD == 0.7366197119237633
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_ACF_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_RANK_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_DIV_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_REAS_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_ACF_BNN_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_ACF_IPC_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_ACF_GACF_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_ACF_GELF_HASH
assert LOCKED_ORGDET_FNN_ACF_SCF_HASH != LOCKED_ORGDET_FNN_ACF_TAC_HASH
assert LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE == 0.7346197119237633
assert LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT == 0.7555029702166214
assert MAX_DT1_EDGE_UM == 8.0


def _clean_stem(raw: str) -> str:
    stem = Path(raw).name
    for suffix in (".zarr", ".geff", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def refuse_non_tuning_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) != TUNING_SPECIMEN or stem not in INFER_STEMS:
            raise ValueError(f"learned_unet_orgdet_fnn_acf_scf_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_acf_scf_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_acf_scf_v1 holdout refuses non-holdout stem {stem}")


def hop_speed_um(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> float:
    """Physical hop length in µm. dt=1 so this is speed in µm/frame."""
    delta = np.asarray(dst_xyz, dtype=np.float64) - np.asarray(src_xyz, dtype=np.float64)
    return float(np.linalg.norm(delta))


def speed_change_um(
    pred_xyz: np.ndarray,
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    speed_eps: float = SPEED_EPS_UM,
) -> tuple[float, float, float, float]:
    """Return (speed1, speed2, |Δv|, relative |Δv| / max(s1, s2, eps))."""
    speed1 = hop_speed_um(pred_xyz, src_xyz)
    speed2 = hop_speed_um(src_xyz, dst_xyz)
    delta = abs(speed2 - speed1)
    rel = delta / max(speed1, speed2, float(speed_eps))
    return speed1, speed2, delta, float(rel)


def _gt_edge_set(graph: TrackGraph) -> set[tuple[int, int]]:
    return {(int(s), int(d)) for s, d in graph.consecutive_edges()}


def _gt_partners(gt: TrackGraph) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    sources: dict[int, set[int]] = {}
    targets: dict[int, set[int]] = {}
    for src, dst in gt.consecutive_edges():
        sources.setdefault(int(dst), set()).add(int(src))
        targets.setdefault(int(src), set()).add(int(dst))
    return sources, targets


def extract_dt1_speed_changes(
    graph: TrackGraph,
    speed_eps: float = SPEED_EPS_UM,
) -> list[dict[str, Any]]:
    """dt=1 continuation edges whose source has a unique dt=1 predecessor."""
    nids = [int(x) for x in graph.node_id]
    if not nids:
        return []
    index = graph.id_to_index()
    ins = graph.in_edges()
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    rows: list[dict[str, Any]] = []
    raw_edges = np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else np.zeros((0, 2), dtype=np.int64)
    for src, dst in raw_edges:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("scf saw a dangling edge; refusing")
        i, j = index[s], index[d]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt != 1:
            continue
        preds = []
        for pred in ins.get(s, []):
            p = int(pred)
            if p not in index:
                raise RuntimeError("scf saw a dangling predecessor; refusing")
            if int(graph.t[i]) - int(graph.t[index[p]]) == 1:
                preds.append(p)
        if len(preds) != 1:
            continue
        pred = preds[0]
        speed1, speed2, delta, rel = speed_change_um(
            xyz[index[pred]], xyz[i], xyz[j], speed_eps=speed_eps
        )
        rows.append(
            {
                "dataset": graph.dataset,
                "pred": int(pred),
                "src": int(s),
                "dst": int(d),
                "dt": 1,
                "speed1_um": speed1,
                "speed2_um": speed2,
                "speed_change_um": delta,
                "rel_speed_change": rel,
            }
        )
    return rows


def classify_dt1_speed_changes(
    graph: TrackGraph,
    gt: TrackGraph,
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach official sparse-GT hop classes. Leftovers are not negatives."""
    mapping = match_nodes(graph, gt)
    gt_edge_set = _gt_edge_set(gt)
    gt_sources, gt_targets = _gt_partners(gt)
    out: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["edge_class"] = hop_class(
            int(row["src"]),
            int(row["dst"]),
            mapping,
            gt_edge_set,
            gt_sources,
            gt_targets,
        )
        out.append(item)
    return out


def propose_speed_change_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One-shot 0.1 µm/frame round-up above TP max if conflict FPs sit above it.

    Leftover unmatched edges are reported but never used as negatives.
    Relative |Δv| is recorded, not gated. No threshold sweep.
    """
    tp = [float(r["speed_change_um"]) for r in rows if r.get("edge_class") == "tp"]
    cfp = [float(r["speed_change_um"]) for r in rows if r.get("edge_class") == "conflict_fp"]
    leftover = [float(r["speed_change_um"]) for r in rows if r.get("edge_class") == "leftover"]
    tp_rel = [float(r["rel_speed_change"]) for r in rows if r.get("edge_class") == "tp"]
    cfp_rel = [float(r["rel_speed_change"]) for r in rows if r.get("edge_class") == "conflict_fp"]
    leftover_rel = [float(r["rel_speed_change"]) for r in rows if r.get("edge_class") == "leftover"]
    summary = {
        "n_hops": len(rows),
        "n_tp": len(tp),
        "n_conflict_fp": len(cfp),
        "n_leftover": len(leftover),
        "tp_max_um": max(tp) if tp else None,
        "tp_min_um": min(tp) if tp else None,
        "conflict_fp_max_um": max(cfp) if cfp else None,
        "conflict_fp_min_um": min(cfp) if cfp else None,
        "leftover_max_um": max(leftover) if leftover else None,
        "tp_max_rel": max(tp_rel) if tp_rel else None,
        "conflict_fp_max_rel": max(cfp_rel) if cfp_rel else None,
        "leftover_max_rel": max(leftover_rel) if leftover_rel else None,
        "leftover_not_negatives": True,
        "supported": False,
        "max_speed_change_um": None,
        "reason": "unscored",
        "n_conflict_fp_above_gate": 0,
        "n_tp_above_gate": 0,
        "n_leftover_above_gate": 0,
        "retunes_max_dt1_edge_um": False,
        "retunes_max_rel_appearance": False,
        "relative_not_gated": True,
    }
    if not tp:
        summary["reason"] = "no_tp_dt1_speed_changes"
        return summary
    tp_max = max(tp)
    if float(tp_max) >= float(MAX_DT1_EDGE_UM) - 1e-9:
        summary["reason"] = "tp_covers_full_speed_change"
        summary["max_speed_change_um"] = None
        return summary
    gate = round_up_tenth_um(tp_max)
    if gate > float(MAX_DT1_EDGE_UM):
        summary["reason"] = "tp_covers_full_speed_change"
        summary["max_speed_change_um"] = None
        return summary
    n_tp_over = sum(1 for x in tp if x > gate)
    n_cfp_over = sum(1 for x in cfp if x > gate)
    n_left_over = sum(1 for x in leftover if x > gate)
    summary["max_speed_change_um"] = gate
    summary["n_conflict_fp_above_gate"] = int(n_cfp_over)
    summary["n_tp_above_gate"] = int(n_tp_over)
    summary["n_leftover_above_gate"] = int(n_left_over)
    if n_tp_over:
        summary["reason"] = "gate_would_drop_tp"
        return summary
    if n_cfp_over <= 0:
        summary["reason"] = "no_conflict_fp_above_gate"
        return summary
    summary["supported"] = True
    summary["reason"] = "tp_max_round_up_separates_conflict_fp"
    return summary


def _empty_scf_graph(graph: TrackGraph, max_um: float) -> TrackGraph:
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=np.zeros((0,), dtype=np.int64),
        t=np.zeros((0,), dtype=np.int64),
        z=np.zeros((0,), dtype=np.int64),
        y=np.zeros((0,), dtype=np.int64),
        x=np.zeros((0,), dtype=np.int64),
        edges=np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    out.metadata["n_dropped_abrupt_dt1_speed_changes"] = 0
    out.metadata["n_checked_dt1_speed_changes"] = 0
    out.metadata["n_kept_dt1_speed_changes"] = 0
    out.metadata["n_kept_non_dt1_edges"] = 0
    out.metadata["max_speed_change_um"] = float(max_um)
    return out


def drop_abrupt_dt1_speed_changes(
    graph: TrackGraph,
    max_um: float,
    speed_eps: float = SPEED_EPS_UM,
) -> TrackGraph:
    """Drop dt=1 continuation edges whose unique-predecessor |Δv| exceeds ``max_um``.

    Edges without a unique dt=1 predecessor and gap-close hops are kept.
    When the diagnostic has locked ``MAX_SPEED_CHANGE_UM``, refuse any
    other gate.
    """
    if MAX_SPEED_CHANGE_UM is not None and float(max_um) != float(MAX_SPEED_CHANGE_UM):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_scf_v1 would retune max_speed_change_um; refusing")
    nids = [int(x) for x in graph.node_id]
    if not nids:
        return _empty_scf_graph(graph, max_um)
    hops = extract_dt1_speed_changes(graph, speed_eps=speed_eps)
    drop: set[tuple[int, int]] = set()
    kept_hops = 0
    for hop in hops:
        if float(hop["speed_change_um"]) > float(max_um):
            drop.add((int(hop["src"]), int(hop["dst"])))
        else:
            kept_hops += 1
    raw_edges = np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else np.zeros((0, 2), dtype=np.int64)
    index = graph.id_to_index()
    kept: list[tuple[int, int]] = []
    kept_non_dt1 = 0
    for src, dst in raw_edges:
        edge = (int(src), int(dst))
        if edge in drop:
            continue
        kept.append(edge)
        i, j = index[edge[0]], index[edge[1]]
        if int(graph.t[j]) - int(graph.t[i]) != 1:
            kept_non_dt1 += 1
    out = TrackGraph(
        dataset=graph.dataset,
        node_id=graph.node_id,
        t=graph.t,
        z=graph.z,
        y=graph.y,
        x=graph.x,
        edges=np.asarray(kept, dtype=np.int64).reshape(-1, 2) if kept else np.zeros((0, 2), dtype=np.int64),
        estimated_n_nodes=graph.estimated_n_nodes,
    )
    out.metadata = dict(graph.metadata)
    out.metadata["n_dropped_abrupt_dt1_speed_changes"] = len(drop)
    out.metadata["n_checked_dt1_speed_changes"] = len(hops)
    out.metadata["n_kept_dt1_speed_changes"] = int(kept_hops)
    out.metadata["n_kept_non_dt1_edges"] = int(kept_non_dt1)
    out.metadata["n_nodes_before_scf"] = int(graph.n_nodes)
    out.metadata["n_edges_before_scf"] = int(graph.n_edges)
    out.metadata["max_speed_change_um"] = float(max_um)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("scf kept a dangling edge; refusing")
    return out


def apply_orgdet_fnn_acf_scf_overlay(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float] | None = None,
    zarr_path: str | Path | None = None,
) -> TrackGraph:
    """Locked ACF, then drop abrupt dt=1 speed changes and new isolates."""
    if MAX_SPEED_CHANGE_UM is None or LOCKED_GATES["max_speed_change_um"] is None:
        raise RuntimeError(
            "learned_unet_orgdet_fnn_acf_scf_v1 speed-change gate is unlocked; "
            "run the 44b6 diagnostic and lock before scoring"
        )
    if orgdet_fnn_acf_scf_config_hash() != LOCKED_ORGDET_FNN_ACF_SCF_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_scf_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_fnn_acf":
        raise RuntimeError("learned_unet_orgdet_fnn_acf_scf_v1 would retune assignment; refusing")
    if not LOCKED_GATES["drop_abrupt_dt1_speed_change"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_scf_v1 required filters disabled; refusing")
    if float(LOCKED_GATES["max_speed_change_um"]) != float(MAX_SPEED_CHANGE_UM):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_scf_v1 would retune max_speed_change_um; refusing")
    if int(LOCKED_GATES["max_gap_dt"]) != int(MAX_GAP_DT):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_scf_v1 would retune max_gap_dt; refusing")
    acf = apply_orgdet_fnn_acf_overlay(graph, appearance_by_id=appearance_by_id, zarr_path=zarr_path)
    filtered = drop_abrupt_dt1_speed_changes(acf, max_um=float(LOCKED_GATES["max_speed_change_um"]))
    n_dropped = int(filtered.metadata.get("n_dropped_abrupt_dt1_speed_changes") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_scf_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_SCF_HASH
    out.metadata["parent_orgdet_hash"] = LOCKED_ORGDET_HASH
    out.metadata["parent_orgdet_gap_hash"] = LOCKED_ORGDET_GAP_HASH
    out.metadata["parent_orgdet_gap_prune_hash"] = LOCKED_ORGDET_GAP_PRUNE_HASH
    out.metadata["parent_orgdet_elf_hash"] = LOCKED_ORGDET_ELF_HASH
    out.metadata["parent_orgdet_fnn_hash"] = LOCKED_ORGDET_FNN_HASH
    out.metadata["parent_orgdet_fnn_acf_hash"] = LOCKED_ORGDET_FNN_ACF_HASH
    out.metadata["rejected_orgop_hash"] = REJECTED_ORGOP_HASH
    out.metadata["rejected_fnn_rank_hash"] = LOCKED_ORGDET_FNN_RANK_HASH
    out.metadata["rejected_fnn_div_hash"] = LOCKED_ORGDET_FNN_DIV_HASH
    out.metadata["rejected_fnn_reas_hash"] = LOCKED_ORGDET_FNN_REAS_HASH
    out.metadata["rejected_fnn_acf_bnn_hash"] = LOCKED_ORGDET_FNN_ACF_BNN_HASH
    out.metadata["rejected_fnn_acf_ipc_hash"] = LOCKED_ORGDET_FNN_ACF_IPC_HASH
    out.metadata["rejected_fnn_acf_gacf_hash"] = LOCKED_ORGDET_FNN_ACF_GACF_HASH
    out.metadata["rejected_fnn_acf_gelf_hash"] = LOCKED_ORGDET_FNN_ACF_GELF_HASH
    out.metadata["rejected_fnn_acf_tac_hash"] = LOCKED_ORGDET_FNN_ACF_TAC_HASH
    out.metadata["n_dropped_abrupt_dt1_speed_changes"] = n_dropped
    out.metadata["n_checked_dt1_speed_changes"] = int(filtered.metadata.get("n_checked_dt1_speed_changes") or 0)
    out.metadata["n_dropped_inconsistent_dt1_edges"] = int(
        acf.metadata.get("n_dropped_inconsistent_dt1_edges") or 0
    )
    out.metadata["n_dropped_non_forward_nn_edges"] = int(
        acf.metadata.get("n_dropped_non_forward_nn_edges") or 0
    )
    out.metadata["n_dropped_long_dt1_edges"] = int(acf.metadata.get("n_dropped_long_dt1_edges") or 0)
    out.metadata["max_speed_change_um"] = float(filtered.metadata.get("max_speed_change_um") or MAX_SPEED_CHANGE_UM)
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
    out.metadata["uses_velocity"] = True
    out.metadata["retunes_detection"] = False
    out.metadata["retunes_consecutive_links"] = False
    out.metadata["retunes_gap_gates"] = False
    out.metadata["retunes_max_dt1_edge_um"] = False
    out.metadata["retunes_nn_tie_eps"] = False
    out.metadata["retunes_max_rel_appearance"] = False
    out.metadata["is_12um_assignment"] = False
    out.metadata["is_global_backward_nn"] = False
    return out


def overlay_summary(before: TrackGraph, after: TrackGraph) -> dict[str, Any]:
    return {
        "dataset": after.dataset,
        "n_nodes_before": before.n_nodes,
        "n_nodes_after": after.n_nodes,
        "n_edges_before": before.n_edges,
        "n_edges_after": after.n_edges,
        "n_dropped_abrupt_dt1_speed_changes": int(after.metadata.get("n_dropped_abrupt_dt1_speed_changes") or 0),
        "n_checked_dt1_speed_changes": int(after.metadata.get("n_checked_dt1_speed_changes") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_dropped_inconsistent_dt1_edges": int(after.metadata.get("n_dropped_inconsistent_dt1_edges") or 0),
        "n_dropped_non_forward_nn_edges": int(after.metadata.get("n_dropped_non_forward_nn_edges") or 0),
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_nodes_before_scf": int(after.metadata.get("n_nodes_before_scf") or after.n_nodes),
        "n_edges_before_scf": int(after.metadata.get("n_edges_before_scf") or after.n_edges),
        "max_speed_change_um": after.metadata.get("max_speed_change_um"),
        "overlay": "learned_unet_orgdet_fnn_acf_scf_v1",
        "config_hash": LOCKED_ORGDET_FNN_ACF_SCF_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_acf_scf_combined") is not None:
        raise SystemExit(
            f"refusing to score frozen holdout twice; existing decision={payload.get('decision')}"
        )


def require_tuning_keep(path: str | Path) -> dict[str, Any]:
    dest = Path(path)
    if not dest.exists():
        raise SystemExit("refusing holdout; tuning report missing. Score 44b6 first.")
    payload = json.loads(dest.read_text())
    decision = str(payload.get("decision") or "")
    if not decision.startswith("keep_"):
        raise SystemExit(f"refusing holdout; tuning decision is {decision}")
    return payload


def require_supported_gate() -> float:
    if MAX_SPEED_CHANGE_UM is None:
        raise SystemExit(
            "refusing official overlay score; 44b6 speed-change diagnostic did not lock a gate"
        )
    return float(MAX_SPEED_CHANGE_UM)


def tuning_decision(learned_combined: float, n_dropped_edges: int) -> str:
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_fnn_acf_scf_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_acf_scf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_scf_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_ACF_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_acf_scf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_scf_v1"


def diagnostic_decision(proposal: Mapping[str, Any]) -> str:
    if proposal.get("supported"):
        return "lock_learned_unet_orgdet_fnn_acf_scf_v1"
    return "reject_learned_unet_orgdet_fnn_acf_scf_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 8.0,
    scf_overhead: float = 1.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1100.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(scf_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "scf_overhead": float(scf_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN+ACF then drop dt=1 "
            "continuation edges whose unique-predecessor speed-change "
            "exceeds the locked 44b6 TP-derived gate and new degree-0 "
            "leftovers; two sequence Zarrs for the locked ACF parent; no "
            "transformer; no ILP; no new UNet"
        ),
        "prefers_tuning_zarrs": True,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
