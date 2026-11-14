"""Drop long gap-close chains after locked ACF (length, not appearance).

Official 44b6 ACF residue is still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. ELF only
drops dt=1 edges longer than 8.0 µm. Gap-close still links leftover
detections across dt∈{2,3} (interpolated to dt=1 hops). Those real
endpoints stay unfiltered by ELF.

This overlay keeps locked ACF, then drops a gap-close chain when the
two real endpoints are farther than the locked gap-edge length gate,
and drops any remaining dt≥2 edges under the same gate. Newly created
degree-0 leftovers are dropped. It does not reassign, does not shrink
the 12 µm linker/gap gates, and is not an ELF / FNN / ACF / BNN /
IPC / TCF / GACF / rank / division retune. Do not retune
``max_dt1_edge_um`` or ``max_rel_appearance``.

The length gate is locked from 44b6 ACF official-matched TP gap-chain
real-endpoint displacements only. Leftover unmatched chains are not
negatives. If that diagnostic does not separate conflict FPs above a
0.1 µm round-up of TP max, the overlay is rejected without a threshold
sweep and without an official 44b6 overlay score.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number. Do not overwrite locked slugs. Do not retune locked
parent hashes. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
import math
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
from .learned_orgdet_fnn_acf_ipc import LOCKED_ORGDET_FNN_ACF_IPC_HASH
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
# Locked from 44b6 ACF TP gap-chain real-endpoint lengths only. None
# means the diagnostic has not supported a gate. Do not invent one.
MAX_GAP_EDGE_UM: float | None = None

LOCKED_GATES = {
    "drop_long_gap": True,
    "drop_degree0_after": True,
    "max_gap_edge_um": MAX_GAP_EDGE_UM,
    "max_gap_dt": MAX_GAP_DT,
    "assignment": "unchanged_locked_orgdet_fnn_acf",
}


def orgdet_fnn_acf_gelf_config_payload() -> dict[str, Any]:
    """Gap-edge length filter after locked ACF. Lock the µm gate before 44b6."""
    return {
        "label": "learned_unet_orgdet_fnn_acf_gelf_v1",
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
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_fnn_acf_then_drop_long_gap_length",
        "drop_long_gap": LOCKED_GATES["drop_long_gap"],
        "drop_degree0_after": LOCKED_GATES["drop_degree0_after"],
        "max_gap_edge_um": LOCKED_GATES["max_gap_edge_um"],
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


def orgdet_fnn_acf_gelf_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_acf_gelf_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_ACF_GELF_HASH = orgdet_fnn_acf_gelf_config_hash(orgdet_fnn_acf_gelf_config_payload())
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH == "59812f90385ed4d2"
assert LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_FNN_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
assert LOCKED_ORGDET_FNN_ACF_BNN_HASH == "bdb38cf13e226ced"
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH == "7302409595bee523"
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH == "5ce49a6d175c8032"
assert TUNE_KEEP_THRESHOLD == 0.7366197119237633
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_ACF_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_RANK_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_DIV_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_REAS_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_ACF_BNN_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_ACF_IPC_HASH
assert LOCKED_ORGDET_FNN_ACF_GELF_HASH != LOCKED_ORGDET_FNN_ACF_GACF_HASH
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
            raise ValueError(f"learned_unet_orgdet_fnn_acf_gelf_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_acf_gelf_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_acf_gelf_v1 holdout refuses non-holdout stem {stem}")


def round_up_tenth_um(value: float) -> float:
    """Smallest 0.1 µm gate strictly above ``value``."""
    return math.ceil((float(value) + 1e-12) * 10.0) / 10.0


def _gt_edge_set(graph: TrackGraph) -> set[tuple[int, int]]:
    return {(int(s), int(d)) for s, d in graph.consecutive_edges()}


def _gt_partners(gt: TrackGraph) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    sources: dict[int, set[int]] = {}
    targets: dict[int, set[int]] = {}
    for src, dst in gt.consecutive_edges():
        sources.setdefault(int(dst), set()).add(int(src))
        targets.setdefault(int(src), set()).add(int(dst))
    return sources, targets


def hop_class(
    src: int,
    dst: int,
    mapping: Mapping[int, int],
    gt_edge_set: set[tuple[int, int]],
    gt_sources: Mapping[int, set[int]],
    gt_targets: Mapping[int, set[int]],
) -> str:
    """Official sparse-GT hop class: tp / conflict_fp / leftover."""
    ms = mapping.get(int(src))
    md = mapping.get(int(dst))
    if ms is not None and md is not None and (ms, md) in gt_edge_set:
        return "tp"
    conflict = False
    if md is not None and gt_sources.get(md) and (ms is None or ms not in gt_sources[md]):
        conflict = True
    if ms is not None and gt_targets.get(ms) and (md is None or md not in gt_targets[ms]):
        conflict = True
    if conflict:
        return "conflict_fp"
    return "leftover"


def chain_class_from_hops(hop_classes: Sequence[str]) -> str:
    if any(cls == "tp" for cls in hop_classes):
        return "tp"
    if any(cls == "conflict_fp" for cls in hop_classes):
        return "conflict_fp"
    return "leftover"


def extract_gap_chains(
    graph: TrackGraph,
    source_ids: Iterable[int],
    max_gap_dt: int = MAX_GAP_DT,
) -> list[dict[str, Any]]:
    """Walk pre-gap sources through interpolants to real endpoints (dt 2..3)."""
    if int(max_gap_dt) != int(MAX_GAP_DT):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 would retune max_gap_dt; refusing")
    nids = [int(x) for x in graph.node_id]
    if not nids:
        return []
    source = {int(x) for x in source_ids}
    index = graph.id_to_index()
    outs = graph.out_edges()
    xyz = physical_coords(graph.coords_zyx(), SCALE_ZYX)
    raw_edges = np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else np.zeros((0, 2), dtype=np.int64)
    chains: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()

    def _record(start: int, end: int, edges: list[tuple[int, int]]) -> None:
        key = (int(start), int(end))
        if key in seen:
            return
        i, j = index[start], index[end]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt < 2 or dt > int(max_gap_dt):
            return
        seen.add(key)
        length = float(np.linalg.norm(xyz[i] - xyz[j]))
        chains.append(
            {
                "dataset": graph.dataset,
                "start": int(start),
                "end": int(end),
                "dt": int(dt),
                "length_um": length,
                "edges": [(int(a), int(b)) for a, b in edges],
                "n_hops": len(edges),
                "interpolated": any(int(a) not in source or int(b) not in source for a, b in edges),
            }
        )

    for src, dst in raw_edges:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("gelf saw a dangling edge; refusing")
        if s in source and d in source:
            _record(s, d, [(s, d)])

    for start in nids:
        if start not in source:
            continue
        for first in outs.get(start, []):
            child = int(first)
            if child in source:
                continue
            chain = [(start, child)]
            cur = child
            visited = {start, child}
            complete = True
            while cur not in source:
                nxts = outs.get(cur, [])
                if len(nxts) != 1:
                    complete = False
                    break
                nxt = int(nxts[0])
                if nxt in visited:
                    complete = False
                    break
                chain.append((cur, nxt))
                visited.add(nxt)
                cur = nxt
            if not complete or cur not in source:
                continue
            _record(start, cur, chain)
    return chains


def classify_gap_chains(
    graph: TrackGraph,
    gt: TrackGraph,
    chains: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Attach official sparse-GT hop/chain classes. Leftovers are not negatives."""
    mapping = match_nodes(graph, gt)
    gt_edge_set = _gt_edge_set(gt)
    gt_sources, gt_targets = _gt_partners(gt)
    out: list[dict[str, Any]] = []
    for chain in chains:
        hops = [tuple(edge) for edge in chain["edges"]]
        hop_classes = [
            hop_class(src, dst, mapping, gt_edge_set, gt_sources, gt_targets) for src, dst in hops
        ]
        row = dict(chain)
        row["hop_classes"] = hop_classes
        row["chain_class"] = chain_class_from_hops(hop_classes)
        row["start_matched"] = int(chain["start"]) in mapping
        row["end_matched"] = int(chain["end"]) in mapping
        out.append(row)
    return out


def propose_gap_length_gate(chains: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One-shot 0.1 µm round-up above TP max if conflict FPs sit above it.

    Leftover unmatched chains are reported but never used as negatives.
    No threshold sweep.
    """
    tp = [float(c["length_um"]) for c in chains if c.get("chain_class") == "tp"]
    cfp = [float(c["length_um"]) for c in chains if c.get("chain_class") == "conflict_fp"]
    leftover = [float(c["length_um"]) for c in chains if c.get("chain_class") == "leftover"]
    by_dt: dict[int, dict[str, int]] = {}
    for chain in chains:
        dt = int(chain.get("dt") or 0)
        bucket = by_dt.setdefault(dt, {"tp": 0, "conflict_fp": 0, "leftover": 0})
        cls = str(chain.get("chain_class") or "leftover")
        if cls not in bucket:
            cls = "leftover"
        bucket[cls] += 1
    summary = {
        "n_chains": len(chains),
        "n_tp": len(tp),
        "n_conflict_fp": len(cfp),
        "n_leftover": len(leftover),
        "tp_max_um": max(tp) if tp else None,
        "tp_min_um": min(tp) if tp else None,
        "conflict_fp_max_um": max(cfp) if cfp else None,
        "conflict_fp_min_um": min(cfp) if cfp else None,
        "leftover_max_um": max(leftover) if leftover else None,
        "leftover_not_negatives": True,
        "by_dt": {str(k): v for k, v in sorted(by_dt.items())},
        "supported": False,
        "max_gap_edge_um": None,
        "reason": "unscored",
        "n_conflict_fp_above_gate": 0,
        "n_tp_above_gate": 0,
        "n_leftover_above_gate": 0,
        "retunes_max_dt1_edge_um": False,
    }
    if not tp:
        summary["reason"] = "no_tp_gap_chains"
        return summary
    gate = round_up_tenth_um(max(tp))
    n_tp_over = sum(1 for x in tp if x > gate)
    n_cfp_over = sum(1 for x in cfp if x > gate)
    n_left_over = sum(1 for x in leftover if x > gate)
    summary["max_gap_edge_um"] = gate
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


def _empty_gelf_graph(graph: TrackGraph, max_um: float) -> TrackGraph:
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
    out.metadata["n_dropped_long_gap_edges"] = 0
    out.metadata["n_dropped_long_gap_chains"] = 0
    out.metadata["n_kept_gap_chains"] = 0
    out.metadata["n_checked_gap_chains"] = 0
    out.metadata["n_kept_dt1_edges"] = 0
    out.metadata["max_gap_edge_um"] = float(max_um)
    return out


def _assemble_gelf_graph(
    graph: TrackGraph,
    kept: list[tuple[int, int]],
    dropped_edges: int,
    dropped_chains: int,
    kept_chains: int,
    checked_chains: int,
    kept_dt1: int,
    max_um: float,
) -> TrackGraph:
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
    out.metadata["n_dropped_long_gap_edges"] = int(dropped_edges)
    out.metadata["n_dropped_long_gap_chains"] = int(dropped_chains)
    out.metadata["n_kept_gap_chains"] = int(kept_chains)
    out.metadata["n_checked_gap_chains"] = int(checked_chains)
    out.metadata["n_kept_dt1_edges"] = int(kept_dt1)
    out.metadata["n_nodes_before_gelf"] = int(graph.n_nodes)
    out.metadata["n_edges_before_gelf"] = int(graph.n_edges)
    out.metadata["max_gap_edge_um"] = float(max_um)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("gelf kept a dangling edge; refusing")
    return out


def drop_long_gap_edges(
    graph: TrackGraph,
    source_ids: Iterable[int],
    max_um: float,
    max_gap_dt: int = MAX_GAP_DT,
) -> TrackGraph:
    """Drop gap-close chains / dt≥2 edges whose real-endpoint length exceeds ``max_um``.

    ``source_ids`` are the pre-gap detection node ids. Interpolants added
    by locked gap-close are not in that set. Consecutive dt=1 ACF/FNN/ELF
    edges are kept. Do not retune ``max_gap_dt``. When the diagnostic has
    locked ``MAX_GAP_EDGE_UM``, refuse any other gate.
    """
    if MAX_GAP_EDGE_UM is not None and float(max_um) != float(MAX_GAP_EDGE_UM):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 would retune max_gap_edge_um; refusing")
    if int(max_gap_dt) != int(MAX_GAP_DT):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 would retune max_gap_dt; refusing")
    nids = [int(x) for x in graph.node_id]
    if not nids:
        return _empty_gelf_graph(graph, max_um)
    chains = extract_gap_chains(graph, source_ids, max_gap_dt=max_gap_dt)
    drop: set[tuple[int, int]] = set()
    dropped_chains = 0
    kept_chains = 0
    for chain in chains:
        if float(chain["length_um"]) > float(max_um):
            drop.update((int(a), int(b)) for a, b in chain["edges"])
            dropped_chains += 1
        else:
            kept_chains += 1
    raw_edges = np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else np.zeros((0, 2), dtype=np.int64)
    index = graph.id_to_index()
    kept: list[tuple[int, int]] = []
    kept_dt1 = 0
    for src, dst in raw_edges:
        edge = (int(src), int(dst))
        if edge in drop:
            continue
        kept.append(edge)
        i, j = index[edge[0]], index[edge[1]]
        if int(graph.t[j]) - int(graph.t[i]) == 1:
            kept_dt1 += 1
    return _assemble_gelf_graph(
        graph,
        kept,
        len(drop),
        dropped_chains,
        kept_chains,
        len(chains),
        kept_dt1,
        max_um,
    )


def apply_orgdet_fnn_acf_gelf_overlay(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float] | None = None,
    zarr_path: str | Path | None = None,
) -> TrackGraph:
    """Locked ACF, then drop long gap-close edges and new isolates."""
    if MAX_GAP_EDGE_UM is None or LOCKED_GATES["max_gap_edge_um"] is None:
        raise RuntimeError(
            "learned_unet_orgdet_fnn_acf_gelf_v1 length gate is unlocked; "
            "run the 44b6 diagnostic and lock before scoring"
        )
    if orgdet_fnn_acf_gelf_config_hash() != LOCKED_ORGDET_FNN_ACF_GELF_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_fnn_acf":
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 would retune assignment; refusing")
    if not LOCKED_GATES["drop_long_gap"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 required filters disabled; refusing")
    if float(LOCKED_GATES["max_gap_edge_um"]) != float(MAX_GAP_EDGE_UM):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 would retune max_gap_edge_um; refusing")
    if int(LOCKED_GATES["max_gap_dt"]) != int(MAX_GAP_DT):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gelf_v1 would retune max_gap_dt; refusing")
    source_ids = {int(x) for x in graph.node_id}
    acf = apply_orgdet_fnn_acf_overlay(graph, appearance_by_id=appearance_by_id, zarr_path=zarr_path)
    filtered = drop_long_gap_edges(
        acf,
        source_ids,
        max_um=float(LOCKED_GATES["max_gap_edge_um"]),
        max_gap_dt=LOCKED_GATES["max_gap_dt"],
    )
    n_dropped_edges = int(filtered.metadata.get("n_dropped_long_gap_edges") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_gelf_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_GELF_HASH
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
    out.metadata["n_dropped_long_gap_edges"] = n_dropped_edges
    out.metadata["n_dropped_long_gap_chains"] = int(filtered.metadata.get("n_dropped_long_gap_chains") or 0)
    out.metadata["n_checked_gap_chains"] = int(filtered.metadata.get("n_checked_gap_chains") or 0)
    out.metadata["n_dropped_inconsistent_dt1_edges"] = int(
        acf.metadata.get("n_dropped_inconsistent_dt1_edges") or 0
    )
    out.metadata["n_dropped_non_forward_nn_edges"] = int(
        acf.metadata.get("n_dropped_non_forward_nn_edges") or 0
    )
    out.metadata["n_dropped_long_dt1_edges"] = int(acf.metadata.get("n_dropped_long_dt1_edges") or 0)
    out.metadata["max_gap_edge_um"] = float(filtered.metadata.get("max_gap_edge_um") or MAX_GAP_EDGE_UM)
    out.metadata["uses_intensity"] = False
    out.metadata["uses_transformer"] = False
    out.metadata["uses_ilp"] = False
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
        "n_dropped_long_gap_edges": int(after.metadata.get("n_dropped_long_gap_edges") or 0),
        "n_dropped_long_gap_chains": int(after.metadata.get("n_dropped_long_gap_chains") or 0),
        "n_checked_gap_chains": int(after.metadata.get("n_checked_gap_chains") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_dropped_inconsistent_dt1_edges": int(after.metadata.get("n_dropped_inconsistent_dt1_edges") or 0),
        "n_dropped_non_forward_nn_edges": int(after.metadata.get("n_dropped_non_forward_nn_edges") or 0),
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_nodes_before_gelf": int(after.metadata.get("n_nodes_before_gelf") or after.n_nodes),
        "n_edges_before_gelf": int(after.metadata.get("n_edges_before_gelf") or after.n_edges),
        "max_gap_edge_um": after.metadata.get("max_gap_edge_um"),
        "overlay": "learned_unet_orgdet_fnn_acf_gelf_v1",
        "config_hash": LOCKED_ORGDET_FNN_ACF_GELF_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_acf_gelf_combined") is not None:
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
    if MAX_GAP_EDGE_UM is None:
        raise SystemExit(
            "refusing official overlay score; 44b6 gap-length diagnostic did not lock a gate"
        )
    return float(MAX_GAP_EDGE_UM)


def tuning_decision(learned_combined: float, n_dropped_edges: int) -> str:
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_fnn_acf_gelf_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_acf_gelf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_gelf_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_ACF_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_acf_gelf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_gelf_v1"


def diagnostic_decision(proposal: Mapping[str, Any]) -> str:
    if proposal.get("supported"):
        return "lock_learned_unet_orgdet_fnn_acf_gelf_v1"
    return "reject_learned_unet_orgdet_fnn_acf_gelf_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 8.0,
    gelf_overhead: float = 1.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1100.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(gelf_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "gelf_overhead": float(gelf_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN+ACF then drop gap-close "
            "chains whose real-endpoint physical displacement exceeds the "
            "locked 44b6 TP-derived gate and new degree-0 leftovers; two "
            "sequence Zarrs for the locked ACF parent; no transformer; "
            "no ILP; no new UNet"
        ),
        "prefers_tuning_zarrs": True,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
