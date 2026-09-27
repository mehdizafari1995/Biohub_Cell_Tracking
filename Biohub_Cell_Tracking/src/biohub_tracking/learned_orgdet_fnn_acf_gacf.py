"""Drop appearance-inconsistent gap-close edges after locked ACF.

Official 44b6 ACF residue is still association-dominated (160 conflict
FPs / 143 FNs; 0 predicted divisions). Locked ACF only drops dt=1
edges whose 3×3×3-mean relative appearance exceeds 0.38. Gap-close
still links leftover detections across dt∈{2,3} (interpolated to
dt=1 hops). Those real endpoints stay unfiltered. Incoming-parent
IPC is a no-op after ACF; global backward-NN is closed.

This overlay keeps locked ACF, then drops a gap-close chain when the
two real endpoints have relative appearance > 0.38, and drops any
remaining dt≥2 edges under the same locked threshold. Newly created
degree-0 leftovers are dropped. It does not reassign, does not shrink
the 12 µm linker/gap gates, and is not an ELF / FNN / ACF / BNN /
IPC / TCF / rank / division retune. Do not retune
``max_rel_appearance``.

Never opens ``6bba`` during tuning. Does not write ``submission.csv``.
Not a competition submission. A local official score is not a
leaderboard number. Do not overwrite locked slugs. Do not retune locked
parent hashes. Do not use public 0.602 / 0.612 / 0.553.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .graphs import TrackGraph
from .learned_infer import LOCKED_INFER_HASH, TRAIN_CONFIG_HASH
from .learned_orgdet import LOCKED_ORGDET_HASH
from .learned_orgdet_elf import LOCKED_ORGDET_ELF_HASH
from .learned_orgdet_fnn import (
    HOLDOUT_STEMS,
    INFER_STEMS,
    LOCKED_ORGDET_FNN_HASH,
    ensure_orgdet_tuning_graph,
    load_role_graphs,
    refuse_non_orgdet_node_count,
)
from .learned_orgdet_fnn_acf import (
    APPEARANCE_EPS,
    APPEARANCE_FEATURE,
    LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT,
    LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE,
    LOCKED_ORGDET_FNN_ACF_HASH,
    MAX_REL_APPEARANCE,
    apply_orgdet_fnn_acf_overlay,
    relative_appearance_change,
    resolve_sequence_zarr,
    sample_graph_appearances,
)
from .learned_orgdet_fnn_acf_bnn import LOCKED_ORGDET_FNN_ACF_BNN_HASH
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

SCALE_ZYX = (1.625, 0.40625, 0.40625)
ORGDET_FNN_ACF_LOCKED_TUNE = LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE
ORGDET_FNN_ACF_LOCKED_HOLDOUT = LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT
TUNE_MARGIN = 0.002000
TUNE_KEEP_THRESHOLD = ORGDET_FNN_ACF_LOCKED_TUNE + TUNE_MARGIN
# Locked association_v2 / learned gap-close span. Do not retune.
MAX_GAP_DT = 3

LOCKED_GATES = {
    "drop_inconsistent_gap": True,
    "drop_degree0_after": True,
    "max_rel_appearance": MAX_REL_APPEARANCE,
    "appearance_eps": APPEARANCE_EPS,
    "appearance_feature": APPEARANCE_FEATURE,
    "max_gap_dt": MAX_GAP_DT,
    "assignment": "unchanged_locked_orgdet_fnn_acf",
}


def orgdet_fnn_acf_gacf_config_payload() -> dict[str, Any]:
    """First gap-edge appearance filter after locked ACF. Lock before 44b6."""
    return {
        "label": "learned_unet_orgdet_fnn_acf_gacf_v1",
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
        "source_graphs": "locked_learned_unet_orgdet_fnn_acf_graphs",
        "source_zarrs": "locked_tuning_zarr_export_two_44b6",
        "retunes_detection": False,
        "retunes_consecutive_links": False,
        "retunes_gap_gates": False,
        "retunes_max_dt1_edge_um": False,
        "retunes_nn_tie_eps": False,
        "retunes_max_rel_appearance": False,
        "uses_intensity": True,
        "uses_transformer": False,
        "uses_velocity": False,
        "uses_ilp": False,
        "linker": "orgdet_fnn_acf_then_drop_inconsistent_gap_appearance",
        "drop_inconsistent_gap": LOCKED_GATES["drop_inconsistent_gap"],
        "drop_degree0_after": LOCKED_GATES["drop_degree0_after"],
        "max_rel_appearance": LOCKED_GATES["max_rel_appearance"],
        "appearance_eps": LOCKED_GATES["appearance_eps"],
        "appearance_feature": LOCKED_GATES["appearance_feature"],
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


def orgdet_fnn_acf_gacf_config_hash(payload: Mapping[str, Any] | None = None) -> str:
    blob = json.dumps(
        payload or orgdet_fnn_acf_gacf_config_payload(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


LOCKED_ORGDET_FNN_ACF_GACF_HASH = orgdet_fnn_acf_gacf_config_hash(orgdet_fnn_acf_gacf_config_payload())
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH == "5ce49a6d175c8032"
assert LOCKED_ORGDET_FNN_ACF_HASH == "a66075085ac64400"
assert LOCKED_ORGDET_FNN_HASH == "95e9300eb84ba71c"
assert LOCKED_ORGDET_FNN_RANK_HASH == "2748c4a2b2823c0a"
assert LOCKED_ORGDET_FNN_DIV_HASH == "e184e6ce5bd90e46"
assert LOCKED_ORGDET_FNN_REAS_HASH == "39fb20d675e21d0e"
assert LOCKED_ORGDET_FNN_ACF_BNN_HASH == "bdb38cf13e226ced"
assert LOCKED_ORGDET_FNN_ACF_IPC_HASH == "7302409595bee523"
assert TUNE_KEEP_THRESHOLD == 0.7366197119237633
assert REJECTED_ORGOP_HASH == "250c5bfacd4a8198"
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_ACF_HASH
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_HASH
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_RANK_HASH
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_DIV_HASH
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_REAS_HASH
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_ACF_BNN_HASH
assert LOCKED_ORGDET_FNN_ACF_GACF_HASH != LOCKED_ORGDET_FNN_ACF_IPC_HASH
assert LEARNED_ORGDET_FNN_ACF_LOCKED_TUNE == 0.7346197119237633
assert LEARNED_ORGDET_FNN_ACF_LOCKED_HOLDOUT == 0.7555029702166214


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
            raise ValueError(f"learned_unet_orgdet_fnn_acf_gacf_v1 tuning refuses stem {stem}")


def refuse_non_holdout_stems(stems: Iterable[str]) -> None:
    for raw in stems:
        stem = _clean_stem(str(raw))
        if specimen_of(stem) in HOLDOUT_SPECIMENS or stem in HOLDOUT_STEMS:
            if stem not in HOLDOUT_STEMS:
                raise ValueError(f"learned_unet_orgdet_fnn_acf_gacf_v1 holdout refuses extra stem {stem}")
            continue
        raise ValueError(f"learned_unet_orgdet_fnn_acf_gacf_v1 holdout refuses non-holdout stem {stem}")


def _empty_gacf_graph(graph: TrackGraph, max_rel: float) -> TrackGraph:
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
    out.metadata["n_dropped_inconsistent_gap_edges"] = 0
    out.metadata["n_dropped_inconsistent_gap_chains"] = 0
    out.metadata["n_kept_gap_chains"] = 0
    out.metadata["n_checked_gap_chains"] = 0
    out.metadata["n_kept_dt1_edges"] = 0
    out.metadata["max_rel_appearance"] = float(max_rel)
    return out


def _assemble_gacf_graph(
    graph: TrackGraph,
    kept: list[tuple[int, int]],
    dropped_edges: int,
    dropped_chains: int,
    kept_chains: int,
    checked_chains: int,
    kept_dt1: int,
    max_rel: float,
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
    out.metadata["n_dropped_inconsistent_gap_edges"] = int(dropped_edges)
    out.metadata["n_dropped_inconsistent_gap_chains"] = int(dropped_chains)
    out.metadata["n_kept_gap_chains"] = int(kept_chains)
    out.metadata["n_checked_gap_chains"] = int(checked_chains)
    out.metadata["n_kept_dt1_edges"] = int(kept_dt1)
    out.metadata["n_nodes_before_gacf"] = int(graph.n_nodes)
    out.metadata["n_edges_before_gacf"] = int(graph.n_edges)
    out.metadata["max_rel_appearance"] = float(max_rel)
    leftover = set(int(x) for x in out.node_id)
    if any(src not in leftover or dst not in leftover for src, dst in kept):
        raise RuntimeError("gacf kept a dangling edge; refusing")
    return out


def drop_inconsistent_gap_edges(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float],
    source_ids: Iterable[int],
    max_rel: float = MAX_REL_APPEARANCE,
    max_gap_dt: int = MAX_GAP_DT,
) -> TrackGraph:
    """Drop gap-close chains / dt≥2 edges whose real-endpoint appearance exceeds ``max_rel``.

    ``source_ids`` are the pre-gap detection node ids. Interpolants added
    by locked gap-close are not in that set. Do not retune
    ``max_rel_appearance`` or ``max_gap_dt``.
    """
    if float(max_rel) != float(MAX_REL_APPEARANCE):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 would retune max_rel_appearance; refusing")
    if int(max_gap_dt) != int(MAX_GAP_DT):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 would retune max_gap_dt; refusing")
    nids = [int(x) for x in graph.node_id]
    if not nids:
        return _empty_gacf_graph(graph, max_rel)
    source = {int(x) for x in source_ids}
    index = graph.id_to_index()
    outs = graph.out_edges()
    raw_edges = np.asarray(graph.edges).reshape(-1, 2) if graph.n_edges else np.zeros((0, 2), dtype=np.int64)
    drop: set[tuple[int, int]] = set()
    checked: set[tuple[int, int]] = set()
    dropped_chains = 0
    kept_chains = 0

    def _require_appearance(nid: int) -> float:
        if nid not in appearance_by_id:
            raise RuntimeError("gacf missing appearance for a real gap endpoint; refusing")
        return float(appearance_by_id[nid])

    def _consider_chain(start: int, end: int, chain_edges: list[tuple[int, int]]) -> None:
        nonlocal dropped_chains, kept_chains
        key = (int(start), int(end))
        if key in checked:
            return
        i, j = index[start], index[end]
        dt = int(graph.t[j]) - int(graph.t[i])
        if dt < 2 or dt > int(max_gap_dt):
            return
        checked.add(key)
        rel = relative_appearance_change(_require_appearance(start), _require_appearance(end))
        if rel > float(max_rel):
            drop.update(chain_edges)
            dropped_chains += 1
        else:
            kept_chains += 1

    for src, dst in raw_edges:
        s, d = int(src), int(dst)
        if s not in index or d not in index:
            raise RuntimeError("gacf saw a dangling edge; refusing")
        if s in source and d in source:
            _consider_chain(s, d, [(s, d)])

    for start in nids:
        if start not in source:
            continue
        for first in outs.get(start, []):
            child = int(first)
            if child in source:
                continue
            chain = [(start, child)]
            cur = child
            seen = {start, child}
            complete = True
            while cur not in source:
                nxts = outs.get(cur, [])
                if len(nxts) != 1:
                    complete = False
                    break
                nxt = int(nxts[0])
                if nxt in seen:
                    complete = False
                    break
                chain.append((cur, nxt))
                seen.add(nxt)
                cur = nxt
            if not complete or cur not in source:
                continue
            _consider_chain(start, cur, chain)

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
    return _assemble_gacf_graph(
        graph,
        kept,
        len(drop),
        dropped_chains,
        kept_chains,
        len(checked),
        kept_dt1,
        max_rel,
    )


def apply_orgdet_fnn_acf_gacf_overlay(
    graph: TrackGraph,
    appearance_by_id: Mapping[int, float] | None = None,
    zarr_path: str | Path | None = None,
) -> TrackGraph:
    """Locked ACF, then drop appearance-inconsistent gap-close edges and new isolates."""
    if orgdet_fnn_acf_gacf_config_hash() != LOCKED_ORGDET_FNN_ACF_GACF_HASH:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 hash drifted; refusing to retune")
    if LOCKED_GATES["assignment"] != "unchanged_locked_orgdet_fnn_acf":
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 would retune assignment; refusing")
    if not LOCKED_GATES["drop_inconsistent_gap"] or not LOCKED_GATES["drop_degree0_after"]:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 required filters disabled; refusing")
    if float(LOCKED_GATES["max_rel_appearance"]) != float(MAX_REL_APPEARANCE):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 would retune max_rel_appearance; refusing")
    if LOCKED_GATES["appearance_feature"] != APPEARANCE_FEATURE:
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 would retune appearance_feature; refusing")
    if int(LOCKED_GATES["max_gap_dt"]) != int(MAX_GAP_DT):
        raise RuntimeError("learned_unet_orgdet_fnn_acf_gacf_v1 would retune max_gap_dt; refusing")
    source_ids = {int(x) for x in graph.node_id}
    acf = apply_orgdet_fnn_acf_overlay(graph, appearance_by_id=appearance_by_id, zarr_path=zarr_path)
    sampled = appearance_by_id
    if sampled is None:
        if zarr_path is None:
            raise RuntimeError("gacf needs appearance_by_id or zarr_path; refusing")
        sampled = sample_graph_appearances(acf, zarr_path)
    filtered = drop_inconsistent_gap_edges(
        acf,
        sampled,
        source_ids,
        max_rel=LOCKED_GATES["max_rel_appearance"],
        max_gap_dt=LOCKED_GATES["max_gap_dt"],
    )
    n_dropped_edges = int(filtered.metadata.get("n_dropped_inconsistent_gap_edges") or 0)
    out = drop_degree0(filtered)
    out.metadata["overlay"] = "learned_unet_orgdet_fnn_acf_gacf_v1"
    out.metadata["config_hash"] = LOCKED_ORGDET_FNN_ACF_GACF_HASH
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
    out.metadata["n_dropped_inconsistent_gap_edges"] = n_dropped_edges
    out.metadata["n_dropped_inconsistent_gap_chains"] = int(
        filtered.metadata.get("n_dropped_inconsistent_gap_chains") or 0
    )
    out.metadata["n_checked_gap_chains"] = int(filtered.metadata.get("n_checked_gap_chains") or 0)
    out.metadata["n_dropped_inconsistent_dt1_edges"] = int(
        acf.metadata.get("n_dropped_inconsistent_dt1_edges") or 0
    )
    out.metadata["n_dropped_non_forward_nn_edges"] = int(
        acf.metadata.get("n_dropped_non_forward_nn_edges") or 0
    )
    out.metadata["n_dropped_long_dt1_edges"] = int(acf.metadata.get("n_dropped_long_dt1_edges") or 0)
    out.metadata["max_rel_appearance"] = float(filtered.metadata.get("max_rel_appearance") or MAX_REL_APPEARANCE)
    out.metadata["uses_intensity"] = True
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
        "n_dropped_inconsistent_gap_edges": int(after.metadata.get("n_dropped_inconsistent_gap_edges") or 0),
        "n_dropped_inconsistent_gap_chains": int(after.metadata.get("n_dropped_inconsistent_gap_chains") or 0),
        "n_checked_gap_chains": int(after.metadata.get("n_checked_gap_chains") or 0),
        "n_dropped_isolates": int(after.metadata.get("n_dropped_isolates") or 0),
        "n_dropped_inconsistent_dt1_edges": int(after.metadata.get("n_dropped_inconsistent_dt1_edges") or 0),
        "n_dropped_non_forward_nn_edges": int(after.metadata.get("n_dropped_non_forward_nn_edges") or 0),
        "n_dropped_long_dt1_edges": int(after.metadata.get("n_dropped_long_dt1_edges") or 0),
        "n_nodes_before_gacf": int(after.metadata.get("n_nodes_before_gacf") or after.n_nodes),
        "n_edges_before_gacf": int(after.metadata.get("n_edges_before_gacf") or after.n_edges),
        "max_rel_appearance": float(after.metadata.get("max_rel_appearance") or MAX_REL_APPEARANCE),
        "overlay": "learned_unet_orgdet_fnn_acf_gacf_v1",
        "config_hash": LOCKED_ORGDET_FNN_ACF_GACF_HASH,
    }


def refuse_existing_holdout_score(path: str | Path) -> None:
    dest = Path(path)
    if not dest.exists():
        return
    payload = json.loads(dest.read_text())
    if payload.get("status") == "holdout_scored" and payload.get("learned_orgdet_fnn_acf_gacf_combined") is not None:
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


def tuning_decision(learned_combined: float, n_dropped_edges: int) -> str:
    if float(learned_combined) < float(TUNE_KEEP_THRESHOLD):
        return "reject_learned_unet_orgdet_fnn_acf_gacf_v1"
    if int(n_dropped_edges) == 0:
        return "keep_as_noop_learned_unet_orgdet_fnn_acf_gacf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_gacf_v1"


def holdout_decision(learned_combined: float) -> str:
    if float(learned_combined) <= float(ORGDET_FNN_ACF_LOCKED_HOLDOUT):
        return "reject_learned_unet_orgdet_fnn_acf_gacf_v1"
    return "keep_candidate_learned_unet_orgdet_fnn_acf_gacf_v1"


def runtime_estimate(
    n_volumes: int = 2,
    overlay_sec_per_volume: float = 8.0,
    gacf_overhead: float = 4.0,
    limit_hours: float = 12.0,
    peak_memory_mb: float = 1100.0,
) -> dict[str, Any]:
    n = int(n_volumes)
    sec = float(n) * (float(overlay_sec_per_volume) + float(gacf_overhead))
    return {
        "n_volumes": n,
        "overlay_sec_per_volume": float(overlay_sec_per_volume),
        "gacf_overhead": float(gacf_overhead),
        "estimate_sec": sec,
        "estimate_hours": sec / 3600.0,
        "kaggle_gpu_limit_hours": float(limit_hours),
        "under_limit": sec < float(limit_hours) * 3600.0,
        "peak_memory_mb_budget": float(peak_memory_mb),
        "source": (
            "CPU locked orgdet+gap+prune+ELF+FNN+ACF then drop gap-close "
            "chains whose real-endpoint 3x3x3-mean relative appearance "
            f"exceeds {MAX_REL_APPEARANCE} and new degree-0 leftovers; "
            "two sequence Zarrs for the locked ACF parent; no transformer; "
            "no ILP; no new UNet"
        ),
        "prefers_tuning_zarrs": True,
        "is_kaggle_run": False,
        "is_leaderboard_score": False,
    }
