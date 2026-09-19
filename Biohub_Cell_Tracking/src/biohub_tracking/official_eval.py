from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl

from .config import DEFAULT_MATCH_DISTANCE_UM, DEFAULT_SCALE_ZYX
from .evaluate import evaluate_graph
from .graphs import TrackGraph

OFFICIAL_REPO = "https://github.com/royerlab/kaggle-cell-tracking-competition"
OFFICIAL_COMMIT = "075fc5f5a52d11077f9dc2b074644618f26939e2"
OFFICIAL_TRACKSDATA_REPO = "https://github.com/royerlab/tracksdata"
OFFICIAL_TRACKSDATA_COMMIT = "7523bb39ed9ab1bf88512911fd6c69e46fcad188"
OFFICIAL_SRC = Path(__file__).resolve().parents[2] / "third_party" / "kaggle-cell-tracking-competition" / "src"


def official_dependency_versions() -> dict[str, str]:
    import importlib.metadata as md

    versions: dict[str, str] = {
        "official_repo": OFFICIAL_REPO,
        "official_commit": OFFICIAL_COMMIT,
        "tracksdata_repo": OFFICIAL_TRACKSDATA_REPO,
        "tracksdata_commit": OFFICIAL_TRACKSDATA_COMMIT,
    }
    for name in ("tracksdata", "polars", "geff", "numpy", "scipy"):
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = "missing"
    return versions


def official_scorer_available() -> bool:
    return (OFFICIAL_SRC / "tracking_cellmot" / "metrics.py").exists()


def _ensure_official_path() -> None:
    src = str(OFFICIAL_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def load_tracksdata_geff(geff_path: str | Path):
    import tracksdata as td

    result = td.graph.IndexedRXGraph.from_geff(Path(geff_path))
    return result[0] if isinstance(result, tuple) else result


def trackgraph_to_tracksdata_mapped(graph: TrackGraph):
    """Convert a TrackGraph and return (tracksdata graph, original_id -> td_id)."""
    import tracksdata as td

    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, 0.0)
    rows = [
        {"t": int(t), "z": float(z), "y": float(y), "x": float(x)}
        for t, z, y, x in zip(graph.t, graph.z, graph.y, graph.x, strict=True)
    ]
    assigned = g.bulk_add_nodes(rows) if rows else []
    id_map = dict(zip((int(i) for i in graph.node_id.tolist()), assigned, strict=True))
    if graph.n_edges:
        g.bulk_add_edges(
            [{"source_id": id_map[int(s)], "target_id": id_map[int(d)]} for s, d in graph.edges]
        )
    return g, id_map


def trackgraph_to_tracksdata(graph: TrackGraph):
    g, _ = trackgraph_to_tracksdata_mapped(graph)
    return g


def estimated_n_nodes(geff_path: str | Path) -> float:
    from geff import GeffMetadata

    meta = GeffMetadata.read(Path(geff_path))
    val = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


@dataclass
class OfficialSampleResult:
    dataset: str
    edge_tp: int
    edge_fp: int
    edge_fn: int
    division_tp: int
    division_fp: int
    division_fn: int
    num_pred_nodes: int
    node_recall: float
    edge_jaccard: float
    adj_edge_jaccard: float
    division_jaccard: float
    score: float
    total_node_ratio: float
    estimated_n_nodes: float
    scorer: str
    official_commit: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_official(
    pred: TrackGraph,
    gt_geff: str | Path | TrackGraph,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance_um: float = DEFAULT_MATCH_DISTANCE_UM,
) -> OfficialSampleResult:
    """Score with the organizer tracking_cellmot.metrics implementation."""
    if not official_scorer_available():
        raise RuntimeError("Official scorer source is missing under third_party/")
    _ensure_official_path()
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics

    pred_graph = trackgraph_to_tracksdata(pred)
    if isinstance(gt_geff, TrackGraph):
        gt_graph = trackgraph_to_tracksdata(gt_geff)
        n_total = float(gt_geff.estimated_n_nodes) if gt_geff.estimated_n_nodes is not None else float("nan")
        name = pred.dataset
    else:
        gt_graph = load_tracksdata_geff(gt_geff)
        n_total = estimated_n_nodes(gt_geff)
        name = Path(gt_geff).name.removesuffix(".geff")

    er = evaluate(pred_graph, gt_graph, scale=scale_zyx, max_distance=max_distance_um)
    recall = (
        node_recall(pred_graph, gt_graph)
        if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0
        else 0.0
    )
    row = per_sample_metrics(er, n_total, recall)
    div_j = _safe_div_jaccard(er.division_tp, er.division_fp, er.division_fn)
    adj = row["adj_edge_jaccard"]
    if adj == adj and div_j == div_j:
        score = adj + 0.1 * div_j
    else:
        score = adj
    return OfficialSampleResult(
        dataset=name,
        edge_tp=int(er.edge_tp),
        edge_fp=int(er.edge_fp),
        edge_fn=int(er.edge_fn),
        division_tp=int(er.division_tp),
        division_fp=int(er.division_fp),
        division_fn=int(er.division_fn),
        num_pred_nodes=int(er.num_pred_nodes),
        node_recall=float(recall),
        edge_jaccard=float(row["edge_jaccard"]) if row["edge_jaccard"] == row["edge_jaccard"] else float("nan"),
        adj_edge_jaccard=float(adj) if adj == adj else float("nan"),
        division_jaccard=float(div_j) if div_j == div_j else float("nan"),
        score=float(score) if score == score else float("nan"),
        total_node_ratio=float(row["total_node_ratio"]) if row["total_node_ratio"] == row["total_node_ratio"] else float("nan"),
        estimated_n_nodes=n_total,
        scorer="royerlab/tracking_cellmot.metrics",
        official_commit=OFFICIAL_COMMIT,
    )


def compare_official_and_diagnostic(
    pred: TrackGraph,
    gt: TrackGraph,
    gt_geff: str | Path | None = None,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
) -> dict[str, Any]:
    diagnostic = evaluate_graph(pred, gt, scale_zyx=scale_zyx)
    official = evaluate_official(pred, gt_geff if gt_geff is not None else gt, scale_zyx=scale_zyx)
    return {
        "dataset": pred.dataset,
        "official": official.as_dict(),
        "diagnostic": diagnostic.as_dict(),
        "score_gap": _nan_sub(official.score, diagnostic.score),
        "edge_jaccard_gap": _nan_sub(official.edge_jaccard, diagnostic.edge_jaccard),
        "adj_gap": _nan_sub(official.adj_edge_jaccard, diagnostic.adj_edge_jaccard),
        "division_gap": _nan_sub(official.division_jaccard, diagnostic.division_jaccard),
    }


def summarise_official(rows: list[OfficialSampleResult]) -> dict[str, Any]:
    _ensure_official_path()
    from tracking_cellmot.metrics import summarise

    payload = [
        {
            "edge_tp": r.edge_tp,
            "edge_fp": r.edge_fp,
            "edge_fn": r.edge_fn,
            "division_tp": r.division_tp,
            "division_fp": r.division_fp,
            "division_fn": r.division_fn,
            "num_pred_nodes": r.num_pred_nodes,
            "node_recall": r.node_recall,
            "total_node_ratio": r.total_node_ratio,
            "edge_jaccard": r.edge_jaccard,
            "adj_edge_jaccard": r.adj_edge_jaccard,
        }
        for r in rows
    ]
    summary = summarise(payload)
    summary["scorer"] = "royerlab/tracking_cellmot.metrics"
    summary["official_commit"] = OFFICIAL_COMMIT
    summary["source_url"] = OFFICIAL_REPO
    return summary


def score_official_division_events(
    pred: TrackGraph,
    gt: TrackGraph,
    scale_zyx: tuple[float, float, float] = DEFAULT_SCALE_ZYX,
    max_distance_um: float = DEFAULT_MATCH_DISTANCE_UM,
) -> dict[str, Any]:
    """Score each annotated GT division with the organizer division metric.

    Uses ``tracking_cellmot.division_metrics.score_divisions`` (local
    grandparent → parent → children → grandchildren window). Returns original
    TrackGraph node IDs, not the remapped tracksdata IDs.
    """
    if not official_scorer_available():
        raise RuntimeError("Official scorer source is missing under third_party/")
    _ensure_official_path()
    from tracking_cellmot.division_metrics import score_divisions

    pred_td, pred_map = trackgraph_to_tracksdata_mapped(pred)
    gt_td, gt_map = trackgraph_to_tracksdata_mapped(gt)
    inv_gt = {td_id: orig for orig, td_id in gt_map.items()}
    inv_pred = {td_id: orig for orig, td_id in pred_map.items()}
    scored = score_divisions(pred_td, gt_td, scale=scale_zyx, max_distance=max_distance_um)
    events = []
    for td_id, recovered in scored.scores.items():
        events.append(
            {
                "dataset": gt.dataset,
                "parent_id": int(inv_gt.get(td_id, td_id)),
                "official_recovered": bool(recovered),
                "scorer": "royerlab/tracking_cellmot.division_metrics.score_divisions",
                "official_commit": OFFICIAL_COMMIT,
            }
        )
    events.sort(key=lambda row: (row["dataset"], row["parent_id"]))
    tp = int(sum(1 for row in events if row["official_recovered"]))
    fn = int(len(events) - tp)
    fp_forks = [int(inv_pred.get(i, i)) for i in scored.fp_forks]
    tp_forks = [int(inv_pred.get(i, i)) for i in scored.tp_forks]
    return {
        "dataset": gt.dataset,
        "events": events,
        "division_tp": tp,
        "division_fn": fn,
        "division_fp": int(len(fp_forks)),
        "tp_forks": sorted(tp_forks),
        "fp_forks": sorted(fp_forks),
        "scorer": "royerlab/tracking_cellmot.division_metrics.score_divisions",
        "official_commit": OFFICIAL_COMMIT,
    }


def _safe_div_jaccard(tp: int, fp: int, fn: int) -> float:
    denom = tp + fp + fn
    return tp / denom if denom > 0 else float("nan")


def _nan_sub(a: float, b: float) -> float:
    if a != a or b != b:
        return float("nan")
    return a - b
