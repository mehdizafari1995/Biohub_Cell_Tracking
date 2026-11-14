"""Independent Biohub cell-tracking pipeline.

Heavy tracking imports stay lazy so helper scripts (Kaggle poll/collect)
can run on a stdlib-only interpreter without scipy.
"""

from typing import Any

__all__ = ["PipelineConfig", "TrackGraph", "load_config", "track_volume"]
__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    if name in {"PipelineConfig", "load_config"}:
        from .config import PipelineConfig, load_config

        return PipelineConfig if name == "PipelineConfig" else load_config
    if name == "TrackGraph":
        from .graphs import TrackGraph

        return TrackGraph
    if name == "track_volume":
        from .track import track_volume

        return track_volume
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
