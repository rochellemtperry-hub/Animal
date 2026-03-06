from __future__ import annotations

from pathlib import Path

from trailcam_filter.analytics import generate_wildlife_analytics


def generate_graph_datasets(observations_csv: Path, graphs_dir: Path) -> None:
    generate_wildlife_analytics(observations_csv=observations_csv, graphs_dir=graphs_dir)
