from __future__ import annotations

import csv
from pathlib import Path

from trailcam_filter.graphs import generate_graph_datasets
from trailcam_filter.observations import ObservationRow, append_observations, ensure_observations_file


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_observation_append_and_graph_generation(tmp_path: Path) -> None:
    observations_csv = tmp_path / "animal_observations.csv"
    graphs_dir = tmp_path / "graphs"

    ensure_observations_file(observations_csv)
    append_observations(
        observations_csv,
        [
            ObservationRow(
                observation_id="obs-1",
                timestamp="2026-03-06T18:00:00",
                camera_id="cam-a",
                image_path="data/outputs/animals/a.jpg",
                source_path="data/raw/cam-a/a.jpg",
                species_label="deer",
                confidence=0.91,
                run_id="run-1",
                temperature_c="17.4",
                gps_lat="",
                gps_lon="",
            ),
            ObservationRow(
                observation_id="obs-2",
                timestamp="2026-03-06T18:05:00",
                camera_id="cam-b",
                image_path="data/outputs/animals/b.jpg",
                source_path="data/raw/cam-b/b.jpg",
                species_label="deer",
                confidence=0.87,
                run_id="run-1",
                temperature_c="18.1",
                gps_lat="",
                gps_lon="",
            ),
        ],
    )

    generate_graph_datasets(observations_csv, graphs_dir)

    daily = _read_csv(graphs_dir / "daily_counts_by_camera.csv")
    species = _read_csv(graphs_dir / "species_by_camera.csv")
    cooccurrence = _read_csv(graphs_dir / "camera_cooccurrence.csv")
    points = _read_csv(graphs_dir / "observation_points.csv")

    assert daily[0] == ["date", "camera_id", "observation_count"]
    assert ["2026-03-06", "cam-a", "1"] in daily
    assert ["2026-03-06", "cam-b", "1"] in daily

    assert species[0] == ["camera_id", "species_label", "observation_count"]
    assert ["cam-a", "deer", "1"] in species
    assert ["cam-b", "deer", "1"] in species

    assert cooccurrence[0] == ["camera_a", "camera_b", "overlap_count"]
    assert ["cam-a", "cam-b", "1"] in cooccurrence

    assert points[0][-1] == "image_path"
    assert any("data/outputs/animals/a.jpg" in row for row in points[1:])
