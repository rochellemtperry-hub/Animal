from __future__ import annotations

import csv
from pathlib import Path

from trailcam_filter.graphs import generate_graph_datasets
from trailcam_filter.observations import ObservationRecord, upsert_and_save


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def test_observation_upsert_and_graph_generation(tmp_path: Path) -> None:
    observations_csv = tmp_path / "animal_observations.csv"
    graphs_dir = tmp_path / "graphs"

    records = [
        ObservationRecord(
            observation_id="",
            file_name="a.jpg",
            camera_id="cam-a",
            location_name="north",
            photo_datetime="2026-03-06T18:00:00",
            date="2026-03-06",
            time="18:00:00",
            temperature_c="17.4",
            animal_detected="yes",
            species="deer",
            species_confidence="0.91",
            count="1",
            image_path="data/outputs/animals/a.jpg",
            source_path="data/raw/cam-a/a.jpg",
        ),
        ObservationRecord(
            observation_id="",
            file_name="b.jpg",
            camera_id="cam-b",
            location_name="south",
            photo_datetime="2026-03-06T18:05:00",
            date="2026-03-06",
            time="18:05:00",
            temperature_c="18.1",
            animal_detected="yes",
            species="deer",
            species_confidence="0.87",
            count="2",
            image_path="data/outputs/animals/b.jpg",
            source_path="data/raw/cam-b/b.jpg",
        ),
    ]
    upsert_and_save(observations_csv, records)

    generate_graph_datasets(observations_csv, graphs_dir)

    daily = _read_csv(graphs_dir / "daily_counts_by_camera.csv")
    species = _read_csv(graphs_dir / "species_by_camera.csv")
    points = _read_csv(graphs_dir / "observation_points.csv")
    temp_species = _read_csv(graphs_dir / "temperature_by_species.csv")
    hourly = _read_csv(graphs_dir / "hourly_activity_by_species.csv")

    assert daily[0] == ["date", "camera_id", "observation_count"]
    assert ["2026-03-06", "cam-a", "1"] in daily
    assert ["2026-03-06", "cam-b", "1"] in daily

    assert species[0] == ["camera_id", "species", "observation_count"]
    assert ["cam-a", "deer", "1"] in species
    assert ["cam-b", "deer", "1"] in species

    assert points[0] == ["observation_id", "date", "time", "camera_id", "species", "temperature_c", "image_path"]
    assert any("data/outputs/animals/a.jpg" in row for row in points[1:])

    assert temp_species[0] == ["species", "camera_id", "temperature_c"]
    assert ["deer", "cam-a", "17.400"] in temp_species
    assert ["18", "deer", "2"] in hourly
