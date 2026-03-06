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

    by_month = _read_csv(graphs_dir / "species_by_month.csv")
    by_hour = _read_csv(graphs_dir / "species_by_hour.csv")
    by_camera = _read_csv(graphs_dir / "species_by_camera.csv")
    by_temp = _read_csv(graphs_dir / "species_by_temperature.csv")
    by_season = _read_csv(graphs_dir / "species_by_season.csv")
    summary = _read_csv(graphs_dir / "activity_summary.csv")
    matrix = _read_csv(graphs_dir / "species_presence_matrix.csv")

    assert by_month[0] == ["species", "year", "month", "sighting_count"]
    assert ["deer", "2026", "03", "3"] in by_month

    assert by_hour[0] == ["species", "hour", "sighting_count"]
    assert ["deer", "18", "3"] in by_hour

    assert by_camera[0] == ["species", "camera_id", "location_name", "sighting_count"]
    assert ["deer", "cam-a", "north", "1"] in by_camera
    assert ["deer", "cam-b", "south", "2"] in by_camera

    assert by_temp[0] == ["species", "temperature_band", "sighting_count"]
    assert ["deer", "15-20", "3"] in by_temp

    assert by_season[0] == ["species", "season", "sighting_count"]
    assert ["deer", "Autumn", "3"] in by_season

    assert summary[0] == ["metric", "value"]
    assert ["total_rows", "2"] in summary
    assert matrix[0] == ["species", "cam-a", "cam-b"]
    assert ["deer", "1", "1"] in matrix
