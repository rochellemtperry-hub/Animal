from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations
from pathlib import Path


def _load_observations(observations_csv: Path) -> list[dict[str, str]]:
    if not observations_csv.exists():
        return []
    with observations_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _date_key(timestamp: str) -> str:
    value = (timestamp or "").strip()
    if not value:
        return "unknown"
    return value[:10]


def generate_graph_datasets(observations_csv: Path, graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    observations = _load_observations(observations_csv)

    daily_counts = Counter(
        (_date_key(row.get("timestamp", "")), row.get("camera_id", "unknown"))
        for row in observations
    )
    with (graphs_dir / "daily_counts_by_camera.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "camera_id", "observation_count"])
        for (date, camera_id), count in sorted(daily_counts.items()):
            writer.writerow([date, camera_id, count])

    species_counts = Counter(
        (row.get("camera_id", "unknown"), row.get("species_label", "unknown"))
        for row in observations
    )
    with (graphs_dir / "species_by_camera.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera_id", "species_label", "observation_count"])
        for (camera_id, species_label), count in sorted(species_counts.items()):
            writer.writerow([camera_id, species_label, count])

    camera_presence: dict[tuple[str, str], set[str]] = {}
    for row in observations:
        key = (_date_key(row.get("timestamp", "")), row.get("species_label", "unknown"))
        camera_presence.setdefault(key, set()).add(row.get("camera_id", "unknown"))

    pair_counts = Counter()
    for _, cameras in camera_presence.items():
        for pair in combinations(sorted(cameras), 2):
            pair_counts[pair] += 1
    with (graphs_dir / "camera_cooccurrence.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera_a", "camera_b", "overlap_count"])
        for (camera_a, camera_b), count in sorted(pair_counts.items()):
            writer.writerow([camera_a, camera_b, count])

    with (graphs_dir / "observation_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "observation_id",
                "date",
                "camera_id",
                "species_label",
                "confidence",
                "temperature_c",
                "image_path",
            ]
        )
        for row in observations:
            writer.writerow(
                [
                    row.get("observation_id", ""),
                    _date_key(row.get("timestamp", "")),
                    row.get("camera_id", ""),
                    row.get("species_label", ""),
                    row.get("confidence", ""),
                    row.get("temperature_c", ""),
                    row.get("image_path", ""),
                ]
            )
