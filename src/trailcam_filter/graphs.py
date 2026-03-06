from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


def _load_observations(observations_csv: Path) -> list[dict[str, str]]:
    if not observations_csv.exists():
        return []
    with observations_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def generate_graph_datasets(observations_csv: Path, graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    observations = _load_observations(observations_csv)

    daily_counts = Counter((r.get("date", "") or "unknown", r.get("camera_id", "unknown")) for r in observations)
    with (graphs_dir / "daily_counts_by_camera.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "camera_id", "observation_count"])
        for (date, camera_id), count in sorted(daily_counts.items()):
            writer.writerow([date, camera_id, count])

    species_counts = Counter((r.get("camera_id", "unknown"), r.get("species", "unknown")) for r in observations)
    with (graphs_dir / "species_by_camera.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["camera_id", "species", "observation_count"])
        for (camera_id, species), count in sorted(species_counts.items()):
            writer.writerow([camera_id, species, count])

    with (graphs_dir / "observation_points.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["observation_id", "date", "time", "camera_id", "species", "temperature_c", "image_path"])
        for row in observations:
            writer.writerow(
                [
                    row.get("observation_id", ""),
                    row.get("date", ""),
                    row.get("time", ""),
                    row.get("camera_id", ""),
                    row.get("species", ""),
                    row.get("temperature_c", ""),
                    row.get("image_path", ""),
                ]
            )

    temp_rows = []
    for row in observations:
        temp_c = _safe_float(row.get("temperature_c"))
        if temp_c is None:
            continue
        temp_rows.append((row.get("species", "unknown"), row.get("camera_id", "unknown"), temp_c))
    with (graphs_dir / "temperature_by_species.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", "camera_id", "temperature_c"])
        for species, camera_id, temp_c in temp_rows:
            writer.writerow([species, camera_id, f"{temp_c:.3f}"])

    hourly_counts = Counter((r.get("time", "")[:2] if r.get("time", "") else "unknown", r.get("species", "unknown")) for r in observations)
    with (graphs_dir / "hourly_activity_by_species.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hour", "species", "observation_count"])
        for (hour, species), count in sorted(hourly_counts.items()):
            writer.writerow([hour, species, count])
