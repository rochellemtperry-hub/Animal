from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import uuid


OBSERVATION_HEADERS = [
    "observation_id",
    "timestamp",
    "camera_id",
    "image_path",
    "source_path",
    "species_label",
    "confidence",
    "run_id",
    "temperature_c",
    "gps_lat",
    "gps_lon",
]

CAMERA_HEADERS = [
    "camera_id",
    "camera_name",
    "location_lat",
    "location_lon",
    "timezone",
    "active",
]


@dataclass(slots=True)
class ObservationRow:
    observation_id: str
    timestamp: str
    camera_id: str
    image_path: str
    source_path: str
    species_label: str
    confidence: float
    run_id: str
    temperature_c: str
    gps_lat: str
    gps_lon: str


def new_run_id() -> str:
    return uuid.uuid4().hex


def ensure_camera_metadata_file(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CAMERA_HEADERS)


def ensure_observations_file(path: Path) -> None:
    if path.exists():
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            existing_rows = list(reader)
        if not existing_rows:
            pass
        elif existing_rows[0] == OBSERVATION_HEADERS:
            return
        else:
            legacy_header = existing_rows[0]
            migrated: list[dict[str, str]] = []
            for legacy_row in existing_rows[1:]:
                if not any(cell.strip() for cell in legacy_row):
                    continue
                row_map = {
                    col: legacy_row[i] if i < len(legacy_row) else ""
                    for i, col in enumerate(legacy_header)
                }
                migrated.append(
                    {
                        "observation_id": row_map.get("observation_id", ""),
                        "timestamp": row_map.get("timestamp", ""),
                        "camera_id": row_map.get("camera_id", ""),
                        "image_path": row_map.get("image_path", ""),
                        "source_path": row_map.get("source_path", ""),
                        "species_label": row_map.get("species_label", ""),
                        "confidence": row_map.get("confidence", ""),
                        "run_id": row_map.get("run_id", ""),
                        "temperature_c": row_map.get("temperature_c", ""),
                        "gps_lat": row_map.get("gps_lat", ""),
                        "gps_lon": row_map.get("gps_lon", ""),
                    }
                )
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(OBSERVATION_HEADERS)
                for row in migrated:
                    writer.writerow([row[col] for col in OBSERVATION_HEADERS])
            return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(OBSERVATION_HEADERS)


def load_known_camera_ids(path: Path) -> set[str]:
    ensure_camera_metadata_file(path)
    ids: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            camera_id = (row.get("camera_id") or "").strip()
            if camera_id:
                ids.add(camera_id)
    return ids


def camera_id_for_image(image_path: Path, input_root: Path) -> str:
    try:
        relative = image_path.relative_to(input_root)
    except ValueError:
        return "unknown"
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return "unknown"


def append_observations(path: Path, rows: list[ObservationRow]) -> None:
    if not rows:
        return
    ensure_observations_file(path)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(
                [
                    row.observation_id,
                    row.timestamp,
                    row.camera_id,
                    row.image_path,
                    row.source_path,
                    row.species_label,
                    f"{row.confidence:.6f}",
                    row.run_id,
                    row.temperature_c,
                    row.gps_lat,
                    row.gps_lon,
                ]
            )
