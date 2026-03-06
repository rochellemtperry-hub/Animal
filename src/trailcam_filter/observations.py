from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


OBSERVATION_HEADERS = [
    "observation_id",
    "file_name",
    "camera_id",
    "location_name",
    "photo_datetime",
    "date",
    "time",
    "temperature_c",
    "animal_detected",
    "species",
    "species_confidence",
    "count",
    "reviewed",
    "review_status",
    "ai_corrected",
    "notes",
    "image_path",
    "source_path",
]

CAMERA_HEADERS = [
    "camera_id",
    "location_name",
    "camera_name",
    "location_lat",
    "location_lon",
    "timezone",
    "active",
]

AI_MANAGED_FIELDS = {
    "file_name",
    "camera_id",
    "location_name",
    "photo_datetime",
    "date",
    "time",
    "temperature_c",
    "animal_detected",
    "species",
    "species_confidence",
    "count",
    "image_path",
    "source_path",
}


@dataclass(slots=True)
class ObservationRecord:
    observation_id: str
    file_name: str
    camera_id: str
    location_name: str
    photo_datetime: str
    date: str
    time: str
    temperature_c: str
    animal_detected: str
    species: str
    species_confidence: str
    count: str
    reviewed: str = "no"
    review_status: str = "pending"
    ai_corrected: str = "no"
    notes: str = ""
    image_path: str = ""
    source_path: str = ""

    def to_row(self) -> dict[str, str]:
        return {
            "observation_id": self.observation_id,
            "file_name": self.file_name,
            "camera_id": self.camera_id,
            "location_name": self.location_name,
            "photo_datetime": self.photo_datetime,
            "date": self.date,
            "time": self.time,
            "temperature_c": self.temperature_c,
            "animal_detected": self.animal_detected,
            "species": self.species,
            "species_confidence": self.species_confidence,
            "count": self.count,
            "reviewed": self.reviewed,
            "review_status": self.review_status,
            "ai_corrected": self.ai_corrected,
            "notes": self.notes,
            "image_path": self.image_path,
            "source_path": self.source_path,
        }


def _empty_row() -> dict[str, str]:
    return {key: "" for key in OBSERVATION_HEADERS}


def ensure_camera_metadata_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CAMERA_HEADERS)
        return

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = [dict(row) for row in reader]
        header = reader.fieldnames or []
    if header == CAMERA_HEADERS:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CAMERA_HEADERS)
        writer.writeheader()
        for row in existing:
            writer.writerow({key: (row.get(key) or "") for key in CAMERA_HEADERS})


def ensure_observations_file(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(OBSERVATION_HEADERS)
        return

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        existing = [dict(row) for row in reader]
        header = reader.fieldnames or []

    if header == OBSERVATION_HEADERS:
        return

    migrated: list[dict[str, str]] = []
    for row in existing:
        out = _empty_row()
        for key in OBSERVATION_HEADERS:
            out[key] = (row.get(key) or "").strip()
        if not out["review_status"]:
            out["review_status"] = "pending"
        if not out["reviewed"]:
            out["reviewed"] = "no"
        if not out["ai_corrected"]:
            out["ai_corrected"] = "no"
        migrated.append(out)

    write_observations(path, migrated)


def load_observations(path: Path) -> list[dict[str, str]]:
    ensure_observations_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            normalized = _empty_row()
            for key in OBSERVATION_HEADERS:
                normalized[key] = (row.get(key) or "").strip()
            rows.append(normalized)
        return rows


def write_observations(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OBSERVATION_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in OBSERVATION_HEADERS})


def load_camera_lookup(path: Path) -> dict[str, str]:
    ensure_camera_metadata_file(path)
    lookup: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            camera_id = (row.get("camera_id") or "").strip()
            if not camera_id:
                continue
            location_name = (row.get("location_name") or row.get("camera_name") or "").strip()
            lookup[camera_id] = location_name
    return lookup


def camera_id_for_image(image_path: Path, input_root: Path, default_camera_id: str = "unknown") -> str:
    try:
        relative = image_path.relative_to(input_root)
    except ValueError:
        return default_camera_id
    if len(relative.parts) >= 2:
        return relative.parts[0]
    return default_camera_id


def find_observation_index(rows: list[dict[str, str]], image_path: str, file_name: str, camera_id: str) -> int | None:
    normalized_path = image_path.strip()
    for i, row in enumerate(rows):
        if row.get("image_path", "").strip() == normalized_path:
            return i
    for i, row in enumerate(rows):
        if row.get("file_name", "").strip() == file_name and row.get("camera_id", "").strip() == camera_id:
            return i
    return None


def _next_observation_id(rows: list[dict[str, str]]) -> str:
    year = datetime.now().year
    prefix = f"OBS-{year}-"
    max_index = 0
    for row in rows:
        obs_id = (row.get("observation_id") or "").strip()
        if not obs_id.startswith(prefix):
            continue
        try:
            max_index = max(max_index, int(obs_id.split("-")[-1]))
        except ValueError:
            continue
    return f"{prefix}{max_index + 1:05d}"


def upsert_observation(rows: list[dict[str, str]], record: ObservationRecord) -> None:
    new_row = record.to_row()
    idx = find_observation_index(rows, new_row["image_path"], new_row["file_name"], new_row["camera_id"])
    if idx is None:
        if not new_row["observation_id"]:
            new_row["observation_id"] = _next_observation_id(rows)
        rows.append(new_row)
        return

    existing = rows[idx]
    for field in AI_MANAGED_FIELDS:
        existing[field] = new_row[field]
    if not existing.get("observation_id"):
        existing["observation_id"] = _next_observation_id(rows)
    if not existing.get("review_status"):
        existing["review_status"] = "pending"
    if not existing.get("reviewed"):
        existing["reviewed"] = "no"
    if not existing.get("ai_corrected"):
        existing["ai_corrected"] = "no"


def upsert_and_save(path: Path, records: list[ObservationRecord]) -> int:
    if not records:
        return 0
    rows = load_observations(path)
    for record in records:
        upsert_observation(rows, record)
    write_observations(path, rows)
    return len(records)
