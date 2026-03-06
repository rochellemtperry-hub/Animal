from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def _load_observations(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _parse_int(value: str | None, default: int = 1) -> int:
    if value is None:
        return default
    text = value.strip()
    if not text:
        return default
    try:
        return max(1, int(float(text)))
    except ValueError:
        return default


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_timestamp(row: dict[str, str]) -> datetime | None:
    raw = (row.get("photo_datetime") or "").strip()
    if raw:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw[:19], fmt)
            except ValueError:
                continue
    date = (row.get("date") or "").strip()
    time = (row.get("time") or "").strip() or "00:00:00"
    if date:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y/%m/%d %H:%M"):
            try:
                return datetime.strptime(f"{date} {time}", fmt)
            except ValueError:
                continue
    return None


def _season(month: int) -> str:
    if month in (12, 1, 2):
        return "Summer"
    if month in (3, 4, 5):
        return "Autumn"
    if month in (6, 7, 8):
        return "Winter"
    return "Spring"


def _temperature_band(temp_c: float | None) -> str:
    if temp_c is None:
        return "unknown"
    if temp_c < 10:
        return "<10"
    if temp_c < 15:
        return "10-15"
    if temp_c < 20:
        return "15-20"
    if temp_c < 25:
        return "20-25"
    if temp_c < 30:
        return "25-30"
    return "30+"


def _species_value(row: dict[str, str]) -> str:
    species = (row.get("species") or "").strip().lower()
    if species:
        return species
    return "unknown"


def _write_counter(path: Path, headers: list[str], items: list[tuple[tuple[str, ...], int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for key, count in items:
            writer.writerow([*key, count])


def generate_wildlife_analytics(observations_csv: Path, graphs_dir: Path) -> None:
    graphs_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_observations(observations_csv)

    by_month: Counter[tuple[str, str, str]] = Counter()
    by_hour: Counter[tuple[str, str]] = Counter()
    by_camera: Counter[tuple[str, str, str]] = Counter()
    by_temp: Counter[tuple[str, str]] = Counter()
    by_season: Counter[tuple[str, str]] = Counter()
    reviewed_counter: Counter[str] = Counter()
    species_camera_presence: defaultdict[str, set[str]] = defaultdict(set)

    total_rows = 0
    for row in rows:
        if (row.get("animal_detected") or "yes").strip().lower() in {"no", "false", "0"}:
            continue

        total_rows += 1
        species = _species_value(row)
        count = _parse_int(row.get("count"), default=1)
        camera_id = (row.get("camera_id") or "unknown").strip() or "unknown"
        location_name = (row.get("location_name") or "").strip()
        ts = _parse_timestamp(row)
        temp_c = _parse_float(row.get("temperature_c"))

        year = str(ts.year) if ts else "unknown"
        month = f"{ts.month:02d}" if ts else "unknown"
        hour = f"{ts.hour:02d}" if ts else "unknown"
        season = _season(ts.month) if ts else "unknown"
        temp_band = _temperature_band(temp_c)

        by_month[(species, year, month)] += count
        by_hour[(species, hour)] += count
        by_camera[(species, camera_id, location_name)] += count
        by_temp[(species, temp_band)] += count
        by_season[(species, season)] += count
        reviewed_counter[(row.get("review_status") or "pending").strip().lower() or "pending"] += 1
        species_camera_presence[species].add(camera_id)

    _write_counter(
        graphs_dir / "species_by_month.csv",
        ["species", "year", "month", "sighting_count"],
        sorted(by_month.items()),
    )
    _write_counter(
        graphs_dir / "species_by_hour.csv",
        ["species", "hour", "sighting_count"],
        sorted(by_hour.items()),
    )
    _write_counter(
        graphs_dir / "species_by_camera.csv",
        ["species", "camera_id", "location_name", "sighting_count"],
        sorted(by_camera.items()),
    )
    _write_counter(
        graphs_dir / "species_by_temperature.csv",
        ["species", "temperature_band", "sighting_count"],
        sorted(by_temp.items()),
    )
    _write_counter(
        graphs_dir / "species_by_season.csv",
        ["species", "season", "sighting_count"],
        sorted(by_season.items()),
    )

    with (graphs_dir / "activity_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerow(["total_rows", total_rows])
        writer.writerow(["unique_species", len(species_camera_presence)])
        writer.writerow(["reviewed_rows", reviewed_counter.get("reviewed", 0)])
        writer.writerow(["pending_rows", reviewed_counter.get("pending", 0)])

    all_cameras = sorted({camera for cameras in species_camera_presence.values() for camera in cameras})
    with (graphs_dir / "species_presence_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species", *all_cameras])
        for species in sorted(species_camera_presence):
            present = species_camera_presence[species]
            writer.writerow([species, *["1" if cam in present else "0" for cam in all_cameras]])
