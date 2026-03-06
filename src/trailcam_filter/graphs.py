from __future__ import annotations

import csv
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable


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


def _hour_key(timestamp: str, overlay_time: str) -> str:
    stamp = (timestamp or "").strip()
    if len(stamp) >= 13 and "T" in stamp:
        return stamp[11:13]
    time_value = (overlay_time or "").strip()
    if len(time_value) >= 2:
        return time_value[:2]
    return "unknown"


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


def _render_png_charts(graphs_dir: Path, observations: Iterable[dict[str, str]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    species_temp: dict[str, list[float]] = {}
    hourly_counts: Counter[str] = Counter()
    for row in observations:
        species = row.get("species_label", "unknown")
        temp = _safe_float(row.get("temperature_c"))
        if temp is not None:
            species_temp.setdefault(species, []).append(temp)
        hour = _hour_key(row.get("timestamp", ""), row.get("overlay_time", ""))
        if hour != "unknown":
            hourly_counts[hour] += 1

    if species_temp:
        species_names = sorted(species_temp)
        avg_temps = [sum(species_temp[s]) / len(species_temp[s]) for s in species_names]
        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.bar(species_names, avg_temps, color="#2a9d8f")
        ax.set_title("Average Temperature by Species")
        ax.set_ylabel("Temperature (C)")
        ax.set_xlabel("Species")
        ax.tick_params(axis="x", rotation=35)
        for bar, temp in zip(bars, avg_temps):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{temp:.1f}", ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(graphs_dir / "temperature_by_species.png", dpi=180)
        plt.close(fig)

    if hourly_counts:
        hours = [f"{h:02d}" for h in range(24)]
        counts = [hourly_counts.get(hour, 0) for hour in hours]
        plt.style.use("seaborn-v0_8-darkgrid")
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(hours, counts, marker="o", linewidth=2.5, color="#e76f51")
        ax.set_title("Animal Activity by Hour")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Observations")
        fig.tight_layout()
        fig.savefig(graphs_dir / "activity_by_hour.png", dpi=180)
        plt.close(fig)


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
                "hour",
                "camera_id",
                "species_label",
                "confidence",
                "temperature_c",
                "temperature_f",
                "image_path",
            ]
        )
        for row in observations:
            writer.writerow(
                [
                    row.get("observation_id", ""),
                    _date_key(row.get("timestamp", "")),
                    _hour_key(row.get("timestamp", ""), row.get("overlay_time", "")),
                    row.get("camera_id", ""),
                    row.get("species_label", ""),
                    row.get("confidence", ""),
                    row.get("temperature_c", ""),
                    row.get("temperature_f", ""),
                    row.get("image_path", ""),
                ]
            )

    temp_rows: list[tuple[str, str, float, float | None]] = []
    for row in observations:
        temp_c = _safe_float(row.get("temperature_c"))
        if temp_c is None:
            continue
        temp_rows.append(
            (
                row.get("species_label", "unknown"),
                row.get("camera_id", "unknown"),
                temp_c,
                _safe_float(row.get("temperature_f")),
            )
        )
    with (graphs_dir / "temperature_by_species.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["species_label", "camera_id", "temperature_c", "temperature_f"])
        for species, camera_id, temp_c, temp_f in temp_rows:
            writer.writerow([species, camera_id, f"{temp_c:.3f}", "" if temp_f is None else f"{temp_f:.3f}"])

    hourly_species = Counter(
        (_hour_key(row.get("timestamp", ""), row.get("overlay_time", "")), row.get("species_label", "unknown"))
        for row in observations
    )
    with (graphs_dir / "hourly_activity_by_species.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["hour", "species_label", "observation_count"])
        for (hour, species_label), count in sorted(hourly_species.items()):
            writer.writerow([hour, species_label, count])

    _render_png_charts(graphs_dir, observations)
