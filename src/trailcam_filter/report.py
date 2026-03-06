from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ReportRow:
    run_id: str
    camera_id: str
    source_path: str
    output_path: str
    has_animal: bool
    top_animal_confidence: float
    total_detections: int
    animal_detections: int


def write_report(rows: list[ReportRow], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "source_path",
                "output_path",
                "run_id",
                "camera_id",
                "has_animal",
                "top_animal_confidence",
                "total_detections",
                "animal_detections",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.source_path,
                    row.output_path,
                    row.run_id,
                    row.camera_id,
                    row.has_animal,
                    f"{row.top_animal_confidence:.6f}",
                    row.total_detections,
                    row.animal_detections,
                ]
            )
