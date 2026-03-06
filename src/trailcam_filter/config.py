from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    input_dir: Path
    output_animals_dir: Path
    output_non_animals_dir: Path
    report_csv: Path
    observations_csv: Path
    cameras_csv: Path
    graphs_dir: Path
    model_path: Path
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    recursive: bool = True
    action: str = "copy"
    keep_structure: bool = True
    device: str | None = None
    batch_size: int = 16
    dry_run: bool = False
