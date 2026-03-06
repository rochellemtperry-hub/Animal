# Trailcam Animal Filter

Local image-sorting CLI that runs YOLOv5 inference to split trail camera photos into:
- images with animals
- images without animals

Everything runs locally.

## What It Produces

Given an input folder, the tool writes:
- `data/outputs/animals/`
- `data/outputs/non_animals/`
- `data/reports/detections.csv`
- `data/observations/animal_observations.csv` (append-only canonical observations)
- `data/graphs/*.csv` (graph-ready aggregate datasets)

CSV columns:
- `source_path`
- `output_path`
- `has_animal`
- `top_animal_confidence`
- `total_detections`
- `animal_detections`

## Requirements

- Python 3.10+ (project currently tested with Python 3.12)
- PyTorch installed
- YOLOv5 code available at `vendor/yolov5`
- A YOLOv5-compatible model weights file (default: `models/weights/megadetector.pt`)

This repo imports YOLOv5 internals directly (`models.*`, `utils.*`), so include `vendor/yolov5` on `PYTHONPATH`.

## Model Setup

Default model path used by the CLI:
- `models/weights/megadetector.pt`

This repository currently already includes that file. Verify before running:

```bash
ls -lh models/weights/megadetector.pt
```

If you want to use a different YOLOv5 `.pt` file:
1. Place it in `models/weights/` (or any path you prefer).
2. Pass it with `--model`.

Example:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --model models/weights/your_model.pt
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
# Install torch + other runtime deps you use in your environment.
```

Example runtime invocation from repo root:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw
```

## Quick Start

1. Put camera images under `data/raw/`.
2. Run:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw
```

3. Check:
- `data/outputs/animals/`
- `data/outputs/non_animals/`
- `data/reports/detections.csv`

## CLI Options

- `--input` input image directory (required)
- `--model` model file path (default: `models/weights/megadetector.pt`)
- `--output-animals` output directory for animal images
- `--output-non-animals` output directory for non-animal images
- `--report-csv` CSV report path
- `--observations-csv` append-only observations CSV path
- `--cameras-csv` camera metadata CSV path
- `--graphs-dir` output directory for graph-ready CSVs
- `--conf-threshold` detection confidence threshold (default: `0.25`)
- `--iou-threshold` NMS IOU threshold (default: `0.45`)
- `--device` inference device (examples: `cpu`, `cuda:0`)
- `--action {copy,move}` copy or move source files
- `--no-recursive` disable recursive image discovery
- `--flat-output` do not preserve input subdirectory layout
- `--dry-run` run inference/reporting without copying or moving files

## CUDA Usage

- Install a CUDA-enabled PyTorch build that matches your NVIDIA driver/runtime.
- If `--device` is omitted, the app auto-selects:
  - `cuda:0` when CUDA is available
  - CPU otherwise
- You can force GPU:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --device cuda:0
```

- You can force CPU:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --device cpu
```

- On startup, the CLI prints `Inference device: ...` so you can verify which device is active.

## Examples

Default recursive run:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw
```

Move files instead of copy:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --action move
```

Non-recursive run:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --no-recursive
```

Dry-run (no file operations):

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --dry-run
```

## Observation + Graph Pipeline

- Animal detections are appended to `data/observations/animal_observations.csv`.
- Metadata extraction attempts to capture:
  - `timestamp` (EXIF date/time)
  - `temperature_c` (when embedded in EXIF description/comments)
  - `gps_lat`, `gps_lon`
- Camera ID is inferred from the first folder under input root.
  - Example: `data/raw/cam_01/image.jpg` -> `camera_id=cam_01`
- Graph-ready datasets are regenerated each run in `data/graphs/`:
  - `daily_counts_by_camera.csv`
  - `species_by_camera.csv`
  - `camera_cooccurrence.csv`
  - `observation_points.csv` (includes image paths for linking back to files)

## Supported Image Types

The scanner currently includes:
- `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff`, `.webp`

## Troubleshooting

- `ModuleNotFoundError: No module named 'trailcam_filter'`
  - Run with `PYTHONPATH=src` (or install package metadata when packaging is added).
- YOLOv5 import errors (`models.common`, `utils.general`, etc.)
  - Add `vendor/yolov5` to `PYTHONPATH`.
- CLI says CPU when you expected CUDA
  - Confirm CUDA-enabled torch is installed and `torch.cuda.is_available()` is `True` in your environment.
