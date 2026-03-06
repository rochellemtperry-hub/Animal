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
- `data/graphs/*.png` (visual charts when plotting deps are installed)

CSV columns:
- `source_path`
- `output_path`
- `has_animal`
- `top_animal_confidence`
- `total_detections`
- `animal_detections`

## Requirements

- Python 3.10+ (project currently tested with Python 3.12)
- Runtime dependencies installed from `requirements.txt`
- YOLOv5 code available at `vendor/yolov5`
- A YOLOv5-compatible model weights file (default: `models/weights/megadetector.pt`)
- For overlay bar extraction: `pytesseract`, `Pillow`, and system `tesseract-ocr`
- For styled chart rendering: `matplotlib`

This repo imports YOLOv5 internals directly (`models.*`, `utils.*`), so include `vendor/yolov5` on `PYTHONPATH`.

## Model Setup

Default model path used by the CLI:
- `models/weights/megadetector.pt`

This repository does not commit model weights. Create the directory and download the default model before running:

```bash
mkdir -p models/weights
wget -O models/weights/megadetector.pt \
  https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt
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
pip install -r requirements.txt
git clone https://github.com/ultralytics/yolov5.git vendor/yolov5
```

For local test/dev work:

```bash
pip install -r requirements-dev.txt
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
  - overlay OCR fields: `overlay_date`, `overlay_time`, `temperature_c`, `temperature_f`
  - EXIF fallback: `timestamp`, `temperature_c`
  - `gps_lat`, `gps_lon`
- Camera ID is inferred from the first folder under input root.
  - Example: `data/raw/cam_01/image.jpg` -> `camera_id=cam_01`
- Graph-ready datasets are regenerated each run in `data/graphs/`:
  - `daily_counts_by_camera.csv`
  - `species_by_camera.csv`
  - `camera_cooccurrence.csv`
  - `observation_points.csv` (includes image paths for linking back to files)
  - `temperature_by_species.csv`
  - `hourly_activity_by_species.csv`
- Visual chart outputs (for quick presentation / channel usage):
  - `temperature_by_species.png`
  - `activity_by_hour.png`

## OCR Setup (Overlay Bar)

Ubuntu/Debian example:

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr
pip install pillow pytesseract matplotlib
```

If OCR dependencies are missing, the pipeline still runs but overlay fields remain blank.
## Testing

Run the smoke test suite from the repo root:

```bash
pytest -q
```

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
