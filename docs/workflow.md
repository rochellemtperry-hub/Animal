# Workflow Guide

## Purpose

This document describes a repeatable process for sorting trail camera photos with the CLI and validating the results.

## 1. Prepare Environment

From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies used in your environment (at minimum: PyTorch + NumPy + OpenCV).

## 2. Prepare Input Data

Copy camera exports into a working input directory, for example:

```text
data/raw/
```

Subfolders are supported by default (recursive scan).

## 3. Run Inference

Standard run:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw
```

Key runtime behavior:
- Creates output folders if they do not exist.
- Preserves source subfolder structure by default.
- Avoids filename collisions by adding suffixes like `_1`, `_2`, etc.

## 4. Check Outputs

Expected outputs:
- `data/outputs/animals/`
- `data/outputs/non_animals/`
- `data/reports/detections.csv`

CSV is the audit trail for each processed image and includes confidence and detection counts.

## 5. Validate Device Selection

At startup, the CLI prints:

```text
Inference device: ...
```

Device selection rules:
- No `--device` provided: auto-select CUDA (`cuda:0`) when available, else CPU.
- `--device cuda:0`: requests GPU.
- `--device cpu`: forces CPU.

If GPU is requested but not usable, runtime falls back to CPU.

## 6. Common Run Variants

Dry run (no copy/move):

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --dry-run
```

Move files instead of copy:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --action move
```

Disable recursion:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --no-recursive
```

Flatten output structure:

```bash
PYTHONPATH=src:vendor/yolov5 python -m src.cli --input data/raw --flat-output
```

## 7. Post-Run Spot Check

Recommended quick checks:
- Open a random sample from `animals/` and `non_animals/`.
- Scan CSV rows where `has_animal=false` but `total_detections>0`.
- Scan low-confidence positives for possible threshold tuning.

## 8. Threshold Tuning

Useful flags:
- `--conf-threshold` (default `0.25`)
- `--iou-threshold` (default `0.45`)

Practical approach:
1. Start with defaults.
2. Run on a representative subset.
3. Increase confidence threshold to reduce false positives.
4. Re-run and compare CSV metrics.
