# Trailcam Animal Filter

Local AI tool to automatically identify and separate wildlife photos from trail camera dumps.

## 📂 Workflow

1. Copy SD card photos into:
   data/raw/

2. Run the CLI:
   python -m src.cli --input data/raw

3. Results:
   - data/outputs/animals/
   - data/reports/detections.csv

## 🚀 Goal

- Fully local inference
- No cloud uploads
- Fast filtering of empty frames

