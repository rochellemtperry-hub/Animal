from __future__ import annotations

import argparse
from pathlib import Path

from trailcam_filter.config import AppConfig
from trailcam_filter.infer import UltraLyticsDetector, infer_image
from trailcam_filter.io import discover_images, route_image
from trailcam_filter.postprocess import RunSummary
from trailcam_filter.report import ReportRow, write_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trailcam animal image sorter")
    parser.add_argument("--input", required=True, type=Path, help="Input image directory")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/weights/megadetector.pt"),
        help="Path to detector model (.pt)",
    )
    parser.add_argument(
        "--output-animals",
        type=Path,
        default=Path("data/outputs/animals"),
        help="Output directory for animal images",
    )
    parser.add_argument(
        "--output-non-animals",
        type=Path,
        default=Path("data/outputs/non_animals"),
        help="Output directory for non-animal images",
    )
    parser.add_argument(
        "--report-csv",
        type=Path,
        default=Path("data/reports/detections.csv"),
        help="CSV report output path",
    )
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.25,
        help="Detector confidence threshold",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.45,
        help="Detector IOU threshold",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, e.g. 'cpu', 'cuda:0'. Defaults to library auto-selection.",
    )
    parser.add_argument(
        "--action",
        choices=["copy", "move"],
        default="copy",
        help="How to place files in outputs",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Disable recursive image discovery under input directory",
    )
    parser.add_argument(
        "--flat-output",
        action="store_true",
        help="Do not preserve subdirectory structure in output directories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run inference and reporting without copying/moving files",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        input_dir=args.input,
        output_animals_dir=args.output_animals,
        output_non_animals_dir=args.output_non_animals,
        report_csv=args.report_csv,
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        recursive=not args.no_recursive,
        action=args.action,
        keep_structure=not args.flat_output,
        device=args.device,
        dry_run=args.dry_run,
    )


def run(config: AppConfig) -> int:
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {config.model_path}")

    images = discover_images(config.input_dir, recursive=config.recursive)
    if not images:
        print(f"No images found in {config.input_dir}")
        write_report([], config.report_csv)
        return 0

    detector = UltraLyticsDetector(
        model_path=config.model_path,
        conf_threshold=config.conf_threshold,
        iou_threshold=config.iou_threshold,
        device=config.device,
    )
    print(f"Inference device: {detector.device_name}")

    summary = RunSummary()
    rows: list[ReportRow] = []

    for image_path in images:
        result = infer_image(detector, image_path)
        out_path = route_image(
            src_path=image_path,
            input_root=config.input_dir,
            output_animals_dir=config.output_animals_dir,
            output_non_animals_dir=config.output_non_animals_dir,
            has_animal=result.has_animal,
            action=config.action,
            keep_structure=config.keep_structure,
            dry_run=config.dry_run,
        )
        summary.on_result(result.has_animal)
        rows.append(
            ReportRow(
                source_path=str(image_path),
                output_path=str(out_path),
                has_animal=result.has_animal,
                top_animal_confidence=result.top_animal_confidence,
                total_detections=result.total_detections,
                animal_detections=result.animal_detections,
            )
        )

    write_report(rows, config.report_csv)

    print(
        f"Processed {summary.processed} images | "
        f"animals: {summary.animals} | non_animals: {summary.non_animals}"
    )
    print(f"Report written to: {config.report_csv}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
