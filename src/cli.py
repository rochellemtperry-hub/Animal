from __future__ import annotations

import argparse
from itertools import islice
from typing import Iterator
from pathlib import Path
import uuid

from trailcam_filter.config import AppConfig
from trailcam_filter.graphs import generate_graph_datasets
from trailcam_filter.infer import Detector, ImageInference, UltraLyticsDetector, infer_detections, infer_image, is_animal_label
from trailcam_filter.io import discover_images, route_image
from trailcam_filter.metadata import extract_image_metadata
from trailcam_filter.observations import (
    ObservationRow,
    append_observations,
    camera_id_for_image,
    ensure_observations_file,
    load_known_camera_ids,
)
from trailcam_filter.postprocess import RunSummary
from trailcam_filter.report import ReportRow, write_report
from tqdm import tqdm


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
        "--observations-csv",
        type=Path,
        default=Path("data/observations/animal_observations.csv"),
        help="Append-only animal observations CSV path",
    )
    parser.add_argument(
        "--cameras-csv",
        type=Path,
        default=Path("data/metadata/cameras.csv"),
        help="Camera metadata CSV path",
    )
    parser.add_argument(
        "--graphs-dir",
        type=Path,
        default=Path("data/graphs"),
        help="Directory for graph-ready aggregated CSV outputs",
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
        "--batch-size",
        type=int,
        default=16,
        help="Images per inference batch. Increase to improve GPU utilization (default: 16).",
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
        observations_csv=args.observations_csv,
        cameras_csv=args.cameras_csv,
        graphs_dir=args.graphs_dir,
        model_path=args.model,
        conf_threshold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        recursive=not args.no_recursive,
        action=args.action,
        keep_structure=not args.flat_output,
        device=args.device,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


def _chunked(items: list[Path], size: int) -> Iterator[list[Path]]:
    if size <= 0:
        raise ValueError("batch_size must be >= 1")
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def _infer_batch_resilient(
    detector: Detector,
    image_batch: list[Path],
) -> tuple[list[tuple[Path, ImageInference]], list[tuple[Path, str]]]:
    if hasattr(detector, "predict_batch"):
        try:
            detections_by_image = detector.predict_batch(image_batch)  # type: ignore[attr-defined]
            batch_results = [infer_detections(detections) for detections in detections_by_image]
            return list(zip(image_batch, batch_results)), []
        except Exception as exc:
            tqdm.write(f"Warning: batch inference failed; retrying individually: {exc}")

    results: list[tuple[Path, ImageInference]] = []
    skipped: list[tuple[Path, str]] = []
    for image_path in image_batch:
        try:
            results.append((image_path, infer_image(detector, image_path)))
        except Exception as exc:
            skipped.append((image_path, str(exc)))
    return results, skipped


def run(config: AppConfig) -> int:
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {config.model_path}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

    ensure_observations_file(config.observations_csv)
    known_camera_ids = load_known_camera_ids(config.cameras_csv)
    run_id = uuid.uuid4().hex

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
    print(f"Inference device: {detector.device_name} | batch_size: {config.batch_size}")

    summary = RunSummary()
    rows: list[ReportRow] = []
    observation_rows: list[ObservationRow] = []
    unknown_camera_ids: set[str] = set()

    with tqdm(total=len(images), desc="Processing images", unit="img") as progress:
        for image_batch in _chunked(images, config.batch_size):
            successful_results, skipped_results = _infer_batch_resilient(detector, image_batch)

            for image_path, error in skipped_results:
                summary.on_skipped()
                progress.update(1)
                progress.set_postfix(
                    animals=summary.animals,
                    non_animals=summary.non_animals,
                    skipped=summary.skipped,
                    refresh=False,
                )
                tqdm.write(f"Warning: skipping image {image_path}: {error}")

            for image_path, result in successful_results:
                metadata = extract_image_metadata(image_path)
                camera_id = camera_id_for_image(image_path, config.input_dir)
                if known_camera_ids and camera_id not in known_camera_ids:
                    unknown_camera_ids.add(camera_id)

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
                if result.has_animal:
                    timestamp = metadata.timestamp or ""
                    temperature_c = "" if metadata.temperature_c is None else f"{metadata.temperature_c:.3f}"
                    gps_lat = "" if metadata.gps_lat is None else f"{metadata.gps_lat:.6f}"
                    gps_lon = "" if metadata.gps_lon is None else f"{metadata.gps_lon:.6f}"
                    for detection in result.detections:
                        if not is_animal_label(detection.label):
                            continue
                        observation_rows.append(
                            ObservationRow(
                                observation_id=uuid.uuid4().hex,
                                timestamp=timestamp,
                                camera_id=camera_id,
                                image_path=str(out_path),
                                source_path=str(image_path),
                                species_label=detection.label,
                                confidence=detection.confidence,
                                run_id=run_id,
                                temperature_c=temperature_c,
                                gps_lat=gps_lat,
                                gps_lon=gps_lon,
                            )
                        )
                progress.update(1)
                progress.set_postfix(
                    animals=summary.animals,
                    non_animals=summary.non_animals,
                    skipped=summary.skipped,
                    refresh=False,
                )

    write_report(rows, config.report_csv)
    append_observations(config.observations_csv, observation_rows)
    generate_graph_datasets(config.observations_csv, config.graphs_dir)

    print(
        f"Processed {summary.processed} images | "
        f"animals: {summary.animals} | non_animals: {summary.non_animals} | skipped: {summary.skipped}"
    )
    print(f"Report written to: {config.report_csv}")
    print(f"Observations appended: {len(observation_rows)} -> {config.observations_csv}")
    print(f"Graph datasets written to: {config.graphs_dir}")
    if unknown_camera_ids:
        sorted_ids = ", ".join(sorted(unknown_camera_ids))
        print(f"Warning: camera_id(s) not found in {config.cameras_csv}: {sorted_ids}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = config_from_args(args)
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
