from __future__ import annotations

import argparse
from itertools import islice
from pathlib import Path
import sys
from typing import Iterator
import uuid

from trailcam_filter.config import AppConfig
from trailcam_filter.graphs import generate_graph_datasets
from trailcam_filter.infer import UltraLyticsDetector, infer_detections, infer_image, is_animal_label
from trailcam_filter.io import discover_images, route_image
from trailcam_filter.metadata import extract_image_metadata
from trailcam_filter.observations import (
    ObservationRecord,
    camera_id_for_image,
    ensure_observations_file,
    find_observation_index,
    load_camera_lookup,
    load_observations,
    upsert_and_save,
    write_observations,
)
from trailcam_filter.overlay import extract_overlay_readout
from trailcam_filter.postprocess import RunSummary
from trailcam_filter.report import ReportRow, write_report


def _chunked(items: list[Path], size: int) -> Iterator[list[Path]]:
    if size <= 0:
        raise ValueError("batch_size must be >= 1")
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            break
        yield batch


def _split_datetime(value: str) -> tuple[str, str]:
    if not value:
        return "", ""
    if "T" in value:
        date, time = value.split("T", 1)
        return date[:10], time[:8]
    if " " in value:
        date, time = value.split(" ", 1)
        return date[:10], time[:8]
    return value[:10], ""


def _to_yes_no(value: str) -> str:
    return "yes" if value.strip().lower() in {"y", "yes", "true", "1"} else "no"


def _prompt(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    return value if value else default


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trailcam animal image sorter and review workflow")
    subparsers = parser.add_subparsers(dest="command")

    process = subparsers.add_parser("process", help="Run AI processing pipeline")
    process.add_argument("--input", required=True, type=Path, help="Input image directory")
    process.add_argument("--model", type=Path, default=Path("models/weights/megadetector.pt"))
    process.add_argument("--output-animals", type=Path, default=Path("data/outputs/animals"))
    process.add_argument("--output-non-animals", type=Path, default=Path("data/outputs/non_animals"))
    process.add_argument("--report-csv", type=Path, default=Path("data/reports/detections.csv"))
    process.add_argument("--observations-csv", type=Path, default=Path("data/observations/animal_observations.csv"))
    process.add_argument("--cameras-csv", type=Path, default=Path("data/metadata/cameras.csv"))
    process.add_argument("--graphs-dir", type=Path, default=Path("data/graphs"))
    process.add_argument("--conf-threshold", type=float, default=0.25)
    process.add_argument("--iou-threshold", type=float, default=0.45)
    process.add_argument("--device", default=None)
    process.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Images per inference batch. Default is 1 for safe processing.",
    )
    process.add_argument("--default-camera-id", default="unknown")
    process.add_argument("--camera-id", default=None, help="Force one camera_id for all input images")
    process.add_argument("--action", choices=["copy", "move"], default="copy")
    process.add_argument("--no-recursive", action="store_true")
    process.add_argument("--flat-output", action="store_true")
    process.add_argument("--dry-run", action="store_true")

    review = subparsers.add_parser("review", help="Review and tag images in animal outputs")
    review.add_argument("--input", required=True, type=Path, help="Directory of animal images to review")
    review.add_argument("--observations-csv", type=Path, default=Path("data/observations/animal_observations.csv"))
    review.add_argument("--default-camera-id", default="unknown")
    review.add_argument("--camera-id", default=None, help="Force one camera_id for all reviewed images")
    review.add_argument("--no-recursive", action="store_true", help="Disable recursive review scan")
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
        default_camera_id=args.default_camera_id,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


def run_process(config: AppConfig, forced_camera_id: str | None = None) -> int:
    if not config.input_dir.exists() or not config.input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {config.input_dir}")
    if not config.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {config.model_path}")
    if config.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {config.batch_size}")

    ensure_observations_file(config.observations_csv)
    camera_lookup = load_camera_lookup(config.cameras_csv)
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
    report_rows: list[ReportRow] = []
    observations: list[ObservationRecord] = []

    for image_batch in _chunked(images, config.batch_size):
        if hasattr(detector, "predict_batch"):
            detections_by_image = detector.predict_batch(image_batch)  # type: ignore[attr-defined]
            batch_results = [infer_detections(detections) for detections in detections_by_image]
        else:
            batch_results = [infer_image(detector, image_path) for image_path in image_batch]

        for image_path, result in zip(image_batch, batch_results):
            metadata = extract_image_metadata(image_path)
            overlay = extract_overlay_readout(image_path)
            camera_id = forced_camera_id or camera_id_for_image(
                image_path, config.input_dir, default_camera_id=config.default_camera_id
            )
            location_name = camera_lookup.get(camera_id, "")

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
            report_rows.append(
                ReportRow(
                    run_id=run_id,
                    camera_id=camera_id,
                    source_path=str(image_path),
                    output_path=str(out_path),
                    has_animal=result.has_animal,
                    top_animal_confidence=result.top_animal_confidence,
                    total_detections=result.total_detections,
                    animal_detections=result.animal_detections,
                )
            )

            if not result.has_animal:
                continue

            timestamp = overlay.timestamp_iso or metadata.timestamp or ""
            date, time = _split_datetime(timestamp)
            temperature_c_value = overlay.temperature_c if overlay.temperature_c is not None else metadata.temperature_c
            animal_detections = [d for d in result.detections if is_animal_label(d.label)]
            animal_detections.sort(key=lambda d: d.confidence, reverse=True)
            top_species = animal_detections[0].label if animal_detections else "animal"
            top_conf = animal_detections[0].confidence if animal_detections else 0.0
            observations.append(
                ObservationRecord(
                    observation_id="",
                    file_name=out_path.name,
                    camera_id=camera_id,
                    location_name=location_name,
                    photo_datetime=timestamp,
                    date=date,
                    time=time,
                    temperature_c="" if temperature_c_value is None else f"{temperature_c_value:.3f}",
                    animal_detected="yes",
                    species=top_species,
                    species_confidence=f"{top_conf:.6f}",
                    count=str(max(1, result.animal_detections)),
                    image_path=str(out_path),
                    source_path=str(image_path),
                )
            )

    write_report(report_rows, config.report_csv)
    upserted = upsert_and_save(config.observations_csv, observations)
    generate_graph_datasets(config.observations_csv, config.graphs_dir)

    print(
        f"Processed {summary.processed} images | "
        f"animals: {summary.animals} | non_animals: {summary.non_animals}"
    )
    print(f"Raw detections report: {config.report_csv}")
    print(f"Observations upserted: {upserted} -> {config.observations_csv}")
    print(f"Graph datasets written to: {config.graphs_dir}")
    return 0


def run_review(input_dir: Path, observations_csv: Path, recursive: bool, default_camera_id: str, forced_camera_id: str | None) -> int:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Review input directory not found: {input_dir}")

    ensure_observations_file(observations_csv)
    rows = load_observations(observations_csv)
    images = discover_images(input_dir, recursive=recursive)
    if not images:
        print(f"No images found in {input_dir}")
        return 0

    reviewed_count = 0
    for image_path in images:
        camera_id = forced_camera_id or camera_id_for_image(
            image_path=image_path,
            input_root=input_dir,
            default_camera_id=default_camera_id,
        )
        idx = find_observation_index(rows, str(image_path), image_path.name, camera_id)
        if idx is None:
            rows.append(
                ObservationRecord(
                    observation_id="",
                    file_name=image_path.name,
                    camera_id=camera_id,
                    location_name="",
                    photo_datetime="",
                    date="",
                    time="",
                    temperature_c="",
                    animal_detected="yes",
                    species="unknown",
                    species_confidence="",
                    count="1",
                    image_path=str(image_path),
                    source_path="",
                ).to_row()
            )
            idx = len(rows) - 1

        row = rows[idx]
        print("\n--- Review ---")
        print(f"file_name:           {row.get('file_name', '')}")
        print(f"camera_id:           {row.get('camera_id', '')}")
        print(f"photo_datetime:      {row.get('photo_datetime', '')}")
        print(f"temperature_c:       {row.get('temperature_c', '')}")
        print(f"ai species/conf:     {row.get('species', '')} / {row.get('species_confidence', '')}")
        print(f"ai count:            {row.get('count', '')}")
        print(f"image_path:          {row.get('image_path', '')}")

        row["species"] = _prompt("species", row.get("species", ""))
        row["count"] = _prompt("count", row.get("count", "1") or "1")
        row["reviewed"] = _to_yes_no(_prompt("reviewed (yes/no)", row.get("reviewed", "yes") or "yes"))
        row["ai_corrected"] = _to_yes_no(_prompt("ai_corrected (yes/no)", row.get("ai_corrected", "no") or "no"))
        row["notes"] = _prompt("notes", row.get("notes", ""))
        row["review_status"] = "reviewed" if row["reviewed"] == "yes" else "pending"
        if not row.get("observation_id", "").strip():
            row["observation_id"] = ""
        reviewed_count += 1

    # Normalize with upsert path to enforce stable IDs.
    records: list[ObservationRecord] = []
    for row in rows:
        records.append(
            ObservationRecord(
                observation_id=row.get("observation_id", ""),
                file_name=row.get("file_name", ""),
                camera_id=row.get("camera_id", ""),
                location_name=row.get("location_name", ""),
                photo_datetime=row.get("photo_datetime", ""),
                date=row.get("date", ""),
                time=row.get("time", ""),
                temperature_c=row.get("temperature_c", ""),
                animal_detected=row.get("animal_detected", "yes") or "yes",
                species=row.get("species", ""),
                species_confidence=row.get("species_confidence", ""),
                count=row.get("count", "1") or "1",
                reviewed=row.get("reviewed", "no") or "no",
                review_status=row.get("review_status", "pending") or "pending",
                ai_corrected=row.get("ai_corrected", "no") or "no",
                notes=row.get("notes", ""),
                image_path=row.get("image_path", ""),
                source_path=row.get("source_path", ""),
            )
        )
    # write_observations first to persist manual edits, then upsert for ID stabilization.
    write_observations(observations_csv, [r.to_row() for r in records])
    upsert_and_save(observations_csv, records)
    print(f"Review complete. Updated rows: {reviewed_count}")
    print(f"Dataset updated: {observations_csv}")
    return 0


def main() -> int:
    parser = build_parser()
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["process", *argv]
    args = parser.parse_args(argv)

    if args.command == "review":
        return run_review(
            input_dir=args.input,
            observations_csv=args.observations_csv,
            recursive=not args.no_recursive,
            default_camera_id=args.default_camera_id,
            forced_camera_id=args.camera_id,
        )

    config = config_from_args(args)
    return run_process(config=config, forced_camera_id=getattr(args, "camera_id", None))


if __name__ == "__main__":
    raise SystemExit(main())
