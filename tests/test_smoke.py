from __future__ import annotations

from pathlib import Path

from trailcam_filter.infer import Detection, infer_image
from trailcam_filter.io import route_image
from trailcam_filter.report import ReportRow, write_report


class StubDetector:
    def __init__(self, by_name: dict[str, list[Detection]]) -> None:
        self._by_name = by_name

    def predict(self, image_path: Path) -> list[Detection]:
        return self._by_name.get(image_path.name, [])


def test_routing_and_report_smoke(tmp_path: Path) -> None:
    input_root = tmp_path / "raw"
    animals_out = tmp_path / "animals"
    non_animals_out = tmp_path / "non_animals"
    report_csv = tmp_path / "detections.csv"

    input_root.mkdir(parents=True)
    deer_img = input_root / "deer.jpg"
    empty_img = input_root / "empty.jpg"
    deer_img.write_bytes(b"fake-jpg-content")
    empty_img.write_bytes(b"fake-jpg-content")

    detector = StubDetector(
        {
            "deer.jpg": [Detection(label="animal", confidence=0.98)],
            "empty.jpg": [Detection(label="person", confidence=0.34)],
        }
    )

    rows: list[ReportRow] = []
    for image in [deer_img, empty_img]:
        result = infer_image(detector, image)
        out = route_image(
            src_path=image,
            input_root=input_root,
            output_animals_dir=animals_out,
            output_non_animals_dir=non_animals_out,
            has_animal=result.has_animal,
            action="copy",
            keep_structure=True,
            dry_run=False,
        )
        rows.append(
            ReportRow(
                run_id="run-test",
                camera_id="cam-test",
                source_path=str(image),
                output_path=str(out),
                has_animal=result.has_animal,
                top_animal_confidence=result.top_animal_confidence,
                total_detections=result.total_detections,
                animal_detections=result.animal_detections,
            )
        )

    write_report(rows, report_csv)

    assert (animals_out / "deer.jpg").exists()
    assert (non_animals_out / "empty.jpg").exists()
    assert report_csv.exists()
    content = report_csv.read_text(encoding="utf-8")
    assert "deer.jpg" in content
    assert "empty.jpg" in content
