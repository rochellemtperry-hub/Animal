from __future__ import annotations

import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def discover_images(input_dir: Path, recursive: bool = True) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    images: list[Path] = []
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return sorted(images)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_collision(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 1
    while True:
        candidate = parent / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def route_image(
    src_path: Path,
    input_root: Path,
    output_animals_dir: Path,
    output_non_animals_dir: Path,
    has_animal: bool,
    action: str = "copy",
    keep_structure: bool = True,
    dry_run: bool = False,
) -> Path:
    relative = src_path.relative_to(input_root) if keep_structure else Path(src_path.name)
    base_out_dir = output_animals_dir if has_animal else output_non_animals_dir
    dst_path = base_out_dir / relative

    ensure_dir(dst_path.parent)
    dst_path = _resolve_collision(dst_path)

    if dry_run:
        return dst_path

    if action == "move":
        shutil.move(str(src_path), str(dst_path))
    else:
        shutil.copy2(src_path, dst_path)
    return dst_path
