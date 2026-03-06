from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any


@dataclass(slots=True)
class ImageMetadata:
    timestamp: str | None = None
    temperature_c: float | None = None
    gps_lat: float | None = None
    gps_lon: float | None = None


def _to_float_ratio(value: Any) -> float:
    if isinstance(value, tuple) and len(value) == 2 and value[1]:
        return float(value[0]) / float(value[1])
    if hasattr(value, "numerator") and hasattr(value, "denominator") and value.denominator:
        return float(value.numerator) / float(value.denominator)
    return float(value)


def _dms_to_decimal(dms: Any, ref: str | None) -> float | None:
    try:
        deg = _to_float_ratio(dms[0])
        minute = _to_float_ratio(dms[1])
        sec = _to_float_ratio(dms[2])
    except Exception:
        return None
    decimal = deg + (minute / 60.0) + (sec / 3600.0)
    if ref in {"S", "W"}:
        decimal *= -1.0
    return decimal


def _parse_timestamp(value: str | bytes | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    text = value.strip()
    if not text:
        return None
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).isoformat()
        except ValueError:
            continue
    return text


def _parse_temperature_c(text_values: list[str]) -> float | None:
    for text in text_values:
        match = re.search(r"(-?\d+(?:\.\d+)?)\s*°?\s*([CF])\b", text, flags=re.IGNORECASE)
        if match:
            value = float(match.group(1))
            unit = match.group(2).upper()
            if unit == "F":
                return (value - 32.0) * (5.0 / 9.0)
            return value
        alt = re.search(r"\btemp(?:erature)?\s*[:=]?\s*(-?\d+(?:\.\d+)?)\b", text, flags=re.IGNORECASE)
        if alt:
            return float(alt.group(1))
    return None


def extract_image_metadata(image_path: Path) -> ImageMetadata:
    try:
        from PIL import ExifTags, Image
    except ImportError:
        return ImageMetadata()

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
    except Exception:
        return ImageMetadata()

    if not exif:
        return ImageMetadata()

    tags = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif.items()}
    timestamp = (
        _parse_timestamp(tags.get("DateTimeOriginal"))
        or _parse_timestamp(tags.get("DateTimeDigitized"))
        or _parse_timestamp(tags.get("DateTime"))
    )

    gps_info_raw = tags.get("GPSInfo")
    gps_lat: float | None = None
    gps_lon: float | None = None
    if isinstance(gps_info_raw, dict):
        gps = {ExifTags.GPSTAGS.get(tag, tag): value for tag, value in gps_info_raw.items()}
        gps_lat = _dms_to_decimal(gps.get("GPSLatitude"), gps.get("GPSLatitudeRef"))
        gps_lon = _dms_to_decimal(gps.get("GPSLongitude"), gps.get("GPSLongitudeRef"))

    text_values: list[str] = []
    for key in ("ImageDescription", "UserComment", "XPComment"):
        value = tags.get(key)
        if value is None:
            continue
        if isinstance(value, bytes):
            text_values.append(value.decode("utf-8", errors="ignore"))
        else:
            text_values.append(str(value))
    temperature_c = _parse_temperature_c(text_values)

    return ImageMetadata(
        timestamp=timestamp,
        temperature_c=temperature_c,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )
