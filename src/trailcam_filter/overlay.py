from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


DATE_RE = re.compile(r"\b(\d{4}/\d{2}/\d{2})\b")
TIME_RE = re.compile(r"\b([01]\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b")
TEMP_C_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*C\b", re.IGNORECASE)
TEMP_F_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*F\b", re.IGNORECASE)


@dataclass(slots=True)
class OverlayReadout:
    date_ymd: str | None = None
    time_24h: str | None = None
    temperature_c: float | None = None
    temperature_f: float | None = None
    raw_text: str = ""

    @property
    def timestamp_iso(self) -> str | None:
        if not self.date_ymd or not self.time_24h:
            return None
        seconds = self.time_24h if len(self.time_24h.split(":")) == 3 else f"{self.time_24h}:00"
        try:
            return datetime.strptime(f"{self.date_ymd} {seconds}", "%Y/%m/%d %H:%M:%S").isoformat()
        except ValueError:
            return None


def parse_overlay_text(text: str) -> OverlayReadout:
    normalized = " ".join(text.replace("\n", " ").split())
    date_match = DATE_RE.search(normalized)
    time_match = TIME_RE.search(normalized)
    c_match = TEMP_C_RE.search(normalized)
    f_match = TEMP_F_RE.search(normalized)

    temp_c = float(c_match.group(1)) if c_match else None
    temp_f = float(f_match.group(1)) if f_match else None
    if temp_c is None and temp_f is not None:
        temp_c = (temp_f - 32.0) * (5.0 / 9.0)
    if temp_f is None and temp_c is not None:
        temp_f = temp_c * (9.0 / 5.0) + 32.0

    return OverlayReadout(
        date_ymd=date_match.group(1) if date_match else None,
        time_24h=time_match.group(0) if time_match else None,
        temperature_c=temp_c,
        temperature_f=temp_f,
        raw_text=normalized,
    )


def extract_overlay_readout(image_path: Path, strip_ratio: float = 0.2) -> OverlayReadout:
    try:
        from PIL import Image, ImageOps
        import pytesseract
    except ImportError:
        return OverlayReadout()

    try:
        with Image.open(image_path) as img:
            width, height = img.size
            start_y = int(height * (1.0 - strip_ratio))
            strip = img.crop((0, start_y, width, height))
            gray = ImageOps.grayscale(strip)
            # Upscale and binarize to improve OCR on embedded overlay text.
            upscaled = gray.resize((gray.width * 2, gray.height * 2))
            contrasted = ImageOps.autocontrast(upscaled)
            binary = contrasted.point(lambda p: 255 if p > 140 else 0)
    except Exception:
        return OverlayReadout()

    try:
        text = pytesseract.image_to_string(binary, config="--psm 6")
    except Exception:
        return OverlayReadout()

    return parse_overlay_text(text)
