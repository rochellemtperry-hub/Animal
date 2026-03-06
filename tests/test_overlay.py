from __future__ import annotations

from trailcam_filter.overlay import parse_overlay_text


def test_parse_overlay_text_extracts_date_time_and_temp() -> None:
    text = "CAM01 2026/03/06 19:42:15 17C 63F"
    readout = parse_overlay_text(text)
    assert readout.date_ymd == "2026/03/06"
    assert readout.time_24h == "19:42:15"
    assert readout.temperature_c is not None
    assert readout.temperature_f is not None
    assert abs(readout.temperature_c - 17.0) < 0.01
    assert abs(readout.temperature_f - 63.0) < 0.01
    assert readout.timestamp_iso == "2026-03-06T19:42:15"
