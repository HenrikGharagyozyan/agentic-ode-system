"""Plot data from saved recordings (agentic_ode/web/recordings.py): pure numpy, small made-up event arrays."""

from __future__ import annotations

import base64
import json

import numpy as np

from agentic_ode.hardware.devices.evcam.events import EVENT_DTYPE
from agentic_ode.web.recordings import preview, recordings_of


def events(rows):
    ev = np.zeros(len(rows), dtype=EVENT_DTYPE)
    for i, (x, y, p, t) in enumerate(rows):
        ev[i] = (x, y, p, t)
    return ev


def image(p, name):
    return np.frombuffer(base64.b64decode(p["image"][name]["data"]), dtype=np.uint8).reshape(180, 320)


def test_image_bins_keep_sensor_coordinates_and_polarity():
    p = preview(events([(9, 5, 1, 0), (9, 5, 1, 10), (1279, 719, 0, 20)]), {"recorded_s": 1.0})
    on, off = image(p, "on"), image(p, "off")
    assert on[1, 2] == 255 and on.sum() == 255  # (9, 5) is in bin (x 2, y 1); row 0 is the top of the sensor
    assert off[179, 319] == 255 and off.sum() == 255
    assert p["n_events"] == 3


def test_rate_is_events_per_second_per_bin():
    rows = [(0, 0, 1, t) for t in range(0, 20_000, 1_000)]  # 20 ON events in the first 20 ms
    rate = preview(events(rows), {"recorded_s": 0.1})["rate"]
    assert rate["bin_s"] == 0.01
    assert rate["on_hz"][:3] == [1000.0, 1000.0, 0.0]  # 10 events in 10 ms = 1000 ev/s
    assert sum(rate["off_hz"]) == 0


def test_no_events_gives_an_empty_image_and_no_rate():
    p = preview(events([]), {"recorded_s": 0.5})
    assert image(p, "on").sum() == 0 and p["rate"] is None and p["hot_pixels"] == []


def test_recordings_of_reads_only_successful_records():
    ok = {"tool": "record", "ok": True, "result_line": json.dumps({"ok": True, "recording_id": "a"})}
    failed = {"tool": "record", "ok": False, "result_line": json.dumps({"ok": False, "error": "x"})}
    other = {"tool": "spectrum", "ok": True, "result_line": json.dumps({"ok": True, "recording_id": "b"})}
    broken = {"tool": "record", "ok": True, "result_line": "not json"}
    assert recordings_of([ok, failed, other, broken]) == ["a"]
