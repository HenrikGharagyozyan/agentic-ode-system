"""Plot data from saved recordings (agentic_ode/web/recordings.py): pure numpy, small made-up event arrays."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

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


def test_brightness_curve_is_downsampled_and_has_a_full_resolution_zoom():
    # one pixel, 1 s: ON at 0.1 s, OFF at 0.6 s (the meta says 1 s, so the curve covers 0.6 s, the last event)
    p = preview(events([(5, 5, 1, 100_000), (5, 5, 0, 600_000)]), {"recording_id": "r", "recorded_s": 1.0})
    c = p["brightness_curve"]
    assert c["bin_s"] == 0.0001 and c["leak_tau_s"] == 0.2 and c["unit"] == "contrast steps (delta)"
    assert len(c["t_s"]) == len(c["delta_l"]) <= 2000 and c["step_s"] == 0.0004  # 6001 bins -> blocks of 4
    assert len(c["zoom"]["t_s"]) == 1000 and c["zoom"]["t_s"][0] == 0.0001  # 100 ms at 100 us
    assert max(c["delta_l"]) == pytest.approx(1.0, abs=1e-3) and min(c["delta_l"]) < 0  # +1 Δ, then down


def test_delta_image_is_signed_and_time_image_codes_no_events_as_0():
    rows = [(0, 0, 1, 10), (1, 1, 1, 20), (8, 4, 0, 900_000)]  # bin (0, 0): +2, bin (2, 1): -1
    p = preview(events(rows), {"recording_id": "r", "recorded_s": 1.0})
    d = np.frombuffer(base64.b64decode(p["delta_image"]["data"]), dtype=np.int8).reshape(180, 320)
    assert d[0, 0] > 0 and d[1, 2] < 0 and np.count_nonzero(d) == 2
    assert p["delta_image"]["min"] == -1 and p["delta_image"]["max"] == 2
    t = np.frombuffer(base64.b64decode(p["time_image"]["data"]), dtype=np.uint8).reshape(180, 320)
    assert t[0, 0] == 1 and t[1, 2] == 1 + round(0.9 * 254) and np.count_nonzero(t) == 2  # start, 0.9 s, empty


def test_new_plots_drop_hot_pixels_and_are_none_without_events():
    hot = [(50, 50, 1, i * 10) for i in range(30_000)]  # 30,000 events in 1 s: 30 kHz
    p = preview(events(hot + [(0, 0, 0, 5)]), {"recording_id": "r", "recorded_s": 1.0})
    d = np.frombuffer(base64.b64decode(p["delta_image"]["data"]), dtype=np.int8).reshape(180, 320)
    assert d[12, 12] == 0 and d[0, 0] < 0 and p["delta_image"]["max"] == 0
    empty = preview(events([]), {"recording_id": "r", "recorded_s": 1.0})
    assert empty["brightness_curve"] is None and empty["delta_image"] is None and empty["time_image"] is None
