"""The evcam MCP server, called through a real MCP client (in memory) with the fake camera. No hardware needed."""

import asyncio
import json

import numpy as np
import pytest
from fastmcp import Client

from agentic_ode.hardware.evcam import fake, limits, store
from agentic_ode.hardware.evcam.connector import EVENT_DTYPE
from agentic_ode.hardware.evcam.fake import FakeEventCamera
from agentic_ode.mcp_servers import evcam_server

TOOLS = {"camera_info", "get_settings", "set_bias", "find_hot_pixels", "mask_pixels", "clear_mask", "set_roi",
         "clear_roi", "record", "spectrum"}


@pytest.fixture(autouse=True)
def fake_server(tmp_path, monkeypatch):
    """Fake camera (pwm scene), recordings and log in tmp_path, no camera open before and after each test."""
    monkeypatch.setenv("EVCAM_FAKE", "1")
    monkeypatch.delenv("EVCAM_FAKE_SCENE", raising=False)
    monkeypatch.setattr(evcam_server, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(evcam_server, "LOG_FILE", tmp_path / "tool_log.jsonl")
    evcam_server._close_camera()
    yield
    evcam_server._close_camera()


@pytest.fixture
def opens(monkeypatch):
    """Counts how often FakeEventCamera.open() opens a camera."""
    calls = []
    real_open = FakeEventCamera.open

    def counting_open(self):
        calls.append(1)
        return real_open(self)

    monkeypatch.setattr(FakeEventCamera, "open", counting_open)
    return calls


def call(*calls):
    """Run (tool, arguments) calls in ONE client session (= one server run) and return the raw results."""
    async def go():
        async with Client(evcam_server.mcp) as client:
            return [await client.call_tool(name, args, raise_on_error=False) for name, args in calls]
    return asyncio.run(go())


def run(*calls):
    """Like call(), but returns the dict each tool returned."""
    return [r.structured_content for r in call(*calls)]


def log_lines(tmp_path):
    return [json.loads(line) for line in (tmp_path / "tool_log.jsonl").read_text().splitlines()]


# ---- the tool list ----

def test_tool_list():
    async def go():
        async with Client(evcam_server.mcp) as client:
            return await client.list_tools()
    tools = {t.name: t for t in asyncio.run(go())}
    assert set(tools) == TOOLS
    assert all(t.description for t in tools.values())
    assert "never the raw events" in tools["record"].description
    assert tools["set_bias"].input_schema["required"] == ["name", "value"]


def test_camera_info_says_fake():
    (info,) = run(("camera_info", {}))
    assert info["ok"] and info["fake"] is True and info["serial"] == "FAKE"


# ---- a whole measurement ----

def test_hot_pixels_mask_record_spectrum(tmp_path):
    (hot,) = run(("find_hot_pixels", {}))
    assert hot["ok"] and sorted((h["x"], h["y"]) for h in hot["hot_pixels"]) == sorted(fake.HOT_PIXELS)
    pixels = [[h["x"], h["y"]] for h in hot["hot_pixels"]]
    masked, rec, settings = run(("mask_pixels", {"pixels": pixels}),
                                ("record", {"seconds": 2.0, "note": "test"}),
                                ("get_settings", {}))
    # each run() is a new server run, and the camera is reopened: the mask is set in the same run as record
    assert masked["ok"] and masked["slots_used"] == 2 and masked["slots_total"] == 64
    assert rec["ok"] and store.ID_PATTERN.fullmatch(rec["recording_id"]) and "warning" not in rec
    assert all((p["x"], p["y"]) not in fake.HOT_PIXELS for p in rec["summary"]["top_pixels"])
    assert settings["mask"] == pixels
    (spec,) = run(("spectrum", {"recording_id": rec["recording_id"]}))
    assert spec["ok"] and abs(spec["dominant_hz"] - 240) <= spec["resolution_hz"]
    assert spec["fundamental_hz"] == spec["dominant_hz"] and "warning" not in spec  # 2 s: no short warning
    assert spec["recording_id"] == rec["recording_id"] and spec["peaks"][0]["snr"] >= 8


def test_record_returns_no_raw_events(tmp_path):
    (rec,) = run(("record", {"seconds": 1.0}))
    assert set(rec) == {"ok", "recording_id", "summary", "warning"}  # warning: the hot pixels are not masked
    assert len(json.dumps(rec)) < 5_000  # a summary, not ~70,000 events
    events, meta = store.load(rec["recording_id"], root=tmp_path)  # the events are in the file
    assert events.size == rec["summary"]["n_events"] > 0
    assert meta["fake"] is True and meta["seconds"] == 1.0
    assert meta["settings"]["masked_pixels"] == 0 and meta["settings"]["roi"] is None
    assert meta["recorded_s"] == 1.0 and "stopped after" not in rec["warning"]  # not stopped early


def test_record_warns_about_active_hot_pixels():
    (rec,) = run(("record", {"seconds": 0.5}))
    assert rec["ok"] and "2 pixel(s) above 20 kHz still active" in rec["warning"]
    assert all(str([x, y]) in rec["warning"] for x, y in fake.HOT_PIXELS)
    assert "mask_pixels" in rec["warning"]


def test_record_no_warning_when_the_hot_pixels_are_masked():
    _, rec = run(("mask_pixels", {"pixels": [list(p) for p in fake.HOT_PIXELS]}), ("record", {"seconds": 0.5}))
    assert rec["ok"] and "warning" not in rec


def test_record_that_stops_early_reports_the_real_duration(tmp_path, monkeypatch):
    # first real-camera UC2 run: 30 M events in 0.17 s, but the summary said "seconds: 2.0, 14.99 Mev/s"
    monkeypatch.setattr(limits, "MAX_EVENTS", 5_000)  # the fake camera makes ~70,000 events per second
    (rec,) = run(("record", {"seconds": 1.0}))
    s = rec["summary"]
    assert s["n_events"] == 5_000 and 0 < s["seconds"] < 0.2
    assert s["rate_mev_s"] == round(5_000 / s["seconds"] / 1e6, 6)  # rate over the real time
    assert (f"stopped after {s['seconds']:.2f} s at 5000 events ({5_000 / s['seconds'] / 1e6:.1f} Mev/s): too much "
            "light or activity; set an ROI, reduce the light or raise the biases, then record again") in rec["warning"]
    assert "2 pixel(s) above 20 kHz still active" in rec["warning"]  # both warnings, joined by " | "
    _, meta = store.load(rec["recording_id"], root=tmp_path)
    assert meta["seconds"] == 1.0 and meta["recorded_s"] == s["seconds"]


def fake_record(monkeypatch, keep_until_us, last_record):
    """FakeEventCamera.record that keeps only events before keep_until_us and reports `last_record`."""
    real = FakeEventCamera.record

    def record(self, seconds):
        ev = real(self, seconds)
        self.last_record = last_record
        return ev[ev["t"] < keep_until_us]

    monkeypatch.setattr(FakeEventCamera, "record", record)


def test_early_stop_is_decided_by_the_recorded_time_not_the_event_count(monkeypatch):
    # third record of the real UC2 run: 29.95 M events (fewer than MAX_EVENTS) in 0.27 s; the summary said 2.0 s
    fake_record(monkeypatch, 270_000, {"recorded_s": 0.27, "wall_s": 17.4, "stopped_early": True})
    (rec,) = run(("record", {"seconds": 2.0}))
    assert rec["summary"]["seconds"] == 0.27 and rec["summary"]["n_events"] < limits.MAX_EVENTS
    assert f"stopped after 0.27 s at {rec['summary']['n_events']} events" in rec["warning"]


def test_zero_events_much_faster_than_real_time_gives_a_warning(monkeypatch):
    # second record of the real UC2 run: "recorded 2.00 s of camera time in 0.02 s wall", 0 events, ok
    fake_record(monkeypatch, 0, {"recorded_s": 2.0, "wall_s": 0.02, "stopped_early": False})
    (rec,) = run(("record", {"seconds": 2.0}))
    assert rec["ok"] and rec["summary"]["n_events"] == 0
    assert rec["warning"] == ("0 events in 0.02 s of wall time for 2 s of camera time: the camera stream delivered "
                              "no new data; record again")


def test_zero_events_in_real_time_is_no_warning(monkeypatch):
    # e.g. an ROI where nothing happens: 0 events, but the recording took its full time
    fake_record(monkeypatch, 0, {"recorded_s": 2.0, "wall_s": 2.1, "stopped_early": False})
    (rec,) = run(("record", {"seconds": 2.0}))
    assert rec["summary"]["n_events"] == 0 and "warning" not in rec


def test_spectrum_says_which_settings_the_recording_was_made_with():
    # the bug of the UC2 run on 2026-10-15: "spectrum of the masked recording" used the recording made BEFORE masking
    before, _, _, after = run(("record", {"seconds": 0.5}),
                              ("mask_pixels", {"pixels": [list(p) for p in fake.HOT_PIXELS]}),
                              ("set_bias", {"name": "bias_diff_off", "value": 20}),
                              ("record", {"seconds": 0.5}))
    spec_before, spec_after = run(("spectrum", {"recording_id": before["recording_id"]}),
                                  ("spectrum", {"recording_id": after["recording_id"]}))
    zero = {"bias_diff": 0, "bias_diff_on": 0, "bias_diff_off": 0, "bias_fo": 0, "bias_hpf": 0, "bias_refr": 0}
    assert spec_before["recorded_with"] == {"masked_pixels": 0, "biases": zero, "roi": None}
    assert spec_after["recorded_with"] == {"masked_pixels": 2, "biases": {**zero, "bias_diff_off": 20}, "roi": None}


def test_spectrum_of_an_old_recording_has_no_settings(tmp_path):
    ev = np.zeros(2, dtype=EVENT_DTYPE)
    ev["p"], ev["t"] = [1, 0], [0, 1_000_000]
    rid = store.save(ev, {"seconds": 1.0}, root=tmp_path)  # saved without "settings", like the files before the fix
    (spec,) = run(("spectrum", {"recording_id": rid}))
    assert spec["ok"] and spec["recorded_with"] is None


def test_steady_scene_has_no_peak(monkeypatch):
    monkeypatch.setenv("EVCAM_FAKE_SCENE", "steady")
    (rec,) = run(("record", {"seconds": 2.0}))
    (spec,) = run(("spectrum", {"recording_id": rec["recording_id"]}))
    assert spec["ok"] and spec["dominant_hz"] is None


# ---- settings ----

def test_settings_set_and_clear():
    results = run(("set_bias", {"name": "bias_diff_off", "value": 20}),
                  ("mask_pixels", {"pixels": [[1, 2]]}),
                  ("set_roi", {"x": 0, "y": 0, "width": 640, "height": 360}),
                  ("get_settings", {}),
                  ("clear_mask", {}),
                  ("clear_roi", {}),
                  ("get_settings", {}))
    assert all(r["ok"] for r in results)
    bias, _, roi, before, _, _, after = results
    assert bias == {"ok": True, "name": "bias_diff_off", "value": 20}
    assert roi["roi"] == [0, 0, 640, 360]
    assert before["biases"]["bias_diff_off"] == 20 and before["mask"] == [[1, 2]] and before["roi"] == [0, 0, 640, 360]
    assert before["bias_ranges"]["bias_diff"] == {"recommended": [-25, 23], "allowed": [-150, 200]}
    assert after["mask"] == [] and after["roi"] is None


# ---- errors ----

@pytest.mark.parametrize("tool, args, msg", [
    ("set_bias", {"name": "bias_diff", "value": 50}, "recommended range"),
    ("set_bias", {"name": "bias_foo", "value": 0}, "unknown bias"),
    ("record", {"seconds": 10}, "between 0.05 and 5.0"),
    ("find_hot_pixels", {"seconds": 0}, "between 0.05 and 5.0"),
    ("mask_pixels", {"pixels": [[i, 0] for i in range(65)]}, "mask slots"),
    ("mask_pixels", {"pixels": [[1280, 0]]}, "outside the sensor"),
    ("set_roi", {"x": 1000, "y": 0, "width": 400, "height": 10}, "does not fit"),
    ("spectrum", {"recording_id": "../x"}, "bad recording_id"),
    ("spectrum", {"recording_id": "20261015-143012-a3f9c1"}, "no recording"),
])
def test_errors_are_returned_not_raised(tool, args, msg):
    (r,) = run((tool, args))
    assert r["ok"] is False and msg in r["error"]


def test_wrong_argument_type_is_refused_by_fastmcp(tmp_path):
    (r,) = call(("set_bias", {"name": "bias_fo", "value": "abc"}))
    assert r.is_error and "valid integer" in r.content[0].text
    assert not (tmp_path / "tool_log.jsonl").exists()  # the tool never ran, so nothing was logged


def test_a_failed_open_is_retried(monkeypatch):
    real_open = FakeEventCamera.open
    tries = []

    def open_fails_once(self):
        tries.append(1)
        if len(tries) == 1:
            raise RuntimeError("no event camera found")
        return real_open(self)

    monkeypatch.setattr(FakeEventCamera, "open", open_fails_once)
    first, second = run(("camera_info", {}), ("camera_info", {}))
    assert first == {"ok": False, "error": "RuntimeError: no event camera found"}
    assert second["ok"]


# ---- one camera, opened on first use ----

def test_camera_is_opened_once_and_closed_at_the_end(opens):
    results = run(("camera_info", {}), ("get_settings", {}), ("record", {"seconds": 0.1}))
    assert all(r["ok"] for r in results)
    assert len(opens) == 1
    assert evcam_server._cam is None  # closed when the server stopped


def test_spectrum_does_not_open_the_camera(tmp_path, opens):
    ev = np.zeros(2, dtype=EVENT_DTYPE)
    ev["p"], ev["t"] = [1, 0], [0, 1_000_000]
    rid = store.save(ev, {"seconds": 1.0}, root=tmp_path)
    (spec,) = run(("spectrum", {"recording_id": rid}))
    assert spec["ok"] and opens == []


# ---- the log ----

def test_every_call_is_logged(tmp_path):
    run(("set_bias", {"name": "bias_fo", "value": 10}), ("set_bias", {"name": "bias_diff", "value": 50}),
        ("record", {"seconds": 0.1}))
    lines = log_lines(tmp_path)
    assert [line["tool"] for line in lines] == ["set_bias", "set_bias", "record"]
    assert [line["ok"] for line in lines] == [True, False, True]
    assert lines[0]["args"] == {"name": "bias_fo", "value": 10}
    assert "recommended range" in lines[1]["result"]["error"]
    assert lines[2]["result"]["recording_id"] and lines[2]["duration_s"] >= 0
    assert all("time" in line for line in lines)
