"""EventCamera.record() against a scripted HAL stream (no hardware). Bugs of the real-camera UC2 run, 2026-10-15:
1) after an early stop, the next record returned 0 events in 0.02 s; 2) an early stop was missed because the
warm-up window filter left fewer than MAX_EVENTS events."""

import numpy as np
import pytest

from agentic_ode.hardware.devices.evcam import driver, limits
from agentic_ode.hardware.devices.evcam.driver import EventCamera
from agentic_ode.hardware.devices.evcam.events import EVENT_DTYPE

PER_BUFFER = 1000  # events per buffer; one buffer = 1 ms of camera time, so 1 Mev/s


class ScriptedStream:
    """A fake HAL event stream. Each read gives a fresh buffer of 1 ms of camera time. With stale_once, the FIRST
    stop() leaves one old buffer in the driver and the camera time jumps 20 s ahead: what we think happened after
    the early stop on the real camera (Python decoded far slower than the camera produced data)."""

    def __init__(self, stale_once=False):
        self.queue: list[np.ndarray] = []
        self.next_t = 0
        self.stale_once = stale_once
        self.calls: list[str] = []

    def buffer(self, t0: int) -> np.ndarray:
        ev = np.zeros(PER_BUFFER, dtype=EVENT_DTYPE)
        ev["t"] = t0 + np.arange(PER_BUFFER) * 1000 // PER_BUFFER
        ev["x"], ev["p"] = np.arange(PER_BUFFER) % 1280, np.arange(PER_BUFFER) % 2
        return ev

    def start(self):
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")
        if self.stale_once:
            self.stale_once = False
            self.queue.insert(0, self.buffer(self.next_t))  # old data still waiting in the driver
            self.next_t += 20_000_000  # the camera ran ahead while Python was decoding

    def wait_next_buffer(self) -> int:
        self.calls.append("wait")
        return 1

    def get_latest_raw_data(self) -> np.ndarray:
        self.calls.append("raw")
        if self.queue:
            return self.queue.pop(0)
        buf = self.buffer(self.next_t)
        self.next_t += 1000
        return buf


class ScriptedDecoder:
    def __init__(self, stream: ScriptedStream, callback):
        self.stream, self.callback, self.last = stream, callback, 0

    def decode(self, raw: np.ndarray) -> None:
        self.stream.calls.append("decode")
        self.last = int(raw["t"].max())
        self.callback(raw.copy())  # like the real CD decoder: the callback gets each decoded batch

    def get_last_timestamp(self) -> int:
        return self.last


class ScriptedDevice:
    def __init__(self, stream: ScriptedStream, cam: EventCamera):
        self.stream, self.decoder = stream, ScriptedDecoder(stream, cam._chunks.append)

    def get_i_events_stream(self):
        return self.stream

    def get_i_events_stream_decoder(self):
        return self.decoder


def camera(stream: ScriptedStream) -> EventCamera:
    cam = EventCamera()
    cam._dev = ScriptedDevice(stream, cam)  # "open" without hardware
    return cam


@pytest.fixture(autouse=True)
def short_flush(monkeypatch):
    monkeypatch.setattr(driver, "FLUSH_S", 0.01)


def test_normal_record_covers_all_the_time_and_does_not_restart():
    stream = ScriptedStream()
    cam = camera(stream)
    ev = cam.record(0.1)
    assert ev.size == 100 * PER_BUFFER and ev["t"].min() >= 0 and ev["t"].max() < 100_000
    assert cam.last_record["recorded_s"] == 0.1 and cam.last_record["stopped_early"] is False
    assert stream.calls.count("start") == 1


def test_early_stop_is_detected_although_fewer_than_max_events_are_kept(monkeypatch):
    # stops after 51 buffers (51,000 > 50,500), but the first buffer after the warm-up straddles the window start,
    # so the filter keeps only 50,001: fewer than MAX_EVENTS, like 29.95 M on the real camera
    monkeypatch.setattr(limits, "MAX_EVENTS", 50_500)
    cam = camera(ScriptedStream())
    ev = cam.record(0.1)
    assert ev.size == 50_001 < limits.MAX_EVENTS
    assert cam.last_record["stopped_early"] is True
    assert cam.last_record["recorded_s"] == (int(ev["t"].max()) + 1) / 1e6 < 0.1


def test_early_stop_restarts_the_stream_cleanly(monkeypatch):
    monkeypatch.setattr(limits, "MAX_EVENTS", 50_500)
    stream = ScriptedStream()
    camera(stream).record(0.1)
    first_stop = stream.calls.index("stop")
    after = stream.calls[first_stop + 1:]
    # start, read and drop raw data (no decoding), stop
    assert after[0] == "start" and after[-1] == "stop" and "raw" in after and "decode" not in after


def test_stale_buffer_after_an_early_stop_gives_zero_events_without_the_restart(monkeypatch):
    # the bug: the old buffer becomes the start of the next record, the next (current) buffer is far past its end
    monkeypatch.setattr(limits, "MAX_EVENTS", 50_500)
    monkeypatch.setattr(EventCamera, "_restart_stream", lambda self, stream: None)
    cam = camera(ScriptedStream(stale_once=True))
    cam.record(0.1)
    monkeypatch.setattr(limits, "MAX_EVENTS", 30_000_000)  # the next record must not stop early itself
    ev = cam.record(0.05)
    assert ev.size == 0 and cam.last_record["recorded_s"] == 0.05  # like "2.00 s of camera time", nothing kept


def test_the_restart_drains_the_stale_buffer(monkeypatch):
    monkeypatch.setattr(limits, "MAX_EVENTS", 50_500)
    cam = camera(ScriptedStream(stale_once=True))
    cam.record(0.1)  # stops early -> restart reads the old buffer away
    monkeypatch.setattr(limits, "MAX_EVENTS", 30_000_000)  # the next record must not stop early itself
    ev = cam.record(0.05)
    assert ev.size == 50 * PER_BUFFER and ev["t"].max() < 50_000
    assert cam.last_record == {"recorded_s": 0.05, "wall_s": cam.last_record["wall_s"], "stopped_early": False}
