"""Plot data from a SAVED recording (runs/evcam/<id>.npz): an event-count image and the event rate over time.

This reads the file that the MCP record tool saved; it never opens the camera. It is not a live stream: the browser
labels the result with the recording id, its duration and its time.

- image: events per 4 x 4 pixel bin (320 x 180 bins for 1280 x 720), ON and OFF separately, scaled to 0..255 on a
  log scale against the 99.9th percentile, so two hot pixels do not make everything else black. Sent as base64 bytes,
  row by row from the top (y = 0 is the first row, as on the sensor).
- rate: ON and OFF events per second in time bins (10 ms, or longer so there are at most 500 bins).
- hot_pixels: pixels above 20 kHz (the same rule as the record tool's warning), so the browser can mark them.
"""

from __future__ import annotations

import base64
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from agentic_ode.hardware.devices.evcam import analysis, limits, store

BIN_PX = 4
MAX_RATE_BINS = 500
MIN_RATE_BIN_US = 10_000
CACHE_SIZE = 8  # a preview of a large real recording takes a moment; keep the last few


def _scale(counts: np.ndarray) -> tuple[bytes, float]:
    """counts -> uint8 on a log scale; the 99.9th percentile of the non-empty bins becomes 255."""
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return bytes(counts.size), 0.0
    top = float(np.percentile(nonzero, 99.9))
    scaled = np.log1p(counts) / np.log1p(max(top, 1.0)) * 255
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8).tobytes(), top


def preview(events: np.ndarray, meta: dict) -> dict:
    seconds = float(meta.get("recorded_s") or meta.get("seconds") or 0) or None
    width, height = limits.SENSOR_W // BIN_PX, limits.SENSOR_H // BIN_PX
    images = {}
    for name, polarity in (("on", 1), ("off", 0)):
        ev = events[events["p"] == polarity]
        index = (ev["y"].astype(np.int64) // BIN_PX) * width + ev["x"].astype(np.int64) // BIN_PX
        data, top = _scale(np.bincount(index, minlength=width * height)[: width * height])
        images[name] = {"data": base64.b64encode(data).decode("ascii"), "scale_top": top}

    rate = None
    if events.size and seconds:
        span_us = int(seconds * 1e6)
        bin_us = max(MIN_RATE_BIN_US, -(-span_us // MAX_RATE_BINS))  # ceil division
        n_bins = -(-span_us // bin_us)
        t = np.clip(events["t"] // bin_us, 0, n_bins - 1)
        on = np.bincount(t[events["p"] == 1], minlength=n_bins)
        off = np.bincount(t[events["p"] == 0], minlength=n_bins)
        per_s = 1e6 / bin_us
        rate = {"bin_s": bin_us / 1e6, "on_hz": (on * per_s).round(1).tolist(),
                "off_hz": (off * per_s).round(1).tolist()}

    return {
        "recording_id": meta.get("recording_id"),
        "created": meta.get("created"),
        "note": meta.get("note", ""),
        "fake": meta.get("fake"),
        "seconds_asked": meta.get("seconds"),
        "recorded_s": seconds,
        "n_events": int(events.size),
        "settings": meta.get("settings"),
        "image": {"width": width, "height": height, "bin_px": BIN_PX, "sensor_width": limits.SENSOR_W,
                  "sensor_height": limits.SENSOR_H, "scale": "log1p, 99.9th percentile = 255", **images},
        "rate": rate,
        "hot_pixels": analysis.hot_pixels(events, seconds) if seconds and events.size else [],
    }


def recordings_of(camera_records: list[dict]) -> list[str]:
    """The recording ids that successful record calls of one experiment returned (its CameraRecords)."""
    ids = []
    for rec in camera_records:
        if rec.get("tool") == "record" and rec.get("ok"):
            try:
                rid = json.loads(rec["result_line"]).get("recording_id")
            except (ValueError, AttributeError):
                continue
            if isinstance(rid, str):
                ids.append(rid)
    return ids


class PreviewCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: OrderedDict[str, dict] = OrderedDict()

    def get(self, recording_id: str) -> dict:
        """The preview of one saved recording. store.load() checks the id (no paths) and raises if it is missing."""
        if recording_id not in self._cache:
            events, meta = store.load(recording_id, root=self.root)
            self._cache[recording_id] = preview(events, meta)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        self._cache.move_to_end(recording_id)
        return self._cache[recording_id]
