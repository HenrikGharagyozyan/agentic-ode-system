"""Plot data from a SAVED recording (runs/evcam/<id>.npz): event images, the event rate and the brightness change.

This reads the file that the MCP record tool saved; it never opens the camera. It is not a live stream: the browser
labels the result with the recording id, its duration and its time.

- image: events per 4 x 4 pixel bin (320 x 180 bins for 1280 x 720), ON and OFF separately, scaled to 0..255 on a
  log scale against the 99.9th percentile, so two hot pixels do not make everything else black. Sent as base64 bytes,
  row by row from the top (y = 0 is the first row, as on the sensor).
- rate: ON and OFF events per second in time bins (10 ms, or longer so there are at most 500 bins).
- hot_pixels: pixels above 20 kHz (the same rule as the record tool's warning), so the browser can mark them.
- brightness_curve: the brightness change of the whole sensor (analysis.brightness_curve: each ON = +Δ, each
  OFF = -Δ, leaky integral with tau 0.2 s), in units of Δ. At most 2000 points (the mean of each block of bins), plus
  the first 100 ms at the full 100 us resolution as "zoom".
- delta_image: the sum of the polarity per 4 x 4 bin (analysis.delta_image), linear and symmetric: int8, the 99th
  percentile of |value| (non-empty bins) becomes +-127. > 0 = brighter, < 0 = darker.
- time_image: the mean event time per 4 x 4 bin (analysis.time_image) as uint8: 0 = no events, 1..255 = start..end
  of the recording.
- edge_images: "where did the light change". A delta image over a full period sums to ~0, so the rising and the
  falling edges are shown separately: delta images summed over the rise windows and over the fall windows of
  analysis.edge_windows() (first 5 periods, 1/4 period wide). f0 is always computed offline with the current
  analysis.spectrum(fmin_hz=2); a saved spectrum f0 of the same recording is shown next to it, with a note when they
  differ by more than the resolution (old runs were analysed by older code). Same encoding as delta_image, one
  colour scale for both. No f0 (no spectrum peak) or no full period: no edge_images, but edge_images_reason.
- edge_windows: [{t0_us, t1_us, kind}] so the browser can shade them on the brightness curve ([] without f0).
Hot pixels (above 20 kHz) are left out of these, so older recordings without a mask look clean too.

window_preview(): the delta image of one custom time window (t0_ms .. t1_ms), checked against the recording length.
"""

from __future__ import annotations

import base64
import json
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np

from agentic_ode.hardware.devices.evcam import analysis, limits, store

BIN_PX = 4
MAX_RATE_BINS = 500
MIN_RATE_BIN_US = 10_000
CACHE_SIZE = 8  # a preview of a large real recording takes a moment; keep the last few
CURVE_BIN_US = 100  # brightness curve: the same bins as the spectrum
LEAK_TAU_S = 0.2
MAX_CURVE_POINTS = 2000
ZOOM_US = 100_000  # the first 100 ms at full resolution
DELTA_PERCENTILE = 99
SPECTRUM_FMIN_HZ = 2.0  # the offline f0 for the edge windows (as in the lumen baseline)
EDGE_PERIODS = 5


class WindowError(ValueError):
    """A custom time window outside the recording, or t0 >= t1 (the API answers 422)."""


def _scale(counts: np.ndarray) -> tuple[bytes, float]:
    """counts -> uint8 on a log scale; the 99.9th percentile of the non-empty bins becomes 255."""
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return bytes(counts.size), 0.0
    top = float(np.percentile(nonzero, 99.9))
    scaled = np.log1p(counts) / np.log1p(max(top, 1.0)) * 255
    return np.clip(np.round(scaled), 0, 255).astype(np.uint8).tobytes(), top


def _curve(t_s: np.ndarray, level: np.ndarray) -> dict:
    block = -(-t_s.size // MAX_CURVE_POINTS)  # ceil division: so there are at most MAX_CURVE_POINTS blocks
    starts = np.arange(0, t_s.size, block)
    sizes = np.diff(np.append(starts, t_s.size))  # the last block can be shorter
    zoom = ZOOM_US // CURVE_BIN_US
    return {
        "bin_s": CURVE_BIN_US / 1e6,
        "step_s": block * CURVE_BIN_US / 1e6,
        "leak_tau_s": LEAK_TAU_S,
        "unit": "contrast steps (delta)",
        "t_s": (np.add.reduceat(t_s, starts) / sizes).round(6).tolist(),
        "delta_l": (np.add.reduceat(level, starts) / sizes).round(6).tolist(),
        "zoom": {"t_s": t_s[:zoom].round(6).tolist(), "delta_l": level[:zoom].round(6).tolist()},
    }


def _top(*images: np.ndarray) -> float:
    """The DELTA_PERCENTILE-th percentile of |value| over the non-empty bins of all images (0 if all are empty)."""
    nonzero = np.concatenate([np.abs(img[img != 0]) for img in images])
    return float(np.percentile(nonzero, DELTA_PERCENTILE)) if nonzero.size else 0.0


def _signed(img: np.ndarray, top: float) -> dict:
    """A signed image as int8 bytes, linear and symmetric: +-top becomes +-127 (clipped beyond)."""
    scaled = np.clip(np.round(img / max(top, 1.0) * 127), -127, 127).astype(np.int8)
    return {"data": base64.b64encode(scaled.tobytes()).decode("ascii"), "scale_top": top,
            "min": int(img.min()), "max": int(img.max()),
            "scale": f"linear, {DELTA_PERCENTILE}th percentile of |value| = 127"}


def _delta(events: np.ndarray, seconds: float) -> dict:
    img = analysis.delta_image(events, limits.SENSOR_W, limits.SENSOR_H, bin_px=BIN_PX, seconds=seconds)
    return _signed(img, _top(img))


def _edges(events: np.ndarray, seconds: float, curve: tuple[np.ndarray, np.ndarray],
           saved_f0_hz: float | None) -> tuple[dict | None, str | None, list[dict]]:
    """(edge_images, edge_images_reason, edge_windows): exactly one of the first two is None."""
    try:
        spec = analysis.spectrum(events, fmin_hz=SPECTRUM_FMIN_HZ)
    except ValueError as e:  # e.g. too short for the band
        return None, f"No flicker frequency: {e}. The delta image shows the whole recording instead.", []
    f0 = spec["fundamental_hz"]
    if f0 is None:
        return None, (f"No flicker frequency: the spectrum has no peak above the noise (snr >= {spec['min_snr']:g}) "
                      f"between {SPECTRUM_FMIN_HZ:g} and {spec['fmax_hz']:g} Hz, so there are no edges to find. "
                      "The delta image shows the whole recording instead."), []
    windows = analysis.edge_windows(events, f0, EDGE_PERIODS, seconds=seconds, curve=curve)
    listed = [{"t0_us": t0, "t1_us": t1, "kind": kind} for t0, t1, kind in windows]
    if not windows:
        return None, (f"The recording ({seconds:.3f} s) is shorter than one period of f0 = {f0:g} Hz, so there are "
                      "no edge windows. The delta image shows the whole recording instead."), listed
    images = analysis.edge_delta_images(events, limits.SENSOR_W, limits.SENSOR_H, f0, bin_px=BIN_PX,
                                        seconds=seconds, windows=windows)
    top = _top(images["rise"], images["fall"])  # one colour scale, so the two images can be compared
    note = None
    if saved_f0_hz is not None and abs(saved_f0_hz - f0) > spec["resolution_hz"]:
        note = (f"The saved spectrum of this recording says {saved_f0_hz:g} Hz, the current analysis {f0:g} Hz "
                f"(resolution {spec['resolution_hz']:g} Hz). The saved value is outdated (older analysis code); "
                f"the edges use {f0:g} Hz.")
    n_rise = sum(k == "rise" for _, _, k in windows)
    n_fall = len(windows) - n_rise
    return {
        "f0_hz": f0,
        "f0_source": f"offline spectrum (fmin {SPECTRUM_FMIN_HZ:g} Hz, current analysis code)",
        "saved_f0_hz": saved_f0_hz,
        "f0_note": note,
        "resolution_hz": spec["resolution_hz"],
        "window_ms": round(1e3 / f0 / analysis.EDGE_WINDOW_PART, 3),
        "n_periods": max(n_rise, n_fall),
        "n_rise": n_rise,
        "n_fall": n_fall,
        "rise": _signed(images["rise"], top),
        "fall": _signed(images["fall"], top),
    }, None, listed


def _time(events: np.ndarray, seconds: float) -> dict:
    img = analysis.time_image(events, limits.SENSOR_W, limits.SENSOR_H, bin_px=BIN_PX, seconds=seconds)
    frac = np.clip(img / seconds, 0, 1)  # 0 = start, 1 = end of the recording
    coded = np.where(np.isnan(img), 0, 1 + np.round(np.nan_to_num(frac) * 254)).astype(np.uint8)
    return {"data": base64.b64encode(coded.tobytes()).decode("ascii"), "seconds": seconds,
            "scale": "0 = no events, 1..255 = mean event time from start to end"}


def preview(events: np.ndarray, meta: dict, saved_f0_hz: float | None = None) -> dict:
    """All plot data of one saved recording. saved_f0_hz: the f0 of a saved spectrum of it (only shown, see above)."""
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

    result = {
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
        "brightness_curve": None,
        "delta_image": None,
        "time_image": None,
        "edge_windows": [],
    }
    if not (seconds and events.size):
        result["edge_images_reason"] = "No events: no edges to find."
        return result
    curve = analysis.brightness_curve(events, CURVE_BIN_US, LEAK_TAU_S, seconds=seconds)
    edges, reason, windows = _edges(events, seconds, curve, saved_f0_hz)
    result.update(brightness_curve=_curve(*curve), delta_image=_delta(events, seconds),
                  time_image=_time(events, seconds), edge_windows=windows)
    if edges is not None:
        result["edge_images"] = edges
    else:
        result["edge_images_reason"] = reason
    return result


def window_preview(events: np.ndarray, meta: dict, t0_ms: float, t1_ms: float) -> dict:
    """The delta image of the custom window t0_ms <= t < t1_ms (ms from the start of the recording).
    Raises WindowError unless both are finite numbers and 0 <= t0_ms < t1_ms <= the recording length."""
    seconds = float(meta.get("recorded_s") or meta.get("seconds") or 0)
    length_ms = seconds * 1e3
    for name, v in (("t0_ms", t0_ms), ("t1_ms", t1_ms)):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise WindowError(f"{name} must be a finite number, got {v!r}")
    if not 0 <= t0_ms < t1_ms <= length_ms:
        raise WindowError(f"need 0 <= t0_ms < t1_ms <= {length_ms:g} (the recording length), "
                          f"got t0_ms={t0_ms:g}, t1_ms={t1_ms:g}")
    img = analysis.delta_image(events, limits.SENSOR_W, limits.SENSOR_H, t0_us=round(t0_ms * 1e3),
                               t1_us=round(t1_ms * 1e3), bin_px=BIN_PX, seconds=seconds)
    return {"recording_id": meta.get("recording_id"), "t0_ms": t0_ms, "t1_ms": t1_ms,
            "delta_image": _signed(img, _top(img))}


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


def spectrum_f0_of(camera_records: list[dict], recording_id: str) -> float | None:
    """The fundamental_hz of the LAST successful spectrum call of one experiment on this recording, or None."""
    f0 = None
    for rec in camera_records:
        if rec.get("tool") == "spectrum" and rec.get("ok"):
            try:
                result = json.loads(rec["result_line"])
            except (ValueError, KeyError, TypeError):
                continue
            if not isinstance(result, dict) or result.get("recording_id") != recording_id:
                continue
            value = result.get("fundamental_hz")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                f0 = float(value)
    return f0


class PreviewCache:
    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: OrderedDict[tuple[str, float | None], dict] = OrderedDict()
        self._events: tuple[str, np.ndarray, dict] | None = None  # the last loaded recording (custom windows)

    def _load(self, recording_id: str) -> tuple[np.ndarray, dict]:
        """store.load() checks the id (no paths) and raises if it is missing. Keeps ONE recording in memory, so
        moving the custom window does not read the file again (a real recording: ~0.5 GB, ~1 s to read)."""
        if self._events is None or self._events[0] != recording_id:
            self._events = None  # free the old one first
            events, meta = store.load(recording_id, root=self.root)
            self._events = (recording_id, events, meta)
        return self._events[1], self._events[2]

    def get(self, recording_id: str, saved_f0_hz: float | None = None) -> dict:
        """The preview of one saved recording. The key includes saved_f0_hz: a spectrum saved later changes it."""
        key = (recording_id, saved_f0_hz)
        if key not in self._cache:
            self._cache[key] = preview(*self._load(recording_id), saved_f0_hz)
            while len(self._cache) > CACHE_SIZE:
                self._cache.popitem(last=False)
        self._cache.move_to_end(key)
        return self._cache[key]

    def window(self, recording_id: str, t0_ms: float, t1_ms: float) -> dict:
        """The delta image of one custom window (window_preview); not cached, it is quick."""
        return window_preview(*self._load(recording_id), t0_ms, t1_ms)
