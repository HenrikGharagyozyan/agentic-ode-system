"""MCP server for the IDS event camera (FastMCP, stdio). Each tool is one camera action or one reading.

Run:   python -m agentic_ode.mcp_servers.evcam_server                      (real camera: run `evcam` first)
       EVCAM_FAKE=1 python -m agentic_ode.mcp_servers.evcam_server         (FakeEventCamera, no hardware)
       EVCAM_FAKE_SCENE=dark|pwm|steady chooses the fake scene (default pwm)

- One camera object for the whole server: opened on the first tool call that needs it, kept open, closed when the
  server stops. A lock lets only one tool use it at a time (FastMCP runs each tool call in a worker thread).
- Every tool returns {"ok": true, ...} or {"ok": false, "error": "..."}; the server never crashes on a bad call.
  (Arguments of the wrong type, e.g. a string for an int, are refused by FastMCP before the tool runs.)
- record() saves the events with store.save() and returns only a recording_id and a summary, never raw events.
- Every call is appended to runs/evcam/tool_log.jsonl: time, tool, arguments, result, duration.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime

from fastmcp import FastMCP

from agentic_ode.hardware.evcam import analysis, store
from agentic_ode.hardware.evcam.connector import EventCamera
from agentic_ode.hardware.evcam.fake import FakeEventCamera

RECORDINGS_DIR = store.DEFAULT_ROOT  # runs/evcam/ (git-ignored); the tests point this to a temporary folder
LOG_FILE = RECORDINGS_DIR / "tool_log.jsonl"

_cam: EventCamera | FakeEventCamera | None = None  # the one camera; None until the first tool that needs it
_lock = threading.Lock()


def _camera() -> EventCamera | FakeEventCamera:
    """The open camera; opens it on first use. EVCAM_FAKE=1 -> FakeEventCamera."""
    global _cam
    if _cam is None:
        if os.environ.get("EVCAM_FAKE") == "1":
            cam = FakeEventCamera(scene=os.environ.get("EVCAM_FAKE_SCENE", "pwm"))
        else:
            cam = EventCamera()
        cam.open()  # the real camera needs ~9 s here (settle); if it fails, _cam stays None and the next call retries
        _cam = cam
    return _cam


def _close_camera() -> None:
    global _cam
    with _lock:
        if _cam is not None:
            _cam.close()
            _cam = None


def _log(tool: str, args: dict, result: dict, seconds: float) -> None:
    record = {"time": datetime.now().astimezone().isoformat(timespec="milliseconds"), "tool": tool, "args": args,
              "ok": result["ok"], "result": result, "duration_s": round(seconds, 3)}
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")  # default=str: never fail on an unusual value


def _run(tool: str, args: dict, action) -> dict:
    """Run action() under the lock and return {"ok": True, **its result}, or {"ok": False, "error": ...}.
    Every call is logged."""
    t0 = time.monotonic()
    try:
        with _lock:
            result = {"ok": True, **action()}
    except Exception as e:  # every error goes back to the model as text; the server keeps running
        result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    _log(tool, args, result, time.monotonic() - t0)
    return result


@asynccontextmanager
async def _lifespan(server):
    try:
        yield {}
    finally:
        _close_camera()  # release the camera so another process can open it


mcp = FastMCP("evcam", instructions="Tools for an IDS event camera (Sony IMX636, 1280 x 720 pixels, no lens): "
              "set biases, mask pixels, set a region of interest, record events and analyse light flicker.",
              lifespan=_lifespan)


# ---- tools ----

@mcp.tool
def camera_info() -> dict:
    """Camera identity: serial, sensor, width x height, temperature, current biases, and whether it is the fake camera."""  # noqa: E501
    def action():
        cam = _camera()
        return {**cam.info(), "fake": isinstance(cam, FakeEventCamera)}
    return _run("camera_info", {}, action)


@mcp.tool
def get_settings() -> dict:
    """Current settings (biases, masked pixels, ROI) and the allowed range of each bias."""
    def action():
        cam = _camera()
        return {"biases": cam.get_biases(), "mask": cam.state["mask"], "roi": cam.state["roi"],
                "bias_ranges": cam.bias_ranges()}
    return _run("get_settings", {}, action)


@mcp.tool
def set_bias(name: str, value: int) -> dict:
    """Set one bias (bias_diff, bias_diff_on, bias_diff_off, bias_fo, bias_hpf, bias_refr) and return the value read back; higher bias_diff_on/off = less noise, less sensitive."""  # noqa: E501
    return _run("set_bias", {"name": name, "value": value},
                lambda: {"name": name, "value": _camera().set_bias(name, value)})


@mcp.tool
def find_hot_pixels(seconds: float = 0.5, top: int = 64, min_rate_hz: float = 20000) -> dict:
    """Make a short recording (not saved) and list pixels firing at least min_rate_hz, busiest first; it does not mask them."""  # noqa: E501
    def action():
        events = _camera().record(seconds)
        return {"seconds": seconds, "hot_pixels": analysis.hot_pixels(events, seconds, top, min_rate_hz)}
    return _run("find_hot_pixels", {"seconds": seconds, "top": top, "min_rate_hz": min_rate_hz}, action)


@mcp.tool
def mask_pixels(pixels: list[tuple[int, int]]) -> dict:
    """Silence the given (x, y) pixels (at most 64 in total); returns the newly masked pixels and the slots used."""
    def action():
        cam = _camera()
        new = cam.mask_pixels(pixels)
        return {"newly_masked": new, "slots_used": len(cam.state["mask"]), "slots_total": 64}
    return _run("mask_pixels", {"pixels": pixels}, action)


@mcp.tool
def clear_mask() -> dict:
    """Unmask all masked pixels."""
    def action():
        _camera().clear_mask()
        return {"mask": []}
    return _run("clear_mask", {}, action)


@mcp.tool
def set_roi(x: int, y: int, width: int, height: int) -> dict:
    """Only pixels inside this window (sensor 1280 x 720) send events; one window at most."""
    def action():
        _camera().set_roi(x, y, width, height)
        return {"roi": [x, y, width, height]}
    return _run("set_roi", {"x": x, "y": y, "width": width, "height": height}, action)


@mcp.tool
def clear_roi() -> dict:
    """Remove the ROI: the whole sensor sends events again."""
    def action():
        _camera().clear_roi()
        return {"roi": None}
    return _run("clear_roi", {}, action)


@mcp.tool
def record(seconds: float = 1.0, note: str = "") -> dict:
    """Record 0.05..5 s of events, save them, and return recording_id plus a summary (rate, ON fraction, active pixels, busiest pixels); never the raw events."""  # noqa: E501
    def action():
        cam = _camera()
        events = cam.record(seconds)  # checks 0.05..5 s itself
        meta = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": note,
            "seconds": seconds,
            "fake": isinstance(cam, FakeEventCamera),
            "camera": cam.info(),
            "state": cam.state,  # biases, mask and ROI at the time of the recording
        }
        recording_id = store.save(events, meta, root=RECORDINGS_DIR)
        return {"recording_id": recording_id, "summary": analysis.summary(events, seconds)}
    return _run("record", {"seconds": seconds, "note": note}, action)


@mcp.tool
def spectrum(recording_id: str, bin_us: int = 100, fmin_hz: float = 20, fmax_hz: float = 2000,
             min_snr: float = 8) -> dict:
    """Flicker spectrum of a saved recording: dominant frequency (null if no peak stands out of the noise), top peaks and harmonics, each with snr."""  # noqa: E501
    def action():  # reads the saved file only: does not open the camera
        events, _ = store.load(recording_id, root=RECORDINGS_DIR)
        return {"recording_id": recording_id,
                **analysis.spectrum(events, bin_us=bin_us, fmax_hz=fmax_hz, fmin_hz=fmin_hz, min_snr=min_snr)}
    return _run("spectrum", {"recording_id": recording_id, "bin_us": bin_us, "fmin_hz": fmin_hz,
                             "fmax_hz": fmax_hz, "min_snr": min_snr}, action)


if __name__ == "__main__":
    # stdout carries the MCP messages, so all logging goes to stderr
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    mcp.run()  # default transport: stdio
