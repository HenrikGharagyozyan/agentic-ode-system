"""EvcamInstrument: the IDS event camera as one Instrument (Model Hardware Standard) with 10 actions.

Two backends with the same methods: EventCamera (driver.py, the real camera through the SDK) and FakeEventCamera
(fake.py, made-up events). EVCAM_FAKE=1 chooses the fake one; EVCAM_FAKE_SCENE=dark|pwm|steady its scene.

- One camera object for the instrument: opened on the first action that needs it, kept open, closed by close().
  The backend is chosen at that moment, not when the instrument is made.
- record() saves the events with store.save() and returns only a recording_id and a summary, never raw events.
- Each method's docstring is the MCP tool description the model reads; its type hints are the tool's JSON schema.
"""

from __future__ import annotations

import inspect
import os
from datetime import datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from agentic_ode.hardware.devices.evcam import analysis, limits, store
from agentic_ode.hardware.devices.evcam.driver import EventCamera
from agentic_ode.hardware.devices.evcam.fake import FakeEventCamera
from agentic_ode.hardware.standard import ActionSpec, Instrument, Result

INSTRUCTIONS = ("Tools for an IDS event camera (Sony IMX636, 1280 x 720 pixels, no lens): set biases, mask pixels, "
                "set a region of interest, record events and analyse light flicker.")


class CameraBackend(Protocol):
    """The methods EventCamera and FakeEventCamera both have (written down here; nothing checks it at run time)."""

    state: dict
    last_record: dict | None

    def open(self) -> CameraBackend: ...
    def close(self) -> None: ...
    def info(self) -> dict: ...
    def get_biases(self) -> dict[str, int]: ...
    def bias_ranges(self) -> dict[str, dict]: ...
    def set_bias(self, name: str, value: int) -> int: ...
    def mask_pixels(self, pixels: list[tuple[int, int]]) -> list[tuple[int, int]]: ...
    def clear_mask(self) -> None: ...
    def set_roi(self, x: int, y: int, w: int, h: int) -> None: ...
    def clear_roi(self) -> None: ...
    def record(self, seconds: float) -> np.ndarray: ...


class EvcamInstrument(Instrument):
    name = "evcam"

    def __init__(self, recordings_dir: Path = store.DEFAULT_ROOT):
        self.recordings_dir = Path(recordings_dir)  # runs/evcam/ (git-ignored); the tests use a temporary folder
        self._cam: CameraBackend | None = None  # the one camera; None until the first action that needs it

    # ---- the Instrument contract ----

    def describe(self) -> dict:
        return {"name": self.name, "instructions": INSTRUCTIONS}

    def actions(self) -> list[ActionSpec]:
        def spec(handler, kind: str) -> ActionSpec:  # name and description come from the method itself
            return ActionSpec(handler.__name__, inspect.getdoc(handler), kind, handler)

        return [
            spec(self.camera_info, "read"),
            spec(self.get_settings, "read"),
            spec(self.set_bias, "setting"),
            spec(self.find_hot_pixels, "measurement"),
            spec(self.mask_pixels, "setting"),
            spec(self.clear_mask, "setting"),
            spec(self.set_roi, "setting"),
            spec(self.clear_roi, "setting"),
            spec(self.record, "measurement"),
            spec(self.spectrum, "measurement"),
        ]

    @property
    def is_open(self) -> bool:
        return self._cam is not None

    def camera(self) -> CameraBackend:
        """The open camera; opens it on first use. EVCAM_FAKE=1 -> FakeEventCamera."""
        if self._cam is None:
            if os.environ.get("EVCAM_FAKE") == "1":
                cam = FakeEventCamera(scene=os.environ.get("EVCAM_FAKE_SCENE", "pwm"))
            else:
                cam = EventCamera()
            # the real camera needs ~9 s here (settle); if it fails, _cam stays None and the next call retries
            cam.open()
            self._cam = cam
        return self._cam

    def close(self) -> None:
        """Release the camera so another process can open it."""
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    # ---- actions ----

    def camera_info(self) -> Result:
        """Camera identity: serial, sensor, width x height, temperature, current biases, and whether it is the fake camera."""  # noqa: E501
        cam = self.camera()
        return Result.success({**cam.info(), "fake": isinstance(cam, FakeEventCamera)})

    def get_settings(self) -> Result:
        """Current settings (biases, masked pixels, ROI) and the allowed range of each bias."""
        cam = self.camera()
        return Result.success({"biases": cam.get_biases(), "mask": cam.state["mask"], "roi": cam.state["roi"],
                               "bias_ranges": cam.bias_ranges()})

    def set_bias(self, name: str, value: int) -> Result:
        """Set one bias (bias_diff, bias_diff_on, bias_diff_off, bias_fo, bias_hpf, bias_refr) and return the value read back; higher bias_diff_on/off = less noise, less sensitive."""  # noqa: E501
        return Result.success({"name": name, "value": self.camera().set_bias(name, value)})

    def find_hot_pixels(self, seconds: float = 0.5, top: int = limits.MAX_MASK_SLOTS,
                        min_rate_hz: float = limits.MAX_PIXEL_RATE_HZ) -> Result:
        """Make a short recording (not saved) and list pixels firing at least min_rate_hz, busiest first; it does not mask them."""  # noqa: E501
        events = self.camera().record(seconds)
        hot = analysis.hot_pixels(events, seconds, top, min_rate_hz)
        return Result.success({"seconds": seconds, "hot_pixels": hot})

    def mask_pixels(self, pixels: list[tuple[int, int]]) -> Result:
        """Silence the given (x, y) pixels (at most 64 in total); returns the newly masked pixels and the slots used."""
        cam = self.camera()
        new = cam.mask_pixels(pixels)
        return Result.success({"newly_masked": new, "slots_used": len(cam.state["mask"]),
                               "slots_total": limits.MAX_MASK_SLOTS})

    def clear_mask(self) -> Result:
        """Unmask all masked pixels."""
        self.camera().clear_mask()
        return Result.success({"mask": []})

    def set_roi(self, x: int, y: int, width: int, height: int) -> Result:
        """Only pixels inside this window (sensor 1280 x 720) send events; one window at most."""
        self.camera().set_roi(x, y, width, height)
        return Result.success({"roi": [x, y, width, height]})

    def clear_roi(self) -> Result:
        """Remove the ROI: the whole sensor sends events again."""
        self.camera().clear_roi()
        return Result.success({"roi": None})

    def record(self, seconds: float = 1.0, note: str = "") -> Result:
        """Record 0.05..5 s of events, save them, and return recording_id plus a summary (rate, ON fraction, active pixels, busiest pixels); never the raw events. Adds a "warning" if it stopped early (too many events; the summary then covers the real, shorter time) or if pixels above 20 kHz are still active."""  # noqa: E501
        cam = self.camera()
        events = cam.record(seconds)  # checks 0.05..5 s itself; stops early at limits.MAX_EVENTS events
        # what the camera really did; ONE rule for an early stop: less camera time than asked
        recorded_s, wall_s = cam.last_record["recorded_s"], cam.last_record["wall_s"]
        stopped_early = recorded_s < seconds
        meta = {
            "created": datetime.now().astimezone().isoformat(timespec="seconds"),
            "note": note,
            "seconds": seconds,  # asked for
            "recorded_s": recorded_s,  # really covered (less than seconds if stopped early)
            "fake": isinstance(cam, FakeEventCamera),
            "camera": cam.info(),
            "state": cam.state,  # biases, mask and ROI at the time of the recording
            # the settings this recording was made with; spectrum returns them as "recorded_with"
            "settings": {"masked_pixels": len(cam.state["mask"]), "biases": cam.get_biases(),
                         "roi": cam.state["roi"]},
        }
        recording_id = store.save(events, meta, root=self.recordings_dir)
        result = {"recording_id": recording_id, "summary": analysis.summary(events, recorded_s)}
        warnings = []
        if events.size == 0 and wall_s < 0.5 * seconds:  # a real recording takes at least its camera time
            warnings.append(f"0 events in {wall_s:.2f} s of wall time for {seconds:g} s of camera time: the camera "
                            "stream delivered no new data; record again")
        if stopped_early:  # the summary above already uses the real duration
            warnings.append(f"stopped after {recorded_s:.2f} s at {events.size} events "
                            f"({events.size / recorded_s / 1e6:.1f} Mev/s): too much light or activity; set an ROI, "
                            "reduce the light or raise the biases, then record again")
        # safety net: a pixel above 20 kHz is a defect, not light (find_hot_pixels once missed one, 2026-10-15)
        hot = analysis.hot_pixels(events, recorded_s)
        if hot:
            pixels = [[h["x"], h["y"]] for h in hot]
            warnings.append(f"{len(hot)} pixel(s) above 20 kHz still active (defects, not light): {pixels}; "
                            "mask them with mask_pixels and record again")
        return Result.success(result, warnings)

    def spectrum(self, recording_id: str, bin_us: int = 100, fmin_hz: float = 20, fmax_hz: float = 2000,
                 min_snr: float = 8) -> Result:
        """Flicker spectrum of a saved recording: report fundamental_hz as the flicker frequency (dominant_hz is only the strongest peak, which can be a harmonic; both null if no peak stands out of the noise); top peaks and harmonics of the fundamental, each with snr; "recorded_with" gives the settings the recording was made with; a "warning" if the recording is shorter than 1 s."""  # noqa: E501
        # reads the saved file only: does not open the camera
        events, meta = store.load(recording_id, root=self.recordings_dir)
        spec = analysis.spectrum(events, bin_us=bin_us, fmax_hz=fmax_hz, fmin_hz=fmin_hz, min_snr=min_snr)
        warning = spec.pop("warning", None)  # analysis puts it last; Result.to_wire() puts it last again
        recorded_with = meta.get("settings")  # None for recordings saved before 2026-10-15 16:00
        return Result.success({"recording_id": recording_id, "recorded_with": recorded_with, **spec},
                              [warning] if warning else [])
