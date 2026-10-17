"""Offline Event Lab exporter (DV-EV-EXPORT-01): one saved recording -> one lumen.flicker.v1 document.

saved .npz (from store.save()) -> store.load() -> analysis.spectrum(fmin_hz=2, fmax_hz=2000) -> a JSON-compatible dict.
It reads a saved file only: no camera, no SDK, no MCP, no model, no network. The recording file is never changed.

The science is analysis.spectrum(), unchanged: f0_hz is its fundamental_hz (null stays null, the operator's
reference_hz never replaces it), resolution_hz is its resolution_hz, and snr is the snr of its peak at the
fundamental (null when the fundamental is not one of its peaks: no snr is computed here). The analyzer's warning is
copied word for word. fmin_hz is 2 on purpose: the MCP default of 20 Hz would hide a 10 Hz fundamental.

Acquisition (EvcamInstrument.record() is a duration request): requested_s is meta "seconds" (asked for), actual_s is
meta "recorded_s" (camera time really covered), samples is the number of saved events. early_stop uses the
instrument's one rule: recorded_s < seconds. A recording without seconds, recorded_s or n_events is refused
(ExportError): nothing is guessed, and an incomplete recording is never written as a complete one.

What the operator must give (nothing of it is read from the file, a file name, a file date or the sensor clock):
reference_hz, local_time with a UTC offset, run_id and source_type; trial_id, duty and distance_cm are optional.
trial_id is written only when it is given.

Privacy: the output has no hardware serial, no absolute path and no events. hardware.real is true only when the
recording says it was not made by the fake camera (meta "fake" is false) and nothing in it says "simulated".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

from agentic_ode.dualvision.comparison import SCHEMA
from agentic_ode.hardware.devices.evcam import analysis, limits, store

FMIN_HZ = 2.0  # explicit: the default 20 Hz of analysis.spectrum() and of the MCP tool is above a 10 Hz fundamental
FMAX_HZ = 2000.0
FAKE_SERIAL = "FAKE"  # FakeEventCamera.info()["serial"]; only compared, never copied


class ExportError(ValueError):
    """A recording or an operator value that cannot be exported truthfully. The message never holds a serial or an
    absolute path."""


def _text(name: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExportError(f"{name} must be a non-empty text")
    return value


def _positive(name: str, value) -> float:
    # NaN fails every comparison, so `not 0 < value < inf` also rejects it
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < math.inf:
        raise ExportError(f"{name} must be a positive finite number")
    return value


def _local_time(value) -> str:
    """value if it is an ISO 8601 date and time with a UTC offset, e.g. 2026-10-16T14:34:49+04:00."""
    parsed = None
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            parsed = None
    if parsed is None or parsed.tzinfo is None:  # raised here, outside except: no chained parser error
        raise ExportError("local_time must be ISO 8601 with a UTC offset, e.g. 2026-10-16T14:34:49+04:00")
    return value


def _real(meta: dict) -> bool | None:
    """hardware.real from the stored evidence: False for the fake camera, True only for a recording that says it
    is not fake and has no simulation mark, None when the recording does not say."""
    fake = meta.get("fake")
    if fake is True:
        return False
    camera = meta.get("camera") if isinstance(meta.get("camera"), dict) else {}
    simulated = camera.get("serial") == FAKE_SERIAL or "simulated" in str(camera.get("sensor", ""))
    return True if fake is False and not simulated else None


def _load(recording_id: str, root: Path):
    """(events, meta) from store.load(); its errors become ExportError without the folder path."""
    error = None
    try:
        events, meta = store.load(recording_id, root=root)
    except ValueError:
        error = f"bad recording_id {recording_id!r}; expected the form 20261015-143012-a3f9c1"
    except FileNotFoundError:
        error = f"no recording {recording_id} in the recordings folder"
    if error:
        raise ExportError(error)
    if not isinstance(meta, dict):
        raise ExportError("the recording meta is not a JSON object")
    return events, meta


def export_event_flicker(recording_id: str, root: Path, *, reference_hz: float, local_time: str, run_id: str,
                         source_type: str, trial_id: str | None = None, duty: float | None = None,
                         distance_cm: float | None = None) -> dict:
    """The lumen.flicker.v1 document of the saved recording <root>/<recording_id>.npz. Raises ExportError if an
    operator value or a mandatory recording field is missing or wrong, or if the analyzer rejects the events."""
    # 1. the operator's values, checked before the file is read
    reference_hz = _positive("reference_hz", reference_hz)
    local_time = _local_time(local_time)
    run_id = _text("run_id", run_id)
    source_type = _text("source_type", source_type)
    if trial_id is not None:
        trial_id = _text("trial_id", trial_id)
    if duty is not None and not (_positive("duty", duty) < 1):
        raise ExportError("duty must be a fraction between 0 and 1 (e.g. 0.5)")
    if distance_cm is not None:
        distance_cm = _positive("distance_cm", distance_cm)

    # 2. the recording and its mandatory metadata
    events, meta = _load(recording_id, Path(root))
    for key in ("seconds", "recorded_s", "n_events"):
        if key not in meta:
            raise ExportError(f"the recording meta has no {key!r}: it cannot be exported truthfully")
    if meta.get("recording_id") != recording_id:
        raise ExportError("the recording meta has another recording_id than its file name")
    requested_s = _positive("meta seconds", meta["seconds"])
    actual_s = _positive("meta recorded_s", meta["recorded_s"])
    if actual_s > requested_s:
        raise ExportError("the recording meta has recorded_s larger than seconds")
    n_events = meta["n_events"]
    if isinstance(n_events, bool) or not isinstance(n_events, int) or n_events != events.size:
        raise ExportError("the recording meta n_events does not match the saved events")
    early_stop = actual_s < requested_s  # the instrument's one rule for an early stop

    # 3. the analysis, unchanged; only fmin_hz and fmax_hz are set
    try:
        spec = analysis.spectrum(events, fmin_hz=FMIN_HZ, fmax_hz=FMAX_HZ)
    except ValueError as e:
        raise ExportError(f"the analyzer rejected the recording: {e}") from None
    f0 = spec["fundamental_hz"]
    snr = None if f0 is None else next((p["snr"] for p in spec["peaks"] if p["hz"] == f0), None)
    harmonics_hz = [h["hz"] for h in spec["harmonics"] if h["snr"] is not None and h["snr"] >= spec["min_snr"]]

    warnings = []
    if early_stop:
        cap = f" (the {limits.MAX_EVENTS} event cap was reached)" if n_events >= limits.MAX_EVENTS else ""
        warnings.append(f"stopped early: {n_events} events in {actual_s:.3f} s of the requested {requested_s:g} s"
                        f"{cap}")
    if spec.get("warning"):
        warnings.append(spec["warning"])

    real = _real(meta)
    camera = meta.get("camera") if isinstance(meta.get("camera"), dict) else {}
    sensor = camera.get("sensor") if isinstance(camera.get("sensor"), str) else None
    kind = {True: "real", False: "simulated", None: "undeclared"}[real]
    document = {
        "schema": SCHEMA,
        "lab": "event",
        "run_id": run_id,
        "recording_id": recording_id,
        "local_time": local_time,
        "hardware": {"model": None, "sensor": sensor, "real": real},  # the model is not stored; serial never copied
        "source": {"type": source_type, "reference_hz": reference_hz, "duty": duty, "distance_cm": distance_cm,
                   "set_by": "manual"},
        "acquisition": {"request_mode": "duration", "requested_s": requested_s, "actual_s": actual_s,
                        "samples": n_events, "sample_kind": "events", "fps": None, "exposure_us": None,
                        "early_stop": early_stop},
        "estimate": {"f0_hz": f0, "resolution_hz": spec["resolution_hz"], "snr": snr, "method": "fft",
                     "harmonics_hz": harmonics_hz},
        "status": "ok",  # saved and analysed; an early stop is shown by early_stop and its warning
        "warnings": warnings,
        "analysis": (f"offline re-analysis of a saved {kind} recording: analysis.spectrum(signal={spec['signal']}, "
                     f"bin_us={spec['bin_us']}, fmin_hz={spec['fmin_hz']:g}, fmax_hz={spec['fmax_hz']:g}, "
                     f"min_snr={spec['min_snr']:g})"),
    }
    if trial_id is not None:
        document["trial_id"] = trial_id
    return document


def write_json_create_only(document: dict, path: Path) -> None:
    """Write document as JSON to a new .json file; an existing file is never overwritten."""
    path = Path(path)
    if path.suffix != ".json":
        raise ExportError("the output file must end with .json")
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"  # before opening: no half-written file
    try:
        with path.open("x", encoding="utf-8", newline="\n") as f:  # "x": fails if the file exists
            f.write(text)
    except FileExistsError:
        raise ExportError(f"{path.name} already exists: refusing to overwrite it") from None
    except FileNotFoundError:
        raise ExportError(f"the folder for {path.name} does not exist") from None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export one saved Event Lab recording as a lumen.flicker.v1 JSON file "
                                            "(offline: the camera is not opened).")
    p.add_argument("--recording-id", required=True, help="e.g. 20261016-143449-ed6b74")
    p.add_argument("--recordings-dir", type=Path, required=True, help="folder with <recording-id>.npz")
    p.add_argument("--out", type=Path, required=True, help="new .json file; an existing file is not overwritten")
    p.add_argument("--reference-hz", type=float, required=True, help="the generator setting, entered by hand")
    p.add_argument("--local-time", required=True, help="ISO 8601 with UTC offset, e.g. 2026-10-16T14:34:49+04:00")
    p.add_argument("--run-id", required=True)
    p.add_argument("--source-type", required=True, help="e.g. LED_PWM")
    p.add_argument("--trial-id", help="an externally assigned trial id; only this pairs records")
    p.add_argument("--duty", type=float, help="duty cycle as a fraction, e.g. 0.5")
    p.add_argument("--distance-cm", type=float)
    args = p.parse_args(argv)
    try:
        document = export_event_flicker(args.recording_id, args.recordings_dir, reference_hz=args.reference_hz,
                                        local_time=args.local_time, run_id=args.run_id,
                                        source_type=args.source_type, trial_id=args.trial_id, duty=args.duty,
                                        distance_cm=args.distance_cm)
        write_json_create_only(document, args.out)
    except ExportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(f"wrote {args.out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
