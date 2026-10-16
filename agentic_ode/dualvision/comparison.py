"""Cross-laboratory comparison of two lumen.flicker.v1 results: one from the Frame Lab, one from the Event Lab.

Pure and offline: it reads two JSON-like dicts and returns a new JSON-compatible dict. No hardware, no network, no
model calls, and the input dicts are never changed. The output has no raw events, no frames and no hardware.serial.

Eligibility of one record (the first group that matches wins, in this order; all matching reasons are kept):
- invalid: the record is not a well-formed lumen.flicker.v1 result of this lab: wrong schema, lab or sample_kind;
  a required field is missing or has the wrong type; a number is not finite or is negative, or is zero where it
  must be positive (reference_hz, requested_s, actual_s, fps, exposure_us, f0_hz, resolution_hz).
- not_recoverable: the frequency cannot be read from this record:
  REFERENCE_AT_OR_ABOVE_NYQUIST (frame only, fps known: reference_hz >= fps / 2, even if f0 seems to match), or
  F0_NOT_DETECTED (f0_hz is null: a legitimate "no flicker found"; its acquisition status is kept as it is).
- provisional: the measurement is kept, but it is not good enough for a firm comparison:
  EARLY_STOP, SHORT_RECORDING (actual_s < 1 s), RESOLUTION_UNKNOWN, FPS_UNKNOWN and EXPOSURE_UNKNOWN (frame only;
  Nyquist is then not checked), HARMONIC_AMBIGUITY (f0 within resolution_hz of k * reference_hz or reference_hz / k,
  k = 2..15), ACQUISITION_STATUS_NOT_OK (status is not "ok"), SIGNIFICANT_WARNING.
  Every Event Lab warning is significant. A Frame Lab warning is significant unless its code (the text before the
  first ":") is in FRAME_INFORMATIONAL_WARNINGS and reference_hz is strictly below a known Nyquist.
- valid: none of the above.

Eligibility judges the quality of the measurement, not its result: a valid record whose f0_hz is far from
reference_hz is a real sensor failure, shown by its frequency error, not an invalid input.
"""

from __future__ import annotations

import math

SCHEMA = "lumen.flicker.v1"
RESULT_SCHEMA = "lumen.dualvision.comparison.v1"
SAMPLE_KIND = {"frame": "frames", "event": "events"}  # the only sample_kind each lab may have
TOP_LEVEL_FIELDS = ("schema", "lab", "run_id", "recording_id", "local_time", "hardware", "source", "acquisition",
                    "estimate", "status", "warnings", "analysis")
SNR_DEFINITIONS = {
    "event": "spectral amplitude ratio (peak amplitude / median amplitude), Event Lab",
    "frame": "spectral power ratio, Frame Lab DV-01",
}
FRAME_INFORMATIONAL_WARNINGS = frozenset({"ALIASING_NOT_EXCLUDED"})  # DV-01 always gives it
PUBLIC_HARDWARE_FIELDS = ("model", "sensor", "real")  # hardware.serial is never copied
MIN_DURATION_S = 1.0  # shorter recordings give a coarse frequency resolution
MAX_HARMONIC = 15  # the highest k tested by the harmonic-ambiguity check

LIMITATIONS = [
    "reference_hz is the generator setting entered by hand, not a calibrated optical measurement of the light.",
    "local_time is the local clock of each computer: a label only, it gives no physical clock synchronisation.",
    "run_id and recording_id are local to each lab; only an explicit, externally assigned trial_id pairs records.",
    "Event Lab snr is a spectral amplitude ratio and Frame Lab snr is a spectral power ratio: they are not "
    "comparable, so the two sensors are not ranked by snr.",
    "A harmonic relation between f0_hz and reference_hz is only a hint of confusion; f0_hz is never replaced.",
    "ALIASING_NOT_EXCLUDED below Nyquist is kept as a warning; it does not prove that aliasing is impossible.",
    "The Nyquist check needs a measured frame fps; event records have no frame rate and are not checked.",
    "One trial does not show an absolute frequency limit of either sensor.",
]

_MISSING = object()  # marks a field that is not in the record (different from a field that is null)


def _get(record: dict, path: str):
    """The value at a dotted path such as "source.reference_hz"; _MISSING if a part of the path is absent."""
    value = record
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return _MISSING
        value = value[key]
    return value


def _number(record: dict, path: str, problems: list[str], nullable: bool = False, positive: bool = True):
    """The number at path, or None. Adds a problem if it is missing, null (unless nullable), not a number, not
    finite, negative, or zero when positive=True. The bad value itself is not copied into the problem text."""
    value = _get(record, path)
    if value is _MISSING:
        problems.append(f"{path} is missing")
    elif value is None:
        if not nullable:
            problems.append(f"{path} is null")
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        problems.append(f"{path} is not a number ({type(value).__name__})")
    elif not math.isfinite(value):
        problems.append(f"{path} is not finite")
    elif value < 0 or (positive and value == 0):
        problems.append(f"{path} must be {'> 0' if positive else '>= 0'}")
    else:
        return value
    return None


def _text(record: dict, path: str, problems: list[str]) -> str | None:
    """The string at path, or None (and a problem) if it is missing or not a string."""
    value = _get(record, path)
    if isinstance(value, str):
        return value
    problems.append(f"{path} is missing" if value is _MISSING else f"{path} is not a string")
    return None


def _harmonic_relation(f0: float, reference: float, tolerance: float) -> dict | None:
    """The first k (2..MAX_HARMONIC) with f0 within tolerance of k * reference ("harmonic") or of reference / k
    ("subharmonic"); None if there is none. A match is only a hint of harmonic confusion, never a proof."""
    for k in range(2, MAX_HARMONIC + 1):
        if abs(f0 - k * reference) <= tolerance:
            return {"relation": "harmonic", "k": k}
        if abs(f0 - reference / k) <= tolerance:
            return {"relation": "subharmonic", "k": k}
    return None


def assess(record: dict, lab: str) -> dict:
    """Check one lumen.flicker.v1 record of lab ("frame" or "event") and return its public summary: identifiers,
    detected frequency, frequency error, acquisition status, eligibility with its reasons, and every warning."""
    if lab not in SAMPLE_KIND:
        raise ValueError(f"lab must be 'frame' or 'event', got {lab!r}")
    problems: list[str] = []
    if not isinstance(record, dict):
        problems.append("the record is not a JSON object")
        record = {}
    for key in TOP_LEVEL_FIELDS:
        if key not in record:
            problems.append(f"{key} is missing")

    # 1. read and check every field (a bad field gives None and a problem)
    if "schema" in record and record["schema"] != SCHEMA:
        problems.append(f"schema is not {SCHEMA}")
    if "lab" in record and record["lab"] != lab:
        problems.append(f"lab is not {lab!r}")
    sample_kind = _text(record, "acquisition.sample_kind", problems)
    if sample_kind is not None and sample_kind != SAMPLE_KIND[lab]:
        problems.append(f"acquisition.sample_kind is not {SAMPLE_KIND[lab]!r}")
    run_id = _text(record, "run_id", problems)
    recording_id = _text(record, "recording_id", problems)
    local_time = _text(record, "local_time", problems)
    status = _text(record, "status", problems)
    analysis = _text(record, "analysis", problems)
    source_type = _text(record, "source.type", problems)
    set_by = _text(record, "source.set_by", problems)
    reference = _number(record, "source.reference_hz", problems)
    requested_s = _number(record, "acquisition.requested_s", problems)
    actual_s = _number(record, "acquisition.actual_s", problems)
    samples = _number(record, "acquisition.samples", problems, positive=False)
    if samples is not None and samples != int(samples):
        problems.append("acquisition.samples is not a whole number")
    fps = _number(record, "acquisition.fps", problems, nullable=True)
    exposure_us = _number(record, "acquisition.exposure_us", problems, nullable=True)
    early_stop = _get(record, "acquisition.early_stop")
    if not isinstance(early_stop, bool):
        problems.append("acquisition.early_stop is not true or false")
        early_stop = None
    f0 = _number(record, "estimate.f0_hz", problems, nullable=True)
    resolution = _number(record, "estimate.resolution_hz", problems, nullable=True)
    snr = _number(record, "estimate.snr", problems, nullable=True, positive=False)
    hardware = record.get("hardware")
    if "hardware" in record and not isinstance(hardware, dict):
        problems.append("hardware is not a JSON object")
    raw_warnings = record.get("warnings", [])  # missing: already a problem above
    if not (isinstance(raw_warnings, list) and all(isinstance(w, str) for w in raw_warnings)):
        problems.append("warnings is not a list of strings")
        raw_warnings = []
    trial_id = record.get("trial_id")  # optional, set outside both labs; anything but a non-empty string is ignored
    trial_id = trial_id if isinstance(trial_id, str) and trial_id else None

    # 2. derived values (only from numbers that passed the checks)
    nyquist = fps / 2 if lab == "frame" and fps is not None else None
    below_nyquist = nyquist is not None and reference is not None and reference < nyquist
    alias = abs(reference - fps * round(reference / fps)) if nyquist is not None and reference is not None else None
    warnings = []
    for text in raw_warnings:
        informational = lab == "frame" and text.split(":", 1)[0].strip() in FRAME_INFORMATIONAL_WARNINGS
        warnings.append({"text": text, "significant": not (informational and below_nyquist)})
    harmonic = None
    if f0 is not None and reference is not None and resolution is not None:
        harmonic = _harmonic_relation(f0, reference, resolution)
    error = None
    if not problems and f0 is not None and reference is not None:
        signed = f0 - reference
        error = {"signed_hz": signed, "abs_hz": abs(signed), "relative": abs(signed) / reference,
                 "within_resolution": None if resolution is None else abs(signed) <= resolution}

    # 3. eligibility: the worst group with a reason wins
    not_recoverable = []
    if nyquist is not None and reference is not None and reference >= nyquist:
        not_recoverable.append("REFERENCE_AT_OR_ABOVE_NYQUIST")
    if _get(record, "estimate.f0_hz") is None:  # present and null; a missing key is a problem above
        not_recoverable.append("F0_NOT_DETECTED")
    provisional = []
    if early_stop:
        provisional.append("EARLY_STOP")
    if actual_s is not None and actual_s < MIN_DURATION_S:
        provisional.append("SHORT_RECORDING")
    if _get(record, "estimate.resolution_hz") is None:
        provisional.append("RESOLUTION_UNKNOWN")
    if lab == "frame" and _get(record, "acquisition.fps") is None:
        provisional.append("FPS_UNKNOWN")
    if lab == "frame" and _get(record, "acquisition.exposure_us") is None:
        provisional.append("EXPOSURE_UNKNOWN")
    if harmonic is not None:
        provisional.append("HARMONIC_AMBIGUITY")
    if status is not None and status != "ok":
        provisional.append("ACQUISITION_STATUS_NOT_OK")
    if any(w["significant"] for w in warnings):
        provisional.append("SIGNIFICANT_WARNING")
    if problems:
        eligibility = "invalid"
    elif not_recoverable:
        eligibility = "not_recoverable"
    elif provisional:
        eligibility = "provisional"
    else:
        eligibility = "valid"
    if f0 is not None:
        f0_status = "measured"
    elif "F0_NOT_DETECTED" in not_recoverable:
        f0_status = "not_detected"
    else:
        f0_status = "invalid"  # missing or a bad value; never turned into 0

    public_hardware = {}
    for key in PUBLIC_HARDWARE_FIELDS:
        value = hardware.get(key) if isinstance(hardware, dict) else None
        public_hardware[key] = value if isinstance(value, (str, bool)) else None
    return {
        "lab": lab,
        "run_id": run_id,
        "recording_id": recording_id,
        "trial_id": trial_id,
        "local_time": local_time,
        "hardware": public_hardware,
        "source": {"type": source_type, "set_by": set_by},
        "reference_hz": reference,
        "acquisition_status": status,
        "acquisition": {"requested_s": requested_s, "actual_s": actual_s, "samples": samples,
                        "sample_kind": sample_kind, "fps": fps, "exposure_us": exposure_us,
                        "early_stop": early_stop, "nyquist_hz": nyquist, "expected_alias_hz": alias},
        "f0_hz": f0,
        "f0_status": f0_status,
        "resolution_hz": resolution,
        "frequency_error": error,
        "harmonic_relation": harmonic,
        "snr": {"value": snr, "definition": SNR_DEFINITIONS[lab]},
        "eligibility": eligibility,
        "reasons": not_recoverable + provisional,
        "problems": problems,
        "warnings": warnings,
        "analysis": analysis,
    }


def compare(frame_record: dict, event_record: dict) -> dict:
    """Compare one Frame Lab and one Event Lab lumen.flicker.v1 record. Returns a JSON-compatible dict;
    comparison_valid is true only if the references match, an explicit trial_id pairs the records and both
    records are valid. If the references differ (or one is unknown), no frequency error is given."""
    frame = assess(frame_record, "frame")
    event = assess(event_record, "event")
    notes = []

    if frame["reference_hz"] is None or event["reference_hz"] is None:
        reference_check = "unknown"
        notes.append("a reference_hz is missing or invalid: no comparison is computed")
    elif frame["reference_hz"] == event["reference_hz"]:
        reference_check = "match"
    else:
        reference_check = "mismatch"
        notes.append(f"incompatible: the reference frequencies differ (frame {frame['reference_hz']} Hz, "
                     f"event {event['reference_hz']} Hz): no comparison is computed")
    if reference_check != "match":
        frame["frequency_error"] = event["frequency_error"] = None

    if frame["trial_id"] is None or event["trial_id"] is None:
        pairing = "unpaired"
        notes.append("no explicit trial_id on both records: they are not known to be the same trial")
    elif frame["trial_id"] == event["trial_id"]:
        pairing = "paired"
    else:
        pairing = "trial_mismatch"
        notes.append("the trial_ids differ: the records are not the same trial")

    for side in (frame, event):
        if side["eligibility"] != "valid":
            why = ", ".join(side["reasons"] + side["problems"])
            notes.append(f"the {side['lab']} record is {side['eligibility']}: {why}")

    if reference_check != "match":
        error_evidence = "not_computed"
    elif pairing == "paired":
        error_evidence = "paired_trial"
    else:
        error_evidence = "unpaired_evidence"  # per-sensor errors are shown, but they are not a paired comparison
    return {
        "schema": RESULT_SCHEMA,
        "reference_hz": frame["reference_hz"] if reference_check == "match" else None,
        "reference_check": reference_check,
        "pairing": {"status": pairing, "frame_trial_id": frame["trial_id"], "event_trial_id": event["trial_id"],
                    "basis": "explicit trial_id only"},
        "frame": frame,
        "event": event,
        "error_evidence": error_evidence,
        "snr_comparable": False,
        "comparison_valid": reference_check == "match" and pairing == "paired"
        and frame["eligibility"] == "valid" and event["eligibility"] == "valid",
        "notes": notes,
        "limitations": list(LIMITATIONS),
    }
