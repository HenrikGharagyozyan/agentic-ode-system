"""Cross-laboratory comparison core (DV-H01): synthetic records only, plus a read-only check of report/lumen."""

import ast
import copy
import hashlib
import json
import math
import subprocess
import sys

import pytest

from agentic_ode.core.config import PROJECT_ROOT
from agentic_ode.dualvision import comparison
from agentic_ode.dualvision.comparison import assess, compare

FAKE_SERIAL = "TEST-SERIAL-0000"
_DELETE = object()  # a value for _change(): remove this top-level field


def event_record(**changes):
    """An Event Lab record with the shape of report/lumen/*.json (fake serial). changes: "block.key" -> value."""
    record = {
        "schema": "lumen.flicker.v1", "lab": "event", "run_id": "E-RUN", "recording_id": "E-REC",
        "local_time": "2026-10-16T14:27:58+04:00",
        "hardware": {"model": "IDS UE-39B0XCP-E-GL", "sensor": "IMX636", "serial": FAKE_SERIAL, "real": True},
        "source": {"type": "LED_PWM", "reference_hz": 50, "duty": None, "distance_cm": None, "set_by": "manual"},
        "acquisition": {"requested_s": 2.0, "actual_s": 2.0, "samples": 1_000_000, "sample_kind": "events",
                        "fps": None, "exposure_us": None, "early_stop": False},
        "estimate": {"f0_hz": 49.9, "resolution_hz": 0.5, "snr": 40.0, "method": "fft", "harmonics_hz": [150.0]},
        "status": "ok", "warnings": [], "analysis": "synthetic",
    }
    return _change(record, changes)


def frame_record(**changes):
    """A synthetic Frame Lab record: measured fps 200 (Nyquist 100 Hz) and the DV-01 informational warning."""
    record = {
        "schema": "lumen.flicker.v1", "lab": "frame", "run_id": "F-RUN", "recording_id": "F-REC",
        "local_time": "2026-10-16T12:27:58+02:00",
        "hardware": {"model": "MV-A7200MU130", "sensor": "frame", "serial": FAKE_SERIAL, "real": True},
        "source": {"type": "LED_PWM", "reference_hz": 50, "duty": None, "distance_cm": None, "set_by": "manual"},
        "acquisition": {"requested_s": 2.0, "actual_s": 2.0, "samples": 400, "sample_kind": "frames",
                        "fps": 200.0, "exposure_us": 1000.0, "early_stop": False},
        "estimate": {"f0_hz": 50.2, "resolution_hz": 0.5, "snr": 900.0},
        "status": "ok", "warnings": ["ALIASING_NOT_EXCLUDED: a single frame rate cannot exclude aliasing"],
        "analysis": "synthetic",
    }
    return _change(record, changes)


def _change(record, changes):
    for path, value in changes.items():
        block, _, key = path.partition(".")
        if key:
            record[block][key] = value
        elif value is _DELETE:
            del record[block]
        else:
            record[block] = value
    return record



def paired(frame=None, event=None):
    """compare() of the two records, both with trial_id "T1" (the default records when None)."""
    frame = frame or frame_record()
    event = event or event_record()
    frame["trial_id"] = event["trial_id"] = "T1"
    return compare(frame, event)


def all_keys(value):
    """Every dict key anywhere in a nested JSON value."""
    if isinstance(value, dict):
        return set(value) | set().union(*(all_keys(v) for v in value.values()))
    if isinstance(value, list):
        return set().union(*(all_keys(v) for v in value))
    return set()


def test_two_compatible_valid_measurements():
    r = paired()
    assert r["schema"] == "lumen.dualvision.comparison.v1"
    assert r["reference_check"] == "match" and r["reference_hz"] == 50
    assert r["pairing"]["status"] == "paired" and r["error_evidence"] == "paired_trial"
    assert r["frame"]["eligibility"] == r["event"]["eligibility"] == "valid"
    assert r["comparison_valid"] is True and r["notes"] == []
    assert r["frame"]["frequency_error"]["signed_hz"] == pytest.approx(0.2)
    assert r["event"]["frequency_error"]["abs_hz"] == pytest.approx(0.1)
    assert r["event"]["frequency_error"]["relative"] == pytest.approx(0.002)
    assert (r["frame"]["recording_id"], r["event"]["recording_id"]) == ("F-REC", "E-REC")


def test_reliable_measurement_with_wrong_f0_is_a_sensor_failure_not_invalid():
    r = paired(frame=frame_record(**{"estimate.f0_hz": 37.0}))
    assert r["frame"]["eligibility"] == "valid" and r["frame"]["f0_hz"] == 37.0  # the estimate is kept
    assert r["frame"]["frequency_error"]["signed_hz"] == pytest.approx(-13.0)
    assert r["frame"]["frequency_error"]["within_resolution"] is False
    assert r["comparison_valid"] is True  # a valid comparison that shows a failure


def test_event_lab_50hz_sample_shape():
    e = event_record(**{
        "acquisition.actual_s": 0.325007, "acquisition.samples": 30_000_000, "acquisition.early_stop": True,
        "estimate.f0_hz": 49.216, "estimate.resolution_hz": 3.076, "estimate.snr": 42.1,
        "warnings": ["stopped early: the 30000000 event cap was reached after 0.325 s of 2.0 s (light too bright)",
                     "the recording is only 0.33 s long: the frequency resolution is 3.1 Hz"]})
    a = assess(e, "event")
    assert a["eligibility"] == "provisional" and a["problems"] == []
    assert a["reasons"] == ["EARLY_STOP", "SHORT_RECORDING", "SIGNIFICANT_WARNING"]
    assert a["frequency_error"]["signed_hz"] == pytest.approx(-0.784)
    assert a["frequency_error"]["within_resolution"] is True
    assert a["harmonic_relation"] is None
    assert [w["text"] for w in a["warnings"]] == e["warnings"] and all(w["significant"] for w in a["warnings"])


def test_event_lab_10hz_sample_shape():
    e = event_record(**{
        "source.reference_hz": 10, "acquisition.actual_s": 1.549017, "acquisition.samples": 30_000_000,
        "acquisition.early_stop": True, "estimate.f0_hz": 10.329, "estimate.resolution_hz": 0.6455,
        "estimate.snr": 34.2, "warnings": ["stopped early: the 30000000 event cap was reached after 1.549 s"]})
    a = assess(e, "event")
    assert a["eligibility"] == "provisional"
    assert a["reasons"] == ["EARLY_STOP", "SIGNIFICANT_WARNING"]  # 1.55 s is not short
    assert a["frequency_error"]["signed_hz"] == pytest.approx(0.329)
    assert a["harmonic_relation"] is None


def test_event_early_stop_at_the_30m_cap_alone_makes_it_provisional():
    a = assess(event_record(**{"acquisition.samples": 30_000_000, "acquisition.early_stop": True}), "event")
    assert a["eligibility"] == "provisional" and a["reasons"] == ["EARLY_STOP"]
    assert a["f0_hz"] == 49.9 and a["frequency_error"] is not None  # the result is kept


def test_frame_with_sufficient_measured_fps():
    a = assess(frame_record(), "frame")
    assert a["eligibility"] == "valid" and a["reasons"] == []
    assert a["acquisition"]["nyquist_hz"] == 100.0
    assert a["warnings"] == [{"text": frame_record()["warnings"][0], "significant": False}]  # kept, not hidden


@pytest.mark.parametrize("fps", [100.0, 80.0, 30.0])  # Nyquist 50 (equal), 40 and 15 Hz
def test_frame_reference_at_or_above_nyquist_is_not_recoverable(fps):
    a = assess(frame_record(**{"acquisition.fps": fps, "estimate.f0_hz": 50.0}), "frame")  # f0 seems to match
    assert a["eligibility"] == "not_recoverable" and "REFERENCE_AT_OR_ABOVE_NYQUIST" in a["reasons"]
    assert a["f0_hz"] == 50.0  # the estimate is kept, not replaced
    assert a["warnings"][0]["significant"] is True  # ALIASING_NOT_EXCLUDED counts again at/above Nyquist
    r = paired(frame=frame_record(**{"acquisition.fps": fps}))
    assert r["comparison_valid"] is False


def test_frame_expected_alias():
    a = assess(frame_record(**{"acquisition.fps": 80.0}), "frame")
    assert a["acquisition"]["expected_alias_hz"] == pytest.approx(30.0)  # |50 - 80|


def test_missing_frame_fps_does_not_infer_nyquist():
    a = assess(frame_record(**{"acquisition.fps": None}), "frame")
    assert a["acquisition"]["nyquist_hz"] is None and a["acquisition"]["expected_alias_hz"] is None
    assert "REFERENCE_AT_OR_ABOVE_NYQUIST" not in a["reasons"]
    assert a["eligibility"] == "provisional"
    assert a["reasons"] == ["FPS_UNKNOWN", "SIGNIFICANT_WARNING"]  # the aliasing warning is significant now


def test_missing_frame_exposure_is_provisional():
    a = assess(frame_record(**{"acquisition.exposure_us": None}), "frame")
    assert a["eligibility"] == "provisional" and a["reasons"] == ["EXPOSURE_UNKNOWN"]


def test_missing_f0_is_never_zero_or_success():
    a = assess(event_record(**{"estimate.f0_hz": None}), "event")  # a legitimate "no flicker found"
    assert a["eligibility"] == "not_recoverable" and a["reasons"] == ["F0_NOT_DETECTED"]
    assert a["f0_hz"] is None and a["f0_status"] == "not_detected" and a["frequency_error"] is None
    assert a["acquisition_status"] == "ok"  # the original status is kept
    assert paired(event=event_record(**{"estimate.f0_hz": None}))["comparison_valid"] is False
    e = event_record()
    del e["estimate"]["f0_hz"]  # no key at all: malformed
    a = assess(e, "event")
    assert a["eligibility"] == "invalid" and a["f0_status"] == "invalid" and a["f0_hz"] is None


@pytest.mark.parametrize("f0, relation, k", [(150.0, "harmonic", 3), (100.2, "harmonic", 2), (25.0, "subharmonic", 2)])
def test_harmonic_ambiguity_is_a_diagnostic(f0, relation, k):
    a = assess(frame_record(**{"estimate.f0_hz": f0}), "frame")
    assert a["harmonic_relation"] == {"relation": relation, "k": k}
    assert a["eligibility"] == "provisional" and "HARMONIC_AMBIGUITY" in a["reasons"]
    assert a["f0_hz"] == f0  # never replaced by reference_hz


def test_harmonic_ambiguity_warning_is_significant():
    w = ["ALIASING_NOT_EXCLUDED: always", "HARMONIC_AMBIGUITY: the strongest peak may be 3 x f0"]
    a = assess(frame_record(warnings=w), "frame")
    assert [x["significant"] for x in a["warnings"]] == [False, True]
    assert a["eligibility"] == "provisional" and a["reasons"] == ["SIGNIFICANT_WARNING"]


@pytest.mark.parametrize("text", ["DROPPED_FRAMES: 12 of 400", "TIMING_JITTER: 2.1 ms", "SOMETHING_NEW",
                                  "aliasing_not_excluded"])
def test_other_and_unknown_frame_warnings_are_significant(text):
    a = assess(frame_record(warnings=[text]), "frame")
    assert a["warnings"] == [{"text": text, "significant": True}] and a["eligibility"] == "provisional"


def test_mismatched_reference_frequencies_are_incompatible():
    r = paired(event=event_record(**{"source.reference_hz": 10, "estimate.f0_hz": 10.3}))
    assert r["reference_check"] == "mismatch" and r["reference_hz"] is None
    assert r["frame"]["frequency_error"] is None and r["event"]["frequency_error"] is None
    assert r["error_evidence"] == "not_computed" and r["comparison_valid"] is False
    assert any(n.startswith("incompatible") for n in r["notes"])


def test_invalid_reference_gives_unknown_check():
    r = paired(frame=frame_record(**{"source.reference_hz": None}))
    assert r["reference_check"] == "unknown" and r["event"]["frequency_error"] is None
    assert r["comparison_valid"] is False


@pytest.mark.parametrize("path, value", [
    ("source.reference_hz", math.nan), ("source.reference_hz", math.inf), ("source.reference_hz", -50),
    ("source.reference_hz", 0), ("source.reference_hz", "50"), ("source.reference_hz", True),
    ("acquisition.actual_s", -1.0), ("acquisition.requested_s", math.inf), ("acquisition.samples", -5),
    ("acquisition.samples", 2.5), ("acquisition.fps", 0.0), ("acquisition.fps", math.nan),
    ("acquisition.exposure_us", -10.0), ("acquisition.early_stop", "no"), ("estimate.f0_hz", -1.0),
    ("estimate.f0_hz", 0.0), ("estimate.f0_hz", math.nan), ("estimate.resolution_hz", -0.5),
    ("estimate.snr", -3.0), ("estimate.snr", math.inf), ("warnings", "one warning"), ("hardware", "x"),
])
def test_nonfinite_or_negative_numeric_data_is_invalid(path, value):
    r = paired(frame=frame_record(**{path: value}))
    assert r["frame"]["eligibility"] == "invalid" and r["frame"]["problems"]
    assert r["frame"]["frequency_error"] is None and r["comparison_valid"] is False
    json.dumps(r, allow_nan=False)  # a bad number is never copied into the output


@pytest.mark.parametrize("field", comparison.TOP_LEVEL_FIELDS)
def test_missing_top_level_field_is_invalid(field):
    assert assess(event_record(**{field: _DELETE}), "event")["eligibility"] == "invalid"


def test_wrong_lab_or_sample_kind_is_invalid():
    r = compare(event_record(), frame_record())  # swapped
    assert r["frame"]["eligibility"] == r["event"]["eligibility"] == "invalid"
    assert assess(frame_record(**{"acquisition.sample_kind": "events"}), "frame")["eligibility"] == "invalid"
    assert assess(event_record(schema="lumen.flicker.v2"), "event")["eligibility"] == "invalid"
    assert assess("not a dict", "event")["eligibility"] == "invalid"
    with pytest.raises(ValueError):
        assess(event_record(), "thermal")


def test_status_not_ok_is_kept_and_provisional():
    a = assess(event_record(status="degraded"), "event")
    assert a["acquisition_status"] == "degraded" and a["eligibility"] == "provisional"
    assert a["reasons"] == ["ACQUISITION_STATUS_NOT_OK"]


def test_different_local_timestamps_change_nothing():
    a = paired(frame_record(local_time="2026-10-16T09:00:00+00:00"), event_record(local_time="2030-01-01T00:00:00Z"))
    same = "2026-10-16T14:27:58+04:00"
    b = paired(frame_record(local_time=same), event_record(local_time=same))
    for r in (a, b):
        r["frame"]["local_time"] = r["event"]["local_time"] = None
    assert a == b  # local_time is only a label
    assert not any("offset" in k or "sync" in k or "delay" in k for k in all_keys(a))


def test_missing_explicit_trial_id_is_unpaired_evidence():
    r = compare(frame_record(), event_record())  # same run-like ids would not help either
    assert r["pairing"] == {"status": "unpaired", "frame_trial_id": None, "event_trial_id": None,
                            "basis": "explicit trial_id only"}
    assert r["frame"]["eligibility"] == r["event"]["eligibility"] == "valid"
    assert r["comparison_valid"] is False and r["error_evidence"] == "unpaired_evidence"
    assert r["frame"]["frequency_error"] is not None  # shown, labelled as unpaired
    e = event_record(trial_id="T2")
    f = frame_record(trial_id="T1")
    r = compare(f, e)
    assert r["pairing"]["status"] == "trial_mismatch" and r["comparison_valid"] is False
    assert compare(frame_record(trial_id=""), event_record(trial_id=""))["pairing"]["status"] == "unpaired"


def test_same_run_id_does_not_pair():
    r = compare(frame_record(run_id="SAME", recording_id="SAME"), event_record(run_id="SAME", recording_id="SAME"))
    assert r["pairing"]["status"] == "unpaired" and r["comparison_valid"] is False


def test_no_direct_snr_ranking():
    r = paired()
    assert r["snr_comparable"] is False
    assert r["frame"]["snr"] == {"value": 900.0, "definition": comparison.SNR_DEFINITIONS["frame"]}
    assert r["event"]["snr"] == {"value": 40.0, "definition": comparison.SNR_DEFINITIONS["event"]}
    assert "power" in r["frame"]["snr"]["definition"] and "amplitude" in r["event"]["snr"]["definition"]
    assert not any(word in k for k in all_keys(r) for word in ("rank", "better", "winner", "best"))


def test_serial_number_is_redacted():
    text = json.dumps(paired())
    assert FAKE_SERIAL not in text and "serial" not in text
    assert paired()["frame"]["hardware"] == {"model": "MV-A7200MU130", "sensor": "frame", "real": True}


def test_inputs_are_not_mutated():
    f, e = frame_record(trial_id="T1"), event_record(trial_id="T1")
    f_before, e_before = copy.deepcopy(f), copy.deepcopy(e)
    r = compare(f, e)
    assert f == f_before and e == e_before
    r["frame"]["warnings"].clear()
    r["limitations"].clear()
    assert f == f_before and comparison.LIMITATIONS  # the output shares nothing that matters with the inputs


def test_json_serialization():
    cases = [paired(), compare(frame_record(), event_record(**{"source.reference_hz": 10})),
             compare({}, None), paired(frame=frame_record(**{"estimate.f0_hz": math.nan}))]
    for r in cases:
        assert json.loads(json.dumps(r, allow_nan=False)) == r


def test_output_has_no_raw_samples():
    e = event_record(events=list(range(1000)), frames=[[0] * 10] * 10)  # extra fields are never copied
    text = json.dumps(compare(frame_record(), e))
    assert "events\":" not in text and "frames\": [" not in text and len(text) < 10_000


def test_no_hardware_mcp_network_or_model_imports():
    for name in ("__init__.py", "comparison.py"):
        tree = ast.parse((PROJECT_ROOT / "agentic_ode" / "dualvision" / name).read_text())
        modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert modules <= {"__future__", "math", "agentic_ode.dualvision.comparison"}, modules
    code = ("import sys, agentic_ode.dualvision; print(' '.join(sorted(sys.modules)))")
    loaded = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            check=True).stdout.split()
    forbidden = ("metavision", "mcp", "fastmcp", "httpx", "requests", "openai", "langchain", "langgraph",
                 "agentic_ode.hardware", "agentic_ode.mcp", "agentic_ode.agents", "agentic_ode.graph")
    assert not [m for m in loaded if m.startswith(forbidden)]


BASELINES = sorted((PROJECT_ROOT / "report" / "lumen").glob("*.json"))


@pytest.mark.skipif(not BASELINES, reason="report/lumen baseline JSONs are not present")
def test_real_event_lab_baselines_read_only():
    for path in BASELINES:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(path.read_text())
        a = assess(record, "event")
        assert a["eligibility"] == "provisional" and a["problems"] == []  # accepted, all hit the event cap
        assert "EARLY_STOP" in a["reasons"] and a["f0_status"] == "measured"
        assert str(record["hardware"]["serial"]) not in json.dumps(compare(frame_record(), record))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest  # the file is unchanged
