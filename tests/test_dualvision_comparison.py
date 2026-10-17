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
    assert a["warnings"] == [{"code": "ALIASING_NOT_EXCLUDED", "text": frame_record()["warnings"][0],
                              "significant": False}]  # kept, not hidden


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
    code = text.split(":")[0] if text[0].isupper() else None  # "aliasing_not_excluded" is not a code
    assert a["warnings"] == [{"code": code, "text": text, "significant": True}] and a["eligibility"] == "provisional"


# ---- structured warnings (DV-H03): {"code", "message"} objects of the Frame Lab, alone or mixed with strings ----

def warning(code, message="explanation"):
    return {"code": code, "message": message}


def test_event_string_warnings_keep_text_and_get_no_invented_code():
    texts = ["stopped early: the 30000000 event cap was reached after 0.325 s of 2.0 s (light too bright)",
             "the recording is only 0.33 s long: the frequency resolution is 3.1 Hz"]
    a = assess(event_record(warnings=texts), "event")
    assert a["warnings"] == [{"code": None, "text": t, "significant": True} for t in texts]
    assert a["problems"] == [] and a["reasons"] == ["SIGNIFICANT_WARNING"]


def test_structured_frame_warning_is_accepted():
    a = assess(frame_record(warnings=[warning("ALIASING_NOT_EXCLUDED", "one frame rate cannot exclude it")]), "frame")
    assert a["problems"] == [] and a["eligibility"] == "valid"  # below Nyquist (50 < 100): informational
    assert a["warnings"] == [{"code": "ALIASING_NOT_EXCLUDED", "text": "one frame rate cannot exclude it",
                              "significant": False}]


@pytest.mark.parametrize("fps", [100.0, 30.0, None])  # Nyquist 50 (equal), 15 Hz, unknown
def test_structured_aliasing_at_or_above_or_unknown_nyquist_is_significant(fps):
    a = assess(frame_record(warnings=[warning("ALIASING_NOT_EXCLUDED")], **{"acquisition.fps": fps}), "frame")
    assert a["problems"] == [] and a["warnings"][0]["significant"] is True
    assert a["eligibility"] in ("provisional", "not_recoverable") and "SIGNIFICANT_WARNING" in a["reasons"]


def test_structured_aliasing_with_invalid_fps_is_significant():
    a = assess(frame_record(warnings=[warning("ALIASING_NOT_EXCLUDED")], **{"acquisition.fps": math.nan}), "frame")
    assert a["eligibility"] == "invalid" and a["warnings"][0]["significant"] is True


@pytest.mark.parametrize("code", ["HARMONIC_AMBIGUITY", "DEVICE_TICK_UNVERIFIED", "HOST_TIMESTAMPS", "DROPPED_FRAMES",
                                  "TIMING_JITTER", "SOMETHING_NEW", "aliasing_not_excluded", "ALIASING_NOT_EXCLUDED "])
def test_other_and_unknown_structured_frame_warnings_are_significant(code):
    a = assess(frame_record(warnings=[warning(code, "why")]), "frame")
    assert a["problems"] == [] and a["warnings"] == [{"code": code, "text": "why", "significant": True}]
    assert a["eligibility"] == "provisional" and a["reasons"] == ["SIGNIFICANT_WARNING"]
    assert a["f0_hz"] == 50.2  # never replaced by reference_hz


def test_structured_warning_in_an_event_record_is_significant():
    a = assess(event_record(warnings=[warning("ALIASING_NOT_EXCLUDED")]), "event")
    assert a["problems"] == [] and a["warnings"][0]["significant"] is True and a["eligibility"] == "provisional"


def test_mixed_warning_list_keeps_order_and_meaning():
    items = ["ALIASING_NOT_EXCLUDED: a single frame rate cannot exclude aliasing", warning("HOST_TIMESTAMPS", "host"),
             "frames were late", warning("ALIASING_NOT_EXCLUDED", "again")]
    a = assess(frame_record(warnings=items), "frame")
    assert a["problems"] == [] and a["warnings"] == [
        {"code": "ALIASING_NOT_EXCLUDED", "text": items[0], "significant": False},
        {"code": "HOST_TIMESTAMPS", "text": "host", "significant": True},
        {"code": None, "text": "frames were late", "significant": True},
        {"code": "ALIASING_NOT_EXCLUDED", "text": "again", "significant": False},
    ]
    assert a["eligibility"] == "provisional" and a["reasons"] == ["SIGNIFICANT_WARNING"]


def test_early_stop_stays_provisional_whatever_the_warning_format():
    a = assess(frame_record(warnings=[warning("ALIASING_NOT_EXCLUDED")], **{"acquisition.early_stop": True}), "frame")
    assert a["eligibility"] == "provisional" and a["reasons"] == ["EARLY_STOP"]
    a = assess(event_record(warnings=[warning("EARLY_STOP", "event cap")], **{"acquisition.early_stop": True}), "event")
    assert a["reasons"] == ["EARLY_STOP", "SIGNIFICANT_WARNING"]


LEAK = "LEAK-MARKER-1234"


@pytest.mark.parametrize("bad", [
    {}, {"code": "DROPPED_FRAMES"}, {"message": "m"},                             # missing code or message
    {"code": "DROPPED_FRAMES", "message": "m", "extra": LEAK},                    # unexpected key
    {"code": "DROPPED_FRAMES", "message": "m", "severity": "high"},
    {"code": 1, "message": "m"}, {"code": None, "message": "m"}, {"code": True, "message": "m"},  # non-string
    {"code": "DROPPED_FRAMES", "message": 2}, {"code": "DROPPED_FRAMES", "message": [LEAK]},
    {"code": "", "message": "m"}, {"code": "DROPPED_FRAMES", "message": "  "},   # empty
    42, None, True, ["DROPPED_FRAMES", "m"],                                      # not a string or an object
])
def test_malformed_warning_item_is_invalid(bad):
    good = warning("ALIASING_NOT_EXCLUDED")
    r = paired(frame=frame_record(warnings=[good, bad]))
    f = r["frame"]
    assert f["eligibility"] == "invalid" and f["problems"] == ["warnings[1] is not a string or a {code, message} object"]
    assert [w["code"] for w in f["warnings"]] == ["ALIASING_NOT_EXCLUDED"]  # the good item is still listed
    assert f["frequency_error"] is None and r["comparison_valid"] is False
    text = json.dumps(r, allow_nan=False)
    assert LEAK not in text  # the bad value is never copied


def test_warnings_that_are_not_a_list_are_invalid():
    for value in ("one warning", warning("DROPPED_FRAMES"), None, 3):
        a = assess(frame_record(warnings=value), "frame")
        assert a["eligibility"] == "invalid" and a["problems"] == ["warnings is not a list"] and a["warnings"] == []


def test_warning_format_changes_no_pairing_or_frequency_error():
    as_string = ["ALIASING_NOT_EXCLUDED: a single frame rate cannot exclude aliasing", "DROPPED_FRAMES: 3 of 400"]
    as_object = [warning("ALIASING_NOT_EXCLUDED", "a single frame rate cannot exclude aliasing"),
                 warning("DROPPED_FRAMES", "3 of 400")]
    for trials in ((None, None), ("T1", "T1"), ("T1", "T2")):
        results = []
        for items in (as_string, as_object):
            f, e = frame_record(warnings=items), event_record()
            for record, trial in zip((f, e), trials):
                if trial:
                    record["trial_id"] = trial
            r = compare(f, e)
            assert [(w["code"], w["significant"]) for w in r["frame"]["warnings"]] == [
                ("ALIASING_NOT_EXCLUDED", False), ("DROPPED_FRAMES", True)]
            r["frame"]["warnings"] = None  # only the text differs ("CODE: message" or the message)
            results.append(r)
        assert results[0] == results[1]
    r = paired(frame=frame_record(warnings=as_object))
    assert r["pairing"]["status"] == "paired" and r["error_evidence"] == "paired_trial"
    assert r["frame"]["frequency_error"]["signed_hz"] == pytest.approx(0.2) and r["comparison_valid"] is False


def test_structured_warnings_are_not_mutated_shared_or_leaked():
    f = frame_record(trial_id="T1", warnings=[warning("ALIASING_NOT_EXCLUDED"), "HOST_TIMESTAMPS: host clock"])
    e = event_record(trial_id="T1", warnings=[warning("DEVICE_TICK_UNVERIFIED")])
    f_before, e_before = copy.deepcopy(f), copy.deepcopy(e)
    r = compare(f, e)
    assert f == f_before and e == e_before
    r["frame"]["warnings"][0]["code"] = "CHANGED"
    r["event"]["warnings"][0]["text"] = "CHANGED"
    assert f == f_before and e == e_before  # the output shares no warning dict with the inputs
    text = json.dumps(compare(f, e), allow_nan=False)
    assert FAKE_SERIAL not in text and "serial" not in text
    assert json.loads(text) == compare(f, e)


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


# ---- frame-count acquisition requests (DV-H05C): the Frame Lab asks for N frames, not for a duration ----

def frame_count_record(frames=400, samples=None, actual_s=2.0, early_stop=False, **changes):
    """A Frame Lab record that requested `frames` frames (requested_s null); 400 frames at 200 fps = 2.0 s."""
    acquisition = {"acquisition.request_mode": "frame_count", "acquisition.requested_frames": frames,
                   "acquisition.requested_s": None, "acquisition.actual_s": actual_s,
                   "acquisition.samples": frames if samples is None else samples, "acquisition.early_stop": early_stop}
    return frame_record(**{**acquisition, **changes})


def test_event_duration_record_is_unchanged_and_needs_no_frame_count_fields():
    e = event_record()
    assert "request_mode" not in e["acquisition"] and "requested_frames" not in e["acquisition"]
    a = assess(e, "event")
    assert a["eligibility"] == "valid" and a["problems"] == []
    assert a["acquisition"]["request_mode"] == "duration" and a["acquisition"]["requested_s"] == 2.0
    assert a["acquisition"]["requested_frames"] is None and a["acquisition"]["request_completed"] is None
    a = assess(event_record(**{"acquisition.request_mode": "duration"}), "event")  # declared: the same
    assert a["eligibility"] == "valid" and a["acquisition"]["request_mode"] == "duration"


def test_legacy_frame_record_with_positive_requested_s_is_still_accepted():
    a = assess(frame_record(), "frame")
    assert a["eligibility"] == "valid" and a["problems"] == []
    assert a["acquisition"]["request_mode"] == "duration" and a["acquisition"]["requested_s"] == 2.0
    assert a["acquisition"]["requested_frames"] is None


def test_completed_frame_count_record_is_valid():
    f = frame_count_record()
    a = assess(f, "frame")
    assert a["eligibility"] == "valid" and a["problems"] == [] and a["reasons"] == []
    assert a["acquisition"]["request_mode"] == "frame_count" and a["acquisition"]["requested_s"] is None
    assert a["acquisition"]["requested_frames"] == 400 and a["acquisition"]["samples"] == 400
    assert a["acquisition"]["request_completed"] is True and a["acquisition"]["actual_s"] == 2.0
    assert a["frequency_error"]["signed_hz"] == pytest.approx(0.2)


def test_short_complete_128_frame_recording_stays_provisional():
    a = assess(frame_count_record(frames=128, actual_s=0.64), "frame")  # 128 frames at 200 fps
    assert a["problems"] == [] and a["acquisition"]["request_completed"] is True
    assert a["eligibility"] == "provisional" and a["reasons"] == ["SHORT_RECORDING"]
    assert comparison.MIN_DURATION_S == 1.0  # the threshold is unchanged


@pytest.mark.parametrize("mode", [_DELETE, "duration"])
def test_null_requested_s_without_frame_count_mode_is_invalid(mode):
    f = frame_record(**{"acquisition.requested_s": None})
    if mode is _DELETE:
        assert "request_mode" not in f["acquisition"]
    else:
        f["acquisition"]["request_mode"] = mode
    a = assess(f, "frame")
    assert a["eligibility"] == "invalid" and "acquisition.requested_s is null" in a["problems"]


@pytest.mark.parametrize("requested_s", [2.0, 0.0, math.nan, "2"])
def test_frame_count_with_a_requested_s_value_is_invalid(requested_s):
    a = assess(frame_count_record(**{"acquisition.requested_s": requested_s}), "frame")
    assert a["eligibility"] == "invalid" and a["acquisition"]["requested_s"] is None
    json.dumps(a, allow_nan=False)


def test_frame_count_without_requested_frames_is_invalid():
    f = frame_count_record()
    del f["acquisition"]["requested_frames"]
    a = assess(f, "frame")
    assert a["eligibility"] == "invalid" and a["problems"] == ["acquisition.requested_frames is missing"]
    assert a["frequency_error"] is None


@pytest.mark.parametrize("frames, problem", [
    (0, "acquisition.requested_frames must be > 0"), (-128, "acquisition.requested_frames must be > 0"),
    (True, "acquisition.requested_frames is not a whole number (bool)"),
    (False, "acquisition.requested_frames is not a whole number (bool)"),
    (128.0, "acquisition.requested_frames is not a whole number (float)"),
    (2.5, "acquisition.requested_frames is not a whole number (float)"),
    (math.nan, "acquisition.requested_frames is not a whole number (float)"),
    ("128", "acquisition.requested_frames is not a whole number (str)"),
    (None, "acquisition.requested_frames is not a whole number (NoneType)"),
    ([128], "acquisition.requested_frames is not a whole number (list)"),
])
def test_bad_requested_frames_is_invalid(frames, problem):
    a = assess(frame_count_record(samples=1, early_stop=True, **{"acquisition.requested_frames": frames}), "frame")
    assert a["eligibility"] == "invalid" and a["problems"] == [problem]
    assert a["acquisition"]["requested_frames"] is None and a["acquisition"]["request_completed"] is None
    json.dumps(a, allow_nan=False)


@pytest.mark.parametrize("samples, early_stop, problem", [
    (100, False, "acquisition.early_stop is false but samples is less than requested_frames"),
    (0, False, "acquisition.early_stop is false but samples is less than requested_frames"),
    (401, False, "acquisition.samples is more than acquisition.requested_frames"),
    (401, True, "acquisition.samples is more than acquisition.requested_frames"),
    (400, True, "acquisition.early_stop is true but samples equals requested_frames"),
])
def test_contradictory_completion_metadata_is_invalid(samples, early_stop, problem):
    a = assess(frame_count_record(samples=samples, early_stop=early_stop), "frame")
    assert a["eligibility"] == "invalid" and a["problems"] == [problem]
    assert a["acquisition"]["request_completed"] is None


def test_early_stopped_frame_count_is_provisional_and_not_complete():
    a = assess(frame_count_record(samples=300, actual_s=1.5, early_stop=True), "frame")
    assert a["problems"] == [] and a["acquisition_status"] == "ok"  # status ok is not a proof of completion
    assert a["acquisition"]["request_completed"] is False and a["acquisition"]["samples"] == 300
    assert a["eligibility"] == "provisional" and a["reasons"] == ["EARLY_STOP"]
    assert a["f0_hz"] == 50.2 and a["frequency_error"] is not None  # the result is kept
    assert paired(frame=frame_count_record(samples=300, early_stop=True))["comparison_valid"] is False


@pytest.mark.parametrize("mode", ["FRAME_COUNT", "frames", "", None, 1, True, ["frame_count"], {"m": "frame_count"}])
def test_unknown_request_mode_is_invalid(mode):
    a = assess(frame_count_record(**{"acquisition.request_mode": mode}), "frame")
    assert a["eligibility"] == "invalid" and a["acquisition"]["request_mode"] is None
    assert "acquisition.request_mode is not 'duration' or 'frame_count'" in a["problems"]


def test_frame_count_fields_in_an_event_record_are_invalid():
    a = assess(event_record(**{"acquisition.request_mode": "frame_count", "acquisition.requested_s": None,
                               "acquisition.requested_frames": 128}), "event")
    assert a["eligibility"] == "invalid"
    assert "acquisition.request_mode 'frame_count' is only for Frame Lab records" in a["problems"]
    a = assess(event_record(**{"acquisition.requested_frames": 128}), "event")  # a duration request with a count
    assert a["problems"] == ["acquisition.requested_frames is given but request_mode is not 'frame_count'"]
    a = assess(event_record(**{"acquisition.requested_frames": None}), "event")  # null: no count, accepted
    assert a["eligibility"] == "valid"


def test_frame_count_failed_recovery_is_a_measurement_result_not_a_schema_error():
    a = assess(frame_count_record(**{"estimate.f0_hz": None}), "frame")
    assert a["problems"] == [] and a["eligibility"] == "not_recoverable" and a["reasons"] == ["F0_NOT_DETECTED"]
    a = assess(frame_count_record(**{"estimate.f0_hz": 37.0}), "frame")
    assert a["eligibility"] == "valid" and a["frequency_error"]["signed_hz"] == pytest.approx(-13.0)


def test_frame_count_10hz_paired_comparison():
    f = frame_count_record(frames=128, actual_s=2.56, **{"source.reference_hz": 10, "acquisition.fps": 50.0,
                                                          "estimate.f0_hz": 10.1, "estimate.resolution_hz": 0.39})
    e = event_record(**{"source.reference_hz": 10, "estimate.f0_hz": 10.2, "estimate.resolution_hz": 0.5})
    r = paired(frame=f, event=e)
    assert r["reference_check"] == "match" and r["reference_hz"] == 10 and r["comparison_valid"] is True
    assert r["frame"]["frequency_error"]["signed_hz"] == pytest.approx(0.1)
    assert r["event"]["frequency_error"]["signed_hz"] == pytest.approx(0.2)
    assert r["frame"]["acquisition"]["nyquist_hz"] == 25.0
    assert r["frame"]["f0_hz"] == 10.1  # never corrected


@pytest.mark.parametrize("fps", [100.0, 30.0])  # Nyquist 50 (equal) and 15 Hz
def test_frame_count_nyquist_safeguard(fps):
    a = assess(frame_count_record(**{"acquisition.fps": fps, "estimate.f0_hz": 50.0}), "frame")
    assert a["problems"] == [] and a["eligibility"] == "not_recoverable"
    assert "REFERENCE_AT_OR_ABOVE_NYQUIST" in a["reasons"] and a["warnings"][0]["significant"] is True
    assert a["acquisition"]["expected_alias_hz"] == pytest.approx(abs(50.0 - fps * round(50.0 / fps)))


def test_frame_count_warning_normalization():
    w = [warning("ALIASING_NOT_EXCLUDED", "one fps"), "DROPPED_FRAMES: 2 of 400"]
    a = assess(frame_count_record(warnings=w), "frame")
    assert a["warnings"] == [{"code": "ALIASING_NOT_EXCLUDED", "text": "one fps", "significant": False},
                             {"code": "DROPPED_FRAMES", "text": w[1], "significant": True}]
    assert a["eligibility"] == "provisional" and a["reasons"] == ["SIGNIFICANT_WARNING"]


def test_frame_count_trial_matching_and_mismatch():
    assert paired(frame=frame_count_record())["pairing"]["status"] == "paired"
    r = compare(frame_count_record(trial_id="T1"), event_record(trial_id="T2"))
    assert r["pairing"]["status"] == "trial_mismatch" and r["comparison_valid"] is False
    r = compare(frame_count_record(), event_record())
    assert r["pairing"]["status"] == "unpaired" and r["error_evidence"] == "unpaired_evidence"


def test_frame_count_output_is_sanitized_and_explains_the_request():
    f, e = frame_count_record(trial_id="T1"), event_record(trial_id="T1")
    f_before, e_before = copy.deepcopy(f), copy.deepcopy(e)
    r = compare(f, e)
    assert f == f_before and e == e_before
    text = json.dumps(r, allow_nan=False)
    assert FAKE_SERIAL not in text and "serial" not in text and json.loads(text) == r
    assert {k: r["frame"]["acquisition"][k] for k in ("request_mode", "requested_frames", "requested_s")} == {
        "request_mode": "frame_count", "requested_frames": 400, "requested_s": None}
    assert r["event"]["acquisition"]["request_mode"] == "duration"


BASELINES = sorted((PROJECT_ROOT / "report" / "lumen").glob("*.json"))


@pytest.mark.skipif(not BASELINES, reason="report/lumen baseline JSONs are not present")
def test_real_event_lab_baselines_read_only():
    for path in BASELINES:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        record = json.loads(path.read_text())
        a = assess(record, "event")
        assert a["eligibility"] == "provisional" and a["problems"] == []  # accepted, all hit the event cap
        assert "EARLY_STOP" in a["reasons"] and a["f0_status"] == "measured"
        assert "request_mode" not in record["acquisition"] and "requested_frames" not in record["acquisition"]
        assert a["acquisition"]["request_mode"] == "duration" and a["acquisition"]["requested_s"] == 2.0
        assert a["acquisition"]["requested_frames"] is None and a["acquisition"]["request_completed"] is None
        assert str(record["hardware"]["serial"]) not in json.dumps(compare(frame_count_record(), record))
        assert a["warnings"] == [{"code": None, "text": t, "significant": True} for t in record["warnings"]]
        assert str(record["hardware"]["serial"]) not in json.dumps(compare(frame_record(), record))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest  # the file is unchanged
