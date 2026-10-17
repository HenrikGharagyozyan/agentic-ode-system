"""Offline JSON ingestion (DV-H04): synthetic lumen.flicker.v1 documents as bytes, never a file or a camera."""

import ast
import copy
import hashlib
import json
import subprocess
import sys

import pytest

from agentic_ode.core.config import PROJECT_ROOT
from agentic_ode.dualvision import ingest
from agentic_ode.dualvision.ingest import MAX_INPUT_BYTES, IngestError, compare_json_bytes

FAKE_SERIAL = "TEST-SERIAL-0000"
SECRET = "SECRET-MARKER-9876"  # put inside bad inputs: it must never appear in an error message


def frame_record(**top):
    """A synthetic valid Frame Lab record: fps 200 (Nyquist 100 Hz), structured warnings. top: top-level changes."""
    record = {
        "schema": "lumen.flicker.v1", "lab": "frame", "run_id": "F-RUN", "recording_id": "F-REC",
        "local_time": "2026-10-16T12:27:58+02:00",
        "hardware": {"model": "MV-A7200MU130", "sensor": "frame", "serial": FAKE_SERIAL, "real": True},
        "source": {"type": "LED_PWM", "reference_hz": 50, "duty": None, "distance_cm": None, "set_by": "manual"},
        "acquisition": {"requested_s": 2.0, "actual_s": 2.0, "samples": 400, "sample_kind": "frames",
                        "fps": 200.0, "exposure_us": 1000.0, "early_stop": False},
        "estimate": {"f0_hz": 50.2, "resolution_hz": 0.5, "snr": 900.0},
        "status": "ok",
        "warnings": [{"code": "ALIASING_NOT_EXCLUDED", "message": "a single frame rate cannot exclude aliasing"}],
        "analysis": "synthetic",
    }
    record.update(top)
    return record


def event_record(**top):
    """A synthetic valid Event Lab record with the shape of report/lumen/*.json (fake serial, no warnings)."""
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
    record.update(top)
    return record


def as_bytes(record) -> bytes:
    return json.dumps(record).encode("utf-8")


def run(frame=None, event=None) -> dict:
    """compare_json_bytes() of two records (the default records when None)."""
    return compare_json_bytes(as_bytes(frame or frame_record()), as_bytes(event or event_record()))


def rejected(frame_json, event_json=None) -> IngestError:
    """The IngestError raised for these inputs; also checks that it quotes no input and chains no decoder error."""
    with pytest.raises(IngestError) as info:
        compare_json_bytes(frame_json, as_bytes(event_record()) if event_json is None else event_json)
    e = info.value
    assert SECRET not in str(e) and FAKE_SERIAL not in str(e)
    assert e.__cause__ is None and e.__context__ is None
    return e


# ---- valid inputs: the science is compare()'s, unchanged ----

def test_valid_frame_and_event_json():
    r = run(frame_record(trial_id="T1"), event_record(trial_id="T1"))
    assert r["schema"] == "lumen.dualvision.ingest.v1"
    c = r["comparison"]
    assert c["schema"] == "lumen.dualvision.comparison.v1"
    assert c["frame"]["eligibility"] == c["event"]["eligibility"] == "valid"
    assert c["comparison_valid"] is True and c["error_evidence"] == "paired_trial"
    assert c["frame"]["frequency_error"]["signed_hz"] == pytest.approx(0.2)
    assert c["event"]["frequency_error"]["signed_hz"] == pytest.approx(-0.1)
    assert c["snr_comparable"] is False and c["limitations"]


def test_structured_frame_warnings_are_kept():
    w = [{"code": "ALIASING_NOT_EXCLUDED", "message": "one fps"}, {"code": "HOST_TIMESTAMPS", "message": "host"}]
    c = run(frame_record(warnings=w))["comparison"]
    assert c["frame"]["warnings"] == [{"code": "ALIASING_NOT_EXCLUDED", "text": "one fps", "significant": False},
                                      {"code": "HOST_TIMESTAMPS", "text": "host", "significant": True}]
    assert c["frame"]["eligibility"] == "provisional" and c["frame"]["reasons"] == ["SIGNIFICANT_WARNING"]


def test_legacy_event_warnings_are_kept():
    texts = ["stopped early: the 30000000 event cap was reached after 0.325 s of 2.0 s (light too bright)"]
    c = run(event=event_record(warnings=texts))["comparison"]
    assert c["event"]["warnings"] == [{"code": None, "text": texts[0], "significant": True}]
    assert c["event"]["eligibility"] == "provisional"


def test_no_trial_id_is_unpaired_and_none_is_made_up():
    same = {"run_id": "SAME", "recording_id": "SAME", "local_time": "2026-10-16T14:27:58+04:00"}
    c = run(frame_record(**same), event_record(**same))["comparison"]  # same ids and time do not pair
    assert c["pairing"] == {"status": "unpaired", "frame_trial_id": None, "event_trial_id": None,
                            "basis": "explicit trial_id only"}
    assert c["comparison_valid"] is False and c["error_evidence"] == "unpaired_evidence"


def test_matching_trial_id_is_paired():
    c = run(frame_record(trial_id="LUMEN-T01"), event_record(trial_id="LUMEN-T01"))["comparison"]
    assert c["pairing"]["status"] == "paired" and c["pairing"]["frame_trial_id"] == "LUMEN-T01"


def test_mismatched_trial_id():
    c = run(frame_record(trial_id="T1"), event_record(trial_id="T2"))["comparison"]
    assert c["pairing"]["status"] == "trial_mismatch" and c["comparison_valid"] is False


def test_mismatched_reference_frequencies_are_not_compared():
    e = event_record(trial_id="T1")
    e["source"]["reference_hz"] = 10
    e["estimate"]["f0_hz"] = 10.3
    c = run(frame_record(trial_id="T1"), e)["comparison"]
    assert c["reference_check"] == "mismatch" and c["error_evidence"] == "not_computed"
    assert c["frame"]["frequency_error"] is None and c["event"]["frequency_error"] is None
    assert c["frame"]["eligibility"] == c["event"]["eligibility"] == "valid"  # not turned into a sensor error


def test_event_early_stop_is_provisional():
    e = event_record()
    e["acquisition"].update(early_stop=True, samples=30_000_000, actual_s=0.325)
    c = run(event=e)["comparison"]
    assert c["event"]["eligibility"] == "provisional"
    assert c["event"]["reasons"][:2] == ["EARLY_STOP", "SHORT_RECORDING"]
    assert c["event"]["f0_hz"] == 49.9  # kept


@pytest.mark.parametrize("fps", [100.0, 60.0])  # Nyquist 50 (equal) and 30 Hz
def test_frame_reference_at_or_above_nyquist_is_not_recoverable(fps):
    f = frame_record()
    f["acquisition"]["fps"] = fps
    c = run(frame=f)["comparison"]
    assert c["frame"]["eligibility"] == "not_recoverable" and "REFERENCE_AT_OR_ABOVE_NYQUIST" in c["frame"]["reasons"]
    assert c["frame"]["f0_hz"] == 50.2 and c["comparison_valid"] is False


def test_field_defects_are_compares_invalid_not_ingest_errors():
    f = frame_record()
    del f["estimate"]
    c = run(frame=f)["comparison"]
    assert c["frame"]["eligibility"] == "invalid" and "estimate is missing" in c["frame"]["problems"]
    assert c["comparison_valid"] is False


def frame_count_record(**top):
    """frame_record() as a frame-count request (DV-H05C): 400 frames requested and acquired, requested_s null."""
    record = frame_record(**top)
    record["acquisition"].update(request_mode="frame_count", requested_frames=400, requested_s=None, samples=400)
    return record


def test_valid_frame_count_json_is_accepted():
    r = run(frame_count_record(trial_id="T1"), event_record(trial_id="T1"))
    c = r["comparison"]
    assert c["frame"]["eligibility"] == "valid" and c["comparison_valid"] is True
    acq = c["frame"]["acquisition"]
    assert (acq["request_mode"], acq["requested_frames"], acq["requested_s"], acq["request_completed"]) == (
        "frame_count", 400, None, True)
    assert c["event"]["acquisition"]["request_mode"] == "duration"
    assert FAKE_SERIAL not in json.dumps(r)


def test_frame_count_defects_are_compares_invalid_not_ingest_errors():
    f = frame_count_record()
    f["acquisition"]["requested_frames"] = True
    c = run(frame=f)["comparison"]
    assert c["frame"]["eligibility"] == "invalid" and c["comparison_valid"] is False
    f = frame_record()
    f["acquisition"]["requested_s"] = None  # no declared frame_count request
    c = run(frame=f)["comparison"]
    assert c["frame"]["eligibility"] == "invalid" and "acquisition.requested_s is null" in c["frame"]["problems"]


def test_early_stopped_frame_count_json_is_provisional():
    f = frame_count_record()
    f["acquisition"].update(samples=128, actual_s=0.64, early_stop=True)
    c = run(frame=f)["comparison"]
    assert c["frame"]["eligibility"] == "provisional" and c["frame"]["reasons"][:2] == ["EARLY_STOP", "SHORT_RECORDING"]
    assert c["frame"]["acquisition"]["request_completed"] is False


# ---- inputs that are not accepted (fail closed) ----

@pytest.mark.parametrize("data", [
    b"", b"   ", b"{", b'{"schema": "' + SECRET.encode() + b'"',                        # cut off
    as_bytes(frame_record()) + b" {}", as_bytes(frame_record()) + b"\n" + as_bytes(frame_record()),  # two values
    b'{"schema": "lumen.flicker.v1", "x": NaN}', b'{"x": Infinity}', b'{"x": -Infinity}',  # not JSON constants
    b'{"lab": "frame", "lab": "event"}', b'{"a": {"k": 1, "k": "' + SECRET.encode() + b'"}}',  # duplicate keys
    b"\xef\xbb\xbf" + as_bytes(frame_record()),                                              # UTF-8 BOM
    b"{'schema': 'lumen.flicker.v1'}",                                                       # not JSON quotes
])
def test_malformed_json_is_rejected(data):
    e = rejected(data)
    assert (e.side, e.code) == ("frame", "malformed_json")


def test_too_deep_nesting_is_rejected():
    for depth in (100_000, 1_000_000):  # 0.2 and 2 MB: under the size limit, far over Python's recursion limit
        e = rejected(b"[" * depth + b"]" * depth)
        assert (e.side, e.code) == ("frame", "malformed_json")


@pytest.mark.parametrize("data", [
    b"\xff\xfe{}", b'{"x": "\xe9' + SECRET.encode() + b'"}', b'{"x": "\xc3"}',              # bad bytes, cut sequence
    json.dumps(frame_record()).encode("utf-16"), b'{"x": "\xed\xa0\x80"}',                  # UTF-16, a surrogate
])
def test_invalid_utf8_is_rejected(data):
    e = rejected(data)
    assert (e.side, e.code) == ("frame", "not_utf8")


@pytest.mark.parametrize("data", [b"[]", b"[" + as_bytes(frame_record()) + b"]", b'"text"', b"42", b"null", b"true"])
def test_arrays_and_other_values_instead_of_an_object_are_rejected(data):
    e = rejected(data)
    assert (e.side, e.code) == ("frame", "not_object")


def test_oversized_input_is_rejected_before_decoding():
    body = as_bytes(frame_record())
    exact = body + b" " * (MAX_INPUT_BYTES - len(body))  # valid JSON of exactly 2 MiB: accepted
    assert compare_json_bytes(exact, as_bytes(event_record()))["inputs"]["frame"]["bytes"] == MAX_INPUT_BYTES
    e = rejected(exact + b" ")
    assert (e.side, e.code) == ("frame", "too_large")
    e = rejected(as_bytes(frame_record()), b"\xff" * (MAX_INPUT_BYTES + 1))  # not even decoded
    assert (e.side, e.code) == ("event", "too_large")


@pytest.mark.parametrize("data", [str(PROJECT_ROOT / "report" / "lumen"), "report/lumen/x.json",
                                  PROJECT_ROOT / "report", None, 42, memoryview(b"{}"), json.dumps(frame_record())])
def test_paths_text_and_other_types_are_rejected(data):
    e = rejected(data)
    assert (e.side, e.code) == ("frame", "not_bytes")


def test_wrong_lab_identities_are_rejected():
    e = rejected(as_bytes(event_record()), as_bytes(event_record()))  # an event record in the frame slot
    assert (e.side, e.code) == ("frame", "wrong_lab")
    e = rejected(as_bytes(frame_record()), as_bytes(frame_record()))  # a frame record in the event slot
    assert (e.side, e.code) == ("event", "wrong_lab")
    e = rejected(as_bytes(event_record()), as_bytes(frame_record()))  # swapped: the frame slot is checked first
    assert (e.side, e.code) == ("frame", "wrong_lab")
    for lab in (None, "Frame", "thermal", SECRET):
        assert rejected(as_bytes(frame_record(lab=lab))).code == "wrong_lab"
    f = frame_record()
    del f["lab"]
    assert rejected(as_bytes(f)).code == "wrong_lab"


@pytest.mark.parametrize("schema", ["lumen.flicker.v2", "", None, SECRET, ["lumen.flicker.v1"]])
def test_wrong_or_missing_schema_is_rejected(schema):
    assert rejected(as_bytes(frame_record(schema=schema))).code == "wrong_schema"
    e = event_record()
    del e["schema"]
    e = rejected(as_bytes(frame_record()), as_bytes(e))
    assert (e.side, e.code) == ("event", "wrong_schema")


# ---- privacy, raw data, side effects ----

def test_no_serial_in_the_output():
    text = json.dumps(run(frame_record(trial_id="T1"), event_record(trial_id="T1")))
    assert FAKE_SERIAL not in text and "serial" not in text


def test_no_raw_data_or_input_copy_in_the_output():
    e = event_record(events=[[i, i, 1, i] for i in range(5000)], image=[[0] * 64] * 64, secret_note=SECRET)
    f = frame_record(frames=[[[0] * 32] * 32] * 4)
    r = run(f, e)
    text = json.dumps(r, allow_nan=False)
    assert SECRET not in text and not any(f'"{key}":' in text for key in ("events", "frames", "image"))
    assert len(text) < 15_000  # a summary, not the input (the event input alone is far larger)
    assert len(as_bytes(e)) > 5 * len(text)


def test_input_bytes_remain_unchanged():
    frame_bytes = as_bytes(frame_record(trial_id="T1"))
    event_array = bytearray(as_bytes(event_record(trial_id="T1")))
    frame_before, event_before = bytes(frame_bytes), bytes(event_array)
    r = compare_json_bytes(frame_bytes, event_array)
    assert frame_bytes == frame_before and bytes(event_array) == event_before
    event_array[:] = b"x" * len(event_array)  # changing the caller's buffer later changes nothing in the result
    assert r["comparison"]["event"]["recording_id"] == "E-REC"


def test_result_is_strict_json_and_shares_nothing_with_module_constants():
    r = run()
    assert json.loads(json.dumps(r, allow_nan=False)) == r
    r["presentation"]["live_session"] = True
    r["comparison"]["limitations"].clear()
    assert run()["presentation"]["live_session"] is False and run()["comparison"]["limitations"]


def test_no_hardware_mcp_network_model_or_file_access_in_the_module():
    tree = ast.parse((PROJECT_ROOT / "agentic_ode" / "dualvision" / "ingest.py").read_text(encoding="utf-8"))
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules <= {"__future__", "hashlib", "json", "agentic_ode.dualvision.comparison"}, modules
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert not names & {"open", "Path", "read_bytes", "read_text", "glob", "listdir", "scandir", "walk"}
    code = "import sys, agentic_ode.dualvision.ingest; print(' '.join(sorted(sys.modules)))"
    loaded = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            check=True).stdout.split()
    forbidden = ("metavision", "mcp", "fastmcp", "httpx", "requests", "urllib3", "openai", "langchain", "langgraph",
                 "fastapi", "uvicorn", "agentic_ode.hardware", "agentic_ode.mcp", "agentic_ode.agents",
                 "agentic_ode.graph", "agentic_ode.web")
    assert not [m for m in loaded if m.startswith(forbidden)]


# ---- recorded-data display context ----

def test_recorded_data_display_context():
    f, e = frame_record(), event_record()
    r = run(f, e)
    assert r["presentation"] == {"mode": "recorded_data", "live_session": False,
                                 "note": ingest.PRESENTATION["note"]}
    assert "not synchronised" in r["presentation"]["note"] and "no camera" in r["presentation"]["note"]
    for lab, record in (("frame", f), ("event", e)):
        data = as_bytes(record)
        assert r["inputs"][lab] == {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                                    "acquisition": {"declared": "real_hardware",
                                                    "basis": "hardware.real as written in the record"}}
    assert set(r) == {"schema", "presentation", "inputs", "comparison"}


@pytest.mark.parametrize("hardware, declared", [
    ({"model": "m", "sensor": "s", "real": False}, "not_real_hardware"),
    ({"model": "m", "sensor": "s"}, "not_declared"),            # absent: nothing is invented
    ({"model": "m", "sensor": "s", "real": "yes"}, "not_declared"),
    ({"model": "m", "sensor": "s", "real": 1}, "not_declared"),
])
def test_acquisition_provenance_is_only_what_the_record_declares(hardware, declared):
    r = run(event=event_record(hardware=hardware))
    assert r["inputs"]["event"]["acquisition"]["declared"] == declared
    assert r["presentation"]["mode"] == "recorded_data"  # shown later whatever the acquisition was


def test_real_event_lab_baselines_are_accepted_as_bytes_and_stay_unchanged():
    paths = sorted((PROJECT_ROOT / "report" / "lumen").glob("*.json"))  # the TEST reads them, never the module
    if not paths:
        pytest.skip("report/lumen baseline JSONs are not present")
    for path in paths:
        data = path.read_bytes()
        record = json.loads(data)
        r = compare_json_bytes(as_bytes(frame_record()), data)
        assert r["comparison"]["event"]["eligibility"] == "provisional"
        assert "EARLY_STOP" in r["comparison"]["event"]["reasons"]
        assert r["inputs"]["event"]["acquisition"]["declared"] == "real_hardware"
        assert str(record["hardware"]["serial"]) not in json.dumps(r)
        assert path.read_bytes() == data


def test_module_constants_are_not_changed_by_use():
    before = copy.deepcopy(ingest.PRESENTATION)
    run()["presentation"]["note"] = "changed"
    assert ingest.PRESENTATION == before
