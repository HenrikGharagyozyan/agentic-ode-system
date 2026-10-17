"""The Dual-Vision HTTP route (DV-UI-01): a bare FastAPI app with only its router; synthetic records, no camera."""

from __future__ import annotations

import ast
import asyncio
import base64
import itertools
import json
import subprocess
import sys
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agentic_ode.core.config import PROJECT_ROOT
from agentic_ode.dualvision.ingest import MAX_INPUT_BYTES, compare_json_bytes
from agentic_ode.web import dualvision_api
from agentic_ode.web.dualvision_api import MAX_BODY_BYTES, MAX_ENCODED_CHARS, router

URL = "/api/dualvision/compare"
FAKE_SERIAL = "TEST-SERIAL-0000"
SECRET = "SECRET-MARKER-9876"  # put inside bad inputs: it must never come back in a response
JSON_TYPE = {"Content-Type": "application/json"}


def frame_record(**top):
    """A synthetic Frame Lab frame-count record (DV-H05C): 400 frames requested and acquired at 200 fps."""
    record = {
        "schema": "lumen.flicker.v1", "lab": "frame", "run_id": "F-RUN", "recording_id": "F-REC",
        "local_time": "2026-10-16T12:27:58+02:00",
        "hardware": {"model": "MV-A7200MU130", "sensor": None, "serial": FAKE_SERIAL, "real": True},
        "source": {"type": "LED_PWM", "reference_hz": 50, "duty": None, "distance_cm": None, "set_by": "manual"},
        "acquisition": {"request_mode": "frame_count", "requested_frames": 400, "requested_s": None,
                        "actual_s": 2.0, "samples": 400, "sample_kind": "frames", "fps": 200.0,
                        "exposure_us": 1000.0, "early_stop": False},
        "estimate": {"f0_hz": 50.2, "resolution_hz": 0.5, "snr": 900.0},
        "status": "ok",
        "warnings": [{"code": "ALIASING_NOT_EXCLUDED", "message": "a single frame rate cannot exclude aliasing"},
                     {"code": "HOST_TIMESTAMPS", "message": "host arrival times"}],
        "analysis": "synthetic",
    }
    record.update(top)
    return record


def event_record(**top):
    """A synthetic Event Lab duration record with the shape of report/lumen/*.json (fake serial)."""
    record = {
        "schema": "lumen.flicker.v1", "lab": "event", "run_id": "E-RUN", "recording_id": "E-REC",
        "local_time": "2026-10-16T14:27:58+04:00",
        "hardware": {"model": "IDS UE-39B0XCP-E-GL", "sensor": "IMX636", "serial": FAKE_SERIAL, "real": True},
        "source": {"type": "LED_PWM", "reference_hz": 50, "duty": None, "distance_cm": None, "set_by": "manual"},
        "acquisition": {"requested_s": 2.0, "actual_s": 2.0, "samples": 1_000_000, "sample_kind": "events",
                        "fps": None, "exposure_us": None, "early_stop": False},
        "estimate": {"f0_hz": 49.9, "resolution_hz": 0.5, "snr": 40.0},
        "status": "ok", "warnings": ["stopped early: the 30000000 event cap was reached after 1.5 s of 2.0 s"],
        "analysis": "synthetic",
    }
    record.update(top)
    return record


def as_bytes(record) -> bytes:
    return json.dumps(record).encode("utf-8")


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def envelope(frame: bytes | None = None, event: bytes | None = None) -> dict:
    return {"frame_b64": b64(as_bytes(frame_record()) if frame is None else frame),
            "event_b64": b64(as_bytes(event_record()) if event is None else event)}


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def post(client, body, headers=JSON_TYPE):
    """POST raw bytes (body) or a JSON-encoded dict."""
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return client.post(URL, content=content, headers=headers)


def assert_clean(r):
    """An error body: only fixed fields, and no input content."""
    assert SECRET not in r.text and b64(SECRET.encode())[:16] not in r.text and FAKE_SERIAL not in r.text
    assert "Traceback" not in r.text and set(r.json()) <= {"detail", "side", "code"}


# ---- valid requests: compare_json_bytes() decides, the route passes its result on ----

def test_valid_pair_returns_the_exact_ingest_result(client):
    frame, event = as_bytes(frame_record(trial_id="T1")), as_bytes(event_record(trial_id="T1"))
    r = post(client, envelope(frame, event))
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/json")
    assert r.json() == compare_json_bytes(frame, event)
    assert r.json()["schema"] == "lumen.dualvision.ingest.v1"


def test_frame_count_and_duration_semantics_are_kept(client):
    c = post(client, envelope()).json()["comparison"]
    f, e = c["frame"]["acquisition"], c["event"]["acquisition"]
    assert (f["request_mode"], f["requested_frames"], f["requested_s"], f["request_completed"]) == (
        "frame_count", 400, None, True)
    assert (e["request_mode"], e["requested_s"], e["requested_frames"]) == ("duration", 2.0, None)
    assert c["pairing"]["status"] == "unpaired" and c["comparison_valid"] is False  # no trial_id: never paired


def test_warnings_of_both_labs_are_kept(client):
    c = post(client, envelope()).json()["comparison"]
    assert c["frame"]["warnings"] == [
        {"code": "ALIASING_NOT_EXCLUDED", "text": "a single frame rate cannot exclude aliasing", "significant": False},
        {"code": "HOST_TIMESTAMPS", "text": "host arrival times", "significant": True}]
    assert c["event"]["warnings"] == [{"code": None, "text": event_record()["warnings"][0], "significant": True}]


def test_original_acquisition_and_recorded_display_are_distinguished(client):
    sim = frame_record(hardware={"model": "sim", "sensor": None, "real": False})
    r = post(client, envelope(as_bytes(sim))).json()
    assert r["presentation"]["mode"] == "recorded_data" and r["presentation"]["live_session"] is False
    assert r["inputs"]["frame"]["acquisition"]["declared"] == "not_real_hardware"
    assert r["inputs"]["event"]["acquisition"]["declared"] == "real_hardware"


def test_scientifically_invalid_record_is_http_200(client):
    f = frame_record()
    f["acquisition"]["requested_frames"] = True  # passes ingestion, fails the comparator
    r = post(client, envelope(as_bytes(f)))
    assert r.status_code == 200
    assert r.json()["comparison"]["frame"]["eligibility"] == "invalid" and r.json()["comparison"]["frame"]["problems"]


def test_maximum_size_input_is_accepted_and_one_byte_more_is_an_ingest_error(client):
    body = as_bytes(frame_record())
    exact = body + b" " * (MAX_INPUT_BYTES - len(body))  # valid JSON of exactly 2 MiB
    assert len(b64(exact)) == MAX_ENCODED_CHARS
    r = post(client, envelope(exact))
    assert r.status_code == 200 and r.json()["inputs"]["frame"]["bytes"] == MAX_INPUT_BYTES
    over = exact + b" "  # its base64 still fits MAX_ENCODED_CHARS: ingest's own limit refuses it
    assert len(b64(over)) <= MAX_ENCODED_CHARS
    r = post(client, envelope(over))
    assert r.status_code == 422 and (r.json()["side"], r.json()["code"]) == ("frame", "too_large")


# ---- 422: IngestError, fixed texts only ----

@pytest.mark.parametrize("frame, event, side, code", [
    (as_bytes(event_record()), None, "frame", "wrong_lab"),
    (None, as_bytes(frame_record()), "event", "wrong_lab"),
    (as_bytes(frame_record(schema=SECRET)), None, "frame", "wrong_schema"),
    (b'{"schema": "' + SECRET.encode() + b'"', None, "frame", "malformed_json"),
    (b'{"lab": "frame", "lab": "' + SECRET.encode() + b'"}', None, "frame", "malformed_json"),
    (b'{"x": NaN}', None, "frame", "malformed_json"),
    (None, b'{"x": "\xe9' + SECRET.encode() + b'"}', "event", "not_utf8"),
    (b"[]", None, "frame", "not_object"),
])
def test_ingest_errors_are_422_with_side_and_code(client, frame, event, side, code):
    r = post(client, envelope(frame, event))
    assert r.status_code == 422 and set(r.json()) == {"detail", "side", "code"}
    assert (r.json()["side"], r.json()["code"]) == (side, code)
    assert_clean(r)


# ---- 400: the envelope ----

@pytest.mark.parametrize("body", [
    b"", b"{", b'{"frame_b64": "' + SECRET.encode() + b'"', b"[]", b'"text"', b"null",
    b'{"frame_b64": "", "event_b64": NaN}', b'{"frame_b64": Infinity, "event_b64": ""}',
    b"\xff\xfe{}", b'{"frame_b64": "\xe9", "event_b64": ""}', b"\xef\xbb\xbf{}",
])
def test_malformed_envelope_is_400(client, body):
    r = post(client, body)
    assert r.status_code == 400 and "strict UTF-8 JSON object" in r.json()["detail"]
    assert_clean(r)


def test_too_deep_envelope_nesting_is_400(client):  # under the size limit, far over Python's recursion limit
    r = post(client, b"[" * 1_000_000 + b"]" * 1_000_000)
    assert r.status_code == 400 and "strict UTF-8 JSON object" in r.json()["detail"]


def test_duplicate_envelope_keys_are_400(client):
    e = envelope()
    second = b64(SECRET.encode())
    body = f'{{"frame_b64": "{e["frame_b64"]}", "event_b64": "{e["event_b64"]}", "frame_b64": "{second}"}}'.encode()
    r = post(client, body)
    assert r.status_code == 400 and "duplicate" in r.json()["detail"]
    assert_clean(r)


@pytest.mark.parametrize("body", [
    {}, {"frame_b64": "e30="}, {"event_b64": "e30="},
    {"frame_b64": "e30=", "event_b64": "e30=", "extra": SECRET},
    {"frame": "e30=", "event": "e30="}, {"frame_path": "report/lumen/x.json", "event_b64": "e30="},
    {"frame_b64": 1, "event_b64": "e30="}, {"frame_b64": None, "event_b64": "e30="},
    {"frame_b64": frame_record(), "event_b64": "e30="}, {"frame_b64": ["e30="], "event_b64": "e30="},
])
def test_missing_extra_or_wrong_fields_are_400(client, body):
    r = post(client, body)
    assert r.status_code == 400 and "frame_b64 and event_b64" in r.json()["detail"]
    assert_clean(r)


@pytest.mark.parametrize("value", ["not base64!", "e30", "e30=\n", " e30=", "e3=0", "ZXhh" + SECRET, "é30="])
def test_malformed_base64_is_400(client, value):
    r = post(client, {"frame_b64": value, "event_b64": envelope()["event_b64"]})
    assert r.status_code == 400 and r.json() == {"detail": "frame_b64 is not valid base64."}
    assert_clean(r)


# ---- 415: media type ----

@pytest.mark.parametrize("content_type", [None, "text/plain", "application/x-www-form-urlencoded",
                                          "multipart/form-data; boundary=x", "application/json; charset=latin-1",
                                          "application/jsonx", "application/json; boundary=x"])
def test_wrong_content_type_is_415(client, content_type):
    headers = {} if content_type is None else {"Content-Type": content_type}
    r = post(client, json.dumps(envelope()).encode(), headers=headers)
    assert r.status_code == 415 and r.json() == {"detail": "Content-Type must be application/json."}


@pytest.mark.parametrize("content_type", ["application/json; charset=utf-8", "Application/JSON; Charset=UTF-8",
                                          'application/json;charset="utf-8"'])
def test_json_with_utf8_charset_is_accepted(client, content_type):
    assert post(client, envelope(), headers={"Content-Type": content_type}).status_code == 200


# ---- 413: size limits, enforced while reading ----

def test_limits_follow_the_ingest_limit():
    assert MAX_ENCODED_CHARS == 4 * -(-MAX_INPUT_BYTES // 3) == 2_796_204
    assert MAX_BODY_BYTES == 2 * MAX_ENCODED_CHARS + 1024


def raw_post(headers, chunks):
    """Call the ASGI app directly: chunks are given one per receive(). Returns status, body and the chunks read."""
    app = FastAPI()
    app.include_router(router)
    source, read, sent = iter(chunks), [], []

    async def receive():
        chunk = next(source, None)
        if chunk is None:
            return {"type": "http.request", "body": b"", "more_body": False}
        read.append(len(chunk))
        return {"type": "http.request", "body": chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
             "path": URL, "raw_path": URL.encode(), "query_string": b"", "root_path": "",
             "headers": [(k.lower().encode(), v.encode()) for k, v in headers], "client": ("127.0.0.1", 1),
             "server": ("127.0.0.1", 8000)}
    asyncio.run(app(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, json.loads(body), read


def test_declared_content_length_over_the_limit_is_413_before_any_body_is_read():
    chunks = itertools.repeat(b"x" * 65536)  # endless: reading it would never stop
    status, body, read = raw_post([("Content-Type", "application/json"),
                                   ("Content-Length", str(MAX_BODY_BYTES + 1))], chunks)
    assert status == 413 and "larger than" in body["detail"] and read == []


def test_bad_content_length_is_400():
    status, body, read = raw_post([("Content-Type", "application/json"), ("Content-Length", "12abc")], [b"{}"])
    assert status == 400 and read == []


def test_stream_without_content_length_stops_right_after_the_limit():
    chunk = 65536
    status, body, read = raw_post([("Content-Type", "application/json")], itertools.repeat(b"x" * chunk))
    assert status == 413 and "larger than" in body["detail"]
    assert MAX_BODY_BYTES < sum(read) <= MAX_BODY_BYTES + chunk  # stopped at the first chunk over the limit


def test_stream_that_lies_about_its_length_is_still_limited():
    status, _, read = raw_post([("Content-Type", "application/json"), ("Content-Length", "10")],
                               itertools.repeat(b"x" * 65536))
    assert status == 413 and sum(read) <= MAX_BODY_BYTES + 65536


def test_chunked_valid_request_is_accepted():
    data = json.dumps(envelope()).encode()
    chunks = [data[i:i + 100] for i in range(0, len(data), 100)]
    status, body, _ = raw_post([("Content-Type", "application/json")], chunks)
    assert status == 200 and body["schema"] == "lumen.dualvision.ingest.v1"


def test_over_long_base64_value_is_413_before_decoding(client, monkeypatch):
    decoded = []
    monkeypatch.setattr(dualvision_api.base64, "b64decode", lambda *a, **k: decoded.append(1) or b"")
    long = "A" * (MAX_ENCODED_CHARS + 4)
    for body in ({"frame_b64": long, "event_b64": "e30="}, {"frame_b64": "e30=", "event_b64": long}):
        r = post(client, body)
        assert r.status_code == 413 and "base64 characters" in r.json()["detail"]
    assert decoded == [1]  # only the short frame value of the second request was decoded


def test_whole_body_over_the_limit_is_413(client):
    r = post(client, b" " * (MAX_BODY_BYTES + 1))
    assert r.status_code == 413


# ---- privacy, isolation, no side effects ----

def test_no_serial_or_raw_data_in_the_output(client):
    e = event_record(events=[[i, i, 1, i] for i in range(2000)], image=[[0] * 64] * 64, note=SECRET)
    f = frame_record(frames=[[[0] * 16] * 16] * 4)
    r = post(client, envelope(as_bytes(f), as_bytes(e)))
    assert r.status_code == 200
    assert FAKE_SERIAL not in r.text and "serial" not in r.text and SECRET not in r.text
    assert not any(f'"{key}":' in r.text for key in ("events", "frames", "image"))
    assert len(r.text) < 15_000


def test_the_route_does_not_need_or_take_the_camera_lock():
    app = FastAPI()
    app.include_router(router)
    app.state.camera_lock = lock = threading.Lock()
    lock.acquire()  # like a running experiment
    try:
        r = TestClient(app).post(URL, content=json.dumps(envelope()).encode(), headers=JSON_TYPE)
        assert r.status_code == 200 and lock.locked()
    finally:
        lock.release()


def module_tree():
    return ast.parse((PROJECT_ROOT / "agentic_ode" / "web" / "dualvision_api.py").read_text(encoding="utf-8"))


def test_module_imports_and_names_are_limited():
    tree = module_tree()
    modules = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    modules |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert modules == {"__future__", "base64", "json", "fastapi", "fastapi.responses", "starlette.concurrency",
                       "agentic_ode.dualvision.ingest"}, modules
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    forbidden = {"open", "Path", "read_bytes", "read_text", "write_text", "write_bytes", "glob", "listdir", "scandir",
                 "walk", "remove", "unlink", "logging", "logger", "print", "camera_lock", "settings", "Settings",
                 "evcam_session", "store", "urlopen", "subprocess"}
    assert not names & forbidden, names & forbidden


def test_importing_the_router_loads_no_hardware_mcp_model_or_agent_module():
    code = "import sys, agentic_ode.web.dualvision_api; print(' '.join(sorted(sys.modules)))"
    loaded = subprocess.run([sys.executable, "-B", "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True,
                            check=True).stdout.split()
    forbidden = ("metavision", "mcp", "fastmcp", "langchain", "langgraph", "openai", "requests", "urllib3", "httpx",
                 "uvicorn", "torch", "agentic_ode.hardware", "agentic_ode.mcp", "agentic_ode.agents",
                 "agentic_ode.graph", "agentic_ode.runtime", "agentic_ode.web.app", "agentic_ode.web.experiments",
                 "agentic_ode.web.recordings")
    assert not [m for m in loaded if m.startswith(forbidden)]


def test_app_registers_the_router_once_statically():
    """app.py cannot be imported on every machine (it needs fastmcp, LangChain, ...): its source is checked instead."""
    source = (PROJECT_ROOT / "agentic_ode" / "web" / "app.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [n for n in ast.walk(tree)
               if isinstance(n, ast.ImportFrom) and n.module == "agentic_ode.web.dualvision_api"]
    assert len(imports) == 1 and [(a.name, a.asname) for a in imports[0].names] == [("router", "dualvision_router")]
    create_app = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "create_app")
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "include_router"]
    assert len(calls) == 1 and calls[0] in list(ast.walk(create_app))
    assert isinstance(calls[0].func.value, ast.Name) and calls[0].func.value.id == "app"
    assert [a.id for a in calls[0].args if isinstance(a, ast.Name)] == ["dualvision_router"]
    assert "/api/dualvision/compare" in ast.get_docstring(tree)
