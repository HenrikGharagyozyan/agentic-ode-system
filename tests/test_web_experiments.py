"""EventLab experiments through the web API: the real graph with a scripted fake model and the fake camera (MCP in
memory, files in tmp_path). Approval, refusal, the camera lock, cancel, the event stream and reloading from disk."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from agentic_ode.mcp import evcam_server
from agentic_ode.mcp.client import McpSession
from agentic_ode.runtime import runner
from agentic_ode.web.app import create_app
from tests.conftest import FakeLLM
from tests.test_hardware_turn import HwLLM, hw

HOT = [[1053, 704], [926, 151]]


@pytest.fixture
def fake_camera(tmp_path, monkeypatch):
    """Every run_task(camera=True) gets the evcam server with the fake camera, in memory; files in tmp_path."""
    monkeypatch.setenv("EVCAM_FAKE", "1")
    monkeypatch.setenv("EVCAM_FAKE_SCENE", "pwm")
    monkeypatch.setattr(evcam_server.instrument, "recordings_dir", tmp_path / "evcam")
    monkeypatch.setattr(evcam_server.tool_log, "path", tmp_path / "evcam" / "tool_log.jsonl")
    monkeypatch.setattr(runner, "evcam_session", lambda: McpSession(transport=evcam_server.mcp, timeout_s=60))
    evcam_server.instrument.close()
    yield
    evcam_server.instrument.close()


def mask_then_record_llm():
    """mask the hot pixels (needs approval), record 0.5 s, finish."""
    return HwLLM([hw("mask"), hw("record 0.5 s"), "finish"],
                 [[("mask_pixels", {"pixels": HOT})], [("record", {"seconds": 0.5})]])


def wait_for(client, exp_id, condition, timeout=30.0):
    """Poll GET /api/experiments/{id} until condition(body) is true; return the body."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        body = client.get(f"/api/experiments/{exp_id}").json()
        if condition(body):
            return body
        time.sleep(0.05)
    raise AssertionError(f"timeout; last status {body['experiment']['status']}")


def pending(body):
    return body["experiment"]["pending"]


def ended(body):
    return body["experiment"]["status"] in ("completed", "stopped", "failed", "cancelled")


def types(body):
    return [e["type"] for e in body["events"]]


def start(client, question="Mask the hot pixels, then record."):
    r = client.post("/api/experiments", json={"question": question})
    assert r.status_code == 202, r.text
    return r.json()["id"]


# ---- the runner hooks (the CLI does not use them) ----

def test_run_task_uses_the_given_run_dir_and_calls_on_state(tmp_path, settings):
    task = tmp_path / "task.md"
    task.write_text("fit")
    run_dir = tmp_path / "mine"
    run_dir.mkdir()
    seen = []
    outcome = runner.run_task(task, settings, llm=FakeLLM(["executor", "finish"]), run_dir=run_dir,
                              on_state=lambda s: seen.append(len(s["dialogue"])))
    assert outcome.run_dir == run_dir
    assert (run_dir / "transcript.json").is_file()
    assert seen and seen[-1] == len(outcome.state["dialogue"])
    assert not settings.runs_dir.exists()  # no other run folder was made


# ---- a whole experiment ----

def test_approved_setting_change_runs_and_completes(fake_camera, settings):
    app = create_app(settings, llm=mask_then_record_llm())
    client = TestClient(app)
    exp_id = start(client)
    assert client.get("/api/camera/status").json()["busy"] is True

    body = wait_for(client, exp_id, pending)
    request = pending(body)
    assert body["experiment"]["status"] == "awaiting_input"
    assert request["source"] == "camera_setting"
    assert request["call"] == {"name": "mask_pixels", "args": {"pixels": HOT}}
    assert "tool.pending" in types(body)
    assert not any(e["type"] == "tool.result" for e in body["events"])  # nothing ran before the approval

    url = f"/api/experiments/{exp_id}/reply"
    assert client.post(url, json={"request_id": "other", "reply": "yes"}).status_code == 409
    assert client.post(url, json={"request_id": request["request_id"], "reply": "yes please"}).status_code == 422
    assert client.post(url, json={"request_id": request["request_id"], "reply": "yes"}).json() == {"ok": True}

    body = wait_for(client, exp_id, ended)
    exp = body["experiment"]
    assert exp["status"] == "completed" and exp["stop_reason"] == "finish"
    assert exp["mode"] == "simulated" and exp["pending"] is None
    assert [r["tool"] for r in exp["camera"]] == ["mask_pixels", "record"]
    resolved = next(e for e in body["events"] if e["type"] == "input.resolved")["data"]
    assert resolved == {"request_id": request["request_id"], "reply": "yes", "accepted": True}
    results = [e["data"] for e in body["events"] if e["type"] == "tool.result"]
    assert [(r["tool"], r["kind"], r["ok"]) for r in results] == [("mask_pixels", "setting", True),
                                                                 ("record", "measurement", True)]
    assert results[1]["result"]["recording_id"]  # parsed JSON, not text
    assert [e["seq"] for e in body["events"]] == list(range(1, len(body["events"]) + 1))
    assert body["events"][-1]["data"]["status"] == "completed"

    run_dir = settings.runs_dir / exp_id
    saved = json.loads((run_dir / "experiment.json").read_text())
    assert saved["status"] == "completed" and len(saved["camera"]) == 2
    assert len((run_dir / "events.jsonl").read_text().splitlines()) == len(body["events"])
    assert (run_dir / "task.md").read_text().startswith("Mask the hot pixels")
    assert (run_dir / "transcript.json").is_file()
    assert client.get("/api/camera/status").json()["busy"] is False


def test_refused_setting_change_never_runs(fake_camera, settings):
    app = create_app(settings, llm=mask_then_record_llm())
    client = TestClient(app)
    exp_id = start(client)
    request = pending(wait_for(client, exp_id, pending))
    client.post(f"/api/experiments/{exp_id}/reply", json={"request_id": request["request_id"], "reply": "no"})

    body = wait_for(client, exp_id, ended)
    exp = body["experiment"]
    assert [r["tool"] for r in exp["camera"]] == ["record"]  # mask_pixels did NOT run
    assert exp["refused_calls"] == [{"name": "mask_pixels", "args": {"pixels": HOT}}]
    resolved = next(e for e in body["events"] if e["type"] == "input.resolved")["data"]
    assert resolved["accepted"] is False
    turns = [e["data"] for e in body["events"] if e["type"] == "turn.added"]
    notes = [t["content"] for t in turns if t["role"] == "system"]
    assert any("The human refused the camera call mask_pixels" in n for n in notes)


def test_one_experiment_at_a_time(fake_camera, settings):
    app = create_app(settings, llm=mask_then_record_llm())
    client = TestClient(app)
    exp_id = start(client)
    wait_for(client, exp_id, pending)

    second = client.post("/api/experiments", json={"question": "another one"})
    assert second.status_code == 409
    assert client.post("/api/camera/check").status_code == 409  # the check never opens a busy camera
    assert client.get("/api/experiments").json()["active"] == exp_id

    client.post(f"/api/experiments/{exp_id}/cancel")
    wait_for(client, exp_id, ended)
    assert client.get("/api/experiments").json()["active"] is None


def test_cancel_while_waiting_for_approval(fake_camera, settings):
    app = create_app(settings, llm=mask_then_record_llm())
    client = TestClient(app)
    exp_id = start(client)
    wait_for(client, exp_id, pending)

    assert client.post(f"/api/experiments/{exp_id}/cancel").status_code == 202
    body = wait_for(client, exp_id, ended)
    assert body["experiment"]["status"] == "cancelled"
    assert body["experiment"]["camera"] == []
    assert not app.state.camera_lock.locked()
    assert client.post(f"/api/experiments/{exp_id}/cancel").status_code == 409  # already ended


class BrokenLLM(FakeLLM):
    def with_structured_output(self, schema, **kw):
        class _Broken:
            def invoke(self, messages):
                raise RuntimeError("provider down")

        return _Broken()


def test_a_failing_model_gives_a_failed_experiment(fake_camera, settings):
    app = create_app(settings, llm=BrokenLLM(["finish"]))
    client = TestClient(app)
    exp_id = start(client)
    exp = wait_for(client, exp_id, ended)["experiment"]
    assert exp["status"] == "failed"
    assert exp["error"] == "RuntimeError: provider down"
    assert not app.state.camera_lock.locked()


# ---- the event stream and reloading ----

def sse_events(client, url, headers=None):
    with client.stream("GET", url, headers=headers or {}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        return [json.loads(line[len("data: "):]) for line in r.iter_lines() if line.startswith("data: ")]


def test_event_stream_replays_and_resumes(fake_camera, settings):
    client = TestClient(create_app(settings, llm=HwLLM([hw("record"), "finish"], [[("record", {"seconds": 0.5})]])))
    exp_id = start(client, "Record half a second.")
    all_events = wait_for(client, exp_id, ended)["events"]

    streamed = sse_events(client, f"/api/experiments/{exp_id}/events")
    assert streamed == all_events  # the stream ends by itself after the final status
    later = sse_events(client, f"/api/experiments/{exp_id}/events?after=3")
    assert [e["seq"] for e in later] == [e["seq"] for e in all_events[3:]]
    resumed = sse_events(client, f"/api/experiments/{exp_id}/events", {"Last-Event-ID": "4"})
    assert [e["seq"] for e in resumed] == [e["seq"] for e in all_events[4:]]


def test_a_new_backend_reads_saved_experiments(fake_camera, settings):
    client = TestClient(create_app(settings, llm=HwLLM([hw("record"), "finish"], [[("record", {"seconds": 0.5})]])))
    exp_id = start(client, "Record half a second.")
    events = wait_for(client, exp_id, ended)["events"]

    fresh = TestClient(create_app(settings, llm=FakeLLM(["finish"])))  # like a restarted server
    body = fresh.get(f"/api/experiments/{exp_id}").json()
    assert body["experiment"]["status"] == "completed"
    assert body["events"] == events
    listed = fresh.get("/api/experiments").json()
    assert listed["active"] is None
    assert [e["id"] for e in listed["experiments"]] == [exp_id]
    assert fresh.post(f"/api/experiments/{exp_id}/reply", json={"request_id": "x", "reply": "yes"}).status_code == 409


def test_an_interrupted_experiment_is_never_shown_as_running(settings):
    folder = settings.runs_dir / "20261016-120000"
    folder.mkdir(parents=True)
    (folder / "experiment.json").write_text(json.dumps({"id": folder.name, "status": "awaiting_input",
                                                        "pending": {"request_id": "x"}}))
    exp = TestClient(create_app(settings, llm=FakeLLM(["finish"]))).get(f"/api/experiments/{folder.name}").json()
    assert exp["experiment"]["status"] == "failed" and exp["experiment"]["pending"] is None


# ---- validation ----

def test_bad_requests(settings):
    client = TestClient(create_app(settings, llm=FakeLLM(["finish"])))
    assert client.post("/api/experiments", json={"question": "  "}).status_code == 422
    assert client.post("/api/experiments", json={}).status_code == 422
    assert client.get("/api/experiments/20990101-000000").status_code == 404
    assert client.get("/api/experiments/..%2F..%2Fetc").status_code == 404
    assert client.get("/api/experiments/nope/events").status_code == 404
    assert client.post("/api/experiments/20990101-000000/cancel").status_code == 404


# ---- plot data from saved recordings ----

def test_recording_preview_only_for_the_experiments_own_recordings(fake_camera, settings, tmp_path):
    client = TestClient(create_app(settings, llm=HwLLM([hw("record"), "finish"], [[("record", {"seconds": 0.5})]]),
                                   recordings_dir=tmp_path / "evcam"))
    exp_id = start(client, "Record half a second.")
    body = wait_for(client, exp_id, ended)
    rid = next(e["data"]["result"]["recording_id"] for e in body["events"] if e["type"] == "tool.result")

    p = client.get(f"/api/experiments/{exp_id}/recordings/{rid}/preview").json()
    assert p["recording_id"] == rid and p["fake"] is True
    assert p["recorded_s"] == 0.5 and p["n_events"] > 0
    assert (p["image"]["width"], p["image"]["height"]) == (320, 180)
    assert {(h["x"], h["y"]) for h in p["hot_pixels"]} == {(1053, 704), (926, 151)}  # the fake camera's defects
    assert len(p["rate"]["on_hz"]) == len(p["rate"]["off_hz"]) == 50  # 0.5 s in 10 ms bins

    other = "20261016-000000-abcdef"  # well-formed, but not made by this experiment
    assert client.get(f"/api/experiments/{exp_id}/recordings/{other}/preview").status_code == 404
    assert client.get(f"/api/experiments/{exp_id}/recordings/..%2Fx/preview").status_code == 404
