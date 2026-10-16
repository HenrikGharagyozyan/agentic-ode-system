"""The EventLab web API (M1): health, camera status (never opens the camera), the explicit camera check."""

from __future__ import annotations

import sys

import pytest
from fastapi.testclient import TestClient

from agentic_ode import __version__
from agentic_ode.hardware.devices.evcam.instrument import EvcamInstrument
from agentic_ode.mcp.client import McpSession
from agentic_ode.mcp.server_factory import ToolLog, build_server
from agentic_ode.web import app as web_app


def _no_session():
    raise AssertionError("the camera must not be opened here")


@pytest.fixture
def fake_env(monkeypatch):
    monkeypatch.setenv("EVCAM_FAKE", "1")
    monkeypatch.delenv("EVCAM_FAKE_SCENE", raising=False)


def test_health():
    r = TestClient(web_app.create_app()).get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"ok": True, "version": __version__}


@pytest.mark.parametrize("fake, mode", [("1", "simulated"), (None, "real")])
def test_status_never_opens_the_camera(monkeypatch, fake, mode):
    if fake:
        monkeypatch.setenv("EVCAM_FAKE", fake)
    else:
        monkeypatch.delenv("EVCAM_FAKE", raising=False)
    monkeypatch.setattr(web_app, "evcam_session", _no_session)
    body = TestClient(web_app.create_app()).get("/api/camera/status").json()
    assert body["mode"] == mode
    assert body["busy"] is False
    assert body["last_check"] is None
    assert isinstance(body["sdk_available"], bool)


def test_check_with_the_fake_camera(fake_env, monkeypatch, tmp_path):
    # in-memory MCP server with recordings and tool log in tmp_path: nothing is written to runs/evcam/
    server = build_server(EvcamInstrument(tmp_path), ToolLog(tmp_path / "tool_log.jsonl"))
    monkeypatch.setattr(web_app, "evcam_session", lambda: McpSession(transport=server, timeout_s=60))
    app = web_app.create_app()
    client = TestClient(app)

    check = client.post("/api/camera/check").json()
    assert check["ok"] is True
    assert check["result"]["fake"] is True
    assert check["result"]["width"] == 1280
    assert "checked_at" in check
    assert (tmp_path / "tool_log.jsonl").read_text().count('"tool": "camera_info"') == 1  # it went through MCP

    status = client.get("/api/camera/status").json()
    assert status["last_check"] == check
    assert status["busy"] is False  # the lock is free again
    assert not app.state.camera_lock.locked()


def test_check_returns_409_when_the_camera_is_busy(monkeypatch):
    monkeypatch.setattr(web_app, "evcam_session", _no_session)
    app = web_app.create_app()
    client = TestClient(app)
    app.state.camera_lock.acquire()  # like a running experiment
    try:
        assert client.get("/api/camera/status").json()["busy"] is True
        r = client.post("/api/camera/check")
        assert r.status_code == 409
        assert client.get("/api/camera/status").json()["last_check"] is None
    finally:
        app.state.camera_lock.release()


def test_check_reports_a_server_that_does_not_start(monkeypatch):
    # a "server" that exits at once: the session cannot open
    monkeypatch.setattr(web_app, "evcam_session",
                        lambda: McpSession(sys.executable, ["-c", "import sys; sys.exit(3)"], timeout_s=30))
    app = web_app.create_app()
    r = TestClient(app).post("/api/camera/check")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "could not open the MCP session" in body["error"]
    assert "result" not in body
    assert not app.state.camera_lock.locked()
