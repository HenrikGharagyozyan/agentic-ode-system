"""The generic entry point `python -m agentic_ode.mcp.serve NAME` (any registered instrument), as a subprocess."""

import json
import os
import sys

import agentic_ode.hardware.devices  # noqa: F401 - registers the known instruments
from agentic_ode.config import PROJECT_ROOT
from agentic_ode.hardware.standard import registry
from agentic_ode.mcp_client import McpSession, evcam_session


def tool_view(session) -> list:
    return [(t.name, t.description, t.args_schema) for t in session.tools()]


def test_evcam_is_registered():
    assert "evcam" in registry.names()


def test_serve_evcam_is_the_same_server_as_the_evcam_entry(tmp_path):
    env = {**os.environ, "EVCAM_FAKE": "1", "PYTHONPATH": str(PROJECT_ROOT)}
    log = tmp_path / "tool_log.jsonl"
    serve = McpSession(sys.executable, ["-m", "agentic_ode.mcp.serve", "evcam", "--log", str(log)], env=env,
                       timeout_s=60)
    with serve:
        generic = tool_view(serve)
        info = serve.call("camera_info")  # opens the fake camera; writes no recording
    with evcam_session(env=env, timeout_s=60) as s:
        assert generic == tool_view(s)
    assert info["ok"] and info["fake"] is True
    assert [json.loads(line)["tool"] for line in log.read_text().splitlines()] == ["camera_info"]
