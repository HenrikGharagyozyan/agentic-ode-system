"""Snapshots of the public behaviour, taken before the refactor (docs/refactor-plan.md, step 0).

The first run writes each file in tests/snapshots/ and passes; every later run fails if anything differs, key
order included. A refactor step must never rewrite these files. Delete a file only when a behaviour change is
wanted on purpose.
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from agentic_ode.agents import prompts
from agentic_ode.config import PROJECT_ROOT, Settings
from agentic_ode.graph import build_graph
from agentic_ode.hardware.evcam import fake
from agentic_ode.mcp_client import McpSession, evcam_session
from agentic_ode.mcp_servers import evcam_server
from tests.conftest import FakeLLM

SNAPSHOTS = Path(__file__).parent / "snapshots"


def check_snapshot(name: str, data) -> None:
    """Write tests/snapshots/<name> on the first run; afterwards the same JSON text must come out."""
    path = SNAPSHOTS / name
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"  # no sort_keys: the key order is checked too
    if not path.exists():
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(text)
        return
    assert text == path.read_text(), f"{name} differs from the snapshot"


# ---- paths ----

def test_project_root_is_the_repo():
    # PROJECT_ROOT is computed from config.py's own path (__file__): a moved config.py must still point here
    assert (PROJECT_ROOT / "pyproject.toml").is_file()


# ---- MCP tools ----

def test_mcp_tools_snapshot():
    # through our own client: the server runs as a subprocess, started by the same command as in a real run
    session = evcam_session(env={**os.environ, "EVCAM_FAKE": "1"}, timeout_s=60)
    with session:
        tools = session.tools()  # listing the tools does not open the camera

    async def output_schemas():  # McpSession.tools() keeps no output schema, so ask the same command once more
        transport = StdioTransport(command=session.command, args=session.args, env=session.env)
        async with Client(transport) as client:
            return {t.name: t.output_schema for t in await client.list_tools()}

    outputs = asyncio.run(output_schemas())
    check_snapshot("mcp_tools.json", [  # in the server's order: the model gets the tools in this order
        {"name": t.name, "description": t.description, "inputSchema": t.args_schema, "outputSchema": outputs[t.name]}
        for t in tools
    ])


# ---- graph ----

def test_graph_snapshot(tmp_path, settings):
    g = build_graph(FakeLLM(["finish"]), tmp_path, settings).get_graph()
    check_snapshot("graph.json", {
        "nodes": sorted(g.nodes),
        # data: the route name when it differs from the target node (e.g. "hardware" -> hw_decide)
        "edges": sorted([e.source, e.target, e.conditional, e.data] for e in g.edges),
    })


# ---- prompts ----

def test_prompts_snapshot():
    s = Settings(model="fake")  # the defaults a real run uses
    texts = {
        "orchestrator": prompts.orchestrator(s),
        "orchestrator_camera": prompts.orchestrator(s, camera=True),
        "scientist": prompts.scientist(s),
        "scientist_camera": prompts.scientist(s, camera=True),
        "general": prompts.general(s),
        "executor": prompts.executor(s),
        "hardware": prompts.hardware(s),
        "finish_not_allowed": prompts.FINISH_NOT_ALLOWED,
        "step_refused": prompts.step_refused("executor"),
    }
    # the data path contains the project folder; replace it so the snapshot is the same on every machine
    check_snapshot("prompts.json", {k: v.replace(str(PROJECT_ROOT), "<PROJECT_ROOT>") for k, v in texts.items()})


# ---- golden results of the fake camera ----

@pytest.fixture
def fake_server(tmp_path, monkeypatch):
    """Fake camera (pwm scene, seed 0), recordings and log in tmp_path, no camera open before and after."""
    monkeypatch.setenv("EVCAM_FAKE", "1")
    monkeypatch.delenv("EVCAM_FAKE_SCENE", raising=False)
    monkeypatch.setattr(evcam_server.instrument, "recordings_dir", tmp_path)
    monkeypatch.setattr(evcam_server.tool_log, "path", tmp_path / "tool_log.jsonl")
    evcam_server.instrument.close()
    yield
    evcam_server.instrument.close()


def test_fake_camera_results_snapshot(fake_server):
    hot = [list(p) for p in fake.HOT_PIXELS]
    results = []
    with McpSession(transport=evcam_server.mcp, timeout_s=60) as s:  # one session = one camera, like a real run

        def call(name, args=None):
            result = s.call(name, args)
            results.append({"tool": name, "args": args or {}, "result": result})
            return result

        call("camera_info")
        call("get_settings")
        call("find_hot_pixels")
        call("record", {"seconds": 1.0, "note": "golden"})  # hot pixels still active: a warning
        call("mask_pixels", {"pixels": hot})
        call("set_bias", {"name": "bias_diff_off", "value": 20})
        call("set_bias", {"name": "bias_diff", "value": 50})  # outside the recommended range: ok false
        call("set_roi", {"x": 440, "y": 160, "width": 400, "height": 400})
        call("get_settings")
        rec = call("record", {"seconds": 1.0})
        call("spectrum", {"recording_id": rec["recording_id"]})
        short = call("record", {"seconds": 0.5})
        call("spectrum", {"recording_id": short["recording_id"]})  # shorter than 1 s: a warning
        call("clear_roi")
        call("clear_mask")
        call("get_settings")

    # recording ids contain the date and random characters; replace them, keep everything else
    text = json.dumps(results)
    for r in results:
        if "recording_id" in r["result"]:
            text = text.replace(r["result"]["recording_id"], "<recording_id>")
    check_snapshot("evcam_results.json", json.loads(text))
