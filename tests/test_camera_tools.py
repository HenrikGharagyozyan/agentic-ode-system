"""HW3 (c): only the executor holds the event camera's MCP tools, in a separate binding (camera_llm) that the
normal tool loop never uses. The real-session tests start the MCP server with the fake camera and only LIST its
tools (no tool is called, so nothing is written to runs/evcam/)."""

import json
import os
import threading

import pytest
from langchain_core.tools import StructuredTool

from agentic_ode.agents import Executor
from agentic_ode.graph import build_graph
from agentic_ode.mcp_client import evcam_session
from agentic_ode.runner import run_task
from agentic_ode.state import initial_state
from tests.conftest import FakeLLM

CODING_TOOLS = ["write_file", "run_file", "run_focus_search"]
CAMERA_TOOLS = ["camera_info", "get_settings", "set_bias", "find_hot_pixels", "mask_pixels", "clear_mask",
                "set_roi", "clear_roi", "record", "spectrum"]


def stub_camera_tools(calls: list) -> list[StructuredTool]:
    """Tools with the camera tool names that only record that they were called."""
    def make(name):
        return StructuredTool.from_function(lambda **kw: calls.append(name) or "{}", name=name,
                                            description=f"stub {name}")
    return [make(n) for n in CAMERA_TOOLS]


def test_only_the_executor_holds_the_camera_tools(tmp_path, settings):
    llm = FakeLLM(["finish"])
    build_graph(llm, tmp_path, settings, camera_tools=stub_camera_tools([]))
    # exactly two bind_tools calls in the whole graph, both from Executor.__init__: the normal tool loop and the
    # separate camera binding. Orchestrator, scientist and general never call bind_tools.
    assert llm.bound_tools == [CODING_TOOLS, CAMERA_TOOLS]


def test_camera_tools_are_a_separate_binding(tmp_path, settings):
    ex = Executor(FakeLLM([]), settings, tmp_path, camera_tools=stub_camera_tools([]))
    assert [t.name for t in ex.tools] == CODING_TOOLS  # the normal tool loop: no camera tools, so no bypass
    assert [t.name for t in ex.camera_tools] == CAMERA_TOOLS
    assert ex.camera_llm is not None


def test_without_camera_nothing_changes(tmp_path, settings):
    llm = FakeLLM([])
    ex = Executor(llm, settings, tmp_path)
    assert ex.camera_tools == [] and ex.camera_llm is None
    assert llm.bound_tools == [CODING_TOOLS]


def test_normal_executor_turn_never_calls_a_camera_tool(tmp_path, settings):
    calls = []
    graph = build_graph(FakeLLM(["executor", "finish"]), tmp_path, settings, camera_tools=stub_camera_tools(calls))
    state = graph.invoke(initial_state("task"), config={"recursion_limit": 60, "configurable": {"thread_id": "t"}})
    assert state["stop_reason"] == "finish" and len(state["runs"]) == 1  # the executor wrote and ran a script
    assert calls == []


# ---- with the real MCP server (fake camera) ----

@pytest.fixture
def fake_camera_env(monkeypatch):
    monkeypatch.setenv("EVCAM_FAKE", "1")


def test_real_session_tools_go_to_the_executor_only(tmp_path, settings, fake_camera_env):
    with evcam_session(env=dict(os.environ), timeout_s=60) as session:
        llm = FakeLLM(["finish"])
        build_graph(llm, tmp_path, settings, camera_tools=session.tools())
    assert llm.bound_tools == [CODING_TOOLS, CAMERA_TOOLS]


def test_run_task_with_camera_opens_and_closes_one_session(tmp_path, settings, fake_camera_env):
    task = tmp_path / "task.md"
    task.write_text("solve it")
    llm = FakeLLM(["executor", "finish"])
    outcome = run_task(task, settings, llm=llm, camera=True)
    assert outcome.state["stop_reason"] == "finish"
    assert llm.bound_tools == [CODING_TOOLS, CAMERA_TOOLS]
    assert json.loads((outcome.run_dir / "transcript.json").read_text())["camera"] is True
    assert [t for t in threading.enumerate() if t.name == "mcp-session"] == []  # the session was closed


def test_run_task_without_camera_starts_no_server(tmp_path, settings):
    task = tmp_path / "task.md"
    task.write_text("solve it")
    llm = FakeLLM(["executor", "finish"])
    outcome = run_task(task, settings, llm=llm)
    assert llm.bound_tools == [CODING_TOOLS]
    assert json.loads((outcome.run_dir / "transcript.json").read_text())["camera"] is False
