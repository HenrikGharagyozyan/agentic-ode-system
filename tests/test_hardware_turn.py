"""HW3 (d): the hardware turn. A scripted fake LLM drives the graph; the camera is the evcam MCP server with the fake
camera, reached through agentic_ode.mcp_client in memory (real MCP protocol, no subprocess, files in tmp_path)."""

import json
import logging

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool
from langgraph.types import Command

from agentic_ode import runner
from agentic_ode.agents import Decision, prompts
from agentic_ode.cli import _auto_ask
from agentic_ode.graph import build_graph
from agentic_ode.mcp_client import McpSession
from agentic_ode.mcp_servers import evcam_server
from agentic_ode.state import Role, initial_state, render_dialogue
from agentic_ode.transcript import save_transcript
from tests.conftest import FakeLLM

CONFIG = {"recursion_limit": 150, "configurable": {"thread_id": "hw"}}
HOT = [[1053, 704], [926, 151]]


class HwLLM(FakeLLM):
    """FakeLLM plus the hardware turn: each hw_decide call takes the next item of camera_script, a list of
    (tool name, args) calls ([] = the model answers in text). Records what the orchestrator read."""

    def __init__(self, decisions, camera_script=(), tool_choice_error=None):
        super().__init__(decisions)
        self.camera_script = list(camera_script)
        self.tool_choice_error = tool_choice_error  # raised by the binding WITH tool_choice, if set
        self.orchestrator_inputs: list[str] = []

    def with_structured_output(self, schema, **kw):
        inner, outer = super().with_structured_output(schema, **kw), self

        class _Recording:
            def invoke(self, messages):
                outer.orchestrator_inputs.append(messages[-1].content)
                return inner.invoke(messages)

        return _Recording()

    def bind_tools(self, tools, **kwargs):
        if "record" not in [t.name for t in tools]:
            return super().bind_tools(tools, **kwargs)  # the normal executor tool loop
        self.bound_tools.append([t.name for t in tools])
        self.tool_choices.append(kwargs.get("tool_choice"))
        outer, forced = self, "tool_choice" in kwargs

        class _Camera:
            def invoke(self, messages):
                if forced and outer.tool_choice_error:
                    raise outer.tool_choice_error
                calls = outer.camera_script.pop(0)
                return AIMessage(content="" if calls else "I would rather not.",
                                 tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(calls)])

        return _Camera()


def hw(instruction: str) -> Decision:
    return Decision(next="hardware", instruction=instruction, reason="test")


@pytest.fixture
def session(tmp_path, monkeypatch):
    """The evcam MCP server with the fake camera (pwm scene), in memory; recordings and log in tmp_path."""
    monkeypatch.setenv("EVCAM_FAKE", "1")
    monkeypatch.setenv("EVCAM_FAKE_SCENE", "pwm")
    monkeypatch.setattr(evcam_server, "RECORDINGS_DIR", tmp_path / "evcam")
    monkeypatch.setattr(evcam_server, "LOG_FILE", tmp_path / "evcam" / "tool_log.jsonl")
    evcam_server._close_camera()
    with McpSession(transport=evcam_server.mcp, timeout_s=60) as s:
        yield s
    evcam_server._close_camera()


def graph_with_camera(llm, session, tmp_path, settings):
    return build_graph(llm, tmp_path, settings, camera_tools=session.tools())


def roles(state):
    return [t["role"] for t in state["dialogue"]]


def notes(state):
    return [t["content"] for t in state["dialogue"] if t["role"] == Role.SYSTEM]


def paused(graph):
    interrupts = graph.get_state(CONFIG).interrupts
    return interrupts[0].value if interrupts else None


# ---- the full loop ----

def test_measurement_runs_directly_and_the_orchestrator_reads_it(session, tmp_path, settings):
    llm = HwLLM([hw("record 0.5 s"), "finish"], [[("record", {"seconds": 0.5})]])
    graph = graph_with_camera(llm, session, tmp_path, settings)
    state = graph.invoke(initial_state("measure the light"), CONFIG)

    assert paused(graph) is None  # a measurement is not asked
    assert roles(state) == ["user", "orchestrator", "hardware", "orchestrator"]
    turn = state["dialogue"][2]["content"]
    assert turn.startswith('camera call: record({"seconds": 0.5})\nresult: {"ok": true, "recording_id": "')
    (rec,) = state["camera"]
    assert rec["tool"] == "record" and rec["ok"] and rec["turn_index"] == 2 and rec["result_line"] in turn
    assert turn in llm.orchestrator_inputs[1]  # the next orchestrator decision read the result
    assert state["stop_reason"] == "finish"  # allowed after a successful measurement
    assert llm.tool_choices == [None, "any"]  # normal loop: free; camera binding: forced to one call


def test_finish_needs_a_measurement_not_just_a_reading(session, tmp_path, settings):
    llm = HwLLM([hw("camera info"), "finish"], [[("camera_info", {})]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert state["camera"][0]["tool"] == "camera_info" and state["camera"][0]["ok"]
    assert state["stop_reason"] == "finish refused by guard (no RMSE from a successful run)"


def test_spectrum_without_a_peak_allows_finish(session, tmp_path, settings, monkeypatch):
    monkeypatch.setenv("EVCAM_FAKE_SCENE", "steady")  # no flicker: dominant_hz is null, still a valid result
    rid = session.call("record", {"seconds": 1.0})["recording_id"]  # a recording made before the graph runs
    llm = HwLLM([hw("spectrum"), "finish"], [[("spectrum", {"recording_id": rid})]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    result = json.loads(state["camera"][0]["result_line"])
    assert result["ok"] and result["dominant_hz"] is None
    assert state["stop_reason"] == "finish"


def test_setting_change_asks_then_runs_and_the_session_keeps_it(session, tmp_path, settings):
    llm = HwLLM([hw("mask the hot pixels"), hw("show settings"), hw("record 0.5 s"), "finish"],
                [[("mask_pixels", {"pixels": HOT})], [("get_settings", {})], [("record", {"seconds": 0.5})]])
    graph = graph_with_camera(llm, session, tmp_path, settings)
    graph.invoke(initial_state("t"), CONFIG)

    question = paused(graph)
    assert question["kind"] == "hard" and 'mask_pixels({"pixels": [[1053, 704], [926, 151]]})' in question["question"]
    state = graph.invoke(Command(resume="yes"), CONFIG)

    assert paused(graph) is None
    assert roles(state) == ["user", "orchestrator", "human", "hardware", "orchestrator", "hardware",
                            "orchestrator", "hardware", "orchestrator"]
    settings_result = json.loads(state["camera"][1]["result_line"])
    assert settings_result["mask"] == HOT  # same session, same camera object: the mask is still set
    record_result = json.loads(state["camera"][2]["result_line"])
    assert "warning" not in record_result  # both hot pixels masked
    assert state["stop_reason"] == "finish"


def test_refusal_blocks_only_that_exact_call(session, tmp_path, settings):
    a, b = {"pixels": [[1, 2]]}, {"pixels": [[3, 4]]}
    llm = HwLLM([hw("mask 1,2")] * 3, [[("mask_pixels", a)], [("mask_pixels", a)], [("mask_pixels", b)]])
    graph = graph_with_camera(llm, session, tmp_path, settings)
    graph.invoke(initial_state("t"), CONFIG)
    assert "[[1, 2]]" in paused(graph)["question"]

    state = graph.invoke(Command(resume="no"), CONFIG)
    # refused -> note; the same call again -> blocked without a question; other args -> asked again
    assert state["refused_calls"] == [{"name": "mask_pixels", "args": a}]
    assert 'The human refused the camera call mask_pixels({"pixels": [[1, 2]]}). It did NOT run.' in notes(state)
    assert any("earlier. It did NOT run and is not asked again" in n for n in notes(state))
    assert "[[3, 4]]" in paused(graph)["question"]
    assert Role.HARDWARE not in roles(state) and state["camera"] == []  # nothing ran


def test_auto_approve_through_the_runner(session, tmp_path, settings, monkeypatch, capsys):
    monkeypatch.setattr(runner, "evcam_session", lambda: McpSession(transport=evcam_server.mcp, timeout_s=60))
    task = tmp_path / "task.md"
    task.write_text("mask the hot pixels and record")
    llm = HwLLM([hw("mask"), hw("record 0.5 s"), "finish"],
                [[("mask_pixels", {"pixels": HOT})], [("record", {"seconds": 0.5})]])
    outcome = runner.run_task(task, settings, llm=llm, ask=_auto_ask, camera=True)

    assert outcome.state["stop_reason"] == "finish"
    assert "your reply> yes   (auto-approve)" in capsys.readouterr().out
    assert ["human", "hardware"] == roles(outcome.state)[2:4]
    assert "of 20 decisions" in llm.orchestrator_inputs[0]  # camera runs get 20 decisions
    assert (runner.CAMERA_MAX_DECISIONS, runner.CAMERA_RECURSION_LIMIT) == (20, 150)
    assert "## Turn 3 [executor (hardware)]" in (outcome.run_dir / "transcript.md").read_text()


# ---- the model does not make exactly one call ----

def test_zero_tool_calls_gives_a_note(session, tmp_path, settings):
    llm = HwLLM([hw("record"), "finish"], [[]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert "The hardware turn made no camera call; the model answered: I would rather not." in notes(state)
    assert state["camera"] == []


def test_more_than_one_call_runs_only_the_first(session, tmp_path, settings):
    llm = HwLLM([hw("info"), "finish"], [[("camera_info", {}), ("get_settings", {})]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert "only the first one runs. Dropped: get_settings({})" in notes(state)[0]
    assert [r["tool"] for r in state["camera"]] == ["camera_info"]


# ---- tool_choice fallback ----

def test_tool_choice_rejected_retries_once_without_it(session, tmp_path, settings, caplog):
    error = ValueError("Error code: 400 - {'error': {'message': 'tool_choice is not supported by this model'}}")
    llm = HwLLM([hw("info"), hw("info again"), "finish"], [[("camera_info", {})], [("get_settings", {})]],
                tool_choice_error=error)
    with caplog.at_level(logging.WARNING):
        state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert [r["tool"] for r in state["camera"]] == ["camera_info", "get_settings"]  # both hardware turns ran
    assert llm.tool_choices == [None, "any", None]  # rebound ONCE without tool_choice, then kept
    assert sum("rejected tool_choice" in r.message for r in caplog.records) == 1


def test_tool_choice_fallback_then_text_answer_gives_the_note(session, tmp_path, settings):
    llm = HwLLM([hw("info"), "finish"], [[]], tool_choice_error=ValueError("tool_choice not supported"))
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert any("made no camera call" in n for n in notes(state))


def test_other_provider_errors_are_not_swallowed(session, tmp_path, settings):
    llm = HwLLM([hw("info"), "finish"], [[("camera_info", {})]], tool_choice_error=RuntimeError("rate limited"))
    with pytest.raises(RuntimeError, match="rate limited"):
        graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)


# ---- edge cases without the real server ----

def test_no_camera_in_this_run(tmp_path, settings):
    llm = HwLLM([hw("record"), "finish"])
    state = build_graph(llm, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert any("There is no camera in this run" in n for n in notes(state))


def test_a_failing_camera_call_is_reported_not_raised(tmp_path, settings):
    def broken(**kw):
        raise TimeoutError("the MCP server did not answer")
    tools = [StructuredTool.from_function(broken, name="record", description="stub")]
    llm = HwLLM([hw("record"), "finish"], [[("record", {"seconds": 1.0})]])
    state = build_graph(llm, tmp_path, settings, camera_tools=tools).invoke(initial_state("t"), CONFIG)
    (rec,) = state["camera"]
    assert rec["ok"] is False and "TimeoutError: the MCP server did not answer" in rec["result_line"]
    assert state["stop_reason"].startswith("finish refused by guard")  # a failed measurement does not count


def test_unknown_tool_name_gives_a_note(session, tmp_path, settings):
    llm = HwLLM([hw("x"), "finish"], [[("launch_rocket", {})]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert "launch_rocket is not a camera tool. No camera call was made." in notes(state)


# ---- prompts and labels ----

def test_camera_paragraphs_only_in_camera_runs(settings):
    assert prompts.CAMERA_ORCHESTRATOR in prompts.orchestrator(settings, camera=True)
    assert "next = hardware" in prompts.CAMERA_ORCHESTRATOR
    assert "hardware" not in prompts.orchestrator(settings)  # HW1/HW2 runs: unchanged
    assert prompts.CAMERA_SCIENTIST in prompts.scientist(settings, camera=True)
    assert "event camera" not in prompts.scientist(settings)  # (the HW2 bench has its own simulated camera)
    assert "exactly ONE tool call" in prompts.hardware(settings)


def test_hardware_turn_label(tmp_path):
    state = {**initial_state("t"), "dialogue": [{"role": "hardware", "content": "camera call: x({})"}]}
    assert "[executor (hardware)]" in render_dialogue(state["dialogue"])
    save_transcript(tmp_path, state, {"model": "m", "task_file": "t.md"})
    assert "## Turn 0 [executor (hardware)]" in (tmp_path / "transcript.md").read_text()
