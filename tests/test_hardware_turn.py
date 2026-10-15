"""HW3 (d): the hardware turn. A scripted fake LLM drives the graph; the camera is the evcam MCP server with the fake
camera, reached through agentic_ode.mcp_client in memory (real MCP protocol, no subprocess, files in tmp_path)."""

import json
import logging
import time

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.types import Command

from agentic_ode.agents import Decision, Executor, prompts
from agentic_ode.core.dialogue import render_dialogue
from agentic_ode.core.state import Role, initial_state
from agentic_ode.graph import build_graph
from agentic_ode.graph.nodes.hardware_turn import schema_errors, schema_of
from agentic_ode.mcp import evcam_server
from agentic_ode.mcp.client import McpSession
from agentic_ode.runtime import runner
from agentic_ode.runtime.cli import _auto_ask
from agentic_ode.runtime.run_store import save_transcript
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
    monkeypatch.setattr(evcam_server.instrument, "recordings_dir", tmp_path / "evcam")
    monkeypatch.setattr(evcam_server.tool_log, "path", tmp_path / "evcam" / "tool_log.jsonl")
    evcam_server.instrument.close()
    with McpSession(transport=evcam_server.mcp, timeout_s=60) as s:
        yield s
    evcam_server.instrument.close()


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


# ---- wrong arguments and repeated failures (first real run, 2026-10-15) ----

def test_unknown_arguments_are_dropped_and_named(session, tmp_path, settings):
    # the real model called camera_info({"serial": "None"}); camera_info has no parameters
    llm = HwLLM([hw("info"), hw("record"), "finish"],
                [[("camera_info", {"serial": "None"})], [("record", {"seconds": 0.5, "foo": 1})]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    info, rec = [t["content"] for t in state["dialogue"] if t["role"] == Role.HARDWARE]
    ok = '\nresult: {"ok": true'
    assert info.startswith("camera call: camera_info({})\nignored unknown argument(s): serial" + ok)
    assert rec.startswith('camera call: record({"seconds": 0.5})\nignored unknown argument(s): foo' + ok)
    assert [(r["tool"], r["args"], r["ok"]) for r in state["camera"]] == [("camera_info", {}, True),
                                                                          ("record", {"seconds": 0.5}, True)]
    assert state["stop_reason"] == "finish"


def test_camera_schemas_say_no_other_arguments(session, tmp_path, settings):
    no_flag = StructuredTool(name="x", description="d", func=lambda a: "{}",
                             args_schema={"type": "object", "properties": {"a": {"type": "integer"}}})
    ex = Executor(FakeLLM([]), settings, tmp_path, camera_tools=session.tools() + [no_flag])
    assert all(t.args_schema["additionalProperties"] is False for t in ex.camera_tools)
    assert convert_to_openai_tool(ex.camera_tools[-1])["function"]["parameters"]["additionalProperties"] is False
    assert "additionalProperties" not in no_flag.args_schema  # the original tool is not changed


def test_the_same_failed_call_is_not_run_again(session, tmp_path, settings):
    llm = HwLLM([hw("record 10 s")] * 3 + ["finish"],
                [[("record", {"seconds": 10})],                # fails: more than 5 s
                 [("record", {"seconds": 10, "extra": 1})],    # the same call after dropping "extra": not run
                 [("record", {"seconds": 1.0})]])              # a changed call: runs
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert [(r["args"], r["ok"]) for r in state["camera"]] == [({"seconds": 10}, False), ({"seconds": 1.0}, True)]
    (note,) = [n for n in notes(state) if "already failed" in n]
    assert note.startswith('The camera call record({"seconds": 10}) did NOT run: this exact call already failed '
                           "with: ValueError: duration must be between 0.05 and 5.0 s")
    assert note.endswith("; change the call or choose another step.")
    assert state["stop_reason"] == "finish"


def test_bad_arguments_never_reach_the_approval_question(session, tmp_path, settings):
    # UC1, decision 2 (run 20261015-171638): pixels as {"x", "y"} objects were APPROVED, then refused by the server
    bad = {"pixels": [{"x": 926, "y": 151}, {"x": 1053, "y": 704}]}
    llm = HwLLM([hw("mask"), hw("mask")], [[("mask_pixels", bad)], [("mask_pixels", {"pixels": HOT})]])
    graph = graph_with_camera(llm, session, tmp_path, settings)
    graph.invoke(initial_state("t"), CONFIG)

    # the graph pauses only for the second, valid call; the bad one became a note
    assert "[[1053, 704], [926, 151]]" in paused(graph)["question"]
    state = graph.get_state(CONFIG).values
    assert roles(state) == ["user", "orchestrator", "system", "orchestrator"]
    (note,) = notes(state)
    assert note.startswith('The camera call mask_pixels({"pixels": [{"x": 926, "y": 151}, {"x": 1053, "y": 704}]}) '
                           "did NOT run: its arguments do not match the tool's schema: pixels/0: ")
    assert "is not of type 'array'" in note and note.endswith("Fix the arguments and try again.")
    assert state["camera"] == [] and state["refused_calls"] == []


def test_bad_arguments_of_a_measurement_are_not_sent(session, tmp_path, settings):
    llm = HwLLM([hw("record"), "finish"], [[("record", {"seconds": "two"})]])
    state = graph_with_camera(llm, session, tmp_path, settings).invoke(initial_state("t"), CONFIG)
    assert any("seconds: 'two' is not of type 'number'" in n for n in notes(state))
    assert state["camera"] == []  # no camera call was made


def test_schema_errors_on_the_real_schemas(session):
    schemas = {t.name: schema_of(t) for t in session.tools()}
    assert schema_errors(schemas["mask_pixels"], {"pixels": [[926, 151]]}) == []
    assert schema_errors(schemas["set_bias"], {"name": "bias_fo", "value": 20.0}) == []  # 20.0 is an integer value
    assert schema_errors(schemas["set_bias"], {"name": "bias_fo"}) == ["arguments: 'value' is a required property"]
    assert schema_errors(schemas["set_bias"], {"name": "bias_fo", "value": "20"}) == [
        "value: '20' is not of type 'integer'"]  # stricter than the server, which would accept "20"


class RetryLLM:
    """A model whose bindings are real LangChain runnables (so with_retry really retries). The binding WITH
    tool_choice fails with `forced_errors` one after the other, then answers; calls are counted."""

    def __init__(self, forced_errors):
        self.forced_errors = list(forced_errors)
        self.calls = {"forced": 0, "free": 0}

    def bind_tools(self, tools, **kwargs):
        kind = "forced" if "tool_choice" in kwargs else "free"

        def run(messages):
            self.calls[kind] += 1
            if kind == "forced" and self.forced_errors:
                raise self.forced_errors.pop(0)
            return AIMessage(content="", tool_calls=[{"name": "camera_info", "args": {}, "id": "c0"}])

        return RunnableLambda(run)


def camera_executor(llm, tmp_path, settings):
    tool = StructuredTool.from_function(lambda: "{}", name="camera_info", description="stub")
    return Executor(llm, settings, tmp_path, camera_tools=[tool])


def test_tool_choice_rejection_is_not_retried(tmp_path, settings):
    llm = RetryLLM([ValueError("Error code: 400 - tool_choice is not supported by this model")] * 10)
    ex = camera_executor(llm, tmp_path, settings)
    t0 = time.monotonic()
    reply = ex.camera_reply([HumanMessage("x")])
    assert reply.tool_calls[0]["name"] == "camera_info"
    assert llm.calls == {"forced": 1, "free": 1}  # one forced try (no retries), then one try without tool_choice
    assert time.monotonic() - t0 < 1  # no retry waits
    ex.camera_reply([HumanMessage("y")])
    assert llm.calls == {"forced": 1, "free": 2}  # the binding without tool_choice is kept


def test_other_errors_still_get_the_normal_retries(tmp_path, settings):
    llm = RetryLLM([RuntimeError("provider overloaded")])  # fails once, then works
    reply = camera_executor(llm, tmp_path, settings).camera_reply([HumanMessage("x")])
    assert reply.tool_calls[0]["name"] == "camera_info"
    assert llm.calls == {"forced": 2, "free": 0}  # retried with tool_choice, never fell back


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
    assert ("After a setting change, record again before analysing; a recording reflects the settings at the time "
            "it was made.") in prompts.CAMERA_ORCHESTRATOR
    assert ("Read every warning in a hardware result and act on it before finishing. Report the fundamental "
            "frequency\n  (fundamental_hz), not the strongest peak.") in prompts.CAMERA_ORCHESTRATOR
    assert "hardware" not in prompts.orchestrator(settings)  # HW1/HW2 runs: unchanged
    assert prompts.CAMERA_SCIENTIST in prompts.scientist(settings, camera=True)
    assert "event camera" not in prompts.scientist(settings)  # (the HW2 bench has its own simulated camera)
    assert "exactly ONE tool call" in prompts.hardware(settings)
    assert "A tool without parameters is called with an empty argument object {}." in prompts.hardware(settings)


def test_hardware_turn_label(tmp_path):
    state = {**initial_state("t"), "dialogue": [{"role": "hardware", "content": "camera call: x({})"}]}
    assert "[executor (hardware)]" in render_dialogue(state["dialogue"])
    save_transcript(tmp_path, state, {"model": "m", "task_file": "t.md"})
    assert "## Turn 0 [executor (hardware)]" in (tmp_path / "transcript.md").read_text()
