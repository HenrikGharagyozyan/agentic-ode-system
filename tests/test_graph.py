"""Finish guard, decision limit and tool ownership, end to end through the compiled graph."""

from agentic_ode.agents import finish_allowed
from agentic_ode.graph import build_graph
from agentic_ode.state import initial_state
from tests.conftest import FakeLLM


def test_finish_guard_blocks_early_finish(run_graph, settings):
    llm = FakeLLM(["finish"])  # always finish, no executor turn yet
    out = run_graph(llm)
    assert out["decisions"] == 1
    assert llm.decision_calls == 1 + settings.max_finish_retries  # re-asked with the "not yet allowed" note
    assert out["stop_reason"].startswith("finish refused")
    assert not finish_allowed(out)


def test_finish_guard_allows_after_real_rmse(run_graph):
    out = run_graph(FakeLLM(["executor", "finish"]))
    assert out["stop_reason"] == "finish"
    assert out["runs"][0]["exit_code"] == 0 and out["runs"][0]["test_rmse"] == 0.02
    assert finish_allowed(out)


def test_guard_rejects_failed_script(run_graph):
    out = run_graph(FakeLLM(["executor", "finish"],
                            executor_script="print('FREE_RUN_TEST_RMSE: 0.02')\nraise SystemExit(1)\n"))
    assert not finish_allowed(out)
    assert out["stop_reason"] != "finish"


def test_rmse_typed_by_model_does_not_count():
    state = initial_state("task")
    state["dialogue"].append({"role": "executor", "content": "FREE_RUN_TEST_RMSE: 0.001"})
    assert not finish_allowed(state)  # no real run record behind it


def test_decision_limit(run_graph, settings):
    llm = FakeLLM(["general"])  # never finishes
    out = run_graph(llm)
    assert out["decisions"] == settings.max_decisions == 8
    assert sum(t["role"] == "orchestrator" for t in out["dialogue"]) == settings.max_decisions
    assert llm.decision_calls == settings.max_decisions
    assert "decision limit" in out["stop_reason"]


def test_only_executor_has_tools(tmp_path, settings):
    llm = FakeLLM(["finish"])
    build_graph(llm, tmp_path, settings)
    assert llm.bound_tools == [["write_file", "run_file"]]  # bound exactly once, by the executor


def test_executor_turn_contains_verbatim_tool_output(run_graph):
    out = run_graph(FakeLLM(["executor", "finish"]))
    executor_turn = next(t for t in out["dialogue"] if t["role"] == "executor")
    assert "=== tool output (verbatim) ===" in executor_turn["content"]
    assert "FREE_RUN_TEST_RMSE: 0.02" in executor_turn["content"]
    assert "Files written: fit_01.py" in executor_turn["content"]
