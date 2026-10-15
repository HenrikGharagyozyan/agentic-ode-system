"""Finish guard, decision limit and tool ownership, end to end through the compiled graph."""

from langgraph.types import Command

from agentic_ode.agents import Decision, finish_allowed
from agentic_ode.core.state import initial_state
from agentic_ode.graph import build_graph
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
    assert llm.bound_tools == [["write_file", "run_file", "run_focus_search"]]  # bound exactly once, by the executor


def test_checkpointer_saves_final_state(tmp_path, settings):
    graph = build_graph(FakeLLM(["executor", "finish"]), tmp_path, settings)
    config = {"recursion_limit": 60, "configurable": {"thread_id": "t1"}}
    out = graph.invoke(initial_state("task"), config=config)
    saved = graph.get_state(config)
    assert saved.values == out  # the checkpointer holds the same final state
    assert saved.next == ()  # the graph reached END; nothing is waiting


def test_human_decision_pauses_the_graph(run_graph):
    out = run_graph(FakeLLM(["human"]))
    assert out["decisions"] == 1
    assert out["route"] == "human"
    assert "__interrupt__" in out  # invoke returned early: the graph waits for a reply


def test_graph_pauses_on_human_and_resumes(tmp_path, settings):
    graph = build_graph(FakeLLM(["human", "finish"]), tmp_path, settings)
    config = {"recursion_limit": 60, "configurable": {"thread_id": "t1"}}

    out = graph.invoke(initial_state("task"), config=config)
    assert graph.get_state(config).next == ("human",)  # paused inside the human node
    assert set(out["__interrupt__"][0].value) == {"question", "kind"}

    out = graph.invoke(Command(resume="use reading A"), config=config)
    assert out["dialogue"][2] == {"role": "human", "content": "use reading A"}
    assert [t["role"] for t in out["dialogue"]] == ["user", "orchestrator", "human", "orchestrator"]


HARD_RUN = Decision(next="human", instruction="write and run fit_01.py", reason="the coding tool needs approval",
                    kind="hard", proposed_next="executor", question="Run the coding tool now?")
HARD_FINISH = Decision(next="human", instruction="finish and report", reason="finishing needs approval",
                       kind="hard", proposed_next="finish", question="Finish and report the RMSE?")


def ask_then_reply(llm, tmp_path, settings, reply):
    """Run until the human node pauses, then resume with `reply`. Returns the state at the end."""
    graph = build_graph(llm, tmp_path, settings)
    config = {"recursion_limit": 60, "configurable": {"thread_id": "t1"}}
    graph.invoke(initial_state("task"), config=config)
    return graph.invoke(Command(resume=reply), config=config)


def test_refused_step_does_not_run(tmp_path, settings):
    llm = FakeLLM([HARD_RUN, "finish"])
    out = ask_then_reply(llm, tmp_path, settings, "no, not yet")
    assert llm.scripts == 0 and not list(tmp_path.glob("*.py"))  # executor never called, no file written
    assert "executor" not in [t["role"] for t in out["dialogue"]]
    note = next(t for t in out["dialogue"] if t["role"] == "system")
    assert "refused the step 'executor'" in note["content"] and "did NOT run" in note["content"]


def test_refused_step_is_not_chosen_again(tmp_path, settings):
    llm = FakeLLM([HARD_RUN, "executor"])  # after the refuse, the model keeps choosing executor
    out = ask_then_reply(llm, tmp_path, settings, "no")
    assert llm.decision_calls == 1 + 1 + settings.max_finish_retries  # question, then blocked and re-asked
    assert out["stop_reason"] == "guard: the model chose the refused step 'executor' again"
    assert llm.scripts == 0 and not list(tmp_path.glob("*.py"))
    assert "executor" not in [t["role"] for t in out["dialogue"]]


def test_accepted_step_runs(tmp_path, settings):
    llm = FakeLLM([HARD_RUN, "finish"])
    out = ask_then_reply(llm, tmp_path, settings, "yes, run it")
    assert [t["role"] for t in out["dialogue"]] == ["user", "orchestrator", "human", "executor", "orchestrator"]
    assert (tmp_path / "fit_01.py").exists() and out["runs"][0]["exit_code"] == 0
    assert out["stop_reason"] == "finish"


def test_accepted_finish_still_needs_the_guard(tmp_path, settings):
    llm = FakeLLM([HARD_FINISH, "general"])  # no executor run, so no real RMSE
    out = ask_then_reply(llm, tmp_path, settings, "yes")
    roles = [t["role"] for t in out["dialogue"]]
    assert roles[:5] == ["user", "orchestrator", "human", "system", "orchestrator"]
    assert "finish guard refused it" in out["dialogue"][3]["content"]
    assert out["stop_reason"] != "finish"


def test_ambiguous_reply_goes_back_to_orchestrator(tmp_path, settings):
    ask = Decision(next="human", instruction="ask which reading", reason="two readings",
                   kind="ambiguous", question="Test RMSE or train RMSE?")
    llm = FakeLLM([ask, "finish"])
    out = ask_then_reply(llm, tmp_path, settings, "yes, test RMSE")
    assert llm.scripts == 0  # starts with "yes", but an ambiguous answer never runs a step itself
    assert [t["role"] for t in out["dialogue"]][:4] == ["user", "orchestrator", "human", "orchestrator"]


def test_decision_as_turn_shows_human_fields():
    ask = Decision(next="human", instruction="run fit_01.py", reason="needs approval",
                   kind="hard", proposed_next="executor", question="Run the coding tool now?")
    text = ask.as_turn()
    assert "kind: hard" in text and "proposed_next: executor" in text
    assert "question: Run the coding tool now?" in text
    normal = Decision(next="scientist", instruction="plan", reason="no plan yet")
    assert normal.as_turn().count("\n") == 2  # still only next / instruction / reason


def test_executor_turn_contains_verbatim_tool_output(run_graph):
    out = run_graph(FakeLLM(["executor", "finish"]))
    executor_turn = next(t for t in out["dialogue"] if t["role"] == "executor")
    assert "=== tool output (verbatim) ===" in executor_turn["content"]
    assert "FREE_RUN_TEST_RMSE: 0.02" in executor_turn["content"]
    assert "Files written: fit_01.py" in executor_turn["content"]
