"""The runner writes transcripts; the report copies RMSE lines from the committed run verbatim."""

import json
from pathlib import Path

import pytest

from agentic_ode.agents import Decision
from agentic_ode.cli import _ask
from agentic_ode.reporting.builder import render_html
from agentic_ode.runner import run_task
from agentic_ode.transcript import load_transcript
from tests.conftest import FakeLLM

COMMITTED_RUN = Path(__file__).resolve().parent.parent / "runs" / "20261009-201921"


def test_run_task_saves_transcript(tmp_path, settings):
    task = tmp_path / "task.md"
    task.write_text("solve it")
    seen = []
    outcome = run_task(task, settings, on_turn=seen.append, llm=FakeLLM(["executor", "finish"]))

    data = json.loads((outcome.run_dir / "transcript.json").read_text())
    assert data["stop_reason"] == "finish" and data["decisions"] == 2
    assert data["dialogue"][0] == {"role": "user", "content": "solve it"}
    assert [t["role"] for t in seen] == [t["role"] for t in data["dialogue"]]
    assert (outcome.run_dir / "transcript.md").exists()
    assert (outcome.run_dir / "fit_01.py").exists()


HARD_RUN = Decision(next="human", instruction="write and run fit_01.py", reason="the coding tool needs approval",
                    kind="hard", proposed_next="executor", question="Run the coding tool now?")


def test_run_task_asks_and_resumes(tmp_path, settings):
    task = tmp_path / "task.md"
    task.write_text("solve it")
    asked = []

    def fake_ask(payload):
        asked.append(payload)
        return "yes"

    outcome = run_task(task, settings, llm=FakeLLM([HARD_RUN, "finish"]), ask=fake_ask)

    assert asked == [{"question": "Run the coding tool now?", "kind": "hard"}]
    data = json.loads((outcome.run_dir / "transcript.json").read_text())
    assert [t["role"] for t in data["dialogue"]] == ["user", "orchestrator", "human", "executor", "orchestrator"]
    assert data["dialogue"][2]["content"] == "yes"
    assert "[human]" in (outcome.run_dir / "transcript.md").read_text()
    assert data["stop_reason"] == "finish"


def test_run_task_without_ask_fails_loudly(tmp_path, settings):
    task = tmp_path / "task.md"
    task.write_text("solve it")
    with pytest.raises(RuntimeError, match="no ask function"):
        run_task(task, settings, llm=FakeLLM([HARD_RUN]))


def test_cli_ask_prints_question_and_reads_reply(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "yes, run it")
    assert _ask({"question": "Run the coding tool now?", "kind": "hard"}) == "yes, run it"
    shown = capsys.readouterr().out
    assert "HUMAN INPUT NEEDED (hard)" in shown and "Run the coding tool now?" in shown

    def end_of_input(prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", end_of_input)
    assert _ask({"question": "Run?", "kind": "hard"}) == ""  # Ctrl-D: empty reply = refuse


def test_report_copies_rmse_from_committed_transcript():
    data = load_transcript(COMMITTED_RUN)
    page, mermaid = render_html(COMMITTED_RUN, data)
    printed = [r for r in data["runs"] if r["exit_code"] == 0][-1]["stdout"]
    test_line = next(line for line in printed.splitlines() if line.startswith("FREE_RUN_TEST_RMSE"))
    assert test_line in page
    assert "orchestrator" in mermaid and "executor" in mermaid
