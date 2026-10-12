"""The runner writes transcripts; the report copies RMSE lines from the committed run verbatim."""

import json
from pathlib import Path

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


def test_report_copies_rmse_from_committed_transcript():
    data = load_transcript(COMMITTED_RUN)
    page, mermaid = render_html(COMMITTED_RUN, data)
    printed = [r for r in data["runs"] if r["exit_code"] == 0][-1]["stdout"]
    test_line = next(line for line in printed.splitlines() if line.startswith("FREE_RUN_TEST_RMSE"))
    assert test_line in page
    assert "orchestrator" in mermaid and "executor" in mermaid
