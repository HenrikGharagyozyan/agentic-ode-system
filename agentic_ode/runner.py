"""Run the graph on a task file: a fresh run directory, streamed turns, transcript saved after every step.

When the graph pauses for a human reply, ask for it and resume the same thread.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from agentic_ode.config import PROJECT_ROOT, Settings
from agentic_ode.graph import build_graph
from agentic_ode.llm import make_llm
from agentic_ode.state import DialogueState, Turn, initial_state
from agentic_ode.transcript import save_transcript

RECURSION_LIMIT = 60


@dataclass(frozen=True)
class RunOutcome:
    run_dir: Path
    state: DialogueState


def new_run_dir(runs_dir: Path) -> Path:
    run_dir = runs_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)
    return run_dir


def run_task(
    task_file: Path,
    settings: Settings,
    on_turn: Callable[[Turn], None] | None = None,
    llm=None,
    ask: Callable[[dict], str] | None = None,
) -> RunOutcome:
    """Send the task text to the orchestrator and run the graph to its end.

    When the graph pauses at the human node, `ask` gets the question ({"question", "kind"}) and
    returns the person's reply; the same thread is then resumed with that reply.
    """
    llm = llm or make_llm(settings)
    run_dir = new_run_dir(settings.runs_dir)
    graph = build_graph(llm, run_dir, settings)
    meta = {
        "model": settings.model,
        "task_file": task_file.name,
        "run_dir": _display_path(run_dir),
        "started": _now(),
    }

    state = initial_state(task_file.read_text())
    config = {"recursion_limit": RECURSION_LIMIT, "configurable": {"thread_id": run_dir.name}}
    graph_input = state
    printed = 0
    while True:
        for snapshot in graph.stream(graph_input, stream_mode="values", config=config):
            state = snapshot
            if on_turn:
                for turn in state["dialogue"][printed:]:
                    on_turn(turn)
            printed = len(state["dialogue"])
            save_transcript(run_dir, state, meta)

        paused = graph.get_state(config).interrupts
        if not paused:
            break  # the graph reached END
        if ask is None:
            raise RuntimeError("the graph is waiting for a human reply, but no ask function was given")
        graph_input = Command(resume=ask(paused[0].value))  # same thread_id: continues the same run

    save_transcript(run_dir, state, {**meta, "finished": _now()})
    return RunOutcome(run_dir=run_dir, state=state)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _display_path(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)
