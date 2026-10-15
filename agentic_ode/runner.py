"""Run the graph on a task file: a fresh run directory, streamed turns, transcript saved after every step.

When the graph pauses for a human reply, ask for it and resume the same thread.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from langgraph.types import Command

from agentic_ode.core.config import PROJECT_ROOT, Settings
from agentic_ode.core.llm import make_llm
from agentic_ode.graph import build_graph
from agentic_ode.mcp_client import evcam_session
from agentic_ode.state import DialogueState, Turn, initial_state
from agentic_ode.transcript import save_transcript

RECURSION_LIMIT = 60
# camera runs: every camera action costs one orchestrator decision and up to 4 graph steps
# (orchestrator -> hw_decide -> hw_approve -> hw_call), so they get more of both
CAMERA_MAX_DECISIONS = 20
CAMERA_RECURSION_LIMIT = 150


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
    camera: bool = False,
) -> RunOutcome:
    """Send the task text to the orchestrator and run the graph to its end.

    When the graph pauses at the human node, `ask` gets the question ({"question", "kind"}) and
    returns the person's reply; the same thread is then resumed with that reply.
    camera=True: start the event camera MCP server (EVCAM_FAKE=1 -> fake camera) with ONE session for the whole
    run, human pauses included, and give its tools to the executor. The session is closed at the end.
    Camera runs get at least CAMERA_MAX_DECISIONS decisions and CAMERA_RECURSION_LIMIT graph steps.
    """
    if camera:
        settings = replace(settings, max_decisions=max(settings.max_decisions, CAMERA_MAX_DECISIONS))
    llm = llm or make_llm(settings)
    run_dir = new_run_dir(settings.runs_dir)
    meta = {
        "model": settings.model,
        "task_file": task_file.name,
        "run_dir": _display_path(run_dir),
        "camera": camera,
        "started": _now(),
    }

    with (evcam_session() if camera else nullcontext()) as session:  # nullcontext() gives session = None
        graph = build_graph(llm, run_dir, settings, camera_tools=session.tools() if session else None)
        state = initial_state(task_file.read_text())
        limit = CAMERA_RECURSION_LIMIT if camera else RECURSION_LIMIT
        config = {"recursion_limit": limit, "configurable": {"thread_id": run_dir.name}}
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
