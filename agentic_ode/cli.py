"""Command line: `agentic-ode run [TASK]` and `agentic-ode report RUN_DIR`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from agentic_ode.core.config import PROJECT_ROOT, Settings
from agentic_ode.core.llm import ConfigurationError
from agentic_ode.core.state import Turn

DEFAULT_TASK = PROJECT_ROOT / "task5.md"


def _print_turn(turn: Turn) -> None:
    print(f"[{turn['role']}] {turn['content']}\n", flush=True)


def _show_question(payload: dict) -> str:
    kind = payload.get("kind") or "none"
    print(f" HUMAN INPUT NEEDED ({kind}) ".center(75, "="))
    print(payload.get("question") or "(the orchestrator gave no question text)")
    if kind == "hard":
        print('(Start your answer with "yes" to accept. Anything else refuses.)')
    print("=" * 75, flush=True)
    return kind


def _ask(payload: dict) -> str:
    """Show the graph's question to the person and read the reply. End of input (Ctrl-D) gives "" = refuse."""
    _show_question(payload)
    try:
        return input("your reply> ")
    except EOFError:
        print()
        return ""


def _auto_ask(payload: dict) -> str:
    """--auto-approve: answer "yes" to hard-decision questions (printed, so the run shows them); ask otherwise."""
    if payload.get("kind") != "hard":
        return _ask(payload)
    _show_question(payload)
    print("your reply> yes   (auto-approve)\n", flush=True)
    return "yes"


def cmd_run(args: argparse.Namespace) -> int:
    from agentic_ode.runner import run_task

    if args.auto_approve:
        print("auto-approve is ON: hard-decision questions are answered 'yes' automatically.\n", flush=True)
    try:
        ask = _auto_ask if args.auto_approve else _ask
        outcome = run_task(args.task, Settings.from_env(), on_turn=_print_turn, ask=ask, camera=args.camera)
    except ConfigurationError as e:
        print(f"configuration error: {e}", file=sys.stderr)
        return 2
    print(f"stop reason: {outcome.state['stop_reason']}; decisions: {outcome.state['decisions']}")
    print(f"transcript: {outcome.run_dir / 'transcript.json'}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from agentic_ode.reporting import build_report

    print(build_report(args.run_dir.resolve(), args.out.resolve()))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agentic-ode", description=__doc__)
    p.add_argument("-v", "--verbose", action="count", default=0, help="log decisions (-v) and tool calls (-vv)")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="send a task to the orchestrator and run the graph")
    run.add_argument("task", nargs="?", type=Path, default=DEFAULT_TASK, help="task file (default: task5.md)")
    run.add_argument("--auto-approve", action="store_true",
                     help='answer "yes" to hard-decision questions automatically (they are still printed)')
    run.add_argument("--camera", action="store_true",
                     help="start the event camera MCP server and give its tools to the executor "
                          "(EVCAM_FAKE=1: fake camera)")
    run.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="build the PDF report from a run's transcript.json")
    rep.add_argument("run_dir", type=Path, help="runs/<timestamp>")
    rep.add_argument("--out", type=Path, default=PROJECT_ROOT / "report", help="output directory (default: report/)")
    rep.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = {0: logging.WARNING, 1: logging.INFO}.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    if args.verbose < 2:
        for noisy in ("httpx", "openai", "httpcore"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
    return args.func(args)
