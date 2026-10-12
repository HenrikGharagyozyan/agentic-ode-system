"""The executor: the only node bound to the coding tool. Runs an internal tool loop, appends one turn."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import ToolMessage

from agentic_ode.agents import prompts
from agentic_ode.agents.context import specialist_messages
from agentic_ode.config import Settings
from agentic_ode.llm import text_of, with_retry
from agentic_ode.state import DialogueState, Role, RunRecord
from agentic_ode.tools import (
    RUN_FILE,
    TEST_RMSE_RE,
    TRAIN_RMSE_RE,
    WRITE_FILE,
    RunResult,
    format_run,
    make_executor_tools,
    parse_rmse,
    run_file,
    write_file,
)

log = logging.getLogger(__name__)

ROUND_LIMIT_NOTE = "(tool-call limit reached before a final report)"


@dataclass
class _TurnLog:
    """What happened inside one executor turn."""

    turn_index: int
    runs: list[RunRecord] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)  # run outputs as shown to the model
    written: list[str] = field(default_factory=list)


class Executor:
    def __init__(self, llm, settings: Settings, workdir: Path):
        self.settings = settings
        self.workdir = workdir
        self.prompt = prompts.executor(settings)
        self.tools = make_executor_tools(workdir, settings.run_timeout_s)
        self.llm = with_retry(llm.bind_tools(self.tools), settings.provider_retries)

    def __call__(self, state: DialogueState) -> dict:
        messages = specialist_messages(self.prompt, state, Role.EXECUTOR)
        turn = _TurnLog(turn_index=len(state["dialogue"]))
        report = ROUND_LIMIT_NOTE
        for _ in range(self.settings.max_tool_rounds):
            reply = self.llm.invoke(messages)
            messages.append(reply)
            if not reply.tool_calls:
                report = text_of(reply)
                break
            for call in reply.tool_calls:
                messages.append(ToolMessage(content=self._dispatch(call, turn), tool_call_id=call["id"]))
        return {
            "dialogue": [{"role": Role.EXECUTOR.value, "content": self._compose_turn(report, turn)}],
            "runs": turn.runs,
        }

    # ------------------------------------------------------------ tool calls

    def _dispatch(self, call: dict, turn: _TurnLog) -> str:
        args = call["args"]
        log.debug("tool call %s(%s)", call["name"], args.get("filename"))
        try:
            if call["name"] == WRITE_FILE:
                return self._write(args["filename"], args["content"], turn)
            if call["name"] == RUN_FILE:
                return self._run(args["filename"], turn)
            return f"ERROR: unknown tool {call['name']}"
        except (KeyError, ValueError) as e:
            return f"ERROR: {e}"

    def _write(self, filename: str, content: str, turn: _TurnLog) -> str:
        out = write_file(self.workdir, filename, content)
        if not out.startswith("ERROR"):
            turn.written.append(filename)
        return out

    def _run(self, filename: str, turn: _TurnLog) -> str:
        limit = self.settings.max_runs_per_turn
        if len(turn.runs) >= limit:
            return f"ERROR: run limit of {limit} runs in this turn reached; report now."
        result = run_file(self.workdir, filename, self.settings.run_timeout_s)
        log.info("ran %s: exit code %s", filename, "timeout" if result.timed_out else result.exit_code)
        shown = format_run(
            filename,
            RunResult(result.exit_code, self._clip(result.stdout), self._clip(result.stderr), result.timed_out),
        )
        turn.outputs.append(shown)
        turn.runs.append({
            "turn_index": turn.turn_index,
            "filename": filename,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "test_rmse": parse_rmse(result.stdout, TEST_RMSE_RE),
            "train_rmse": parse_rmse(result.stdout, TRAIN_RMSE_RE),
        })
        return shown

    # ------------------------------------------------------------ the turn text

    @staticmethod
    def _compose_turn(report: str, turn: _TurnLog) -> str:
        content = report.strip() or "(no report text)"
        if turn.written:
            content += "\n\nFiles written: " + ", ".join(turn.written)
        if turn.outputs:
            content += "\n\n=== tool output (verbatim) ===\n" + "\n\n".join(turn.outputs)
        else:
            content += "\n\n=== tool output ===\n(no script was run in this turn)"
        return content

    def _clip(self, s: str) -> str:
        """Keep head and tail of very long output in the dialogue; the run record keeps it all."""
        n = self.settings.max_output_chars
        if len(s) <= n:
            return s
        half = n // 2
        return f"{s[:half]}\n[... {len(s) - n} characters omitted ...]\n{s[-half:]}"
