"""The executor: the only node bound to tools (coding tool + focusing tool). Runs a tool loop, appends one turn."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.messages import ToolMessage

from agentic_ode.agents import prompts
from agentic_ode.agents.context import specialist_messages
from agentic_ode.core.config import Settings
from agentic_ode.core.llm import text_of, with_retry
from agentic_ode.state import DialogueState, ExperimentRecord, Role, RunRecord
from agentic_ode.tools import (
    FOCUS_RESULT_KEY,
    RUN_FILE,
    RUN_FOCUS_SEARCH,
    TEST_RMSE_RE,
    TRAIN_RMSE_RE,
    WRITE_FILE,
    RunResult,
    format_focus,
    format_run,
    make_executor_tools,
    parse_rmse,
    run_file,
    run_focus_search,
    write_file,
)

log = logging.getLogger(__name__)

ROUND_LIMIT_NOTE = "(tool-call limit reached before a final report)"


@dataclass
class _TurnLog:
    """What happened inside one executor turn."""

    turn_index: int
    runs: list[RunRecord] = field(default_factory=list)
    experiments: list[ExperimentRecord] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)  # run outputs as shown to the model
    written: list[str] = field(default_factory=list)


def closed_schema(tool):
    """The tool with "additionalProperties": false in its JSON schema, so the model is told that no other arguments
    exist. Tools with a pydantic schema are returned unchanged."""
    if isinstance(tool.args_schema, dict):
        return tool.model_copy(update={"args_schema": {**tool.args_schema, "additionalProperties": False}})
    return tool


class Executor:
    def __init__(self, llm, settings: Settings, workdir: Path, camera_tools: list | None = None):
        self.settings = settings
        self.workdir = workdir
        self.prompt = prompts.executor(settings)
        self.tools = make_executor_tools(workdir, settings.run_timeout_s)
        self.llm = with_retry(llm.bind_tools(self.tools), settings.provider_retries)
        # The event camera's MCP tools (HW3), in a SEPARATE binding of the same model: only the hardware turn
        # uses it. The normal tool loop above never sees them, so no camera call can bypass the approval step.
        # tool_choice="any" forces the model to make a tool call (see camera_reply() for models that reject it).
        # _camera_forced has NO retries, camera_llm is the same binding WITH retries.
        self._raw_llm = llm
        self.camera_tools = [closed_schema(t) for t in camera_tools or []]
        self._camera_forced = llm.bind_tools(self.camera_tools, tool_choice="any") if self.camera_tools else None
        self.camera_llm = (with_retry(self._camera_forced, settings.provider_retries)
                           if self.camera_tools else None)

    def camera_reply(self, messages: list):
        """One model call with the camera tools (the hardware turn), forced to make a tool call.

        The first try has no retries, so a provider that rejects tool_choice is seen at once (not after ~15 s of
        useless retries). Then: retry ONCE without tool_choice and keep that binding for the rest of the run.
        Any other error gets the normal retries. A text answer is handled by the caller (hw_decide)."""
        if self._camera_forced is None:  # tool_choice was rejected earlier in this run
            return self.camera_llm.invoke(messages)
        try:
            return self._camera_forced.invoke(messages)
        except Exception as e:
            if "tool_choice" not in str(e).lower():
                log.info("camera model call failed (%s); retrying", e)
                return self.camera_llm.invoke(messages)
            log.warning("the provider rejected tool_choice (%s); retrying once without it", e)
            self._camera_forced = None
            self.camera_llm = with_retry(self._raw_llm.bind_tools(self.camera_tools), self.settings.provider_retries)
            return self.camera_llm.invoke(messages)

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
            "experiments": turn.experiments,
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
            if call["name"] == RUN_FOCUS_SEARCH:
                return self._focus(args, turn)
            return f"ERROR: unknown tool {call['name']}"
        except (KeyError, ValueError, TypeError) as e:
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

    def _focus(self, args: dict, turn: _TurnLog) -> str:
        limit = self.settings.max_focus_calls_per_turn
        if len(turn.experiments) >= limit:
            return f"ERROR: limit of {limit} focus searches in this turn reached; report now."
        result = run_focus_search(**{k: int(v) for k, v in args.items()}, workdir=self.workdir)
        shown = format_focus(result)
        log.info("focus search %s: enhancement %s", result["settings"], result["final_enhancement"])
        turn.outputs.append(shown)
        turn.experiments.append({
            "turn_index": turn.turn_index,
            "settings": result["settings"],
            "n_macropixels": result["n_macropixels"],
            "initial_enhancement": result["initial_enhancement"],
            "final_enhancement": result["final_enhancement"],
            "result_line": f"{FOCUS_RESULT_KEY}: {result['final_enhancement']}",
            "images": result["images"],
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
            content += "\n\n=== tool output ===\n(no tool was run in this turn)"
        return content

    def _clip(self, s: str) -> str:
        """Keep head and tail of very long output in the dialogue; the run record keeps it all."""
        n = self.settings.max_output_chars
        if len(s) <= n:
            return s
        half = n // 2
        return f"{s[:half]}\n[... {len(s) - n} characters omitted ...]\n{s[-half:]}"
