"""The executor's coding tool (write_file + run_file) and the RMSE output contract."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from agentic_ode.tools.metrics import TEST_RMSE_RE, TRAIN_RMSE_RE, last_rmse_line, parse_rmse, rmse_lines
from agentic_ode.tools.workspace import RunResult, format_run, run_file, write_file

WRITE_FILE = "write_file"
RUN_FILE = "run_file"


def make_executor_tools(workdir: Path, timeout: float) -> list[StructuredTool]:
    """Tool schemas bound to the executor's model. No other node receives them.

    The executor dispatches the calls itself (to record each run); these wrappers are
    still fully functional so the tools can be used on their own.
    """

    def _write(filename: str, content: str) -> str:
        """Write a new Python file into this run's working directory. Existing files are never overwritten."""
        return write_file(workdir, filename, content)

    def _run(filename: str) -> str:
        """Run a Python file of the working directory (time limit 300 s). Returns exit code, stdout and stderr."""
        return format_run(filename, run_file(workdir, filename, timeout))

    return [
        StructuredTool.from_function(_write, name=WRITE_FILE),
        StructuredTool.from_function(_run, name=RUN_FILE),
    ]


__all__ = [
    "RUN_FILE",
    "TEST_RMSE_RE",
    "TRAIN_RMSE_RE",
    "WRITE_FILE",
    "RunResult",
    "format_run",
    "last_rmse_line",
    "make_executor_tools",
    "parse_rmse",
    "rmse_lines",
    "run_file",
    "write_file",
]
