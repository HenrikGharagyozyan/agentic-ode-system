"""The coding tool of the executor: write a Python file, run it with a time limit."""

import re
import subprocess
import sys
from pathlib import Path

from langchain_core.tools import StructuredTool

RUN_TIMEOUT_S = 300

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.py$")
_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
TEST_RMSE_RE = re.compile(r"^FREE_RUN_TEST_RMSE:\s*" + _FLOAT + r"\s*$", re.MULTILINE)
TRAIN_RMSE_RE = re.compile(r"^FREE_RUN_TRAIN_RMSE:\s*" + _FLOAT + r"\s*$", re.MULTILINE)


def parse_rmse(stdout: str, pattern: re.Pattern = TEST_RMSE_RE) -> float | None:
    """Last RMSE line of the given kind printed on stdout, or None."""
    found = pattern.findall(stdout)
    return float(found[-1]) if found else None


def _check_name(filename: str) -> None:
    if not _NAME_RE.match(filename):
        raise ValueError(f"invalid filename {filename!r}: use a plain name like 'fit_01.py'")


def write_file(workdir: Path, filename: str, content: str) -> str:
    _check_name(filename)
    path = workdir / filename
    if path.exists():
        return f"ERROR: {filename} already exists. Every script goes into a new file; pick a new name."
    path.write_text(content)
    return f"wrote {path} ({len(content)} characters)"


def run_file(workdir: Path, filename: str, timeout: float = RUN_TIMEOUT_S) -> dict:
    """Run a file of the working directory. Returns exit code (None on timeout), stdout, stderr."""
    _check_name(filename)
    path = workdir / filename
    if not path.exists():
        return {"exit_code": None, "stdout": "", "stderr": f"ERROR: {filename} does not exist", "timed_out": False}
    try:
        p = subprocess.run(
            [sys.executable, filename], cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
        return {"exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        def _s(x):
            return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")

        return {
            "exit_code": None,
            "stdout": _s(e.stdout),
            "stderr": _s(e.stderr) + f"\nTIMEOUT: killed after {timeout:g} s",
            "timed_out": True,
        }


def format_run(filename: str, result: dict) -> str:
    """Tool output as text; this exact text goes to the model and into the executor turn."""
    code = "TIMEOUT" if result["timed_out"] else result["exit_code"]
    return (
        f"run_file({filename!r})\nexit_code: {code}\n"
        f"--- stdout ---\n{result['stdout']}\n--- stderr ---\n{result['stderr']}"
    )


def make_executor_tools(workdir: Path, timeout: float = RUN_TIMEOUT_S) -> list[StructuredTool]:
    """The two tools bound to the executor. No other node receives them."""

    def _write(filename: str, content: str) -> str:
        """Write a new Python file into this run's working directory. Existing files are never overwritten."""
        return write_file(workdir, filename, content)

    def _run(filename: str) -> str:
        """Run a Python file of the working directory (time limit 300 s). Returns exit code, stdout and stderr."""
        return format_run(filename, run_file(workdir, filename, timeout))

    return [
        StructuredTool.from_function(_write, name="write_file"),
        StructuredTool.from_function(_run, name="run_file"),
    ]
