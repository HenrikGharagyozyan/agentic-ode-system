"""The run's working directory: write a new Python file into it, run a file of it with a time limit."""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.py$")


@dataclass(frozen=True)
class RunResult:
    exit_code: int | None  # None when the run was killed by the time limit or never started
    stdout: str
    stderr: str
    timed_out: bool = False


def check_filename(filename: str) -> None:
    """Only plain `name.py` files directly inside the working directory."""
    if not _NAME_RE.match(filename):
        raise ValueError(f"invalid filename {filename!r}: use a plain name like 'fit_01.py'")


def write_file(workdir: Path, filename: str, content: str) -> str:
    """Write a new file. Existing files are never overwritten; every script goes into a new file."""
    check_filename(filename)
    path = workdir / filename
    if path.exists():
        return f"ERROR: {filename} already exists. Every script goes into a new file; pick a new name."
    path.write_text(content)
    return f"wrote {path} ({len(content)} characters)"


def run_file(workdir: Path, filename: str, timeout: float) -> RunResult:
    """Run a file of the working directory with the current interpreter."""
    check_filename(filename)
    if not (workdir / filename).exists():
        return RunResult(exit_code=None, stdout="", stderr=f"ERROR: {filename} does not exist")
    try:
        p = subprocess.run(
            [sys.executable, filename], cwd=workdir, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired as e:
        return RunResult(
            exit_code=None,
            stdout=_decode(e.stdout),
            stderr=_decode(e.stderr) + f"\nTIMEOUT: killed after {timeout:g} s",
            timed_out=True,
        )
    return RunResult(exit_code=p.returncode, stdout=p.stdout, stderr=p.stderr)


def format_run(filename: str, result: RunResult) -> str:
    """Tool output as text; this exact text goes to the model and into the executor turn."""
    code = "TIMEOUT" if result.timed_out else result.exit_code
    return (
        f"run_file({filename!r})\nexit_code: {code}\n"
        f"--- stdout ---\n{result.stdout}\n--- stderr ---\n{result.stderr}"
    )


def _decode(x: bytes | str | None) -> str:
    return x.decode(errors="replace") if isinstance(x, bytes) else (x or "")
