"""The output contract between generated scripts and the system.

A script reports its result by printing, on their own lines,
    FREE_RUN_TRAIN_RMSE: <float>
    FREE_RUN_TEST_RMSE: <float>
Only values parsed from real stdout count; the finish guard relies on this.
"""

from __future__ import annotations

import re

_FLOAT = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
TEST_RMSE_RE = re.compile(r"^FREE_RUN_TEST_RMSE:[ \t]*" + _FLOAT + r"[ \t]*$", re.MULTILINE)
TRAIN_RMSE_RE = re.compile(r"^FREE_RUN_TRAIN_RMSE:[ \t]*" + _FLOAT + r"[ \t]*$", re.MULTILINE)


def parse_rmse(stdout: str, pattern: re.Pattern = TEST_RMSE_RE) -> float | None:
    """Last RMSE value of the given kind printed on stdout, or None."""
    found = pattern.findall(stdout)
    return float(found[-1]) if found else None


def last_rmse_line(stdout: str, pattern: re.Pattern = TEST_RMSE_RE) -> str | None:
    """The last matching line exactly as printed, or None."""
    lines = [m.group(0) for m in pattern.finditer(stdout)]
    return lines[-1] if lines else None


def rmse_lines(stdout: str) -> list[str]:
    """All train/test RMSE lines exactly as printed, in order."""
    return [m.group(0) for m in re.finditer(r"^FREE_RUN_(TRAIN|TEST)_RMSE:.*$", stdout, re.MULTILINE)]
