"""The executor's coding tool (write_file + run_file) and the RMSE output contract."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from agentic_ode.core.metrics import TEST_RMSE_RE, TRAIN_RMSE_RE, last_rmse_line, parse_rmse, rmse_lines
from agentic_ode.tools.focus import RESULT_KEY as FOCUS_RESULT_KEY
from agentic_ode.tools.focus import format_focus, run_focus_search
from agentic_ode.tools.workspace import RunResult, format_run, run_file, write_file

WRITE_FILE = "write_file"
RUN_FILE = "run_file"
RUN_FOCUS_SEARCH = "run_focus_search"


def make_executor_tools(workdir: Path, timeout: float) -> list[StructuredTool]:
    """Tool schemas bound to the executor's model (coding tool + focusing tool). No other node receives them.

    The executor dispatches the calls itself (to record each run); these wrappers are
    still fully functional so the tools can be used on their own.
    """

    def _write(filename: str, content: str) -> str:
        """Write a new Python file into this run's working directory. Existing files are never overwritten."""
        return write_file(workdir, filename, content)

    def _run(filename: str) -> str:
        """Run a Python file of the working directory (time limit 300 s). Returns exit code, stdout and stderr."""
        return format_run(filename, run_file(workdir, filename, timeout))

    def _focus(macropixel: int, focus_size: int, n_background_regions: int, iterations: int,
               flips_per_step: int, seed: int) -> str:
        """Focus light through the scattering medium of the simulated optical bench (laser -> DMD -> medium -> camera).

        The 64x64 binary-amplitude DMD is grouped into square macropixels; the search starts from a random
        pattern and, each iteration, flips `flips_per_step` random macropixels, takes one noisy camera shot and
        keeps the flip only if the enhancement grows. Enhancement = mean intensity in the focus square at the
        camera centre / mean of the sector means of the background ring around it.

        Args:
            macropixel: macropixel side in DMD pixels, one of 2, 4, 8, 16 (controlled macropixels: 1024, 256, 64, 16).
            focus_size: side of the focus square, 1 to 8 camera pixels.
            n_background_regions: number of sectors of the background ring, 2 to 16.
            iterations: number of flip-and-measure iterations, 1 to 20000 (about 0.7 ms each).
            flips_per_step: macropixels flipped per iteration, at least 1.
            seed: random seed of the start pattern, the flips and the camera noise.

        Returns the settings, the initial and final enhancement (FOCUS_ENHANCEMENT, a fresh shot of the kept
        pattern), a short enhancement history and the names of the saved camera / DMD / curve images.
        """
        return format_focus(run_focus_search(macropixel, focus_size, n_background_regions, iterations,
                                             flips_per_step, seed, workdir=workdir))

    return [
        StructuredTool.from_function(_write, name=WRITE_FILE),
        StructuredTool.from_function(_run, name=RUN_FILE),
        StructuredTool.from_function(_focus, name=RUN_FOCUS_SEARCH),
    ]


__all__ = [
    "FOCUS_RESULT_KEY",
    "RUN_FOCUS_SEARCH",
    "format_focus",
    "run_focus_search",
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
