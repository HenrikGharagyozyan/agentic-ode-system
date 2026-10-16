"""Send task5.md to the orchestrator and run the graph. Same as `agentic-ode run [TASK]`."""

import sys

from agentic_ode.runtime.cli import main

if __name__ == "__main__":
    sys.exit(main(["run", *sys.argv[1:]]))
