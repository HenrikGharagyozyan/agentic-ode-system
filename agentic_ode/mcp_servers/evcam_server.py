"""Alias: moved to agentic_ode.mcp.evcam_server (refactor step 10).

Kept for good: `python -m agentic_ode.mcp_servers.evcam_server` is the documented command, and evcam_session() uses
it. Run as a program, it starts the moved server; imported, it is the moved module itself (same object).
"""

import sys

if __name__ == "__main__":
    from agentic_ode.mcp.evcam_server import main

    main()
else:
    from agentic_ode.mcp import evcam_server as _moved

    sys.modules[__name__] = _moved
