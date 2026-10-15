"""Alias: moved to agentic_ode.mcp.client (refactor step 10). Keeps the old import path working."""

import sys

from agentic_ode.mcp import client as _moved

sys.modules[__name__] = _moved
