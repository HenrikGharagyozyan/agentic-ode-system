"""Moved to agentic_ode.graph.nodes.hardware_turn (refactor step 7). This alias keeps the old import path working."""

import sys

from agentic_ode.graph.nodes import hardware_turn as _moved

sys.modules[__name__] = _moved
