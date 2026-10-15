"""Alias: moved to agentic_ode.runtime.cli (refactor step 11). Keeps the old import path working."""

import sys

from agentic_ode.runtime import cli as _moved

sys.modules[__name__] = _moved
