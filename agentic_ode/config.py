"""Moved to agentic_ode.core.config (refactor step 1). This alias keeps the old import path working."""

import sys

from agentic_ode.core import config as _moved

sys.modules[__name__] = _moved
