"""Moved to agentic_ode.core.metrics (refactor step 3). This alias keeps the old import path working."""

import sys

from agentic_ode.core import metrics as _moved

sys.modules[__name__] = _moved
