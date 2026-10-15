"""Alias: moved to agentic_ode.hardware.devices.evcam.fake (refactor step 8). Keeps the old import path working."""

import sys

from agentic_ode.hardware.devices.evcam import fake as _moved

sys.modules[__name__] = _moved
