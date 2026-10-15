"""Alias: moved to agentic_ode.hardware.devices.evcam.driver (refactor step 8). Keeps the old import path working."""

import sys

from agentic_ode.hardware.devices.evcam import driver as _moved

sys.modules[__name__] = _moved
