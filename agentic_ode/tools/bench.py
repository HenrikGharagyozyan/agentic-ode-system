"""Alias: moved to agentic_ode.hardware.devices.optical_bench.simulator (refactor step 9). Keeps the old path."""

import sys

from agentic_ode.hardware.devices.optical_bench import simulator as _moved

sys.modules[__name__] = _moved
