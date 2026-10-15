"""Device drivers. Each device is one folder; hardware/standard/ is the contract they follow.

The list of known instruments is here, and only here: a new device adds one register() line.
Only the "module:attribute" text is stored; the device module is imported when the instrument is created.
"""

from agentic_ode.hardware.standard import registry

registry.register("evcam", "agentic_ode.hardware.devices.evcam.instrument:EvcamInstrument")
