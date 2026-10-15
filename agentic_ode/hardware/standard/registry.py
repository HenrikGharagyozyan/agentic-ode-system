"""The registry: instrument name -> where its factory is ("module:attribute").

The factory is imported only by create(), so listing the instruments never imports a device SDK.
The devices register themselves in agentic_ode/hardware/devices/__init__.py.
"""

from __future__ import annotations

import importlib

from agentic_ode.hardware.standard.instrument import Instrument

_FACTORIES: dict[str, str] = {}


def register(name: str, factory: str) -> None:
    """factory: "package.module:attribute", a class or function that returns an Instrument (no arguments)."""
    if ":" not in factory:
        raise ValueError(f"factory must look like 'package.module:attribute', got {factory!r}")
    if _FACTORIES.get(name, factory) != factory:
        raise ValueError(f"instrument {name!r} is already registered as {_FACTORIES[name]}")
    _FACTORIES[name] = factory


def names() -> list[str]:
    return list(_FACTORIES)


def create(name: str) -> Instrument:
    if name not in _FACTORIES:
        raise KeyError(f"unknown instrument {name!r}; known: {', '.join(_FACTORIES) or 'none'}")
    module, attribute = _FACTORIES[name].split(":")
    instrument = getattr(importlib.import_module(module), attribute)()
    if not isinstance(instrument, Instrument):
        raise TypeError(f"{_FACTORIES[name]} returned {type(instrument).__name__}, not an Instrument")
    return instrument
