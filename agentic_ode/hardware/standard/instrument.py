"""Instrument and ActionSpec: the contract every device driver follows ("Model Hardware Standard").

An instrument is a list of actions. Each action is a normal Python method with type hints that returns a Result;
the type hints define the action's JSON schema. mcp/server_factory.py turns any Instrument into an MCP server.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from agentic_ode.hardware.standard.result import Result

KINDS = ("read", "setting", "measurement")


@dataclass(frozen=True)
class ActionSpec:
    """One thing an instrument can do; one MCP tool.

    kind: "read" changes nothing; "setting" changes the device (the agent asks the human first);
    "measurement" makes a result (a successful one can finish a task).
    """

    name: str  # the MCP tool name, e.g. "set_bias"
    description: str  # the MCP tool description: the model reads it
    kind: Literal["read", "setting", "measurement"]
    handler: Callable[..., Result]  # e.g. def set_bias(self, name: str, value: int) -> Result

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"action {self.name}: kind must be one of {KINDS}, got {self.kind!r}")

    @property
    def input_schema(self) -> dict:
        """The JSON schema of the arguments, made from the handler's type hints by fastmcp's own generator,
        so it is the same schema the MCP server shows."""
        from fastmcp.tools import Tool  # imported here: importing a device driver should not load the MCP library

        return Tool.from_function(self.handler, name=self.name).parameters


class Instrument(ABC):
    """A device (real or simulated) as a list of actions."""

    name: str = ""  # the registry key, e.g. "evcam"

    @abstractmethod
    def describe(self) -> dict:
        """Static facts without touching the hardware: at least "name" and "instructions" (MCP server text)."""

    @abstractmethod
    def actions(self) -> list[ActionSpec]:
        """All actions, in the order the MCP server lists them."""

    def close(self) -> None:  # noqa: B027 - an optional hook: empty on purpose, not abstract
        """Release the hardware. Default: nothing to release."""

    def action(self, name: str) -> ActionSpec:
        for spec in self.actions():
            if spec.name == name:
                return spec
        raise KeyError(f"{self.name} has no action {name!r}")

    def call(self, action: str, args: dict) -> Result:
        """Run one action. An exception becomes Result.failure("TypeName: message"): the caller never sees a crash."""
        try:
            return self.action(action).handler(**args)
        except Exception as e:  # every error goes back to the model as text
            return Result.failure(f"{type(e).__name__}: {e}")
