"""The Model Hardware Standard (agentic_ode/hardware/standard/): Result, ActionSpec, Instrument, registry."""

import asyncio

import pytest
from fastmcp import Client, FastMCP

from agentic_ode.hardware.standard import ActionSpec, Instrument, Result, registry


class Counter(Instrument):
    """A tiny instrument for the tests: a counter that can be set and read."""

    name = "counter"

    def __init__(self):
        self.value = 0

    def describe(self) -> dict:
        return {"name": self.name, "instructions": "a counter"}

    def actions(self) -> list[ActionSpec]:
        return [ActionSpec("read", "Read the counter.", "read", self.read),
                ActionSpec("add", "Add n to the counter.", "setting", self.add)]

    def read(self) -> Result:
        return Result.success({"value": self.value})

    def add(self, n: int, note: str = "") -> Result:
        if n < 0:
            raise ValueError("n must be >= 0")
        self.value += n
        return Result.success({"value": self.value}, warnings=["big step"] if n > 10 else [])


# ---- Result ----

def test_wire_format_is_flat_like_the_tools_today():
    assert Result.success({"a": 1, "b": 2}).to_wire() == {"ok": True, "a": 1, "b": 2}
    assert Result.failure("ValueError: bad").to_wire() == {"ok": False, "error": "ValueError: bad"}
    wire = Result.success({"a": 1}, warnings=["first", "second"]).to_wire()
    assert wire == {"ok": True, "a": 1, "warning": "first | second"}
    assert list(wire) == ["ok", "a", "warning"]  # the key order the model reads: warning last


# ---- Instrument.call ----

def test_call_runs_the_handler():
    c = Counter()
    assert c.call("add", {"n": 3}).to_wire() == {"ok": True, "value": 3}
    assert c.call("add", {"n": 11}).warnings == ["big step"]
    assert c.call("read", {}).result == {"value": 14}


def test_an_exception_becomes_a_failure():
    c = Counter()
    r = c.call("add", {"n": -1})
    assert r.ok is False and r.error == "ValueError: n must be >= 0"  # the same text as the evcam server's _run()
    assert c.call("nope", {}).ok is False


# ---- ActionSpec ----

def test_kind_is_checked():
    with pytest.raises(ValueError, match="kind must be one of"):
        ActionSpec("x", "x", "write", Counter().read)


def test_input_schema_is_the_schema_the_mcp_server_shows():
    c = Counter()
    server = FastMCP("t")
    server.tool(c.add)

    async def listed():
        async with Client(server) as client:
            return {t.name: t.input_schema for t in await client.list_tools()}

    schema = c.action("add").input_schema
    assert schema == asyncio.run(listed())["add"]
    assert schema["required"] == ["n"]


# ---- registry ----

@pytest.fixture
def empty_registry(monkeypatch):
    monkeypatch.setattr(registry, "_FACTORIES", {})


def test_registry_creates_by_name(empty_registry):
    registry.register("counter", "tests.test_hardware_standard:Counter")
    assert registry.names() == ["counter"]
    assert isinstance(registry.create("counter"), Counter)


def test_registry_imports_lazily(empty_registry):
    registry.register("ghost", "no_such_package_xyz.driver:Ghost")  # not imported here
    assert registry.names() == ["ghost"]
    with pytest.raises(ModuleNotFoundError):
        registry.create("ghost")  # imported only now


def test_registry_rejects_bad_entries(empty_registry):
    with pytest.raises(ValueError, match="module:attribute"):
        registry.register("x", "no_colon")
    registry.register("counter", "tests.test_hardware_standard:Counter")
    with pytest.raises(ValueError, match="already registered"):
        registry.register("counter", "other.module:Counter")
    with pytest.raises(KeyError, match="unknown instrument"):
        registry.create("missing")
