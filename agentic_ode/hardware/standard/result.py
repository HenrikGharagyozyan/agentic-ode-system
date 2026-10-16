"""Result: the one envelope every instrument action returns, and its wire format for MCP."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Result:
    """ok + result dict on success, ok=False + error text on failure; warnings can come with a success."""

    ok: bool
    result: dict = field(default_factory=dict)
    error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def success(cls, result: dict | None = None, warnings: list[str] | None = None) -> Result:
        return cls(ok=True, result=dict(result or {}), warnings=list(warnings or []))

    @classmethod
    def failure(cls, error: str) -> Result:
        return cls(ok=False, error=error)

    def to_wire(self) -> dict:
        """The JSON an MCP tool returns. A flat dict, the format the agent and the tests know:
        {"ok": true, **result, "warning": "a | b"} ("warning" only when there are warnings), or
        {"ok": false, "error": "..."}. The envelope itself never reaches the model."""
        if not self.ok:
            return {"ok": False, "error": self.error}
        wire = {"ok": True, **self.result}
        if self.warnings:
            wire["warning"] = " | ".join(self.warnings)
        return wire
