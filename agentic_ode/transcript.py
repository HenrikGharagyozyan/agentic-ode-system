"""transcript.json (dialogue + run records) and a readable transcript.md for a run directory."""

from __future__ import annotations

import json
from pathlib import Path

from agentic_ode.state import role_label

JSON_NAME = "transcript.json"
MD_NAME = "transcript.md"


def save_transcript(run_dir: Path, state: dict, meta: dict) -> None:
    data = {
        **meta,
        "stop_reason": state.get("stop_reason", ""),
        "decisions": state.get("decisions", 0),
        "dialogue": state["dialogue"],
        "runs": state.get("runs", []),
        "experiments": state.get("experiments", []),
    }
    (run_dir / JSON_NAME).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    (run_dir / MD_NAME).write_text(_markdown(run_dir.name, data, meta))


def load_transcript(run_dir: Path) -> dict:
    return json.loads((run_dir / JSON_NAME).read_text())


def _markdown(name: str, data: dict, meta: dict) -> str:
    lines = [
        f"# Transcript {name}",
        "",
        f"- model: `{meta['model']}`",
        f"- task: `{meta['task_file']}`",
        f"- orchestrator decisions: {data['decisions']}",
        f"- stop reason: {data['stop_reason'] or '(running)'}",
        "",
    ]
    for i, t in enumerate(data["dialogue"]):
        lines += [f"## Turn {i} [{role_label(t['role'])}]", "", "````text", t["content"], "````", ""]
    return "\n".join(lines)
