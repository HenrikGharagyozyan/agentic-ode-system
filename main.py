"""Send task5.md to the orchestrator, stream the graph, save the dialogue."""

import json
import sys
from datetime import datetime
from pathlib import Path

from src.agents import make_llm
from src.graph import build_graph
from src.state import initial_state

ROOT = Path(__file__).resolve().parent


def save_transcript(run_dir: Path, state: dict, meta: dict) -> None:
    data = {**meta, "stop_reason": state.get("stop_reason", ""), "decisions": state.get("decisions", 0),
            "dialogue": state["dialogue"], "runs": state.get("runs", [])}
    (run_dir / "transcript.json").write_text(json.dumps(data, indent=2, ensure_ascii=False))

    lines = [f"# Transcript {run_dir.name}", "",
             f"- model: `{meta['model']}`", f"- task: `{meta['task_file']}`",
             f"- orchestrator decisions: {data['decisions']}", f"- stop reason: {data['stop_reason'] or '(running)'}", ""]
    for i, t in enumerate(data["dialogue"]):
        lines += [f"## Turn {i} [{t['role']}]", "", "````text", t["content"], "````", ""]
    (run_dir / "transcript.md").write_text("\n".join(lines))


def main() -> int:
    task_file = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "task5.md")
    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    llm = make_llm()
    graph = build_graph(llm, run_dir)
    state = initial_state(task_file.read_text())
    meta = {"model": llm.model_name if hasattr(llm, "model_name") else str(llm),
            "task_file": task_file.name, "run_dir": str(run_dir.relative_to(ROOT)),
            "started": datetime.now().isoformat(timespec="seconds")}

    printed = 0
    for state in graph.stream(state, stream_mode="values", config={"recursion_limit": 60}):
        for turn in state["dialogue"][printed:]:
            print(f"[{turn['role']}] {turn['content']}\n", flush=True)
        printed = len(state["dialogue"])
        save_transcript(run_dir, state, meta)

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    save_transcript(run_dir, state, meta)
    print(f"stop reason: {state['stop_reason']}; decisions: {state['decisions']}")
    print(f"transcript: {run_dir / 'transcript.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
