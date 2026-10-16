"""Experiments started from the browser: ONE at a time, each one is a normal run_task() camera run in a thread.

    browser --POST--> ExperimentManager.start() --thread--> run_task(camera=True, ask=exp.ask, on_state=exp.on_state)
                                                                   |  graph (unchanged) -> MCP -> camera
    browser <--SSE--  exp.events  <-- on_state() / ask() ----------+

- The camera lock (app.state.camera_lock) is taken in start() and released when the thread ends, whatever happens.
- ask() is the graph's question to a person. It blocks the worker thread until reply() gives the answer for the
  SAME request_id, or until cancel(). The graph decides what the answer means (hw_approve, human node).
- Every event is appended to runs/<id>/events.jsonl; runs/<id>/experiment.json holds the summary and the
  CameraRecords. A finished experiment is read back from these two files.
"""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime
from pathlib import Path

from agentic_ode.agents.guards import accepted
from agentic_ode.core.config import Settings
from agentic_ode.runtime import runner
from agentic_ode.web.events import input_request, state_events

EXPERIMENT_JSON = "experiment.json"
EVENTS_JSONL = "events.jsonl"
TASK_FILE = "task.md"
ID_PATTERN = re.compile(r"\d{8}-\d{6}")  # = runner.new_run_dir(): only these characters, never a path
TERMINAL = ("completed", "stopped", "failed", "cancelled")
MAX_ERROR_CHARS = 500


class Cancelled(Exception):
    """Raised in the worker thread (from ask() or on_state()) to stop the run after cancel()."""


class ConflictError(Exception):
    """The request does not fit the experiment's current state (the API answers 409)."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")  # write, then rename: a reader never sees half a file
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str))
    os.replace(tmp, path)


class Experiment:
    """One experiment: its summary, its ordered events, and the question that waits for a reply (if any)."""

    def __init__(self, run_dir: Path, question: str, mode: str, model: str):
        self.id = run_dir.name
        self.run_dir = run_dir
        self.summary = {
            "id": self.id,
            "question": question,
            "mode": mode,  # "simulated" | "real", fixed when the backend started
            "model": model,
            "created": _now(),
            "finished": None,
            "status": "running",
            "stop_reason": None,
            "error": None,
            "decisions": 0,
            "camera": [],  # the CameraRecords of the run (what the finish guard trusts)
            "refused_calls": [],
            "pending": None,  # the question that waits for a reply, or None
        }
        self.events: list[dict] = []
        self._lock = threading.Lock()  # guards summary, events and the reply fields
        self._state: dict | None = None  # the last state snapshot
        self._reply: str | None = None
        self._replied = threading.Event()
        self._requests = 0
        self.cancel_requested = threading.Event()

    # ---- events ----

    def emit(self, type_: str, data: dict) -> None:
        with self._lock:
            event = {"seq": len(self.events) + 1, "experiment_id": self.id, "time": _now(), "type": type_,
                     "data": data}
            self.events.append(event)
        with open(self.run_dir / EVENTS_JSONL, "a") as f:
            f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def events_after(self, seq: int) -> list[dict]:
        with self._lock:
            return self.events[seq:]  # seq n is at index n-1, so this is every event with seq > n

    def snapshot(self) -> dict:
        with self._lock:
            return json.loads(json.dumps(self.summary, default=str))  # a deep copy

    @property
    def status(self) -> str:
        return self.summary["status"]

    def set_status(self, status: str, **fields) -> None:
        # the event first: whoever reads a final status from the summary then also finds its event
        self.emit("experiment.status", {"status": status, **fields})
        with self._lock:
            self.summary.update(status=status, **fields)
        self.save()

    def save(self) -> None:
        _write_json(self.run_dir / EXPERIMENT_JSON, self.snapshot())

    # ---- called by run_task in the worker thread ----

    def on_state(self, state: dict) -> None:
        for type_, data in state_events(self._state, state):
            self.emit(type_, data)
        self._state = state
        with self._lock:
            self.summary.update(decisions=state.get("decisions", 0), camera=list(state.get("camera", [])),
                                refused_calls=list(state.get("refused_calls", [])))
        self.save()
        if self.cancel_requested.is_set():
            raise Cancelled

    def ask(self, payload: dict) -> str:
        """The graph paused: publish the question, wait for the matching reply (or cancel), return it."""
        with self._lock:
            self._requests += 1
            request = {"request_id": f"{self.id}-q{self._requests}", **input_request(payload, self._state or {})}
            self.summary["pending"] = request
            self._reply = None
            self._replied.clear()
        self.emit("input.required", request)
        self.set_status("awaiting_input")
        while not self._replied.wait(0.2):
            if self.cancel_requested.is_set():
                raise Cancelled
        with self._lock:
            reply, self.summary["pending"] = self._reply or "", None
        resolved = {"request_id": request["request_id"], "reply": reply}
        if request["kind"] == "hard":
            resolved["accepted"] = accepted(reply)  # the same yes rule the graph applies
        self.emit("input.resolved", resolved)
        self.set_status("running")
        return reply

    # ---- called by the API ----

    def reply(self, request_id: str, reply: str) -> None:
        with self._lock:
            pending = self.summary["pending"]
            if pending is None:
                raise ConflictError("No question is waiting for a reply.")
            if pending["request_id"] != request_id:
                raise ConflictError("This question is no longer pending.")
            if pending["kind"] == "hard" and reply not in ("yes", "no"):
                raise ValueError('An approval must be exactly "yes" or "no".')
            self._reply = reply
            self._replied.set()

    def cancel(self) -> None:
        if self.status in TERMINAL:
            raise ConflictError(f"The experiment has already ended ({self.status}).")
        self.cancel_requested.set()


class ExperimentManager:
    """Starts experiments (one at a time) and finds them again, running or saved on disk."""

    def __init__(self, settings: Settings, camera_lock: threading.Lock, mode: str, llm=None):
        self.settings = settings
        self.camera_lock = camera_lock
        self.mode = mode
        self.llm = llm  # None: run_task makes the model from settings; the tests give a fake one
        self.active: Experiment | None = None  # the running experiment, or the last one of this process
        self._experiments: dict[str, Experiment] = {}  # experiments of this process

    def start(self, question: str) -> Experiment:
        if not self.camera_lock.acquire(blocking=False):
            raise ConflictError("The camera is busy (an experiment or a camera check is running).")
        try:
            run_dir = runner.new_run_dir(self.settings.runs_dir)
        except FileExistsError:
            self.camera_lock.release()
            raise ConflictError("An experiment was started in this same second; try again.") from None
        (run_dir / TASK_FILE).write_text(question + "\n")
        exp = Experiment(run_dir, question, self.mode, self.settings.model)
        self._experiments[exp.id] = exp
        self.active = exp
        exp.set_status("running")
        threading.Thread(target=self._run, args=(exp,), name=f"experiment-{exp.id}", daemon=True).start()
        return exp

    def _run(self, exp: Experiment) -> None:
        try:
            outcome = runner.run_task(exp.run_dir / TASK_FILE, self.settings, llm=self.llm, ask=exp.ask, camera=True,
                               run_dir=exp.run_dir, on_state=exp.on_state)
            stop = outcome.state.get("stop_reason") or ""
            # "completed" only when the finish guard accepted real evidence; any other end is "stopped"
            exp.set_status("completed" if stop == "finish" else "stopped", stop_reason=stop, finished=_now())
        except Cancelled:
            exp.set_status("cancelled", stop_reason="cancelled by the operator", finished=_now(), pending=None)
        except Exception as e:  # model, MCP or camera failure: record it, never kill the server
            exp.set_status("failed", error=f"{type(e).__name__}: {e}"[:MAX_ERROR_CHARS], finished=_now(),
                           pending=None)
        finally:
            self.camera_lock.release()

    def get(self, experiment_id: str) -> Experiment | None:
        """A running or recent experiment of this process; None if unknown here (it may be on disk)."""
        return self._experiments.get(experiment_id)

    def summary(self, experiment_id: str) -> dict | None:
        """The summary of any experiment, from memory or from runs/<id>/experiment.json; None if there is none."""
        if not ID_PATTERN.fullmatch(experiment_id):
            return None
        exp = self.get(experiment_id)
        if exp is not None:
            return exp.snapshot()
        path = self.settings.runs_dir / experiment_id / EXPERIMENT_JSON
        if not path.is_file():
            return None
        summary = json.loads(path.read_text())
        if summary["status"] not in TERMINAL:  # the backend stopped during the run: say so, never "running"
            summary.update(status="failed", error="The backend stopped while this experiment was running.",
                           pending=None)
        return summary

    def events(self, experiment_id: str) -> list[dict]:
        """Every event of an experiment in order (from memory, or from runs/<id>/events.jsonl)."""
        exp = self.get(experiment_id)
        if exp is not None:
            return exp.events_after(0)
        path = self.settings.runs_dir / experiment_id / EVENTS_JSONL
        if not ID_PATTERN.fullmatch(experiment_id) or not path.is_file():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def list(self) -> list[dict]:
        """Short summaries of every saved experiment, newest first."""
        out = []
        for path in sorted(self.settings.runs_dir.glob(f"*/{EXPERIMENT_JSON}"), reverse=True):
            s = self.summary(path.parent.name)
            if s:
                out.append({k: s.get(k) for k in ("id", "question", "mode", "created", "finished", "status",
                                                  "stop_reason", "error", "decisions")})
        return out
