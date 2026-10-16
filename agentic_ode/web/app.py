"""The EventLab web API: health, camera status and check, and experiments (start, follow, answer, cancel).

- GET  /api/health                              the server is up
- GET  /api/camera/status                       what we know WITHOUT opening the camera: mode, SDK, busy, last check
- POST /api/camera/check                        open the camera through the normal MCP session, camera_info, close
- POST /api/experiments                         start one experiment (a camera run of the existing graph)
- GET  /api/experiments                         saved experiments, newest first, and the active one
- GET  /api/experiments/{id}                    summary + every event (to rebuild the page after a reload)
- GET  /api/experiments/{id}/events?after=N     Server-Sent Events: the events after seq N, then live until the end
- POST /api/experiments/{id}/reply              the answer to the question that waits (request_id must match)
- POST /api/experiments/{id}/cancel             stop the experiment (it stops at the next step)
- GET  /api/experiments/{id}/recordings/{rid}/preview   image + rate of a recording THIS experiment saved (file only)
- GET  /api/experiments/{id}/download/{name}   transcript.md, transcript.json, experiment.json or events.jsonl

Rules:
- The camera is never opened by itself: not at start, not by /status, never by a timer.
- One owner at a time: camera_lock. A check or an experiment takes it; the other gets 409.
- The API never calls a camera tool for an experiment: the graph does, and the graph asks for approval (hw_approve).
  An approval here is only the reply "yes" or "no" to the graph's own question.
- Real or simulated is read from EVCAM_FAKE once, when the app is made; the MCP server process gets the same
  environment, so both sides agree. It never switches while the server runs.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, StringConstraints

from agentic_ode import __version__
from agentic_ode.core.config import Settings
from agentic_ode.hardware.devices.evcam import store
from agentic_ode.mcp.client import evcam_session
from agentic_ode.web.experiments import TERMINAL, ConflictError, ExperimentManager
from agentic_ode.web.recordings import PreviewCache, recordings_of

SSE_POLL_S = 0.25  # how often the event stream looks for new events
SSE_PING_S = 15.0  # a comment line now and then keeps proxies from closing a quiet stream
# the files of a run folder the browser may download; nothing else in runs/<id>/ (e.g. the executor's scripts)
DOWNLOADS = {"transcript.md": "text/markdown", "transcript.json": "application/json",
             "experiment.json": "application/json", "events.jsonl": "application/x-ndjson"}


class StartRequest(BaseModel):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=4000)]


class ReplyRequest(BaseModel):
    request_id: Annotated[str, StringConstraints(max_length=64)]
    reply: Annotated[str, StringConstraints(strip_whitespace=True, max_length=2000)]


def create_app(settings: Settings | None = None, llm=None, recordings_dir: Path | None = None) -> FastAPI:
    """A new app with its own camera lock, last check and experiments (each test gets a fresh one).
    settings: default Settings.from_env() (.env: MODEL, key, runs dir). llm: a fake model in the tests.
    recordings_dir: where the MCP server saves recordings (default runs/evcam/, the server's own default)."""
    settings = settings or Settings.from_env()
    app = FastAPI(title="EventLab", version=__version__)
    mode = "simulated" if os.environ.get("EVCAM_FAKE") == "1" else "real"
    camera_lock = threading.Lock()
    experiments = ExperimentManager(settings, camera_lock, mode, llm)
    previews = PreviewCache(recordings_dir or store.DEFAULT_ROOT)
    last_check: dict | None = None  # result of the last POST /api/camera/check; None = never checked

    # ---- health and camera ----

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/api/camera/status")
    def camera_status() -> dict:
        """Never opens the camera. find_spec only looks for the SDK module; it does not import it."""
        return {
            "mode": mode,
            "sdk_available": importlib.util.find_spec("metavision_hal") is not None,
            "busy": camera_lock.locked(),
            "last_check": last_check,
        }

    @app.post("/api/camera/check")
    def camera_check() -> dict:
        """A plain def: FastAPI runs it in a worker thread, so the ~10 s of the real camera block no other request."""
        nonlocal last_check
        if not camera_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="The camera is busy; it was not opened.")
        try:
            checked_at = datetime.now().astimezone().isoformat(timespec="seconds")
            try:
                with evcam_session() as session:  # closing the session stops the server and releases the camera
                    result = session.call("camera_info")
            except Exception as e:  # e.g. the server did not start: report it, never crash
                result = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            ok = bool(result.get("ok"))
            info = {k: v for k, v in result.items() if k != "ok"}
            last_check = {"checked_at": checked_at, "ok": ok,
                          **({"result": info} if ok else {"error": str(result.get("error", "unknown error"))})}
            return last_check
        finally:
            camera_lock.release()

    # ---- experiments ----

    def found(experiment_id: str) -> dict:
        summary = experiments.summary(experiment_id)
        if summary is None:
            raise HTTPException(status_code=404, detail=f"No experiment {experiment_id}.")
        return summary

    def running(experiment_id: str):
        found(experiment_id)
        exp = experiments.get(experiment_id)
        if exp is None or exp.status in TERMINAL:
            raise HTTPException(status_code=409, detail="This experiment is not running.")
        return exp

    @app.post("/api/experiments", status_code=202)
    def start_experiment(body: StartRequest) -> dict:
        try:
            exp = experiments.start(body.question)
        except ConflictError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        return {"id": exp.id, "status": exp.status}

    @app.get("/api/experiments")
    def list_experiments() -> dict:
        active = experiments.active
        return {"active": active.id if active and active.status not in TERMINAL else None,
                "experiments": experiments.list()}

    @app.get("/api/experiments/{experiment_id}")
    def get_experiment(experiment_id: str) -> dict:
        return {"experiment": found(experiment_id), "events": experiments.events(experiment_id)}

    @app.post("/api/experiments/{experiment_id}/reply")
    def reply(experiment_id: str, body: ReplyRequest) -> dict:
        exp = running(experiment_id)
        try:
            exp.reply(body.request_id, body.reply)
        except ConflictError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        return {"ok": True}

    @app.post("/api/experiments/{experiment_id}/cancel", status_code=202)
    def cancel(experiment_id: str) -> dict:
        exp = running(experiment_id)
        try:
            exp.cancel()
        except ConflictError as e:
            raise HTTPException(status_code=409, detail=str(e)) from None
        return {"ok": True, "detail": "Cancel requested: the run stops after the current step."}

    @app.get("/api/experiments/{experiment_id}/recordings/{recording_id}/preview")
    def recording_preview(experiment_id: str, recording_id: str) -> dict:
        """Only a recording that this experiment's own record calls returned: one experiment never shows another's
        data. A plain def: a large real recording takes a moment to read."""
        if recording_id not in recordings_of(found(experiment_id)["camera"]):
            raise HTTPException(status_code=404, detail=f"This experiment made no recording {recording_id}.")
        try:
            return previews.get(recording_id)
        except (FileNotFoundError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e)) from None

    @app.get("/api/experiments/{experiment_id}/download/{name}")
    def download(experiment_id: str, name: str) -> FileResponse:
        found(experiment_id)  # also checks the id: only digits and one "-", never a path
        path = settings.runs_dir / experiment_id / name
        if name not in DOWNLOADS or not path.is_file():
            raise HTTPException(status_code=404, detail=f"No file {name} for experiment {experiment_id}.")
        return FileResponse(path, media_type=DOWNLOADS[name], filename=f"eventlab-{experiment_id}-{name}")

    @app.get("/api/experiments/{experiment_id}/events")
    async def stream_events(experiment_id: str, request: Request, after: int = 0,
                            last_event_id: Annotated[str | None, Header()] = None) -> StreamingResponse:
        found(experiment_id)
        if last_event_id and last_event_id.isdigit():  # EventSource sends it by itself when it reconnects
            after = max(after, int(last_event_id))

        async def stream():
            sent, pinged = max(after, 0), time.monotonic()
            while not await request.is_disconnected():
                ended = experiments.summary(experiment_id)["status"] in TERMINAL  # read BEFORE the events
                for event in experiments.events(experiment_id)[sent:]:
                    yield f"id: {event['seq']}\ndata: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"
                    sent = event["seq"]
                if ended:
                    return  # the final status event was emitted before the status changed, so it was sent
                if time.monotonic() - pinged > SSE_PING_S:
                    pinged = time.monotonic()
                    yield ": ping\n\n"
                await asyncio.sleep(SSE_POLL_S)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    app.state.camera_lock = camera_lock  # the tests use the same lock
    app.state.experiments = experiments
    return app
