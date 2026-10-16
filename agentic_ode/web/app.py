"""The EventLab web API (M1): health, camera status, and an explicit camera check.

- GET  /api/health         the server is up
- GET  /api/camera/status  what we know WITHOUT opening the camera: mode, SDK found, busy, the last check
- POST /api/camera/check   open the camera through the normal MCP session, call camera_info, close it again

Rules:
- The camera is never opened by itself: not at start, not by /status, never by a timer. Only /check opens it.
- One owner at a time: camera_lock. /check returns 409 if the lock is taken (later, a running experiment takes it).
- Real or simulated is read from EVCAM_FAKE once, when the app is made; the MCP server process gets the same
  environment, so both sides agree. It never switches while the server runs.
"""

from __future__ import annotations

import importlib.util
import os
import threading
from datetime import datetime

from fastapi import FastAPI, HTTPException

from agentic_ode import __version__
from agentic_ode.mcp.client import evcam_session


def create_app() -> FastAPI:
    """A new app with its own camera lock and last check (each test gets a fresh one)."""
    app = FastAPI(title="EventLab", version=__version__)
    mode = "simulated" if os.environ.get("EVCAM_FAKE") == "1" else "real"
    camera_lock = threading.Lock()
    last_check: dict | None = None  # result of the last POST /api/camera/check; None = never checked

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

    app.state.camera_lock = camera_lock  # the tests (and M3 experiments) use the same lock
    return app
