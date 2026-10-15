"""Save and load recordings: one .npz file per recording in runs/evcam/ (git-ignored).

The file format is the same as in scripts/evcam_capture.py: `events` (EVENT_DTYPE) and `meta` (one JSON string).
"""

from __future__ import annotations

import json
import re
import secrets
from datetime import datetime
from pathlib import Path

import numpy as np

from agentic_ode.config import PROJECT_ROOT
from agentic_ode.hardware.evcam.connector import EVENT_DTYPE

DEFAULT_ROOT = PROJECT_ROOT / "runs" / "evcam"
# e.g. 20261015-143012-a3f9c1: date-time plus 6 random hex characters. Only these characters are allowed, so an id
# from outside (later: from the model, through MCP) can never be a path like ../../x.
ID_PATTERN = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")


def save(events: np.ndarray, meta: dict, root: Path = DEFAULT_ROOT) -> str:
    """Save events and meta as <root>/<recording_id>.npz and return the recording_id.

    meta must be JSON-serialisable; recording_id and n_events are added to it.
    """
    if events.dtype != EVENT_DTYPE:
        raise ValueError(f"events must have EVENT_DTYPE {EVENT_DTYPE}, got {events.dtype}")
    recording_id = f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(3)}"  # token_hex(3) = 6 hex characters
    meta = {**meta, "recording_id": recording_id, "n_events": int(events.size)}  # a new dict: the caller's is kept
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(root / f"{recording_id}.npz", events=events, meta=np.array(json.dumps(meta)))
    return recording_id


def load(recording_id: str, root: Path = DEFAULT_ROOT) -> tuple[np.ndarray, dict]:
    """Return (events, meta) of a recording saved by save()."""
    if not isinstance(recording_id, str) or not ID_PATTERN.fullmatch(recording_id):
        raise ValueError(f"bad recording_id {recording_id!r}; expected the form 20261015-143012-a3f9c1")
    path = Path(root) / f"{recording_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no recording {recording_id} in {root}")
    with np.load(path, allow_pickle=False) as d:  # allow_pickle=False: a file can only hold plain arrays
        return d["events"], json.loads(str(d["meta"]))
