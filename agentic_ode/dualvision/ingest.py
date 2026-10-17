"""Offline ingestion of two lumen.flicker.v1 JSON documents (DV-H04): bytes in, a sanitized comparison out.

The caller gives the bytes of each document (later: a file the operator chose in the browser). This module never
reads a path, never looks for recordings and never opens a camera, an MCP server, a model or the network. The science
is done by comparison.compare(), unchanged.

Each input is checked in this order; the first failure raises IngestError and nothing is compared (fail closed):
bytes or bytearray -> at most MAX_INPUT_BYTES -> strict UTF-8 -> strict JSON (one value, no NaN or Infinity, no
duplicate keys) -> a JSON object -> schema lumen.flicker.v1 -> the lab of its slot ("frame" or "event").
Any other defect (a missing field, a bad number, a malformed warning) is not an ingest error: compare() marks that
record invalid and lists the problems, so the operator sees them.

Error messages are fixed texts: they never quote the input, and the decoder's exception (which holds the input) is
not chained. The output never copies the input: only compare()'s public summary (no hardware.serial, no extra fields
such as raw events or images), the size and SHA-256 of each input, and the provenance context below.

Provenance: a saved JSON may describe a measurement made on real hardware, but showing it later is a recorded-data
view, never a live session. "presentation" says how the result is shown now (always recorded data, no camera);
"inputs.<lab>.acquisition" says what the record itself declares about how it was measured (only hardware.real
today; "not_declared" when it is missing or not true/false).
"""

from __future__ import annotations

import hashlib
import json

from agentic_ode.dualvision.comparison import SCHEMA, compare

INGEST_SCHEMA = "lumen.dualvision.ingest.v1"
MAX_INPUT_BYTES = 2 * 1024 * 1024  # 2 MiB per document; a flicker result is about 1 kB
PRESENTATION = {
    "mode": "recorded_data",
    "live_session": False,
    "note": "saved results shown later: no camera was contacted, and the two cameras were not synchronised",
}


class IngestError(ValueError):
    """An input that is not accepted. side: "frame" or "event"; code: a short fixed reason. Never input content."""

    def __init__(self, side: str, code: str, message: str):
        super().__init__(f"{side} input {message}")
        self.side = side
        self.code = code


def _reject_constant(name: str):
    """NaN, Infinity and -Infinity are accepted by Python's json but are not JSON."""
    raise ValueError("non-standard constant")


def _object_without_duplicates(pairs: list) -> dict:
    """A JSON object; two equal keys (e.g. two "lab") would make the record ambiguous."""
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key")
    return result


def _load(data, side: str) -> tuple[dict, dict]:
    """The record in data and {"bytes", "sha256"} of data, or IngestError. data is only read, never changed."""
    if not isinstance(data, (bytes, bytearray)):
        raise IngestError(side, "not_bytes", "must be bytes (a path or a text is not accepted)")
    if len(data) > MAX_INPUT_BYTES:
        raise IngestError(side, "too_large", f"is larger than {MAX_INPUT_BYTES} bytes")
    data = bytes(data)  # a copy: a bytearray changed later by the caller does not change this result
    record = error = None
    try:
        record = json.loads(data.decode("utf-8"), parse_constant=_reject_constant,
                            object_pairs_hook=_object_without_duplicates)
    except UnicodeDecodeError:
        error = ("not_utf8", "is not valid UTF-8")
    except (ValueError, RecursionError):  # JSONDecodeError is a ValueError; very deep nesting is a RecursionError
        error = ("malformed_json", "is not strict JSON (one value, no NaN or Infinity, no duplicate keys)")
    if error:  # raised here, outside except: the decoder's exception is not attached to IngestError
        raise IngestError(side, *error)
    if not isinstance(record, dict):
        raise IngestError(side, "not_object", "is not a JSON object")
    if record.get("schema") != SCHEMA:
        raise IngestError(side, "wrong_schema", f"is not a {SCHEMA} record")
    if record.get("lab") != side:
        raise IngestError(side, "wrong_lab", f"is not a {side} lab record")
    return record, {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def _acquisition(record: dict) -> dict:
    """What the record declares about its measurement: only hardware.real. Nothing is guessed when it is absent."""
    hardware = record.get("hardware")
    real = hardware.get("real") if isinstance(hardware, dict) else None
    if real is True:
        declared = "real_hardware"
    elif real is False:
        declared = "not_real_hardware"  # e.g. a simulation; the record does not say which
    else:
        declared = "not_declared"
    return {"declared": declared, "basis": "hardware.real as written in the record"}


def compare_json_bytes(frame_json: bytes, event_json: bytes) -> dict:
    """Compare a Frame Lab and an Event Lab lumen.flicker.v1 document, each given as bytes. Raises IngestError for an
    input that is not accepted (the frame input is checked first). Returns a JSON-compatible dict: schema, how it is
    presented (recorded data), size, SHA-256 and declared acquisition of each input, and compare()'s result."""
    frame, frame_input = _load(frame_json, "frame")
    event, event_input = _load(event_json, "event")
    return {
        "schema": INGEST_SCHEMA,
        "presentation": dict(PRESENTATION),
        "inputs": {
            "frame": {**frame_input, "acquisition": _acquisition(frame)},
            "event": {**event_input, "acquisition": _acquisition(event)},
        },
        "comparison": compare(frame, event),
    }
