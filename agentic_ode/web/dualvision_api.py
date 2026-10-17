"""LUMEN Dual-Vision HTTP route (DV-UI-01): POST /api/dualvision/compare, read only.

The browser sends ONE application/json body: {"frame_b64": "<base64>", "event_b64": "<base64>"}, the exact bytes of
a Frame Lab and an Event Lab lumen.flicker.v1 file. Base64 keeps those bytes unchanged, so the strict checks of
ingest.compare_json_bytes() (UTF-8, duplicate keys, NaN, size, SHA-256) see the original files. That function is the
only scientific authority: this route only bounds, unpacks and passes the bytes on, and returns its result unchanged.

Checks, in order (the first failure answers; nothing is compared):
    415  Content-Type is not application/json (an optional charset must be UTF-8)
    400  Content-Length is not a whole number
    413  Content-Length is over MAX_BODY_BYTES: the body is never read
    413  the body grows over MAX_BODY_BYTES while it is read (also without Content-Length): reading stops at once
    400  the envelope is not strict UTF-8 JSON (one object, no NaN or Infinity, no duplicate keys)
    400  the envelope has not exactly the keys frame_b64 and event_b64, or a value is not a string
    413  a base64 value is longer than MAX_ENCODED_CHARS (checked before decoding)
    400  a base64 value is not valid base64 (validate mode: no other characters, correct padding)
    422  compare_json_bytes() raises IngestError: {"detail", "side", "code"}, its fixed texts only
    200  the lumen.dualvision.ingest.v1 result. A record with field defects is still 200: the comparator marks it
         eligibility "invalid" and lists its problems.

Error texts are fixed: they never quote the input. Nothing is written, logged or kept. The route uses no camera lock,
no settings, no file path, no recording lookup, no MCP, no model and no network: it may run during an experiment.
"""

from __future__ import annotations

import base64
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from agentic_ode.dualvision.ingest import MAX_INPUT_BYTES, IngestError, compare_json_bytes

FIELDS = frozenset({"frame_b64", "event_b64"})
MAX_ENCODED_CHARS = 4 * -(-MAX_INPUT_BYTES // 3)  # base64 of MAX_INPUT_BYTES: 4 * ceil(n / 3)
ENVELOPE_ALLOWANCE = 1024  # braces, the two keys, quotes, separators and some whitespace
MAX_BODY_BYTES = 2 * MAX_ENCODED_CHARS + ENVELOPE_ALLOWANCE

router = APIRouter(prefix="/api/dualvision")


class _Rejected(Exception):
    """A request that is not accepted: HTTP status and a fixed text (never input content)."""

    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _is_json_media_type(value: str | None) -> bool:
    """application/json, with at most a UTF-8 charset parameter (case-insensitive, optionally quoted)."""
    if not value:
        return False
    media, *params = (part.strip() for part in value.split(";"))
    if media.lower() != "application/json":
        return False
    for param in params:
        name, _, charset = param.partition("=")
        if name.strip().lower() != "charset" or charset.strip().strip('"').lower() not in ("utf-8", "utf8"):
            return False
    return True


async def _read_body(request: Request) -> bytes:
    """The body, at most MAX_BODY_BYTES. A larger declared length is refused before reading; a larger stream is
    refused as soon as the limit is passed, so an unbounded body is never buffered."""
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.strip().isdigit():
            raise _Rejected(400, "Content-Length is not a whole number.")
        if int(declared) > MAX_BODY_BYTES:
            raise _Rejected(413, f"The request body is larger than {MAX_BODY_BYTES} bytes.")
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY_BYTES:
            raise _Rejected(413, f"The request body is larger than {MAX_BODY_BYTES} bytes.")
        chunks.append(chunk)
    return b"".join(chunks)


def _reject_constant(name: str):
    raise ValueError("non-standard constant")  # NaN, Infinity, -Infinity are not JSON


def _object_without_duplicates(pairs: list) -> dict:
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate key")
    return result


def _unpack(body: bytes) -> tuple[bytes, bytes]:
    """The frame and event bytes from the envelope, or _Rejected. The decoder's exception is never shown."""
    envelope = None
    try:
        envelope = json.loads(body.decode("utf-8"), parse_constant=_reject_constant,
                              object_pairs_hook=_object_without_duplicates)
    except (ValueError, RecursionError):  # UnicodeDecodeError and JSONDecodeError are ValueErrors
        pass
    if not isinstance(envelope, dict):
        raise _Rejected(400, "The body must be one strict UTF-8 JSON object (no NaN or Infinity, no duplicate keys).")
    if set(envelope) != FIELDS or not all(isinstance(envelope[k], str) for k in FIELDS):
        raise _Rejected(400, "The body must have exactly the string fields frame_b64 and event_b64.")
    decoded = {}
    for key in ("frame_b64", "event_b64"):
        text = envelope[key]
        if len(text) > MAX_ENCODED_CHARS:
            raise _Rejected(413, f"{key} is longer than {MAX_ENCODED_CHARS} base64 characters.")
        try:
            decoded[key] = base64.b64decode(text, validate=True)
        except ValueError:  # binascii.Error, or a non-ASCII string
            raise _Rejected(400, f"{key} is not valid base64.") from None
    return decoded["frame_b64"], decoded["event_b64"]


@router.post("/compare")
async def compare_files(request: Request) -> JSONResponse:
    """Compare one Frame Lab and one Event Lab lumen.flicker.v1 file, sent as base64. Read only."""
    try:
        if not _is_json_media_type(request.headers.get("content-type")):
            raise _Rejected(415, "Content-Type must be application/json.")
        frame, event = _unpack(await _read_body(request))
    except _Rejected as e:
        return JSONResponse(status_code=e.status, content={"detail": e.detail})
    try:
        result = await run_in_threadpool(compare_json_bytes, frame, event)
    except IngestError as e:  # its message is a fixed text that never quotes the input
        return JSONResponse(status_code=422, content={"detail": str(e), "side": e.side, "code": e.code})
    return JSONResponse(content=result)
