// LUMEN Dual-Vision client (DV-UI-02): two local lumen.flicker.v1 files -> POST /api/dualvision/compare.
// Pure: no React, no DOM, no file system. fetch is looked up only when a comparison is asked for (or injected by
// the tests), so importing this module sends nothing. The comparison itself is the backend's (ingest.py and
// comparison.py): this file encodes the bytes, checks the shape of the answer and turns errors into fixed texts.
// The ".ts"-free imports elsewhere work in Vite; Node runs this file directly in web/tests/.

export const COMPARE_PATH = "/api/dualvision/compare";
export const MAX_FILE_BYTES = 2 * 1024 * 1024; // = ingest.MAX_INPUT_BYTES

export type Side = "frame" | "event";
export const SIDE_LABEL: Record<Side, string> = { frame: "Frame Lab", event: "Event Lab" };

// ---- the response (agentic_ode/dualvision/ingest.py compare_json_bytes, comparison.py assess and compare) ----

export type Eligibility = "valid" | "provisional" | "not_recoverable" | "invalid";
export type PairingStatus = "paired" | "unpaired" | "trial_mismatch";
export type ReferenceCheck = "match" | "mismatch" | "unknown";

export interface NormalizedWarning {
  code: string | null;
  text: string;
  significant: boolean;
}

export interface LabAcquisition {
  request_mode: "duration" | "frame_count" | null;
  requested_s: number | null;
  requested_frames: number | null;
  request_completed: boolean | null;
  actual_s: number | null;
  samples: number | null;
  sample_kind: string | null;
  fps: number | null;
  exposure_us: number | null;
  early_stop: boolean | null;
  nyquist_hz: number | null;
  expected_alias_hz: number | null;
}

export interface FrequencyError {
  signed_hz: number;
  abs_hz: number;
  relative: number;
  within_resolution: boolean | null;
}

export interface LabAssessment {
  lab: Side;
  run_id: string | null;
  recording_id: string | null;
  trial_id: string | null;
  hardware: { model: string | boolean | null; sensor: string | boolean | null; real: string | boolean | null };
  reference_hz: number | null;
  acquisition_status: string | null;
  acquisition: LabAcquisition;
  f0_hz: number | null;
  f0_status: "measured" | "not_detected" | "invalid";
  resolution_hz: number | null;
  frequency_error: FrequencyError | null;
  harmonic_relation: { relation: string; k: number } | null;
  eligibility: Eligibility;
  reasons: string[];
  problems: string[];
  warnings: NormalizedWarning[];
}

export interface DualVisionComparison {
  schema: "lumen.dualvision.comparison.v1";
  reference_hz: number | null;
  reference_check: ReferenceCheck;
  pairing: { status: PairingStatus; frame_trial_id: string | null; event_trial_id: string | null; basis: string };
  frame: LabAssessment;
  event: LabAssessment;
  error_evidence: "paired_trial" | "unpaired_evidence" | "not_computed";
  snr_comparable: false;
  comparison_valid: boolean;
  notes: string[];
  limitations: string[];
}

export type DeclaredAcquisition = "real_hardware" | "not_real_hardware" | "not_declared";

export interface DualVisionInput {
  bytes: number;
  sha256: string;
  acquisition: { declared: DeclaredAcquisition; basis: string };
}

export interface DualVisionResult {
  schema: "lumen.dualvision.ingest.v1";
  presentation: { mode: "recorded_data"; live_session: false; note: string };
  inputs: { frame: DualVisionInput; event: DualVisionInput };
  comparison: DualVisionComparison;
}

// ---- errors: every message is a fixed text of this file, never server or file content ----

export type DualVisionErrorKind =
  | "no_file" // a side has no file selected
  | "empty" // a file has no bytes
  | "too_large" // a file is over MAX_FILE_BYTES (refused before it is read or sent)
  | "offline" // the backend could not be reached
  | "bad_request" // 400
  | "payload_too_large" // 413
  | "unsupported_media" // 415
  | "invalid_input" // 422: ingest refused a file (side and code say which and why)
  | "server" // any other HTTP status
  | "invalid_response" // 200 but not the expected shape
  | "aborted"; // cancelled or replaced by a newer request

export class DualVisionError extends Error {
  readonly kind: DualVisionErrorKind;
  readonly status: number | null;
  readonly side: Side | null;
  readonly code: string | null;

  constructor(kind: DualVisionErrorKind, message: string, extra: { status?: number; side?: Side; code?: string } = {}) {
    super(message);
    this.name = "DualVisionError";
    this.kind = kind;
    this.status = extra.status ?? null;
    this.side = extra.side ?? null;
    this.code = extra.code ?? null;
  }
}

const FLICKER_REQUIRED =
  "A lumen.flicker.v1 record is required. A Frame Lab frame-series diagnostic (lumen.frame_series_probe.v1) " +
  "must first be converted with the Frame Lab exporter.";

/** The fixed text for an ingest error code (ingest.py IngestError.code). */
function invalidInputMessage(side: Side, code: string | null): string {
  const who = `The ${SIDE_LABEL[side]} file`;
  switch (code) {
    case "wrong_schema":
      return `${who} is not a lumen.flicker.v1 record. ${FLICKER_REQUIRED}`;
    case "wrong_lab":
      return `${who} is a lumen.flicker.v1 record of the other laboratory, or has no lab. Check which file goes where.`;
    case "malformed_json":
      return `${who} is not strict JSON (one object, no NaN or Infinity, no duplicate keys).`;
    case "not_utf8":
      return `${who} is not UTF-8 text.`;
    case "not_object":
      return `${who} is not a JSON object. ${FLICKER_REQUIRED}`;
    case "too_large":
      return `${who} is larger than 2 MiB.`;
    default:
      return `${who} was not accepted by the backend.`;
  }
}

/** Throws a DualVisionError if bytes cannot be sent: empty, or over 2 MiB. */
export function checkFileSize(side: Side, size: number): void {
  if (size === 0) throw new DualVisionError("empty", `The ${SIDE_LABEL[side]} file is empty.`, { side });
  if (size > MAX_FILE_BYTES) {
    throw new DualVisionError("too_large", `The ${SIDE_LABEL[side]} file is larger than 2 MiB.`, { side });
  }
}

// ---- base64 (RFC 4648, standard alphabet, padded); a loop, so no argument-count or stack limit ----

const ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

export function bytesToBase64(bytes: Uint8Array): string {
  const out: string[] = [];
  let i = 0;
  for (; i + 2 < bytes.length; i += 3) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8) | bytes[i + 2];
    out.push(ALPHABET[n >> 18], ALPHABET[(n >> 12) & 63], ALPHABET[(n >> 6) & 63], ALPHABET[n & 63]);
  }
  const rest = bytes.length - i;
  if (rest === 1) {
    const n = bytes[i] << 16;
    out.push(ALPHABET[n >> 18], ALPHABET[(n >> 12) & 63], "==");
  } else if (rest === 2) {
    const n = (bytes[i] << 16) | (bytes[i + 1] << 8);
    out.push(ALPHABET[n >> 18], ALPHABET[(n >> 12) & 63], ALPHABET[(n >> 6) & 63], "=");
  }
  return out.join("");
}

// ---- checks of the response shape (what the panel reads; anything else is refused) ----

type Obj = Record<string, unknown>;
const isObj = (v: unknown): v is Obj => typeof v === "object" && v !== null && !Array.isArray(v);
const isNum = (v: unknown) => typeof v === "number" && Number.isFinite(v);
const isNumOrNull = (v: unknown) => v === null || isNum(v);
const isStrOrNull = (v: unknown) => v === null || typeof v === "string";
const isBoolOrNull = (v: unknown) => v === null || typeof v === "boolean";
const isStrings = (v: unknown) => Array.isArray(v) && v.every((x) => typeof x === "string");
const oneOf = (v: unknown, values: readonly unknown[]) => values.includes(v);

function isWarning(v: unknown): boolean {
  return isObj(v) && isStrOrNull(v.code) && typeof v.text === "string" && typeof v.significant === "boolean";
}

function isAcquisition(v: unknown): boolean {
  if (!isObj(v)) return false;
  const numbers = ["requested_s", "requested_frames", "actual_s", "samples", "fps", "exposure_us", "nyquist_hz",
    "expected_alias_hz"];
  return (
    oneOf(v.request_mode, ["duration", "frame_count", null]) &&
    numbers.every((k) => isNumOrNull(v[k])) &&
    isBoolOrNull(v.request_completed) &&
    isBoolOrNull(v.early_stop) &&
    isStrOrNull(v.sample_kind)
  );
}

function isFrequencyError(v: unknown): boolean {
  return v === null || (isObj(v) && isNum(v.signed_hz) && isNum(v.abs_hz) && isNum(v.relative) &&
    isBoolOrNull(v.within_resolution));
}

function isAssessment(v: unknown, lab: Side): boolean {
  return (
    isObj(v) &&
    v.lab === lab &&
    isStrOrNull(v.run_id) &&
    isStrOrNull(v.recording_id) &&
    isStrOrNull(v.trial_id) &&
    isObj(v.hardware) &&
    isNumOrNull(v.reference_hz) &&
    isStrOrNull(v.acquisition_status) &&
    isAcquisition(v.acquisition) &&
    isNumOrNull(v.f0_hz) &&
    oneOf(v.f0_status, ["measured", "not_detected", "invalid"]) &&
    isNumOrNull(v.resolution_hz) &&
    isFrequencyError(v.frequency_error) &&
    (v.harmonic_relation === null || (isObj(v.harmonic_relation) && isNum(v.harmonic_relation.k))) &&
    oneOf(v.eligibility, ["valid", "provisional", "not_recoverable", "invalid"]) &&
    isStrings(v.reasons) &&
    isStrings(v.problems) &&
    Array.isArray(v.warnings) &&
    v.warnings.every(isWarning)
  );
}

function isInput(v: unknown): boolean {
  return (
    isObj(v) &&
    isNum(v.bytes) &&
    typeof v.sha256 === "string" &&
    isObj(v.acquisition) &&
    oneOf(v.acquisition.declared, ["real_hardware", "not_real_hardware", "not_declared"]) &&
    typeof v.acquisition.basis === "string"
  );
}

export function isDualVisionResult(v: unknown): v is DualVisionResult {
  if (!isObj(v) || v.schema !== "lumen.dualvision.ingest.v1") return false;
  const p = v.presentation;
  if (!isObj(p) || p.mode !== "recorded_data" || p.live_session !== false || typeof p.note !== "string") return false;
  if (!isObj(v.inputs) || !isInput(v.inputs.frame) || !isInput(v.inputs.event)) return false;
  const c = v.comparison;
  return (
    isObj(c) &&
    c.schema === "lumen.dualvision.comparison.v1" &&
    isNumOrNull(c.reference_hz) &&
    oneOf(c.reference_check, ["match", "mismatch", "unknown"]) &&
    isObj(c.pairing) &&
    oneOf(c.pairing.status, ["paired", "unpaired", "trial_mismatch"]) &&
    isStrOrNull(c.pairing.frame_trial_id) &&
    isStrOrNull(c.pairing.event_trial_id) &&
    isAssessment(c.frame, "frame") &&
    isAssessment(c.event, "event") &&
    oneOf(c.error_evidence, ["paired_trial", "unpaired_evidence", "not_computed"]) &&
    c.snr_comparable === false &&
    typeof c.comparison_valid === "boolean" &&
    isStrings(c.notes) &&
    isStrings(c.limitations)
  );
}

// ---- the request ----

export interface CompareOptions {
  fetch?: typeof fetch; // a mock in the tests; default: the browser's fetch, looked up at call time
  signal?: AbortSignal;
}

const isAbort = (e: unknown) => typeof e === "object" && e !== null && (e as { name?: unknown }).name === "AbortError";

/** Compare two files (their exact bytes). The bytes are only read. Resolves with the backend's result, unchanged,
 * or rejects with a DualVisionError whose message is safe to show. */
export async function compareRecordings(
  frame: Uint8Array,
  event: Uint8Array,
  options: CompareOptions = {},
): Promise<DualVisionResult> {
  checkFileSize("frame", frame.length);
  checkFileSize("event", event.length);
  const body = JSON.stringify({ frame_b64: bytesToBase64(frame), event_b64: bytesToBase64(event) });
  const doFetch: typeof fetch = options.fetch ?? ((input, init) => globalThis.fetch(input, init)); // never unbound

  let res: Response;
  try {
    res = await doFetch(COMPARE_PATH, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body,
      signal: options.signal,
    });
  } catch (e) {
    if (isAbort(e) || options.signal?.aborted) throw new DualVisionError("aborted", "The comparison was cancelled.");
    throw new DualVisionError("offline", "The EventLab backend could not be reached. Is it running?");
  }

  let data: unknown = null;
  try {
    data = await res.json();
  } catch (e) {
    if (isAbort(e) || options.signal?.aborted) throw new DualVisionError("aborted", "The comparison was cancelled.");
    data = null; // not JSON, e.g. an HTML error page
  }

  const status = res.status;
  if (res.ok) {
    if (status === 200 && isDualVisionResult(data)) return data;
    throw new DualVisionError("invalid_response", "The backend sent an unexpected comparison result.", { status });
  }
  if (status === 400) {
    throw new DualVisionError("bad_request", "The backend refused the request as malformed.", { status });
  }
  if (status === 413) {
    throw new DualVisionError("payload_too_large", "The files are too large for the backend (2 MiB each).", { status });
  }
  if (status === 415) {
    throw new DualVisionError("unsupported_media", "The backend refused the request format.", { status });
  }
  if (status === 422) {
    const side = isObj(data) && (data.side === "frame" || data.side === "event") ? data.side : null;
    const code = isObj(data) && typeof data.code === "string" && /^[a-z_]{1,32}$/.test(data.code) ? data.code : null;
    if (side === null) throw new DualVisionError("invalid_input", "A file was not accepted by the backend.", { status });
    throw new DualVisionError("invalid_input", invalidInputMessage(side, code), { status, side, code: code ?? undefined });
  }
  if (status >= 502 && status <= 504) {
    throw new DualVisionError("offline", "The EventLab backend could not be reached. Is it running?", { status });
  }
  throw new DualVisionError("server", `The backend answered with HTTP ${status}.`, { status });
}
